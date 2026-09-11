"""Deployable axial force and impact estimator for learned MC admittance.

The estimator consumes only the original HIM proprioceptive history plus the
16-D internal admittance state. Simulator contact force is used only to build
supervised targets during training. Outputs are four normalized per-leg axial
compressive forces followed by four per-leg impact logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .actor_critic import get_activation


class ContactEstimator(nn.Module):
    def __init__(
        self,
        history_dim,
        controller_state_dim,
        output_dim=8,
        hidden_dims=(128, 64),
        activation="elu",
        learning_rate=1.0e-3,
        force_loss_weight=1.0,
        impact_loss_weight=1.0,
        impact_pos_weight=3.0,
        max_grad_norm=10.0,
        initial_output_bias=-3.0,
        replay_ratio=0.0,
        replay_buffer_size=8192,
    ):
        super().__init__()
        if output_dim != 8:
            raise ValueError("MC contact estimator v1 expects 8 outputs")

        act = get_activation(activation)
        input_dim = int(history_dim) + int(controller_state_dim)
        layers = []
        for hidden in hidden_dims:
            layers += [nn.Linear(input_dim, int(hidden)), act]
            input_dim = int(hidden)
        output_layer = nn.Linear(input_dim, int(output_dim))
        # A random Softplus head centered around zero would initially predict
        # Softplus(0)=0.693, i.e. ~69 N / 6.9 kN/s with the current SI scales.
        # Start close to zero instead so baseline migration cannot accidentally
        # activate admittance before the estimator has learned useful structure.
        nn.init.zeros_(output_layer.weight)
        nn.init.constant_(output_layer.bias, float(initial_output_bias))
        layers += [output_layer]
        self.encoder = nn.Sequential(*layers)

        self.force_loss_weight = float(force_loss_weight)
        self.impact_loss_weight = float(impact_loss_weight)
        self.register_buffer(
            "impact_pos_weight", torch.tensor(float(impact_pos_weight))
        )
        self.max_grad_norm = float(max_grad_norm)
        self.learning_rate = float(learning_rate)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.replay_ratio = 0.0
        self.replay_buffer_size = int(replay_buffer_size)
        self.replay_obs_history = None
        self.replay_controller_state = None
        self.replay_target = None
        self.replay_cursor = 0
        self.set_replay_ratio(replay_ratio)

    def set_replay_ratio(self, replay_ratio):
        replay_ratio = float(replay_ratio)
        if replay_ratio < 0.0 or replay_ratio > 1.0:
            raise ValueError("ContactEstimator replay ratio must be in [0, 1]")
        self.replay_ratio = replay_ratio

    def set_replay_buffer(
        self, obs_history, controller_state, target, max_size=None
    ):
        """Install a Stage-0-only buffer for optional continual updates."""
        if (
            obs_history.shape[0] != controller_state.shape[0]
            or obs_history.shape[0] != target.shape[0]
        ):
            raise ValueError("ContactEstimator replay tensors must have equal length")
        if target.shape[-1] != 8:
            raise ValueError("ContactEstimator replay targets must be 8-D")
        limit = self.replay_buffer_size if max_size is None else int(max_size)
        if limit <= 0:
            raise ValueError("ContactEstimator replay buffer size must be positive")
        count = min(int(obs_history.shape[0]), limit)
        if count == 0:
            self.replay_obs_history = None
            self.replay_controller_state = None
            self.replay_target = None
            return
        if obs_history.shape[0] > count:
            indices = torch.linspace(
                0, obs_history.shape[0] - 1, count, dtype=torch.long
            )
            obs_history = obs_history[indices]
            controller_state = controller_state[indices]
            target = target[indices]
        self.replay_obs_history = obs_history.detach().to(self.encoder[0].weight.device)
        self.replay_controller_state = controller_state.detach().to(
            self.encoder[0].weight.device
        )
        self.replay_target = target.detach().to(self.encoder[0].weight.device)
        self.replay_cursor = 0

    def clear_replay_buffer(self):
        self.replay_obs_history = None
        self.replay_controller_state = None
        self.replay_target = None
        self.replay_cursor = 0

    def _mix_replay_batch(self, obs_history, controller_state, target):
        if self.replay_ratio <= 0.0 or self.replay_target is None:
            return obs_history, controller_state, target
        replay_count = max(1, int(round(obs_history.shape[0] * self.replay_ratio)))
        indices = (
            torch.arange(replay_count, device=self.replay_target.device)
            + self.replay_cursor
        ) % self.replay_target.shape[0]
        self.replay_cursor = (
            self.replay_cursor + replay_count
        ) % self.replay_target.shape[0]
        return (
            torch.cat((obs_history, self.replay_obs_history[indices]), dim=0),
            torch.cat(
                (controller_state, self.replay_controller_state[indices]), dim=0
            ),
            torch.cat((target, self.replay_target[indices]), dim=0),
        )

    def set_learning_rate(self, learning_rate):
        """Change only the LR of the existing optimizer, preserving Adam state."""
        learning_rate = float(learning_rate)
        if learning_rate <= 0.0:
            raise ValueError("ContactEstimator learning rate must be positive")
        self.learning_rate = learning_rate
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def _predict(self, obs_history, controller_state):
        x = torch.cat((obs_history.detach(), controller_state.detach()), dim=-1)
        raw = self.encoder(x)
        force = F.softplus(raw[:, :4])
        impact_logits = raw[:, 4:]
        return torch.cat((force, impact_logits), dim=-1)

    def forward(self, obs_history, controller_state):
        return self._predict(obs_history, controller_state).detach()

    def encode(self, obs_history, controller_state):
        return self._predict(obs_history, controller_state)

    def update(self, obs_history, controller_state, target, lr=None):
        if lr is not None:
            self.set_learning_rate(lr)

        obs_history, controller_state, target = self._mix_replay_batch(
            obs_history, controller_state, target
        )

        prediction = self.encode(obs_history, controller_state)
        target = target.detach()
        if target.shape[-1] != 8:
            raise RuntimeError(
                f"Expected 8-D contact target, got {tuple(target.shape)}"
            )

        force_loss = F.mse_loss(prediction[:, :4], target[:, :4])
        impact_loss = F.binary_cross_entropy_with_logits(
            prediction[:, 4:],
            target[:, 4:],
            pos_weight=self.impact_pos_weight,
        )
        loss = (
            self.force_loss_weight * force_loss
            + self.impact_loss_weight * impact_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return force_loss.item(), impact_loss.item()
