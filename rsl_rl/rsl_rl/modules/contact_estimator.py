"""Deployable contact/impact estimator for learned MC admittance.

The estimator consumes only the original HIM proprioceptive history plus the
12-D internal admittance state.  Simulator contact force is used only as the
supervised target during training.
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
        loading_loss_weight=0.5,
        max_grad_norm=10.0,
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
        layers += [nn.Linear(input_dim, int(output_dim))]
        self.encoder = nn.Sequential(*layers)

        self.force_loss_weight = float(force_loss_weight)
        self.loading_loss_weight = float(loading_loss_weight)
        self.max_grad_norm = float(max_grad_norm)
        self.learning_rate = float(learning_rate)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def _predict(self, obs_history, controller_state):
        x = torch.cat((obs_history.detach(), controller_state.detach()), dim=-1)
        # Contact magnitude and positive loading rate are non-negative. Softplus
        # avoids an estimator that spends capacity learning the sign constraint.
        return F.softplus(self.encoder(x))

    def forward(self, obs_history, controller_state):
        return self._predict(obs_history, controller_state).detach()

    def encode(self, obs_history, controller_state):
        return self._predict(obs_history, controller_state)

    def update(self, obs_history, controller_state, target, lr=None):
        if lr is not None:
            self.learning_rate = float(lr)
            for group in self.optimizer.param_groups:
                group["lr"] = self.learning_rate

        prediction = self.encode(obs_history, controller_state)
        target = target.detach()
        if target.shape[-1] != 8:
            raise RuntimeError(
                f"Expected 8-D contact target, got {tuple(target.shape)}"
            )

        force_loss = F.smooth_l1_loss(prediction[:, :4], target[:, :4])
        loading_loss = F.smooth_l1_loss(prediction[:, 4:], target[:, 4:])
        loss = (
            self.force_loss_weight * force_loss
            + self.loading_loss_weight * loading_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return force_loss.item(), loading_loss.item()
