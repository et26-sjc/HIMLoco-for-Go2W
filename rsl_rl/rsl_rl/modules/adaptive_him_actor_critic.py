"""HIM actor-critic extended with learned contact-aware admittance actions.

The original HIM estimator and 16-D locomotion actor are preserved. New modules:
1) a supervised ContactEstimator using deployable inputs only;
2) a zero-initialized 4-D compliance head;
3) a small zero-initialized motion adapter so the locomotion policy can learn
   limited compensation for the controller-induced leg compression.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

from .actor_critic import get_activation
from .him_estimator import HIMEstimator
from .contact_estimator import ContactEstimator


class AdaptiveHIMActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_one_step_obs,
        num_actions,
        num_policy_actions=20,
        controller_state_dim=12,
        contact_estimate_dim=8,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
        init_noise_std=1.0,
        contact_estimator_hidden_dims=(128, 64),
        contact_estimator_lr=1.0e-3,
        contact_estimator_loss_force=1.0,
        contact_estimator_loss_loading=0.5,
        motion_adapter_scale=0.05,
        compliance_init_std=0.15,
        **kwargs,
    ):
        super().__init__()

        if num_actions != 16:
            raise ValueError(
                f"Adaptive MC expects 16 physical motion actions, got {num_actions}"
            )
        if num_policy_actions != 20:
            raise ValueError(
                f"Adaptive MC v1 expects 20 policy actions, got {num_policy_actions}"
            )

        act = get_activation(activation)
        self.history_size = int(num_actor_obs / num_one_step_obs)
        self.num_actor_obs = int(num_actor_obs)
        self.num_one_step_obs = int(num_one_step_obs)
        self.num_motion_actions = int(num_actions)
        self.num_policy_actions = int(num_policy_actions)
        self.num_compliance_actions = self.num_policy_actions - self.num_motion_actions
        self.controller_state_dim = int(controller_state_dim)
        self.contact_estimate_dim = int(contact_estimate_dim)
        self.motion_adapter_scale = float(motion_adapter_scale)

        # Original HIM estimator: history -> base velocity(3) + latent(16).
        self.estimator = HIMEstimator(
            temporal_steps=self.history_size,
            num_one_step_obs=self.num_one_step_obs,
        )

        # Training target is privileged, but estimator inputs are deployable.
        self.contact_estimator = ContactEstimator(
            history_dim=self.num_actor_obs,
            controller_state_dim=self.controller_state_dim,
            output_dim=self.contact_estimate_dim,
            hidden_dims=contact_estimator_hidden_dims,
            activation=activation,
            learning_rate=contact_estimator_lr,
            force_loss_weight=contact_estimator_loss_force,
            loading_loss_weight=contact_estimator_loss_loading,
        )

        # Baseline actor shape is kept exactly compatible: [57 + 3 + 16] -> 16.
        baseline_input_dim = self.num_one_step_obs + 3 + 16
        actor_layers = [nn.Linear(baseline_input_dim, actor_hidden_dims[0]), act]
        for i in range(len(actor_hidden_dims)):
            if i == len(actor_hidden_dims) - 1:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[i], self.num_motion_actions)
                )
            else:
                actor_layers += [
                    nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]),
                    act,
                ]
        self.actor = nn.Sequential(*actor_layers)

        augmented_dim = (
            baseline_input_dim
            + self.contact_estimate_dim
            + self.controller_state_dim
        )
        self.motion_adapter = nn.Sequential(
            nn.Linear(augmented_dim, 128),
            act,
            nn.Linear(128, 64),
            act,
            nn.Linear(64, self.num_motion_actions),
        )
        self.compliance_head = nn.Sequential(
            nn.Linear(augmented_dim, 128),
            act,
            nn.Linear(128, 64),
            act,
            nn.Linear(64, self.num_compliance_actions),
        )

        # Exact zero initial contribution from both new policy heads.
        nn.init.zeros_(self.motion_adapter[-1].weight)
        nn.init.zeros_(self.motion_adapter[-1].bias)
        nn.init.zeros_(self.compliance_head[-1].weight)
        nn.init.zeros_(self.compliance_head[-1].bias)

        critic_layers = [nn.Linear(num_critic_obs, critic_hidden_dims[0]), act]
        for i in range(len(critic_hidden_dims)):
            if i == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], 1))
            else:
                critic_layers += [
                    nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]),
                    act,
                ]
        self.critic = nn.Sequential(*critic_layers)

        # Preserve baseline exploration on the first 16 dimensions but keep the
        # new compliance exploration deliberately small.
        std = torch.ones(self.num_policy_actions) * float(init_noise_std)
        std[self.num_motion_actions :] = float(compliance_init_std)
        self.std = nn.Parameter(std)
        self.distribution = None
        self.last_contact_estimate = None
        Normal.set_default_validate_args = False

        print(f"Adaptive baseline actor: {self.actor}")
        print(f"Motion adapter: {self.motion_adapter}")
        print(f"Compliance head: {self.compliance_head}")
        print(f"Contact estimator: {self.contact_estimator.encoder}")
        print(f"Adaptive critic: {self.critic}")

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        pass

    def _policy_mean(self, obs_history, controller_state):
        with torch.no_grad():
            vel, latent = self.estimator(obs_history)
            contact = self.contact_estimator(obs_history, controller_state)

        baseline_input = torch.cat(
            (obs_history[:, : self.num_one_step_obs], vel, latent), dim=-1
        )
        augmented = torch.cat(
            (baseline_input, contact, controller_state), dim=-1
        )

        base_motion = self.actor(baseline_input)
        motion_delta = self.motion_adapter(augmented)
        motion = base_motion + self.motion_adapter_scale * torch.tanh(motion_delta)

        # Raw compliance action is centered at exactly zero initially. The env
        # clips it to [0,1], so inference at initialization equals baseline.
        compliance = self.compliance_head(augmented)
        mean = torch.cat((motion, compliance), dim=-1)
        self.last_contact_estimate = contact
        return mean

    def update_distribution(self, obs_history, controller_state):
        mean = self._policy_mean(obs_history, controller_state)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, obs_history, controller_state, **kwargs):
        self.update_distribution(obs_history, controller_state)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_history, controller_state):
        return self._policy_mean(obs_history, controller_state)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)
