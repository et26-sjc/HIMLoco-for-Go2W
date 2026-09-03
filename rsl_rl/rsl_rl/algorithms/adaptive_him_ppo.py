"""PPO for learned MC admittance with separate estimator optimization.

Stage 1 intentionally protects the trained HIMLoco locomotion backbone:
* the original HIM estimator is optionally frozen;
* the original 16-D actor has an independent LR scale (zero by default);
* ContactEstimator is never part of PPO's optimizer and retains its own
  supervised learning rate, independent of PPO KL scheduling;
* the adaptive motion/compliance heads, critic and action std remain trainable.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.storage.adaptive_him_rollout_storage import AdaptiveHIMRolloutStorage


class AdaptiveHIMPPO:
    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1.0e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        base_actor_lr_scale=0.0,
        update_him_estimator=False,
        device="cpu",
    ):
        self.device = device
        self.actor_critic = actor_critic.to(device)
        self.base_actor_lr_scale = float(base_actor_lr_scale)
        self.update_him_estimator = bool(update_him_estimator)

        base_actor_params = list(self.actor_critic.actor.parameters())
        adaptive_params = (
            list(self.actor_critic.motion_adapter.parameters())
            + list(self.actor_critic.compliance_head.parameters())
            + list(self.actor_critic.critic.parameters())
            + [self.actor_critic.std]
        )
        self.optimizer = optim.Adam(
            [
                {
                    "params": base_actor_params,
                    "lr": learning_rate * self.base_actor_lr_scale,
                    "lr_scale": self.base_actor_lr_scale,
                    "name": "baseline_actor",
                },
                {
                    "params": adaptive_params,
                    "lr": learning_rate,
                    "lr_scale": 1.0,
                    "name": "adaptive_policy_critic",
                },
            ]
        )
        self.transition = AdaptiveHIMRolloutStorage.Transition()
        self.storage = None

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        print(
            "Adaptive PPO parameter policy: "
            f"base_actor_lr_scale={self.base_actor_lr_scale}, "
            f"update_him_estimator={self.update_him_estimator}; "
            "contact estimator uses independent supervised-only gradients."
        )

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        controller_state_shape,
        critic_obs_shape,
        action_shape,
        contact_target_shape,
    ):
        self.storage = AdaptiveHIMRolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            controller_state_shape,
            critic_obs_shape,
            action_shape,
            contact_target_shape,
            self.device,
        )

    def act(self, obs, controller_state, critic_obs):
        self.transition.actions = self.actor_critic.act(
            obs, controller_state
        ).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.controller_states = controller_state
        self.transition.critic_observations = critic_obs
        return self.transition.actions, self.actor_critic.last_contact_estimate.detach()

    def process_env_step(
        self,
        rewards,
        dones,
        infos,
        next_critic_obs,
        contact_target,
    ):
        self.transition.next_critic_observations = next_critic_obs.clone()
        self.transition.contact_targets = contact_target.clone()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def _apply_learning_rate(self):
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate * float(
                group.get("lr_scale", 1.0)
            )

    def _adapt_learning_rate(self, mu, sigma, old_mu, old_sigma):
        if self.desired_kl is None or self.schedule != "adaptive":
            return
        with torch.inference_mode():
            kl = torch.sum(
                torch.log(sigma / old_sigma + 1.0e-5)
                + (
                    torch.square(old_sigma)
                    + torch.square(old_mu - mu)
                )
                / (2.0 * torch.square(sigma))
                - 0.5,
                axis=-1,
            )
            kl_mean = torch.mean(kl)
            if kl_mean > self.desired_kl * 2.0:
                self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
            elif 0.0 < kl_mean < self.desired_kl / 2.0:
                self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
            self._apply_learning_rate()

    def update(self):
        totals = {
            "value": 0.0,
            "surrogate": 0.0,
            "him_estimation": 0.0,
            "him_swap": 0.0,
            "contact_force": 0.0,
            "contact_loading": 0.0,
        }

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        for batch in generator:
            (
                obs_batch,
                controller_state_batch,
                critic_obs_batch,
                actions_batch,
                next_critic_obs_batch,
                contact_target_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
            ) = batch

            self.actor_critic.act(obs_batch, controller_state_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            self._adapt_learning_rate(
                mu_batch, sigma_batch, old_mu_batch, old_sigma_batch
            )

            if self.update_him_estimator:
                him_estimation, him_swap = self.actor_critic.estimator.update(
                    obs_batch,
                    next_critic_obs_batch,
                    lr=self.learning_rate,
                )
            else:
                him_estimation, him_swap = 0.0, 0.0

            # Deliberately do not pass PPO's adaptive learning rate here. Contact
            # prediction is a separate supervised problem with its own optimizer.
            contact_force, contact_loading = (
                self.actor_critic.contact_estimator.update(
                    obs_batch,
                    controller_state_batch,
                    contact_target_batch,
                )
            )

            ratio = torch.exp(
                actions_log_prob_batch
                - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(
                surrogate, surrogate_clipped
            ).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (
                    value_clipped - returns_batch
                ).pow(2)
                value_loss = torch.max(
                    value_losses, value_losses_clipped
                ).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )
            self.optimizer.zero_grad()
            loss.backward()
            ppo_params = []
            for group in self.optimizer.param_groups:
                ppo_params.extend(group["params"])
            nn.utils.clip_grad_norm_(ppo_params, self.max_grad_norm)
            self.optimizer.step()

            totals["value"] += value_loss.item()
            totals["surrogate"] += surrogate_loss.item()
            totals["him_estimation"] += him_estimation
            totals["him_swap"] += him_swap
            totals["contact_force"] += contact_force
            totals["contact_loading"] += contact_loading

        num_updates = self.num_learning_epochs * self.num_mini_batches
        for key in totals:
            totals[key] /= num_updates
        self.storage.clear()
        return totals
