"""Rollout storage for HIM locomotion plus controller/contact auxiliary state."""

import torch


class AdaptiveHIMRolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.controller_states = None
            self.critic_observations = None
            self.actions = None
            self.contact_targets = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.next_critic_observations = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        controller_state_shape,
        privileged_obs_shape,
        actions_shape,
        contact_target_shape,
        device="cpu",
    ):
        self.device = device
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.step = 0

        core = (num_transitions_per_env, num_envs)
        self.observations = torch.zeros(*core, *obs_shape, device=device)
        self.controller_states = torch.zeros(
            *core, *controller_state_shape, device=device
        )
        self.privileged_observations = torch.zeros(
            *core, *privileged_obs_shape, device=device
        )
        self.next_privileged_observations = torch.zeros(
            *core, *privileged_obs_shape, device=device
        )
        self.contact_targets = torch.zeros(
            *core, *contact_target_shape, device=device
        )
        self.actions = torch.zeros(*core, *actions_shape, device=device)
        self.rewards = torch.zeros(*core, 1, device=device)
        self.dones = torch.zeros(*core, 1, device=device).byte()

        self.actions_log_prob = torch.zeros(*core, 1, device=device)
        self.values = torch.zeros(*core, 1, device=device)
        self.returns = torch.zeros(*core, 1, device=device)
        self.advantages = torch.zeros(*core, 1, device=device)
        self.mu = torch.zeros(*core, *actions_shape, device=device)
        self.sigma = torch.zeros(*core, *actions_shape, device=device)

    def add_transitions(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        s = self.step
        self.observations[s].copy_(transition.observations)
        self.controller_states[s].copy_(transition.controller_states)
        self.privileged_observations[s].copy_(transition.critic_observations)
        self.next_privileged_observations[s].copy_(
            transition.next_critic_observations
        )
        self.contact_targets[s].copy_(transition.contact_targets)
        self.actions[s].copy_(transition.actions)
        self.rewards[s].copy_(transition.rewards.view(-1, 1))
        self.dones[s].copy_(transition.dones.view(-1, 1))
        self.values[s].copy_(transition.values)
        self.actions_log_prob[s].copy_(
            transition.actions_log_prob.view(-1, 1)
        )
        self.mu[s].copy_(transition.action_mean)
        self.sigma[s].copy_(transition.action_sigma)
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_not_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.rewards[step]
                + next_not_terminal * gamma * next_values
                - self.values[step]
            )
            advantage = (
                delta
                + next_not_terminal * gamma * lam * advantage
            )
            self.returns[step] = advantage + self.values[step]

        self.advantages = self.returns - self.values
        self.advantages = (
            self.advantages - self.advantages.mean()
        ) / (self.advantages.std() + 1.0e-8)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size,
            requires_grad=False,
            device=self.device,
        )

        observations = self.observations.flatten(0, 1)
        controller_states = self.controller_states.flatten(0, 1)
        critic_observations = self.privileged_observations.flatten(0, 1)
        next_critic_observations = self.next_privileged_observations.flatten(0, 1)
        contact_targets = self.contact_targets.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]
                yield (
                    observations[batch_idx],
                    controller_states[batch_idx],
                    critic_observations[batch_idx],
                    actions[batch_idx],
                    next_critic_observations[batch_idx],
                    contact_targets[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                )
