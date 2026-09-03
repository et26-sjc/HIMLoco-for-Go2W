"""On-policy runner for sensorless learned MC admittance."""

import os
import time
from collections import deque
import statistics

import torch
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ImportError:
    wandb = None

from rsl_rl.algorithms import AdaptiveHIMPPO
from rsl_rl.modules import AdaptiveHIMActorCritic


class AdaptiveHIMOnPolicyRunner:
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.train_cfg = train_cfg
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.log_dir = log_dir

        num_critic_obs = (
            env.num_privileged_obs
            if env.num_privileged_obs is not None
            else env.num_obs
        )
        self.num_actor_obs = env.num_obs
        self.num_critic_obs = num_critic_obs

        self.actor_critic = AdaptiveHIMActorCritic(
            env.num_obs,
            num_critic_obs,
            env.num_one_step_obs,
            env.num_actions,
            num_policy_actions=env.num_policy_actions,
            controller_state_dim=env.controller_state_dim,
            contact_estimate_dim=env.contact_estimate_dim,
            **self.policy_cfg,
        ).to(device)
        self.alg = AdaptiveHIMPPO(
            self.actor_critic, device=device, **self.alg_cfg
        )

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.alg.init_storage(
            env.num_envs,
            self.num_steps_per_env,
            [env.num_obs],
            [env.controller_state_dim],
            [num_critic_obs],
            [env.num_policy_actions],
            [env.contact_estimate_dim],
        )

        self.writer = None
        self.wandb_run = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        _, _ = self.env.reset()

    def _init_wandb(self):
        if not self.cfg.get("wandb_enabled", False) or self.wandb_run is not None:
            return
        if wandb is None:
            raise RuntimeError(
                "W&B enabled but wandb is not installed; disable it or install wandb."
            )
        run_name = self.cfg.get("run_name", "adaptive-him")
        if self.log_dir:
            run_name = os.path.basename(os.path.normpath(self.log_dir))
        self.wandb_run = wandb.init(
            project=self.cfg.get("wandb_project", "HIMLoco"),
            entity=self.cfg.get("wandb_entity", None),
            name=run_name,
            group=self.cfg.get("wandb_group", None),
            tags=self.cfg.get("wandb_tags", None),
            mode=self.cfg.get("wandb_mode", "online"),
            dir=self.log_dir,
            config={
                "runner": self.cfg,
                "algorithm": self.alg_cfg,
                "policy": self.policy_cfg,
                "environment": {
                    "num_obs": self.env.num_obs,
                    "num_actions_physical": self.env.num_actions,
                    "num_actions_policy": self.env.num_policy_actions,
                    "controller_state_dim": self.env.controller_state_dim,
                    "contact_estimate_dim": self.env.contact_estimate_dim,
                    "dt": float(self.env.dt),
                },
            },
            save_code=True,
        )

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self._init_wandb()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        obs = self.env.get_observations().to(self.device)
        controller_state = self.env.get_controller_state().to(self.device)
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        critic_obs = critic_obs.to(self.device)
        self.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions, contact_estimate = self.alg.act(
                        obs, controller_state, critic_obs
                    )
                    (
                        next_obs,
                        next_privileged_obs,
                        rewards,
                        dones,
                        infos,
                        termination_ids,
                        termination_privileged_obs,
                    ) = self.env.step(actions, contact_estimate)

                    next_obs = next_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    next_controller_state = self.env.get_controller_state().to(
                        self.device
                    )
                    contact_target = self.env.get_contact_estimator_target().to(
                        self.device
                    )

                    next_critic_obs = (
                        next_privileged_obs
                        if next_privileged_obs is not None
                        else next_obs
                    ).to(self.device)
                    termination_ids = termination_ids.to(self.device)
                    termination_privileged_obs = termination_privileged_obs.to(
                        self.device
                    )
                    stored_next_critic_obs = next_critic_obs.clone().detach()
                    stored_next_critic_obs[termination_ids] = (
                        termination_privileged_obs.clone().detach()
                    )

                    self.alg.process_env_step(
                        rewards,
                        dones,
                        infos,
                        stored_next_critic_obs,
                        contact_target,
                    )

                    obs = next_obs
                    controller_state = next_controller_state
                    critic_obs = next_critic_obs

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                collection_time = time.time() - start
                start_learn = time.time()
                self.alg.compute_returns(critic_obs)

            losses = self.alg.update()
            learn_time = time.time() - start_learn
            if self.log_dir is not None:
                self.log(
                    it,
                    losses,
                    collection_time,
                    learn_time,
                    ep_infos,
                    rewbuffer,
                    lenbuffer,
                )
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(
            os.path.join(
                self.log_dir, f"model_{self.current_learning_iteration}.pt"
            )
        )
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None

    def log(
        self,
        it,
        losses,
        collection_time,
        learn_time,
        ep_infos,
        rewbuffer,
        lenbuffer,
    ):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        iteration_time = collection_time + learn_time
        self.tot_time += iteration_time

        metrics = {
            "Loss/value_function": losses["value"],
            "Loss/surrogate": losses["surrogate"],
            "Loss/HIM_estimation": losses["him_estimation"],
            "Loss/HIM_swap": losses["him_swap"],
            "Loss/contact_force": losses["contact_force"],
            "Loss/contact_loading": losses["contact_loading"],
            "Loss/learning_rate": self.alg.learning_rate,
            "Policy/mean_noise_std": self.actor_critic.std.mean().item(),
        }
        if len(rewbuffer) > 0:
            metrics["Train/mean_reward"] = statistics.mean(rewbuffer)
            metrics["Train/mean_episode_length"] = statistics.mean(lenbuffer)

        for key, value in metrics.items():
            self.writer.add_scalar(key, value, it)

        if ep_infos:
            for key in ep_infos[0]:
                vals = []
                for ep_info in ep_infos:
                    value = ep_info[key]
                    if isinstance(value, torch.Tensor):
                        vals.append(value.float().mean().item())
                    else:
                        vals.append(float(value))
                metrics[f"Episode/{key}"] = sum(vals) / len(vals)
                self.writer.add_scalar(
                    f"Episode/{key}", metrics[f"Episode/{key}"], it
                )

        if self.wandb_run is not None:
            metrics["Train/total_timesteps"] = self.tot_timesteps
            metrics["Perf/collection_time"] = collection_time
            metrics["Perf/learning_time"] = learn_time
            self.wandb_run.log(metrics, step=it)

        print(
            f"[adaptive] iter={it} reward="
            f"{metrics.get('Train/mean_reward', float('nan')):.3f} "
            f"contact_F={losses['contact_force']:.4f} "
            f"contact_dF={losses['contact_loading']:.4f} "
            f"time={iteration_time:.2f}s"
        )

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "estimator_optimizer_state_dict": (
                    self.actor_critic.estimator.optimizer.state_dict()
                ),
                "contact_estimator_optimizer_state_dict": (
                    self.actor_critic.contact_estimator.optimizer.state_dict()
                ),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded = torch.load(path, map_location=self.device)
        incoming = loaded["model_state_dict"]
        is_adaptive = any(key.startswith("contact_estimator.") for key in incoming)

        if is_adaptive:
            self.actor_critic.load_state_dict(incoming)
            if load_optimizer and "optimizer_state_dict" in loaded:
                self.alg.optimizer.load_state_dict(loaded["optimizer_state_dict"])
                self.actor_critic.estimator.optimizer.load_state_dict(
                    loaded["estimator_optimizer_state_dict"]
                )
                self.actor_critic.contact_estimator.optimizer.load_state_dict(
                    loaded["contact_estimator_optimizer_state_dict"]
                )
        else:
            # Baseline migration: copy every shape-compatible HIM/actor/critic
            # tensor.  New contact/compliance modules keep their safe init.
            current = self.actor_critic.state_dict()
            copied = []
            skipped = []
            for key, value in incoming.items():
                if key == "std" and key in current:
                    n = min(value.numel(), self.env.num_actions)
                    current[key][:n].copy_(value[:n])
                    copied.append(f"std[:{n}]")
                elif key in current and current[key].shape == value.shape:
                    current[key].copy_(value)
                    copied.append(key)
                else:
                    skipped.append(key)
            self.actor_critic.load_state_dict(current)
            print(
                f"Migrated baseline checkpoint: copied={len(copied)}, "
                f"skipped={len(skipped)}; new compliance/contact modules "
                "remain initialized."
            )
            load_optimizer = False

        self.current_learning_iteration = int(loaded.get("iter", 0))
        return loaded.get("infos", None)

    def get_inference_policy(self, device=None):
        self.actor_critic.eval()
        if device is not None:
            self.actor_critic.to(device)
        return self.actor_critic.act_inference
