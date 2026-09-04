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
        self.contact_warmup_done = False
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

    @staticmethod
    def _accumulate_diagnostics(sums, counts, maxima, diagnostics):
        for key, value in diagnostics.items():
            value = value.detach()
            if (
                key.endswith("_max")
                or key.endswith("_max_n")
                or key.endswith("_max_nps")
                or key.endswith("_max_mps2")
                or key.endswith("_max_rad")
                or key.endswith("_max_mm")
            ):
                if key not in maxima:
                    maxima[key] = value.clone()
                else:
                    maxima[key] = torch.maximum(maxima[key], value)
            else:
                if key not in sums:
                    sums[key] = value.clone()
                    counts[key] = 1
                else:
                    sums[key] = sums[key] + value
                    counts[key] += 1

    @staticmethod
    def _finalize_diagnostics(sums, counts, maxima):
        result = {}
        for key, value in sums.items():
            result[key] = (value / max(counts[key], 1)).item()
        for key, value in maxima.items():
            result[key] = value.item()
        return result

    def _contact_pretrain(self, obs, controller_state, critic_obs):
        """Stage 0: train only ContactEstimator under exact baseline motion."""
        num_steps = int(self.cfg.get("contact_pretrain_steps", 0))
        if num_steps <= 0:
            self.contact_warmup_done = True
            return obs, controller_state, critic_obs
        interval = max(1, int(self.cfg.get("contact_pretrain_log_interval", 50)))

        print(
            f"[contact warmup] starting {num_steps} deterministic baseline "
            "policy steps; compliance is forced to zero."
        )
        self.actor_critic.train()
        running_force = 0.0
        running_loading = 0.0
        window_count = 0
        start_time = time.time()

        for step in range(num_steps):
            with torch.no_grad():
                actions = self.actor_critic.act_inference(
                    obs, controller_state
                ).detach()
                actions[:, self.env.num_actions :] = 0.0
                contact_estimate = self.actor_critic.last_contact_estimate.detach()

                (
                    next_obs,
                    next_privileged_obs,
                    _rewards,
                    _dones,
                    _infos,
                    _termination_ids,
                    _termination_privileged_obs,
                ) = self.env.step(actions, contact_estimate)
                target = self.env.get_contact_estimator_target().to(self.device)
                next_controller_state = self.env.get_controller_state().to(
                    self.device
                )
                next_obs = next_obs.to(self.device)
                next_critic_obs = (
                    next_privileged_obs
                    if next_privileged_obs is not None
                    else next_obs
                ).to(self.device)

            force_loss, loading_loss = self.actor_critic.contact_estimator.update(
                obs, controller_state, target
            )
            running_force += force_loss
            running_loading += loading_loss
            window_count += 1

            obs = next_obs
            controller_state = next_controller_state
            critic_obs = next_critic_obs

            if (step + 1) % interval == 0 or step + 1 == num_steps:
                avg_force = running_force / max(window_count, 1)
                avg_loading = running_loading / max(window_count, 1)
                print(
                    f"[contact warmup] step={step + 1}/{num_steps} "
                    f"force_loss={avg_force:.5f} "
                    f"loading_loss={avg_loading:.5f}"
                )
                if self.writer is not None:
                    self.writer.add_scalar(
                        "Warmup/contact_force_loss", avg_force, step + 1
                    )
                    self.writer.add_scalar(
                        "Warmup/contact_loading_loss", avg_loading, step + 1
                    )
                running_force = 0.0
                running_loading = 0.0
                window_count = 0

        elapsed = time.time() - start_time
        self.tot_timesteps += num_steps * self.env.num_envs
        self.contact_warmup_done = True
        print(
            f"[contact warmup] finished in {elapsed:.2f}s; PPO compliance "
            "learning can now start."
        )
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, "contact_pretrained.pt"))
        return obs, controller_state, critic_obs

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

        if not self.contact_warmup_done:
            obs, controller_state, critic_obs = self._contact_pretrain(
                obs, controller_state, critic_obs
            )

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            diagnostic_sums = {}
            diagnostic_counts = {}
            diagnostic_maxima = {}
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

                    self._accumulate_diagnostics(
                        diagnostic_sums,
                        diagnostic_counts,
                        diagnostic_maxima,
                        self.env.get_admittance_diagnostics(),
                    )

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

            diagnostics = self._finalize_diagnostics(
                diagnostic_sums, diagnostic_counts, diagnostic_maxima
            )
            losses = self.alg.update()
            learn_time = time.time() - start_learn
            self.current_learning_iteration = it + 1

            if self.log_dir is not None:
                self.log(
                    it,
                    losses,
                    collection_time,
                    learn_time,
                    ep_infos,
                    rewbuffer,
                    lenbuffer,
                    diagnostics,
                )
            if self.current_learning_iteration % self.save_interval == 0:
                self.save(
                    os.path.join(
                        self.log_dir,
                        f"model_{self.current_learning_iteration}.pt",
                    )
                )
            ep_infos.clear()

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
        diagnostics,
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
            "Policy/compliance_noise_std": self.actor_critic.std[
                self.env.num_actions :
            ].mean().item(),
        }
        metrics.update(diagnostics)
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
            f"alpha={metrics.get('Admittance/alpha_mean', float('nan')):.3f} "
            f"comp_p95={metrics.get('Admittance/compression_p95_mm', float('nan')):.2f}mm "
            f"F_mae={metrics.get('Estimator/axial_force_mae_n', float('nan')):.1f}N "
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
                "tot_timesteps": self.tot_timesteps,
                "tot_time": self.tot_time,
                "contact_warmup_done": self.contact_warmup_done,
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
            self.contact_warmup_done = bool(
                loaded.get("contact_warmup_done", True)
            )
            if load_optimizer and "optimizer_state_dict" in loaded:
                self.alg.optimizer.load_state_dict(loaded["optimizer_state_dict"])
                self.actor_critic.estimator.optimizer.load_state_dict(
                    loaded["estimator_optimizer_state_dict"]
                )
                self.actor_critic.contact_estimator.optimizer.load_state_dict(
                    loaded["contact_estimator_optimizer_state_dict"]
                )
        else:
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
            self.contact_warmup_done = False
            print(
                f"Migrated baseline checkpoint: copied={len(copied)}, "
                f"skipped={len(skipped)}; new compliance/contact modules "
                "remain initialized."
            )
            load_optimizer = False

        self.current_learning_iteration = int(loaded.get("iter", 0))
        if is_adaptive:
            self.tot_timesteps = int(
                loaded.get(
                    "tot_timesteps",
                    self.current_learning_iteration
                    * self.num_steps_per_env
                    * self.env.num_envs,
                )
            )
            self.tot_time = float(loaded.get("tot_time", 0.0))
            print(
                f"Resumed adaptive checkpoint at iteration "
                f"{self.current_learning_iteration}; contact warmup done="
                f"{self.contact_warmup_done}."
            )
        return loaded.get("infos", None)

    def get_inference_policy(self, device=None):
        self.actor_critic.eval()
        if device is not None:
            self.actor_critic.to(device)
        return self.actor_critic.act_inference
