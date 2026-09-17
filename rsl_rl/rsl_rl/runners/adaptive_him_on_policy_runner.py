"""On-policy runner for sensorless learned MC admittance."""

import math
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

        impact_pos_weight = float(
            self.policy_cfg["contact_estimator_impact_pos_weight"]
        )
        probability_correction_weight = float(
            env.cfg.learned_admittance.impact_probability_pos_weight_correction
        )
        if (
            not math.isfinite(impact_pos_weight)
            or not math.isfinite(probability_correction_weight)
            or impact_pos_weight <= 0.0
            or probability_correction_weight <= 0.0
            or abs(impact_pos_weight - probability_correction_weight) >= 1.0e-6
        ):
            raise RuntimeError(
                "Impact BCE pos_weight and control probability correction must match. "
                f"Got {impact_pos_weight} and {probability_correction_weight}."
            )

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
        legacy_contact_lr = float(
            self.policy_cfg.get("contact_estimator_lr", 1.0e-3)
        )
        self.contact_estimator_warmup_lr = float(
            self.policy_cfg.get(
                "contact_estimator_warmup_lr", legacy_contact_lr
            )
        )
        self.contact_estimator_online_lr = float(
            self.policy_cfg.get(
                "contact_estimator_online_lr", legacy_contact_lr
            )
        )
        if (
            not math.isfinite(self.contact_estimator_warmup_lr)
            or not math.isfinite(self.contact_estimator_online_lr)
            or self.contact_estimator_warmup_lr <= 0.0
            or self.contact_estimator_online_lr <= 0.0
        ):
            raise RuntimeError(
                "ContactEstimator warmup and online learning rates must be "
                "finite and positive."
            )
        self.actor_critic.contact_estimator.set_learning_rate(
            self.contact_estimator_warmup_lr
        )
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
        self.contact_validation_data = None
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
            "policy steps; compliance is forced to zero; "
            f"lr={self.contact_estimator_warmup_lr:.3g}."
        )
        self.actor_critic.train()
        running_force = 0.0
        running_impact = 0.0
        window_count = 0
        start_time = time.time()
        estimator = self.actor_critic.contact_estimator
        replay_enabled = estimator.replay_ratio > 0.0
        replay_samples = {
            "obs_history": [],
            "controller_state": [],
            "contact_target": [],
        }
        replay_per_step = (
            max(1, math.ceil(estimator.replay_buffer_size / num_steps))
            if replay_enabled
            else 0
        )

        for step in range(num_steps):
            with torch.no_grad():
                target = self.env.get_contact_estimator_target().to(self.device)
                if replay_enabled:
                    # Deterministic subsampling avoids perturbing PPO RNG state.
                    stride = max(1, self.env.num_envs // replay_per_step)
                    replay_ids = (
                        torch.arange(replay_per_step, device=self.device) * stride
                        + step
                    ) % self.env.num_envs
                    replay_samples["obs_history"].append(
                        obs[replay_ids].detach().cpu()
                    )
                    replay_samples["controller_state"].append(
                        controller_state[replay_ids].detach().cpu()
                    )
                    replay_samples["contact_target"].append(
                        target[replay_ids].detach().cpu()
                    )
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
                next_controller_state = self.env.get_controller_state().to(
                    self.device
                )
                next_obs = next_obs.to(self.device)
                next_critic_obs = (
                    next_privileged_obs
                    if next_privileged_obs is not None
                    else next_obs
                ).to(self.device)

            force_loss, impact_loss = self.actor_critic.contact_estimator.update(
                obs,
                controller_state,
                target,
                lr=self.contact_estimator_warmup_lr,
            )
            running_force += force_loss
            running_impact += impact_loss
            window_count += 1

            obs = next_obs
            controller_state = next_controller_state
            critic_obs = next_critic_obs

            if (step + 1) % interval == 0 or step + 1 == num_steps:
                avg_force = running_force / max(window_count, 1)
                avg_impact = running_impact / max(window_count, 1)
                print(
                    f"[contact warmup] step={step + 1}/{num_steps} "
                    f"force_loss={avg_force:.5f} "
                    f"impact_loss={avg_impact:.5f}"
                )
                if self.writer is not None:
                    self.writer.add_scalar(
                        "Warmup/contact_force_loss", avg_force, step + 1
                    )
                    self.writer.add_scalar(
                        "Warmup/contact_impact_loss", avg_impact, step + 1
                    )
                running_force = 0.0
                running_impact = 0.0
                window_count = 0

        elapsed = time.time() - start_time
        self.tot_timesteps += num_steps * self.env.num_envs
        self.contact_warmup_done = True
        if replay_enabled:
            replay = {
                key: torch.cat(chunks, dim=0)
                for key, chunks in replay_samples.items()
            }
            replay["metadata"] = {
                "source": "stage0_deterministic_baseline",
                "num_envs": int(self.env.num_envs),
                "num_steps": num_steps,
                "compliance": 0.0,
            }
            replay_path = str(self.cfg.get("contact_replay_path", ""))
            if replay_path:
                replay_path = os.path.abspath(replay_path)
                os.makedirs(os.path.dirname(replay_path), exist_ok=True)
                if os.path.isfile(replay_path):
                    replay = torch.load(replay_path, map_location="cpu")
                    print(f"[contact replay] reused {replay_path}")
                else:
                    torch.save(replay, replay_path)
                    print(f"[contact replay] saved {replay_path}")
            estimator.set_replay_buffer(
                replay["obs_history"],
                replay["controller_state"],
                replay["contact_target"],
            )
            print(
                "[contact replay] installed "
                f"{estimator.replay_target.shape[0]} Stage-0 samples; "
                f"ratio={estimator.replay_ratio:.3g}."
            )
        print(
            f"[contact warmup] finished in {elapsed:.2f}s; PPO compliance "
            "learning can now start."
        )
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, "contact_pretrained.pt"))
        return obs, controller_state, critic_obs

    def _collect_fixed_contact_validation(self, obs, controller_state, critic_obs):
        """Collect a fixed baseline-only dataset that never enters training."""
        num_steps = int(self.cfg.get("contact_validation_steps", 0))
        if num_steps <= 0:
            return obs, controller_state, critic_obs
        max_samples = max(
            1, int(self.cfg.get("contact_validation_max_samples", 51200))
        )
        samples_per_step = min(
            self.env.num_envs,
            max(1, math.ceil(max_samples / num_steps)),
        )

        print(
            f"[contact validation] collecting {num_steps} deterministic "
            "baseline steps; compliance is forced to zero."
        )
        was_training = self.actor_critic.training
        self.actor_critic.eval()
        samples = {
            "obs_history": [],
            "controller_state": [],
            "contact_target": [],
            "axial_loading_rate_nps": [],
            "commands": [],
        }

        with torch.inference_mode():
            for step in range(num_steps):
                # The current post-step observation and controller state are
                # paired with the target/loading rate from that same completed
                # transition. GT values are copied only into this CPU dataset.
                # Deterministic subsampling keeps fixed-validation memory and
                # cost independent of the training environment count.
                stride = max(1, self.env.num_envs // samples_per_step)
                sample_ids = (
                    torch.arange(samples_per_step, device=self.device) * stride
                    + step
                ) % self.env.num_envs
                samples["obs_history"].append(
                    obs[sample_ids].detach().cpu()
                )
                samples["controller_state"].append(
                    controller_state[sample_ids].detach().cpu()
                )
                samples["contact_target"].append(
                    self.env.get_contact_estimator_target()[sample_ids]
                    .detach().cpu()
                )
                samples["axial_loading_rate_nps"].append(
                    self.env.get_contact_validation_loading_rate()[sample_ids]
                    .detach().cpu()
                )
                if hasattr(self.env, "commands"):
                    samples["commands"].append(
                        self.env.commands[sample_ids].detach().cpu()
                    )

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
                obs = next_obs.to(self.device)
                controller_state = self.env.get_controller_state().to(self.device)
                critic_obs = (
                    next_privileged_obs
                    if next_privileged_obs is not None
                    else obs
                ).to(self.device)

        candidate = {}
        for key, chunks in samples.items():
            if chunks:
                candidate[key] = torch.cat(chunks, dim=0)[:max_samples]
        candidate["metadata"] = {
            "num_envs": int(self.env.num_envs),
            "num_steps": num_steps,
            "sample_count": int(candidate["obs_history"].shape[0]),
            "baseline_compliance": 0.0,
        }

        validation_path = str(self.cfg.get("contact_validation_path", ""))
        if not validation_path and self.log_dir is not None:
            validation_path = os.path.join(
                self.log_dir, "validation", "contact_validation.pt"
            )
        if validation_path:
            validation_path = os.path.abspath(validation_path)
            directory = os.path.dirname(validation_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            if os.path.isfile(validation_path):
                fixed = torch.load(validation_path, map_location="cpu")
                required = {
                    "obs_history",
                    "controller_state",
                    "contact_target",
                    "axial_loading_rate_nps",
                }
                missing = sorted(required.difference(fixed))
                if missing:
                    raise RuntimeError(
                        "Fixed contact validation file is missing keys: "
                        + ", ".join(missing)
                    )
                if fixed["obs_history"].shape != candidate["obs_history"].shape:
                    raise RuntimeError(
                        "Fixed contact validation shape does not match this run: "
                        f"saved={tuple(fixed['obs_history'].shape)}, "
                        f"current={tuple(candidate['obs_history'].shape)}"
                    )
                self.contact_validation_data = fixed
                print(
                    "[contact validation] reused fixed dataset "
                    f"{validation_path}; the matching collection trajectory was "
                    "still executed to preserve the Stage-1 start state."
                )
            else:
                torch.save(candidate, validation_path)
                self.contact_validation_data = candidate
                print(
                    f"[contact validation] saved fixed dataset to {validation_path}"
                )
        else:
            self.contact_validation_data = candidate
            print("[contact validation] retained fixed dataset in memory only.")

        if was_training:
            self.actor_critic.train()
        return obs, controller_state, critic_obs

    def _restore_contact_replay_if_needed(self):
        estimator = self.actor_critic.contact_estimator
        if estimator.replay_ratio <= 0.0 or estimator.replay_target is not None:
            return
        replay_path = str(self.cfg.get("contact_replay_path", ""))
        if not replay_path or not os.path.isfile(replay_path):
            raise RuntimeError(
                "Replay is enabled but no Stage-0 replay dataset is available: "
                f"{replay_path or '<unset>'}"
            )
        replay = torch.load(replay_path, map_location="cpu")
        estimator.set_replay_buffer(
            replay["obs_history"],
            replay["controller_state"],
            replay["contact_target"],
        )
        print(f"[contact replay] restored {replay_path}")

    @staticmethod
    def _safe_masked_mean(values, mask):
        selected = values.reshape(-1)[mask.reshape(-1)]
        if selected.numel() == 0:
            return torch.zeros((), dtype=values.dtype)
        return selected.float().mean()

    @staticmethod
    def _binary_metrics(probability, target, threshold=0.5):
        predicted = probability.reshape(-1) >= float(threshold)
        target = target.reshape(-1).bool()
        tp = torch.sum((predicted & target).float())
        pred_n = torch.sum(predicted.float())
        target_n = torch.sum(target.float())
        precision = tp / torch.clamp(pred_n, min=1.0)
        recall = tp / torch.clamp(target_n, min=1.0)
        f1 = 2.0 * precision * recall / torch.clamp(
            precision + recall, min=1.0e-8
        )
        return precision, recall, f1

    @staticmethod
    def _binary_auc(probability, target):
        """Dependency-free ROC AUC and average precision (PR AUC)."""
        probability = probability.reshape(-1).float()
        target = target.reshape(-1).bool()
        positives = torch.sum(target).item()
        negatives = target.numel() - positives
        if positives == 0 or negatives == 0:
            zero = torch.zeros((), dtype=probability.dtype)
            return zero, zero
        order = torch.argsort(probability, descending=True)
        sorted_target = target[order].float()
        true_positive = torch.cumsum(sorted_target, dim=0)
        false_positive = torch.cumsum(1.0 - sorted_target, dim=0)
        recall = true_positive / float(positives)
        false_positive_rate = false_positive / float(negatives)
        zero = torch.zeros(1, dtype=probability.dtype)
        roc_auc = torch.trapz(
            torch.cat((zero, recall)),
            torch.cat((zero, false_positive_rate)),
        )
        precision = true_positive / torch.arange(
            1, target.numel() + 1, dtype=probability.dtype
        )
        recall_delta = recall - torch.cat((zero, recall[:-1]))
        pr_auc = torch.sum(recall_delta * precision)
        return pr_auc, roc_auc

    def _evaluate_fixed_contact_validation(self):
        """Forward-only evaluation; fixed samples cannot reach any optimizer."""
        data = self.contact_validation_data
        if data is None:
            return {}
        batch_size = max(
            1, int(self.cfg.get("contact_validation_batch_size", 8192))
        )
        estimator = self.actor_critic.contact_estimator
        was_training = estimator.training
        estimator.eval()
        outputs = []
        with torch.inference_mode():
            for start in range(0, data["obs_history"].shape[0], batch_size):
                stop = start + batch_size
                outputs.append(
                    estimator(
                        data["obs_history"][start:stop].to(self.device),
                        data["controller_state"][start:stop].to(self.device),
                    ).cpu()
                )
        if was_training:
            estimator.train()

        prediction = torch.cat(outputs, dim=0)
        target = data["contact_target"].float()
        force_prediction = prediction[:, :4]
        force_target = target[:, :4]
        raw_logits = prediction[:, 4:]
        gt_impact = target[:, 4:] >= 0.5
        raw_probability = torch.sigmoid(raw_logits)
        with torch.inference_mode():
            control_probability = self.env.admittance.impact_probability_from_logits(
                raw_logits.to(self.device)
            ).cpu()

        raw_precision, raw_recall, raw_f1 = self._binary_metrics(
            raw_probability, gt_impact
        )
        control_precision, control_recall, control_f1 = self._binary_metrics(
            control_probability, gt_impact
        )
        pr_auc, roc_auc = self._binary_auc(raw_probability, gt_impact)
        positive_logit = self._safe_masked_mean(raw_logits, gt_impact)
        negative_logit = self._safe_masked_mean(raw_logits, ~gt_impact)
        positive_probability = self._safe_masked_mean(
            raw_probability, gt_impact
        )
        negative_probability = self._safe_masked_mean(
            raw_probability, ~gt_impact
        )

        force_x = force_prediction.reshape(-1).float()
        force_y = force_target.reshape(-1).float()
        centered_x = force_x - force_x.mean()
        centered_y = force_y - force_y.mean()
        denominator = torch.sqrt(
            torch.sum(centered_x.square()) * torch.sum(centered_y.square())
        )
        force_corr = torch.where(
            denominator > 1.0e-8,
            torch.sum(centered_x * centered_y)
            / torch.clamp(denominator, min=1.0e-8),
            torch.zeros_like(denominator),
        )
        force_scale_n = float(
            self.env.cfg.learned_admittance.contact_force_scale_n
        )
        metrics = {
            "Validation/force_mae_n": torch.mean(
                torch.abs(force_prediction - force_target)
            ) * force_scale_n,
            "Validation/force_mse": torch.mean(
                torch.square(force_prediction - force_target)
            ),
            "Validation/force_corr": force_corr,
            "Validation/impact_gt_ratio": torch.mean(gt_impact.float()),
            "Validation/raw_precision": raw_precision,
            "Validation/raw_recall": raw_recall,
            "Validation/raw_f1": raw_f1,
            "Validation/control_precision": control_precision,
            "Validation/control_recall": control_recall,
            "Validation/control_f1": control_f1,
            "Validation/raw_logit_positive_mean": positive_logit,
            "Validation/raw_logit_negative_mean": negative_logit,
            "Validation/raw_logit_separation": positive_logit - negative_logit,
            "Validation/raw_probability_positive_mean": positive_probability,
            "Validation/raw_probability_negative_mean": negative_probability,
            "Validation/raw_probability_separation": (
                positive_probability - negative_probability
            ),
            "Validation/PR_AUC": pr_auc,
            "Validation/ROC_AUC": roc_auc,
        }

        loading_rate = data["axial_loading_rate_nps"].float()
        for name, lower, upper in (
            ("0_3000", 0.0, 3000.0),
            ("3000_4000", 3000.0, 4000.0),
            ("4000_4500", 4000.0, 4500.0),
            ("4500_4750", 4500.0, 4750.0),
            ("4750_5000", 4750.0, 5000.0),
            ("5000_5250", 5000.0, 5250.0),
            ("5250_5500", 5250.0, 5500.0),
            ("5500_6000", 5500.0, 6000.0),
            ("6000_8000", 6000.0, 8000.0),
        ):
            metrics[f"Validation/LabelDist/bin_{name}_ratio"] = torch.mean(
                ((loading_rate >= lower) & (loading_rate < upper)).float()
            )
        metrics["Validation/LabelDist/bin_gt_8000_ratio"] = torch.mean(
            (loading_rate >= 8000.0).float()
        )
        metrics["Validation/LabelDist/near_threshold_ratio"] = torch.mean(
            ((loading_rate >= 4500.0) & (loading_rate <= 5500.0)).float()
        )
        metrics[
            "Validation/LabelDist/very_near_threshold_ratio"
        ] = torch.mean(
            ((loading_rate >= 4750.0) & (loading_rate <= 5250.0)).float()
        )
        return {key: value.item() for key, value in metrics.items()}

    def _log_fixed_contact_validation(self, iteration):
        metrics = self._evaluate_fixed_contact_validation()
        if not metrics:
            return
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, iteration)
            self.writer.flush()
        if self.wandb_run is not None:
            # The training log for this same 1-based iteration commits the
            # combined W&B row. This avoids out-of-order validation/training
            # steps, including immediately after resume.
            self.wandb_run.log(metrics, step=iteration, commit=False)
        print(
            f"[contact validation] iter={iteration} "
            f"raw_f1={metrics['Validation/raw_f1']:.4f} "
            f"control_f1={metrics['Validation/control_f1']:.4f} "
            f"separation={metrics['Validation/raw_logit_separation']:.4f}"
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

        if not self.contact_warmup_done:
            obs, controller_state, critic_obs = self._contact_pretrain(
                obs, controller_state, critic_obs
            )
        self._restore_contact_replay_if_needed()

        obs, controller_state, critic_obs = self._collect_fixed_contact_validation(
            obs, controller_state, critic_obs
        )
        self._log_fixed_contact_validation(self.current_learning_iteration)
        # The warm-up snapshot above is evaluated before switching LR. Reuse
        # the same Adam optimizer/state and change only its Stage-1 step size.
        self.actor_critic.contact_estimator.set_learning_rate(
            self.contact_estimator_online_lr
        )
        print(
            "[contact estimator] Stage-1 continual update "
            f"lr={self.contact_estimator_online_lr:.3g}."
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
                    contact_target = self.env.get_contact_estimator_target().to(
                        self.device
                    )
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
                    current_contact_estimate = self.actor_critic.contact_estimator(
                        next_obs, next_controller_state
                    )
                    self.env.cache_contact_estimator_diagnostics(
                        current_contact_estimate
                    )
                    self._accumulate_diagnostics(
                        diagnostic_sums,
                        diagnostic_counts,
                        diagnostic_maxima,
                        self.env.get_admittance_diagnostics(),
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

            validation_interval = max(
                1, int(self.cfg.get("contact_validation_interval", 20))
            )
            if (
                self.contact_validation_data is not None
                and (
                    self.current_learning_iteration % validation_interval == 0
                    or self.current_learning_iteration == tot_iter
                )
            ):
                self._log_fixed_contact_validation(
                    self.current_learning_iteration
                )

            if self.log_dir is not None:
                self.log(
                    self.current_learning_iteration,
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
            "Loss/contact_impact": losses["contact_impact"],
            "Loss/learning_rate": self.alg.learning_rate,
            "Policy/mean_noise_std": self.actor_critic.std.mean().item(),
            "Policy/compliance_noise_std": self.actor_critic.std[
                self.env.num_actions :
            ].mean().item(),
            "Estimator/frozen_after_warmup": float(
                self.alg.freeze_contact_estimator_after_warmup
            ),
            "Estimator/online_learning_rate": (
                self.contact_estimator_online_lr
            ),
            "Estimator/replay_ratio": (
                self.actor_critic.contact_estimator.replay_ratio
            ),
            "Estimator/replay_buffer_size": float(
                0
                if self.actor_critic.contact_estimator.replay_target is None
                else self.actor_critic.contact_estimator.replay_target.shape[0]
            ),
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

        mean_reward = metrics.get("Train/mean_reward")
        reward_text = "n/a" if mean_reward is None else f"{mean_reward:.3f}"
        print(
            f"[adaptive] iter={it} reward={reward_text} "
            f"contact_F={losses['contact_force']:.4f} "
            f"contact_impact={losses['contact_impact']:.4f} "
            f"alpha={metrics.get('Admittance/alpha_mean', float('nan')):.3f} "
            f"comp_p95={metrics.get('Admittance/compression_p95_mm', float('nan')):.2f}mm "
            f"F_mae={metrics.get('Estimator/axial_force_mae_n', float('nan')):.1f}N "
            f"time={iteration_time:.2f}s"
        )

    def save(self, path, infos=None):
        estimator = self.actor_critic.contact_estimator
        contact_replay = None
        if estimator.replay_target is not None:
            contact_replay = {
                "obs_history": estimator.replay_obs_history.detach().cpu(),
                "controller_state": (
                    estimator.replay_controller_state.detach().cpu()
                ),
                "contact_target": estimator.replay_target.detach().cpu(),
                "cursor": int(estimator.replay_cursor),
            }
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
                # Replay is part of the learned estimator state. Keeping it in
                # the checkpoint makes adaptive resume self-contained.
                "contact_replay": contact_replay,
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
            replay = loaded.get("contact_replay")
            if replay is not None:
                estimator = self.actor_critic.contact_estimator
                estimator.set_replay_buffer(
                    replay["obs_history"],
                    replay["controller_state"],
                    replay["contact_target"],
                )
                if estimator.replay_target is not None:
                    estimator.replay_cursor = int(
                        replay.get("cursor", 0)
                    ) % estimator.replay_target.shape[0]
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
