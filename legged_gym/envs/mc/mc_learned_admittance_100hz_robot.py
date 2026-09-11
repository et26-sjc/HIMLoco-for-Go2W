"""100 Hz MC environment for learned sensorless admittance.

The environment separates three information domains explicitly:

* ``self.actions``: original 16-D HIMLoco motion action (the original 57-D
  proprioceptive observation layout remains unchanged);
* policy-only extra action: four compliance activations;
* training-only ground-truth impact signals from Isaac Gym contact tensors.

The deployed admittance never reads ground-truth contact force. During training,
3-D wheel contact force supervises quiet rewards. Force projected onto each
hip-to-wheel axis supervises force regression, and its thresholded positive
loading rate supplies the binary impact label.
"""

from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse
import torch

from .mc_robot import MC
from .mc_learned_admittance import MCLearnedAdmittance


class MCLearnedAdmittance100Hz(MC):
    """MC with 20-D policy action but unchanged 16-D physical actuation."""

    _LEG_SPECS = [
        # leg, wheel body, hip body, hip joint, knee joint
        ("FL", "FL_FOOT_LINK", "FL_HIP_LINK", "FBL_HIP_JOINT", "FBL_KNEE_JOINT"),
        ("FR", "FR_FOOT_LINK", "FR_HIP_LINK", "FAR_HIP_JOINT", "FAR_KNEE_JOINT"),
        ("RR", "RR_FOOT_LINK", "RR_HIP_LINK", "RAR_HIP_JOINT", "RAR_KNEE_JOINT"),
        ("RL", "RL_FOOT_LINK", "RL_HIP_LINK", "RBL_HIP_JOINT", "RBL_KNEE_JOINT"),
    ]

    def _init_buffers(self):
        super()._init_buffers()
        self.num_motion_actions = int(self.cfg.env.num_motion_actions)
        self.num_compliance_actions = int(self.cfg.env.num_compliance_actions)
        self.num_policy_actions = int(self.cfg.env.num_policy_actions)
        self.controller_state_dim = int(self.cfg.env.controller_state_dim)
        self.contact_estimate_dim = int(self.cfg.env.contact_estimate_dim)

        if self.num_motion_actions != self.num_actions or self.num_actions != 16:
            raise RuntimeError(
                "Learned-admittance MC expects physical num_actions=16, got "
                f"num_actions={self.num_actions}, motion={self.num_motion_actions}"
            )
        if self.num_policy_actions != self.num_motion_actions + self.num_compliance_actions:
            raise RuntimeError("Policy action dimensions are inconsistent")

        foot_ids, hip_body_ids, hip_ids, knee_ids = [], [], [], []
        for leg, foot_name, hip_body_name, hip_name, knee_name in self._LEG_SPECS:
            foot_id = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], foot_name
            )
            hip_body_id = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], hip_body_name
            )
            hip_id = self.gym.find_actor_dof_handle(
                self.envs[0], self.actor_handles[0], hip_name
            )
            knee_id = self.gym.find_actor_dof_handle(
                self.envs[0], self.actor_handles[0], knee_name
            )
            if min(foot_id, hip_body_id, hip_id, knee_id) < 0:
                raise RuntimeError(
                    f"Failed to resolve semantic leg {leg}: foot={foot_id}, "
                    f"hip_body={hip_body_id}, hip={hip_id}, knee={knee_id}"
                )
            foot_ids.append(foot_id)
            hip_body_ids.append(hip_body_id)
            hip_ids.append(hip_id)
            knee_ids.append(knee_id)

        self.adm_feet_indices = torch.tensor(foot_ids, dtype=torch.long, device=self.device)
        self.adm_hip_body_indices = torch.tensor(
            hip_body_ids, dtype=torch.long, device=self.device
        )
        self.adm_hip_indices = torch.tensor(hip_ids, dtype=torch.long, device=self.device)
        self.adm_knee_indices = torch.tensor(knee_ids, dtype=torch.long, device=self.device)

        print("### Learned-admittance leg order:")
        for i, spec in enumerate(self._LEG_SPECS):
            print(
                f"  {i}:{spec[0]} foot={foot_ids[i]} hip_body={hip_body_ids[i]} "
                f"hip={hip_ids[i]} knee={knee_ids[i]}"
            )

        self.admittance = MCLearnedAdmittance(
            self.cfg.learned_admittance, self.num_envs, self.device
        )
        self.policy_actions = torch.zeros(
            self.num_envs, self.num_policy_actions, device=self.device
        )
        self.compliance_actions = torch.zeros(
            self.num_envs, self.num_compliance_actions, device=self.device
        )
        self.estimated_contact = torch.zeros(
            self.num_envs, self.contact_estimate_dim, device=self.device
        )

        shape = (self.num_envs, 4)
        self.gt_step_peak_force = torch.zeros(shape, device=self.device)
        self.gt_step_peak_loading_rate = torch.zeros(shape, device=self.device)
        self.gt_prev_force_norm = torch.zeros(shape, device=self.device)
        self.gt_step_peak_axial_force = torch.zeros(shape, device=self.device)
        self.gt_step_peak_axial_loading_rate = torch.zeros(shape, device=self.device)
        self.gt_prev_axial_force = torch.zeros(shape, device=self.device)
        self.gt_skip_rate_once = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self.gt_step_peak_base_acc = torch.zeros(self.num_envs, device=self.device)
        self.gt_prev_base_vel_z = self._base_vel_z().clone()
        self.contact_estimator_target = torch.zeros(
            self.num_envs, self.contact_estimate_dim, device=self.device
        )
        self.transition_contact_estimator_target = torch.zeros_like(
            self.contact_estimator_target
        )
        # Diagnostics/validation only: GT loading rate associated with
        # transition_contact_estimator_target. It is never exposed to policy,
        # estimator, reward, or admittance control inputs.
        self.transition_gt_axial_loading_rate = torch.zeros(
            shape, device=self.device
        )
        # Diagnostics only: y_(t-1), whose post-step proprioception produced the
        # classifier state used to control the transition currently in progress.
        self.control_aligned_gt_impact = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )

        self._last_admittance_diagnostics = {}

    def _base_vel_z(self):
        return quat_rotate_inverse(
            self.root_states[:, 3:7], self.root_states[:, 7:10]
        )[:, 2]

    def get_controller_state(self):
        state = self.admittance.state()
        if state.shape[-1] != self.controller_state_dim:
            raise RuntimeError(
                f"Controller state is {state.shape[-1]}D, expected "
                f"{self.controller_state_dim}D"
            )
        return state

    def get_contact_estimator_target(self):
        return self.transition_contact_estimator_target

    def get_contact_validation_loading_rate(self):
        """Return diagnostic GT loading rate aligned with the current target."""
        return self.transition_gt_axial_loading_rate

    def get_admittance_diagnostics(self):
        return {
            key: value.detach() for key, value in self._last_admittance_diagnostics.items()
        }

    @staticmethod
    def _p95(x):
        flat = x.reshape(-1)
        if flat.numel() == 0:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        return torch.quantile(flat, 0.95)

    @staticmethod
    def _pearson_corr(x, y):
        x = x.reshape(-1).float()
        y = y.reshape(-1).float()
        if x.numel() == 0:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        x = x - torch.mean(x)
        y = y - torch.mean(y)
        denom = torch.sqrt(torch.sum(x * x) * torch.sum(y * y))
        corr = torch.sum(x * y) / torch.clamp(denom, min=1.0e-8)
        return torch.where(denom > 1.0e-8, corr, torch.zeros_like(corr))

    @staticmethod
    def _event_precision_recall(pred, gt):
        pred = pred.reshape(-1)
        gt = gt.reshape(-1)
        tp = torch.sum((pred & gt).float())
        pred_n = torch.sum(pred.float())
        gt_n = torch.sum(gt.float())
        precision = tp / torch.clamp(pred_n, min=1.0)
        recall = tp / torch.clamp(gt_n, min=1.0)
        f1 = 2.0 * precision * recall / torch.clamp(
            precision + recall, min=1.0e-8
        )
        return precision, recall, f1

    @staticmethod
    def _masked_mean(values, mask):
        weights = mask.to(dtype=values.dtype)
        return torch.sum(values * weights) / torch.clamp(
            torch.sum(weights), min=1.0
        )

    @staticmethod
    def _masked_quantile(values, mask, quantile):
        selected = values.reshape(-1)[mask.reshape(-1)]
        if selected.numel() == 0:
            return torch.zeros((), device=values.device, dtype=values.dtype)
        return torch.quantile(selected, float(quantile))

    @classmethod
    def _binary_probability_diagnostics(cls, probability, gt, threshold=0.5):
        predicted = probability >= float(threshold)
        precision, recall, f1 = cls._event_precision_recall(predicted, gt)
        return {
            "pred_ratio": torch.mean(predicted.float()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def cache_contact_estimator_diagnostics(self, current_contact_estimate):
        """Evaluate the classifier against the event represented by current obs."""
        if current_contact_estimate.shape[-1] != self.contact_estimate_dim:
            raise RuntimeError(
                f"Expected {self.contact_estimate_dim}D contact estimate, got "
                f"{current_contact_estimate.shape[-1]}D"
            )

        cfg = self.cfg.learned_admittance
        estimated_force = (
            torch.clamp(current_contact_estimate[:, :4], min=0.0)
            * float(cfg.contact_force_scale_n)
        )
        raw_impact_logits = current_contact_estimate[:, 4:]
        raw_impact_probability = torch.sigmoid(raw_impact_logits)
        impact_probability = self.admittance.impact_probability_from_logits(
            raw_impact_logits
        )
        gt_force = self.gt_step_peak_axial_force
        gt_impact = self.contact_estimator_target[:, 4:] >= 0.5

        impact_threshold = float(
            getattr(cfg, "diagnostic_impact_probability_threshold", 0.5)
        )
        raw_metrics = self._binary_probability_diagnostics(
            raw_impact_probability, gt_impact, impact_threshold
        )
        fullcorr_metrics = self._binary_probability_diagnostics(
            impact_probability, gt_impact, impact_threshold
        )
        predicted_impact = impact_probability >= impact_threshold

        force_event_threshold = float(
            getattr(cfg, "diagnostic_force_event_threshold_n", 60.0)
        )
        predicted_force_event = estimated_force >= force_event_threshold
        gt_force_event = gt_force >= force_event_threshold
        force_precision, force_recall, force_f1 = self._event_precision_recall(
            predicted_force_event, gt_force_event
        )
        target_force_clip_n = (
            float(cfg.contact_target_clip) * float(cfg.contact_force_scale_n)
        )

        self._last_admittance_diagnostics.update(
            {
                "Estimator/axial_force_pred_mean_n": torch.mean(estimated_force),
                "Estimator/axial_force_pred_max_n": torch.max(estimated_force),
                "Estimator/axial_force_gt_mean_n": torch.mean(gt_force),
                "Estimator/axial_force_gt_max_n": torch.max(gt_force),
                "Estimator/axial_force_mae_n": torch.mean(
                    torch.abs(estimated_force - gt_force)
                ),
                "Estimator/axial_force_corr": self._pearson_corr(
                    estimated_force, gt_force
                ),
                "Estimator/force_event_pred_ratio": torch.mean(
                    predicted_force_event.float()
                ),
                "Estimator/force_event_gt_ratio": torch.mean(
                    gt_force_event.float()
                ),
                "Estimator/force_event_precision": force_precision,
                "Estimator/force_event_recall": force_recall,
                "Estimator/force_event_f1": force_f1,
                "Estimator/impact_precision": fullcorr_metrics["precision"],
                "Estimator/impact_recall": fullcorr_metrics["recall"],
                "Estimator/impact_f1": fullcorr_metrics["f1"],
                "Estimator/impact_gt_ratio": torch.mean(gt_impact.float()),
                "Estimator/impact_pred_ratio": torch.mean(
                    predicted_impact.float()
                ),
                "Estimator/raw_logit_positive_mean": self._masked_mean(
                    raw_impact_logits, gt_impact
                ),
                "Estimator/raw_logit_negative_mean": self._masked_mean(
                    raw_impact_logits, ~gt_impact
                ),
                "Estimator/raw_logit_positive_p50": self._masked_quantile(
                    raw_impact_logits, gt_impact, 0.50
                ),
                "Estimator/raw_logit_negative_p50": self._masked_quantile(
                    raw_impact_logits, ~gt_impact, 0.50
                ),
                "Estimator/raw_logit_positive_p95": self._masked_quantile(
                    raw_impact_logits, gt_impact, 0.95
                ),
                "Estimator/raw_logit_negative_p95": self._masked_quantile(
                    raw_impact_logits, ~gt_impact, 0.95
                ),
                "Estimator/raw_logit_separation": self._masked_mean(
                    raw_impact_logits, gt_impact
                ) - self._masked_mean(raw_impact_logits, ~gt_impact),
                "Estimator/raw_probability_positive_mean": self._masked_mean(
                    raw_impact_probability, gt_impact
                ),
                "Estimator/raw_probability_negative_mean": self._masked_mean(
                    raw_impact_probability, ~gt_impact
                ),
                "Estimator/raw_probability_mean": torch.mean(
                    raw_impact_probability
                ),
                "Estimator/raw_probability_separation": self._masked_mean(
                    raw_impact_probability, gt_impact
                ) - self._masked_mean(raw_impact_probability, ~gt_impact),
                "Estimator/raw_impact_pred_ratio": raw_metrics["pred_ratio"],
                "Estimator/raw_impact_precision": raw_metrics["precision"],
                "Estimator/raw_impact_recall": raw_metrics["recall"],
                "Estimator/raw_impact_f1": raw_metrics["f1"],
                "Estimator/fullcorr_probability_positive_mean": self._masked_mean(
                    impact_probability, gt_impact
                ),
                "Estimator/fullcorr_probability_negative_mean": self._masked_mean(
                    impact_probability, ~gt_impact
                ),
                "Estimator/fullcorr_pred_ratio": fullcorr_metrics["pred_ratio"],
                "Estimator/fullcorr_precision": fullcorr_metrics["precision"],
                "Estimator/fullcorr_recall": fullcorr_metrics["recall"],
                "Estimator/fullcorr_f1": fullcorr_metrics["f1"],
                "Estimator/force_target_clip_ratio": torch.mean(
                    (gt_force >= target_force_clip_n).float()
                ),
            }
        )

        # Offline calibration sweep over fractions of log(pos_weight). None of
        # these diagnostic probabilities enters the active controller.
        for label, fraction in (
            ("0p00", 0.00),
            ("0p25", 0.25),
            ("0p50", 0.50),
            ("0p75", 0.75),
            ("1p00", 1.00),
        ):
            probability = torch.sigmoid(
                raw_impact_logits
                - fraction * self.admittance.impact_pos_weight_log_correction
            )
            metrics = self._binary_probability_diagnostics(
                probability, gt_impact, impact_threshold
            )
            for metric_name, value in metrics.items():
                self._last_admittance_diagnostics[
                    f"Calibration/corr_{label}_{metric_name}"
                ] = value

        # Optional raw-probability threshold sweep, also diagnostics only.
        for label, threshold in (
            ("0p30", 0.30),
            ("0p40", 0.40),
            ("0p50", 0.50),
            ("0p60", 0.60),
            ("0p70", 0.70),
        ):
            metrics = self._binary_probability_diagnostics(
                raw_impact_probability, gt_impact, threshold
            )
            for metric_name, value in metrics.items():
                self._last_admittance_diagnostics[
                    f"Threshold/raw_{label}_{metric_name}"
                ] = value

    def _cache_admittance_diagnostics(self):
        cfg = self.cfg.learned_admittance
        alpha = self.admittance.alpha
        effective_alpha = self.admittance.effective_alpha
        compression_m = self.admittance.delta_l
        impact_probability = self.admittance.impact_probability
        transient = self.admittance.transient_force
        drive = self.admittance.drive_force
        stiffness = self.admittance.stiffness
        support_bias = self.admittance.force_bias
        joint_offsets = torch.abs(self.admittance.last_joint_offsets)

        alpha_active_threshold = float(
            getattr(cfg, "diagnostic_alpha_active_threshold", 0.05)
        )
        impact_probability_threshold = float(
            getattr(cfg, "diagnostic_impact_probability_threshold", 0.5)
        )
        predicted_impact = impact_probability >= impact_probability_threshold
        # y_t describes the physical transition that just completed. These two
        # legacy metrics are current-transition correlation diagnostics.
        gt_impact = self.contact_estimator_target[:, 4:] >= 0.5
        # y_(t-1) is causally aligned with the proprioceptive classifier state
        # that controlled this transition; it is not a future-impact target.
        control_aligned_gt_impact = self.control_aligned_gt_impact
        loading_rate = self.gt_step_peak_axial_loading_rate

        self._last_admittance_diagnostics = {
            "Admittance/alpha_mean": torch.mean(alpha),
            "Admittance/alpha_p95": self._p95(alpha),
            "Admittance/alpha_max": torch.max(alpha),
            "Admittance/alpha_active_ratio": torch.mean(
                (alpha > alpha_active_threshold).float()
            ),
            "Admittance/effective_alpha_mean": torch.mean(effective_alpha),
            "Admittance/effective_alpha_p95": self._p95(effective_alpha),
            "Admittance/effective_alpha_max": torch.max(effective_alpha),
            "Admittance/compression_mean_mm": torch.mean(compression_m) * 1000.0,
            "Admittance/compression_p95_mm": self._p95(compression_m) * 1000.0,
            "Admittance/compression_max_mm": torch.max(compression_m) * 1000.0,
            "Admittance/compression_saturation_ratio": torch.mean(
                (
                    compression_m
                    >= 0.95 * float(cfg.max_compression_m)
                ).float()
            ),
            "Admittance/compression_over_10mm_ratio": torch.mean(
                (compression_m >= 0.010).float()
            ),
            "Admittance/compression_when_impact_mm": self._masked_mean(
                compression_m, gt_impact
            ) * 1000.0,
            "Admittance/compression_when_noimpact_mm": self._masked_mean(
                compression_m, ~gt_impact
            ) * 1000.0,
            "Admittance/compression_after_impact_mm": self._masked_mean(
                compression_m, control_aligned_gt_impact
            ) * 1000.0,
            "Admittance/compression_after_noimpact_mm": self._masked_mean(
                compression_m, ~control_aligned_gt_impact
            ) * 1000.0,
            "Admittance/joint_offset_abs_mean_rad": torch.mean(joint_offsets),
            "Admittance/joint_offset_abs_max_rad": torch.max(joint_offsets),
            "Admittance/impact_probability_mean": torch.mean(impact_probability),
            "Admittance/impact_probability_p95": self._p95(impact_probability),
            "Admittance/impact_active_ratio": torch.mean(
                predicted_impact.float()
            ),
            "Admittance/transient_force_mean_n": torch.mean(transient),
            "Admittance/transient_force_max_n": torch.max(transient),
            "Admittance/drive_force_mean_n": torch.mean(drive),
            "Admittance/drive_force_max_n": torch.max(drive),
            "Admittance/stiffness_mean_npm": torch.mean(stiffness),
            "Admittance/support_bias_mean_n": torch.mean(support_bias),
            "Impact/gt_3d_force_peak_mean_n": torch.mean(self.gt_step_peak_force),
            "Impact/gt_3d_force_peak_max_n": torch.max(self.gt_step_peak_force),
            "Impact/gt_3d_loading_peak_mean_nps": torch.mean(
                self.gt_step_peak_loading_rate
            ),
            "Impact/gt_3d_loading_peak_max_nps": torch.max(
                self.gt_step_peak_loading_rate
            ),
            "Impact/gt_base_acc_peak_mean_mps2": torch.mean(
                self.gt_step_peak_base_acc
            ),
            "Impact/gt_base_acc_peak_max_mps2": torch.max(
                self.gt_step_peak_base_acc
            ),
        }

        # Diagnostic-only distribution of the unchanged 5000 N/s hard label.
        # Ratios use every leg sample from the just-completed transition.
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
            in_bin = (loading_rate >= lower) & (loading_rate < upper)
            self._last_admittance_diagnostics[
                f"LabelDist/bin_{name}_ratio"
            ] = torch.mean(in_bin.float())
        self._last_admittance_diagnostics["LabelDist/bin_gt_8000_ratio"] = (
            torch.mean((loading_rate >= 8000.0).float())
        )
        self._last_admittance_diagnostics[
            "LabelDist/near_threshold_ratio"
        ] = torch.mean(
            ((loading_rate >= 4500.0) & (loading_rate <= 5500.0)).float()
        )
        self._last_admittance_diagnostics[
            "LabelDist/very_near_threshold_ratio"
        ] = torch.mean(
            ((loading_rate >= 4750.0) & (loading_rate <= 5250.0)).float()
        )

    def _begin_gt_impact_step(self):
        self.gt_step_peak_force.zero_()
        self.gt_step_peak_loading_rate.zero_()
        self.gt_step_peak_axial_force.zero_()
        self.gt_step_peak_axial_loading_rate.zero_()
        self.gt_step_peak_base_acc.zero_()

    def _ground_truth_contact_signals(self):
        force_vec = self.contact_forces[:, self.adm_feet_indices, :]
        force_norm = torch.norm(force_vec, dim=-1)

        states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        hip_pos = states[:, self.adm_hip_body_indices, 0:3]
        wheel_pos = states[:, self.adm_feet_indices, 0:3]
        leg_vec = wheel_pos - hip_pos
        leg_axis = leg_vec / torch.clamp(
            torch.norm(leg_vec, dim=-1, keepdim=True), min=1.0e-6
        )
        axial_force = torch.clamp(
            torch.sum(force_vec * (-leg_axis), dim=-1), min=0.0
        )
        return force_norm, axial_force

    def _update_gt_impact_substep(self):
        physics_dt = float(self.sim_params.dt)
        force_norm, axial_force = self._ground_truth_contact_signals()

        loading_rate = torch.clamp(
            (force_norm - self.gt_prev_force_norm) / physics_dt, min=0.0
        )
        axial_loading_rate = torch.clamp(
            (axial_force - self.gt_prev_axial_force) / physics_dt, min=0.0
        )
        if torch.any(self.gt_skip_rate_once):
            mask = self.gt_skip_rate_once.unsqueeze(1)
            loading_rate = torch.where(mask, torch.zeros_like(loading_rate), loading_rate)
            axial_loading_rate = torch.where(
                mask, torch.zeros_like(axial_loading_rate), axial_loading_rate
            )
            self.gt_skip_rate_once.zero_()

        self.gt_prev_force_norm.copy_(force_norm)
        self.gt_prev_axial_force.copy_(axial_force)

        base_vel_z = self._base_vel_z()
        base_acc = torch.abs((base_vel_z - self.gt_prev_base_vel_z) / physics_dt)
        self.gt_prev_base_vel_z.copy_(base_vel_z)

        self.gt_step_peak_force = torch.maximum(self.gt_step_peak_force, force_norm)
        self.gt_step_peak_loading_rate = torch.maximum(
            self.gt_step_peak_loading_rate, loading_rate
        )
        self.gt_step_peak_axial_force = torch.maximum(
            self.gt_step_peak_axial_force, axial_force
        )
        self.gt_step_peak_axial_loading_rate = torch.maximum(
            self.gt_step_peak_axial_loading_rate, axial_loading_rate
        )
        self.gt_step_peak_base_acc = torch.maximum(
            self.gt_step_peak_base_acc, base_acc
        )

    def _finish_gt_impact_step(self):
        cfg = self.cfg.learned_admittance
        force = self.gt_step_peak_axial_force / float(cfg.contact_force_scale_n)
        impact_label = (
            self.gt_step_peak_axial_loading_rate
            > float(cfg.contact_impact_threshold_nps)
        ).float()
        clip = float(cfg.contact_target_clip)
        self.contact_estimator_target = torch.cat(
            (torch.clamp(force, 0.0, clip), impact_label),
            dim=-1,
        )

    def _split_policy_action(self, policy_actions):
        if policy_actions.shape[-1] == self.num_motion_actions:
            compliance = torch.zeros(
                policy_actions.shape[0],
                self.num_compliance_actions,
                device=policy_actions.device,
                dtype=policy_actions.dtype,
            )
            return policy_actions, compliance
        if policy_actions.shape[-1] != self.num_policy_actions:
            raise RuntimeError(
                f"Expected {self.num_policy_actions} policy actions or legacy "
                f"{self.num_motion_actions} reset actions, got {policy_actions.shape[-1]}"
            )
        return (
            policy_actions[:, : self.num_motion_actions],
            policy_actions[:, self.num_motion_actions :],
        )

    def _compute_adaptive_torques(
        self, motion_actions, compliance_actions, estimated_contact
    ):
        actions_scaled = motion_actions * self.cfg.control.action_scale
        actions_scaled = actions_scaled.clone()
        actions_scaled[:, self.wheel_indices] = 0.0

        q_target = self.default_dof_pos + actions_scaled
        offsets = self.admittance.step(
            compliance_actions,
            estimated_contact,
            q_target,
            self.adm_hip_indices,
            self.adm_knee_indices,
            float(self.sim_params.dt),
        )
        q_target[:, self.adm_hip_indices] += offsets[:, :, 0]
        q_target[:, self.adm_knee_indices] += offsets[:, :, 1]

        for indices in (self.adm_hip_indices, self.adm_knee_indices):
            q_target[:, indices] = torch.maximum(
                torch.minimum(
                    q_target[:, indices],
                    self.dof_pos_limits[indices, 1].unsqueeze(0),
                ),
                self.dof_pos_limits[indices, 0].unsqueeze(0),
            )

        pos_err = q_target - self.dof_pos
        pos_err[:, self.wheel_indices] = 0.0
        vel_ref = torch.zeros_like(actions_scaled)
        vel_tmp = motion_actions * self.cfg.control.vel_scale
        vel_ref[:, self.wheel_indices] = vel_tmp[:, self.wheel_indices]

        if self.cfg.control.control_type != "P":
            raise RuntimeError("Learned admittance v1 currently requires control_type='P'.")
        torques = (
            self.p_gains * self.Kp_factors * pos_err
            + self.d_gains * self.Kd_factors * (vel_ref - self.dof_vel)
        )
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def step(self, policy_actions, contact_estimate=None):
        # Cache y_(t-1) before this step creates y_t. This buffer is used only
        # to diagnose the compression caused by the classifier state controlling
        # the upcoming physical transition.
        self.control_aligned_gt_impact.copy_(
            self.transition_contact_estimator_target[:, 4:] >= 0.5
        )

        motion_actions, compliance_actions = self._split_policy_action(policy_actions)
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(motion_actions, -clip_actions, clip_actions).to(
            self.device
        )
        self.compliance_actions = torch.clamp(
            compliance_actions.to(self.device), 0.0, 1.0
        )
        self.policy_actions = torch.cat((self.actions, self.compliance_actions), dim=-1)

        if contact_estimate is None:
            self.estimated_contact.zero_()
        else:
            if contact_estimate.shape[-1] != self.contact_estimate_dim:
                raise RuntimeError(
                    f"Expected {self.contact_estimate_dim}D contact estimate, got "
                    f"{contact_estimate.shape[-1]}D"
                )
            self.estimated_contact.copy_(contact_estimate.to(self.device))

        self.delayed_actions = self.actions.clone().view(
            self.num_envs, 1, self.num_actions
        ).repeat(1, self.cfg.control.decimation, 1)
        delay_steps = torch.randint(
            0, self.cfg.control.decimation, (self.num_envs, 1), device=self.device
        )
        if self.cfg.domain_rand.delay:
            for i in range(self.cfg.control.decimation):
                self.delayed_actions[:, i] = self.last_actions + (
                    self.actions - self.last_actions
                ) * (i >= delay_steps)

        self._begin_gt_impact_step()
        self.render()
        for substep in range(self.cfg.control.decimation):
            self.torques = self._compute_adaptive_torques(
                self.delayed_actions[:, substep],
                self.compliance_actions,
                self.estimated_contact,
            ).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self._update_gt_impact_substep()

        self._finish_gt_impact_step()
        self.transition_contact_estimator_target.copy_(self.contact_estimator_target)
        self.transition_gt_axial_loading_rate.copy_(
            self.gt_step_peak_axial_loading_rate
        )
        self._cache_admittance_diagnostics()
        termination_ids, termination_privileged_obs = self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs
            )
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            termination_ids,
            termination_privileged_obs,
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0 or not hasattr(self, "admittance"):
            return
        self.admittance.reset(env_ids)
        self.policy_actions[env_ids] = 0.0
        self.compliance_actions[env_ids] = 0.0
        self.estimated_contact[env_ids] = 0.0
        self.gt_step_peak_force[env_ids] = 0.0
        self.gt_step_peak_loading_rate[env_ids] = 0.0
        self.gt_prev_force_norm[env_ids] = 0.0
        self.gt_step_peak_axial_force[env_ids] = 0.0
        self.gt_step_peak_axial_loading_rate[env_ids] = 0.0
        self.gt_prev_axial_force[env_ids] = 0.0
        self.gt_skip_rate_once[env_ids] = True
        self.gt_step_peak_base_acc[env_ids] = 0.0
        self.contact_estimator_target[env_ids] = 0.0
        # The observation returned for a terminated environment is already the
        # reset state, so it must not be paired with the terminal impact label.
        self.transition_contact_estimator_target[env_ids] = 0.0
        self.transition_gt_axial_loading_rate[env_ids] = 0.0
        self.control_aligned_gt_impact[env_ids] = False
        self.gt_prev_base_vel_z[env_ids] = self._base_vel_z()[env_ids]

    def _reward_quiet_impact_force(self):
        cfg = self.cfg.quiet_training
        excess = torch.clamp(
            self.gt_step_peak_force - float(cfg.force_threshold_n), min=0.0
        ) / float(cfg.force_normalizer_n)
        return torch.mean(excess, dim=1)

    def _reward_quiet_loading_rate(self):
        cfg = self.cfg.quiet_training
        excess = torch.clamp(
            self.gt_step_peak_loading_rate - float(cfg.loading_rate_threshold_nps),
            min=0.0,
        ) / float(cfg.loading_rate_normalizer_nps)
        return torch.mean(excess, dim=1)

    def _reward_quiet_base_acc(self):
        return self.gt_step_peak_base_acc / float(
            self.cfg.quiet_training.base_acc_normalizer_mps2
        )

    def _reward_compliance_usage(self):
        return torch.mean(torch.square(self.admittance.alpha), dim=1)

    def _reward_admittance_displacement(self):
        scale = max(float(self.cfg.learned_admittance.max_compression_m), 1.0e-6)
        return torch.mean(torch.square(self.admittance.delta_l / scale), dim=1)
