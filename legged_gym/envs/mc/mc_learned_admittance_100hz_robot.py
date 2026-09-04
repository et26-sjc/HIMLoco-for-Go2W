"""100 Hz MC environment for learned sensorless admittance.

The environment separates three information domains explicitly:

* ``self.actions``: original 16-D HIMLoco motion action (the original 57-D
  proprioceptive observation layout remains unchanged);
* policy-only extra action: four compliance activations;
* training-only ground-truth impact signals from Isaac Gym contact tensors.

The deployed admittance never reads ground-truth contact force. During training,
3-D wheel contact force is used in two different ways: force norm/loading-rate
norm supervise quiet rewards, while the force projected onto each hip-to-wheel
axis supervises the contact estimator that drives the physical admittance.
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

    def _cache_admittance_diagnostics(self):
        cfg = self.cfg.learned_admittance
        alpha = self.admittance.alpha
        compression_m = self.admittance.delta_l
        est_force = self.admittance.estimated_force
        est_loading = self.admittance.estimated_loading_rate
        gt_force = self.gt_step_peak_axial_force
        gt_loading = self.gt_step_peak_axial_loading_rate
        gate = self.admittance.loading_gate
        transient = self.admittance.transient_force
        drive = self.admittance.drive_force
        stiffness = self.admittance.stiffness
        support_bias = self.admittance.force_bias
        joint_offsets = torch.abs(self.admittance.last_joint_offsets)

        alpha_active_threshold = float(
            getattr(cfg, "diagnostic_alpha_active_threshold", 0.05)
        )
        gate_active_threshold = float(
            getattr(cfg, "diagnostic_gate_active_threshold", 0.5)
        )
        target_force_clip_n = (
            float(cfg.contact_target_clip) * float(cfg.contact_force_scale_n)
        )
        target_loading_clip_nps = (
            float(cfg.contact_target_clip)
            * float(cfg.contact_loading_rate_scale_nps)
        )

        self._last_admittance_diagnostics = {
            "Admittance/alpha_mean": torch.mean(alpha),
            "Admittance/alpha_p95": self._p95(alpha),
            "Admittance/alpha_max": torch.max(alpha),
            "Admittance/alpha_active_ratio": torch.mean(
                (alpha > alpha_active_threshold).float()
            ),
            "Admittance/compression_mean_mm": torch.mean(compression_m) * 1000.0,
            "Admittance/compression_p95_mm": self._p95(compression_m) * 1000.0,
            "Admittance/compression_max_mm": torch.max(compression_m) * 1000.0,
            "Admittance/joint_offset_abs_mean_rad": torch.mean(joint_offsets),
            "Admittance/joint_offset_abs_max_rad": torch.max(joint_offsets),
            "Admittance/gate_mean": torch.mean(gate),
            "Admittance/gate_active_ratio": torch.mean(
                (gate > gate_active_threshold).float()
            ),
            "Admittance/transient_force_mean_n": torch.mean(transient),
            "Admittance/transient_force_max_n": torch.max(transient),
            "Admittance/drive_force_mean_n": torch.mean(drive),
            "Admittance/drive_force_max_n": torch.max(drive),
            "Admittance/stiffness_mean_npm": torch.mean(stiffness),
            "Admittance/support_bias_mean_n": torch.mean(support_bias),
            "Estimator/axial_force_pred_mean_n": torch.mean(est_force),
            "Estimator/axial_force_pred_max_n": torch.max(est_force),
            "Estimator/axial_force_gt_mean_n": torch.mean(gt_force),
            "Estimator/axial_force_gt_max_n": torch.max(gt_force),
            "Estimator/axial_force_mae_n": torch.mean(torch.abs(est_force - gt_force)),
            "Estimator/loading_pred_mean_nps": torch.mean(est_loading),
            "Estimator/loading_pred_max_nps": torch.max(est_loading),
            "Estimator/loading_gt_mean_nps": torch.mean(gt_loading),
            "Estimator/loading_gt_max_nps": torch.max(gt_loading),
            "Estimator/loading_mae_nps": torch.mean(
                torch.abs(est_loading - gt_loading)
            ),
            "Estimator/force_target_clip_ratio": torch.mean(
                (gt_force >= target_force_clip_n).float()
            ),
            "Estimator/loading_target_clip_ratio": torch.mean(
                (gt_loading >= target_loading_clip_nps).float()
            ),
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
        loading = self.gt_step_peak_axial_loading_rate / float(
            cfg.contact_loading_rate_scale_nps
        )
        clip = float(cfg.contact_target_clip)
        self.contact_estimator_target = torch.cat(
            (torch.clamp(force, 0.0, clip), torch.clamp(loading, 0.0, clip)),
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
