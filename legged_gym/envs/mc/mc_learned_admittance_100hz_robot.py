"""100 Hz MC environment for learned sensorless admittance.

The environment separates three information domains explicitly:

* ``self.actions``: original 16-D HIMLoco motion action (and therefore the
  original 57-D proprioceptive observation layout remains unchanged);
* policy-only extra action: four compliance activations;
* training-only ground-truth impact signals from Isaac Gym contact tensors.

The physical admittance never reads ground-truth contact force.  It is driven
only by the contact estimator output supplied by the policy/runner.
"""

from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse
import torch

from .mc_robot import MC
from .mc_learned_admittance import MCLearnedAdmittance


class MCLearnedAdmittance100Hz(MC):
    """MC with 20-D policy action but unchanged 16-D physical actuation."""

    _LEG_SPECS = [
        # semantic leg, foot body, hip joint, knee joint
        ("FL", "FL_FOOT_LINK", "FBL_HIP_JOINT", "FBL_KNEE_JOINT"),
        ("FR", "FR_FOOT_LINK", "FAR_HIP_JOINT", "FAR_KNEE_JOINT"),
        ("RR", "RR_FOOT_LINK", "RAR_HIP_JOINT", "RAR_KNEE_JOINT"),
        ("RL", "RL_FOOT_LINK", "RBL_HIP_JOINT", "RBL_KNEE_JOINT"),
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

        foot_ids, hip_ids, knee_ids = [], [], []
        for leg, foot_name, hip_name, knee_name in self._LEG_SPECS:
            foot_id = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], foot_name
            )
            hip_id = self.gym.find_actor_dof_handle(
                self.envs[0], self.actor_handles[0], hip_name
            )
            knee_id = self.gym.find_actor_dof_handle(
                self.envs[0], self.actor_handles[0], knee_name
            )
            if min(foot_id, hip_id, knee_id) < 0:
                raise RuntimeError(
                    f"Failed to resolve semantic leg {leg}: "
                    f"foot={foot_id}, hip={hip_id}, knee={knee_id}"
                )
            foot_ids.append(foot_id)
            hip_ids.append(hip_id)
            knee_ids.append(knee_id)

        self.adm_feet_indices = torch.tensor(
            foot_ids, dtype=torch.long, device=self.device
        )
        self.adm_hip_indices = torch.tensor(
            hip_ids, dtype=torch.long, device=self.device
        )
        self.adm_knee_indices = torch.tensor(
            knee_ids, dtype=torch.long, device=self.device
        )

        print("### Learned-admittance leg order:")
        for i, spec in enumerate(self._LEG_SPECS):
            print(
                f"  {i}:{spec[0]} foot={foot_ids[i]} "
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

        # Training-only, 200 Hz impact target buffers. These are never returned as
        # actor observations and never used by the deployed admittance controller.
        shape = (self.num_envs, 4)
        self.gt_step_peak_force = torch.zeros(shape, device=self.device)
        self.gt_step_peak_loading_rate = torch.zeros(shape, device=self.device)
        self.gt_prev_force_norm = torch.zeros(shape, device=self.device)
        self.gt_step_peak_base_acc = torch.zeros(self.num_envs, device=self.device)
        self.gt_prev_base_vel_z = self._base_vel_z().clone()
        self.contact_estimator_target = torch.zeros(
            self.num_envs, self.contact_estimate_dim, device=self.device
        )

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
        """Return normalized training-only [peak F(4), peak positive dF(4)]."""
        return self.contact_estimator_target

    def _begin_gt_impact_step(self):
        self.gt_step_peak_force.zero_()
        self.gt_step_peak_loading_rate.zero_()
        self.gt_step_peak_base_acc.zero_()

    def _update_gt_impact_substep(self):
        physics_dt = float(self.sim_params.dt)
        force_vec = self.contact_forces[:, self.adm_feet_indices, :]
        force_norm = torch.norm(force_vec, dim=-1)
        loading_rate = torch.clamp(
            (force_norm - self.gt_prev_force_norm) / physics_dt, min=0.0
        )
        self.gt_prev_force_norm.copy_(force_norm)

        base_vel_z = self._base_vel_z()
        base_acc = torch.abs(
            (base_vel_z - self.gt_prev_base_vel_z) / physics_dt
        )
        self.gt_prev_base_vel_z.copy_(base_vel_z)

        self.gt_step_peak_force = torch.maximum(
            self.gt_step_peak_force, force_norm
        )
        self.gt_step_peak_loading_rate = torch.maximum(
            self.gt_step_peak_loading_rate, loading_rate
        )
        self.gt_step_peak_base_acc = torch.maximum(
            self.gt_step_peak_base_acc, base_acc
        )

    def _finish_gt_impact_step(self):
        cfg = self.cfg.learned_admittance
        force = self.gt_step_peak_force / float(cfg.contact_force_scale_n)
        loading = (
            self.gt_step_peak_loading_rate
            / float(cfg.contact_loading_rate_scale_nps)
        )
        clip = float(cfg.contact_target_clip)
        self.contact_estimator_target = torch.cat(
            (torch.clamp(force, 0.0, clip), torch.clamp(loading, 0.0, clip)),
            dim=-1,
        )

    def _split_policy_action(self, policy_actions):
        if policy_actions.shape[-1] == self.num_motion_actions:
            # BaseTask.reset() still supplies the legacy 16-D zero action. Treat
            # it as an exact baseline reset with compliance disabled.
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
                f"{self.num_motion_actions} reset actions, got "
                f"{policy_actions.shape[-1]}"
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

        # Compliance must never bypass the existing joint safety envelope.
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
            raise RuntimeError(
                "Learned admittance v1 is defined around the MC fixed-PD "
                "controller and currently requires control_type='P'."
            )

        torques = (
            self.p_gains * self.Kp_factors * pos_err
            + self.d_gains * self.Kd_factors * (vel_ref - self.dof_vel)
        )
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def step(self, policy_actions, contact_estimate=None):
        """Execute a 20-D policy action while physically actuating only 16 DOFs."""
        motion_actions, compliance_actions = self._split_policy_action(
            policy_actions
        )

        clip_actions = self.cfg.normalization.clip_actions
        self.policy_actions = torch.clip(
            policy_actions, -clip_actions, clip_actions
        ).to(self.device)
        self.actions = torch.clip(
            motion_actions, -clip_actions, clip_actions
        ).to(self.device)
        self.compliance_actions = torch.clamp(
            compliance_actions.to(self.device), 0.0, 1.0
        )

        if contact_estimate is None:
            self.estimated_contact.zero_()
        else:
            if contact_estimate.shape[-1] != self.contact_estimate_dim:
                raise RuntimeError(
                    f"Expected {self.contact_estimate_dim}D contact estimate, "
                    f"got {contact_estimate.shape[-1]}D"
                )
            self.estimated_contact.copy_(contact_estimate.to(self.device))

        # Preserve the exact baseline motor-delay path for the original 16D
        # motion action. Compliance is a supervisory outer-loop command and is
        # held over the two 200 Hz physics substeps of each 100 Hz policy step.
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
        self.gt_step_peak_base_acc[env_ids] = 0.0
        self.contact_estimator_target[env_ids] = 0.0
        self.gt_prev_base_vel_z[env_ids] = self._base_vel_z()[env_ids]

    # ------------------------------------------------------------------
    # Training-only quiet rewards. All use simulator truth, but none enter
    # inference observations. Values are normalized so reward scales are stable.
    # ------------------------------------------------------------------
    def _reward_quiet_impact_force(self):
        cfg = self.cfg.quiet_training
        excess = torch.clamp(
            self.gt_step_peak_force - float(cfg.force_threshold_n), min=0.0
        ) / float(cfg.force_normalizer_n)
        return torch.mean(excess, dim=1)

    def _reward_quiet_loading_rate(self):
        cfg = self.cfg.quiet_training
        excess = torch.clamp(
            self.gt_step_peak_loading_rate
            - float(cfg.loading_rate_threshold_nps),
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
