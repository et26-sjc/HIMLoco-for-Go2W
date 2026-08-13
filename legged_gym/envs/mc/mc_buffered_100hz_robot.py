"""Contact-aware gain scheduling for the 100 Hz MC experiment.

This is intentionally a minimal buffering controller, not admittance control.
The policy/action space and wheel velocity control remain unchanged.  When a
wheel experiences a new contact or a sharp rise in total contact-force
magnitude, HIP/KNEE gains are softened for a short hold window:

    Kp -> kp_scale * Kp
    Kd -> kd_scale * Kd

The impact detector runs at the 200 Hz physics rate.  The default loading-rate
threshold is chosen above the observed flat-ground p99 (~14 kN/s) and below the
large stair-impact tail, so normal rolling should rarely trigger the buffer.
"""

from isaacgym import gymtorch
import torch

from .mc_robot import MC


class MCBuffered100Hz(MC):
    """MC with a short contact-triggered compliant HIP/KNEE response."""

    def _init_buffers(self):
        super()._init_buffers()
        cfg = self.cfg.buffer_control
        self.buffer_contact_on = float(cfg.contact_on_threshold_n)
        self.buffer_loading_rate_threshold = float(
            cfg.loading_rate_threshold_nps
        )
        self.buffer_hold_steps = max(
            1, int(round(float(cfg.hold_time_s) / float(self.sim_params.dt)))
        )
        self.buffer_kp_scale = float(cfg.hip_knee_kp_scale)
        self.buffer_kd_scale = float(cfg.hip_knee_kd_scale)

        force_norm = torch.norm(
            self.contact_forces[:, self.feet_indices, :], dim=-1
        )
        self.buffer_prev_force_norm = force_norm.clone()
        self.buffer_timer = torch.zeros(
            self.num_envs,
            int(self.feet_indices.numel()),
            dtype=torch.long,
            device=self.device,
        )
        self.buffer_last_loading_rate = torch.zeros_like(force_norm)
        self.buffer_last_impact = torch.zeros_like(force_norm, dtype=torch.bool)

    def _update_buffer_state(self):
        """Update impact detector from the most recently refreshed contact force."""
        force_norm = torch.norm(
            self.contact_forces[:, self.feet_indices, :], dim=-1
        )
        loading_rate = (
            force_norm - self.buffer_prev_force_norm
        ) / float(self.sim_params.dt)

        new_contact = (
            (self.buffer_prev_force_norm < self.buffer_contact_on)
            & (force_norm >= self.buffer_contact_on)
        )
        sharp_loading = loading_rate >= self.buffer_loading_rate_threshold
        impact = new_contact | sharp_loading

        self.buffer_timer = torch.clamp(self.buffer_timer - 1, min=0)
        self.buffer_timer = torch.where(
            impact,
            torch.full_like(self.buffer_timer, self.buffer_hold_steps),
            self.buffer_timer,
        )
        self.buffer_prev_force_norm.copy_(force_norm)
        self.buffer_last_loading_rate.copy_(loading_rate)
        self.buffer_last_impact.copy_(impact)

    def _gain_scales(self):
        """Return per-env/per-DOF Kp and Kd multipliers."""
        soft = self.buffer_timer > 0
        soft_kp = torch.where(
            soft,
            torch.full_like(soft, self.buffer_kp_scale, dtype=torch.float),
            torch.ones_like(soft, dtype=torch.float),
        )
        soft_kd = torch.where(
            soft,
            torch.full_like(soft, self.buffer_kd_scale, dtype=torch.float),
            torch.ones_like(soft, dtype=torch.float),
        )

        kp_scale = torch.ones(
            self.num_envs, self.num_dof, device=self.device, dtype=torch.float
        )
        kd_scale = torch.ones_like(kp_scale)

        # DOF group ordering is FR, FL, RR, RL for both the wheel contacts and
        # semantic HIP/KNEE indices in the imported MC asset.
        kp_scale[:, self.hip_indices] = soft_kp
        kp_scale[:, self.knee_indices] = soft_kp
        kd_scale[:, self.hip_indices] = soft_kd
        kd_scale[:, self.knee_indices] = soft_kd
        return kp_scale, kd_scale

    def _compute_torques(self, actions):
        """Original mixed leg-position/wheel-velocity controller + gain schedule."""
        self._update_buffer_state()
        kp_scale, kd_scale = self._gain_scales()

        dof_err = self.default_dof_pos - self.dof_pos
        dof_err = dof_err.clone()
        dof_err[:, self.wheel_indices] = 0.0

        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled = actions_scaled.clone()
        actions_scaled[:, self.wheel_indices] = 0.0

        vel_ref = torch.zeros_like(actions_scaled)
        vel_tmp = actions * self.cfg.control.vel_scale
        vel_ref[:, self.wheel_indices] = vel_tmp[:, self.wheel_indices]

        control_type = self.cfg.control.control_type
        if control_type == "P":
            p = (
                self.p_gains.unsqueeze(0)
                * self.Kp_factors
                * kp_scale
            )
            d = (
                self.d_gains.unsqueeze(0)
                * self.Kd_factors
                * kd_scale
            )
            torques = p * (actions_scaled + dof_err) + d * (
                vel_ref - self.dof_vel
            )
        elif control_type == "V":
            torques = self.p_gains * (actions_scaled - self.dof_vel) - self.d_gains * (
                self.dof_vel - self.last_dof_vel
            ) / self.sim_params.dt
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def step(self, actions):
        """Base MC step with contact forces refreshed at every physics substep."""
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        self.delayed_actions = self.actions.clone().view(
            self.num_envs, 1, self.num_actions
        ).repeat(1, self.cfg.control.decimation, 1)
        delay_steps = torch.randint(
            0,
            self.cfg.control.decimation,
            (self.num_envs, 1),
            device=self.device,
        )
        if self.cfg.domain_rand.delay:
            for i in range(self.cfg.control.decimation):
                self.delayed_actions[:, i] = self.last_actions + (
                    self.actions - self.last_actions
                ) * (i >= delay_steps)

        self.render()
        for substep in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(
                self.delayed_actions[:, substep]
            ).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            # Needed so the next 5 ms control update sees the latest wheel load.
            self.gym.refresh_net_contact_force_tensor(self.sim)

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
        if len(env_ids) == 0 or not hasattr(self, "buffer_timer"):
            return
        self.buffer_timer[env_ids] = 0
        self.buffer_last_loading_rate[env_ids] = 0.0
        self.buffer_last_impact[env_ids] = False
        force_norm = torch.norm(
            self.contact_forces[:, self.feet_indices, :], dim=-1
        )
        self.buffer_prev_force_norm[env_ids] = force_norm[env_ids]
