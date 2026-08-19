"""Per-leg axial admittance for the 100 Hz MC HIMLoco policy.

The HIM policy is unchanged.  At every 200 Hz physics substep, the controller:

1. projects each wheel contact force onto the current hip-to-wheel leg axis;
2. optionally detects a fast axial loading event and opens a short admittance gate;
3. removes a slowly varying support-force baseline and a small deadband;
4. integrates a virtual mass-spring-damper to obtain extra leg compression;
5. maps that Cartesian compression to HIP/KNEE offsets with a damped Jacobian;
6. executes the modified target with the original fixed PD gains.

The original A0 controller keeps the gate disabled.  A0.1 can enable the gate so
normal flat rolling does not continuously start new admittance events, while
large stair impacts still generate a short-lived compliant displacement.
"""

from isaacgym import gymtorch
import torch

from .mc_robot import MC


class MCAdmittance100Hz(MC):
    """MC with a 200 Hz axial admittance outer loop around the leg PD loops."""

    _HIP_BODY_NAMES = [
        "FR_HIP_LINK",
        "FL_HIP_LINK",
        "RR_HIP_LINK",
        "RL_HIP_LINK",
    ]

    quiet_controller_name = "axial_admittance_outer_loop_fixed_pd_inner_loop"
    quiet_baseline_name = "MC_HIM_100Hz_Minimal_axial_admittance_zeroshot"

    def _init_buffers(self):
        super()._init_buffers()
        cfg = self.cfg.admittance_control

        self.admittance_enabled = bool(cfg.enabled)
        self.admittance_mass = float(cfg.virtual_mass_kg)
        self.admittance_damping = float(cfg.virtual_damping_ns_per_m)
        self.admittance_stiffness = float(cfg.virtual_stiffness_n_per_m)
        self.admittance_force_bias_tau = float(cfg.force_bias_time_constant_s)
        self.admittance_force_deadband = float(cfg.force_deadband_n)
        self.admittance_max_force_input = float(cfg.max_force_input_n)
        self.admittance_max_compression = float(cfg.max_compression_m)
        self.admittance_max_compression_vel = float(
            cfg.max_compression_velocity_mps
        )
        self.admittance_max_joint_offset = float(cfg.max_joint_offset_rad)
        self.admittance_l1 = float(cfg.upper_leg_length_m)
        self.admittance_l2 = float(cfg.lower_leg_length_m)
        self.admittance_jacobian_damping = float(cfg.jacobian_damping)

        # Optional A0.1 impact gate.  getattr keeps the original A0 config
        # backward compatible and behavior-identical when these fields are absent.
        self.admittance_use_loading_rate_gate = bool(
            getattr(cfg, "use_loading_rate_gate", False)
        )
        self.admittance_loading_rate_gate = float(
            getattr(cfg, "loading_rate_gate_nps", 0.0)
        )
        self.admittance_gate_hold_time = float(
            getattr(cfg, "gate_hold_time_s", 0.0)
        )
        self.admittance_freeze_bias_during_gate = bool(
            getattr(cfg, "freeze_force_bias_during_gate", False)
        )

        self.admittance_hip_body_indices = torch.tensor(
            [
                self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0], name
                )
                for name in self._HIP_BODY_NAMES
            ],
            dtype=torch.long,
            device=self.device,
        )
        if torch.any(self.admittance_hip_body_indices < 0):
            raise RuntimeError(
                "Could not resolve MC hip bodies for admittance: "
                f"{self._HIP_BODY_NAMES} -> "
                f"{self.admittance_hip_body_indices.tolist()}"
            )
        if self.admittance_hip_body_indices.numel() != self.feet_indices.numel():
            raise RuntimeError("Admittance hip/wheel body count mismatch")

        shape = (self.num_envs, int(self.feet_indices.numel()))
        self.admittance_delta_l = torch.zeros(
            *shape, dtype=torch.float, device=self.device
        )
        self.admittance_delta_l_dot = torch.zeros_like(self.admittance_delta_l)
        self.admittance_force_bias = torch.zeros_like(self.admittance_delta_l)
        self.admittance_axial_force = torch.zeros_like(self.admittance_delta_l)
        self.admittance_force_input = torch.zeros_like(self.admittance_delta_l)
        self.admittance_loading_rate = torch.zeros_like(self.admittance_delta_l)
        self.admittance_prev_axial_force = torch.zeros_like(self.admittance_delta_l)
        self.admittance_gate_timer = torch.zeros_like(self.admittance_delta_l)
        self.admittance_gate_active = torch.zeros(
            *shape, dtype=torch.bool, device=self.device
        )
        self.admittance_joint_offset = torch.zeros(
            self.num_envs,
            int(self.feet_indices.numel()),
            2,
            dtype=torch.float,
            device=self.device,
        )

    def _axial_contact_force(self):
        """Return positive compressive wheel force along each hip-to-wheel axis."""
        states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        hip_pos = states[:, self.admittance_hip_body_indices, 0:3]
        wheel_pos = states[:, self.feet_indices, 0:3]
        leg_vec = wheel_pos - hip_pos
        leg_axis = leg_vec / torch.clamp(
            torch.norm(leg_vec, dim=-1, keepdim=True), min=1.0e-6
        )

        # leg_axis points hip -> wheel.  A compressive ground reaction force
        # points approximately wheel -> hip, hence the minus sign.
        wheel_force = self.contact_forces[:, self.feet_indices, :]
        axial_force = torch.sum(wheel_force * (-leg_axis), dim=-1)
        return torch.clamp(axial_force, min=0.0)

    def _update_admittance_state(self):
        """Integrate M*xdd + D*xd + K*x = transient axial contact force."""
        if not self.admittance_enabled:
            self.admittance_delta_l.zero_()
            self.admittance_delta_l_dot.zero_()
            self.admittance_axial_force.zero_()
            self.admittance_force_input.zero_()
            self.admittance_loading_rate.zero_()
            self.admittance_gate_active.zero_()
            return

        dt = float(self.sim_params.dt)
        axial_force = self._axial_contact_force()

        # Positive axial force slew is a useful stair-impact discriminator.  The
        # gate only controls whether *new force* drives the admittance; the
        # virtual spring/damper keeps evolving continuously so it can return to
        # zero smoothly after the event.
        loading_rate = torch.clamp(
            (axial_force - self.admittance_prev_axial_force) / dt,
            min=0.0,
        )
        self.admittance_prev_axial_force.copy_(axial_force)
        self.admittance_loading_rate.copy_(loading_rate)

        if self.admittance_use_loading_rate_gate:
            triggered = loading_rate >= self.admittance_loading_rate_gate
            self.admittance_gate_timer = torch.where(
                triggered,
                torch.full_like(
                    self.admittance_gate_timer,
                    self.admittance_gate_hold_time,
                ),
                torch.clamp(self.admittance_gate_timer - dt, min=0.0),
            )
            gate_active = self.admittance_gate_timer > 0.0
        else:
            gate_active = torch.ones_like(self.admittance_gate_active)
        self.admittance_gate_active.copy_(gate_active)

        # Use the *previous* low-pass baseline for the transient-force estimate.
        raw_transient_force = torch.clamp(
            axial_force
            - self.admittance_force_bias
            - self.admittance_force_deadband,
            min=0.0,
            max=self.admittance_max_force_input,
        )
        transient_force = torch.where(
            gate_active,
            raw_transient_force,
            torch.zeros_like(raw_transient_force),
        )

        # Track steady support force while inactive.  For gated A0.1 we can
        # freeze the baseline during an impact window, preventing the fast impact
        # itself from being absorbed into the support-force estimate.
        alpha = dt / max(dt, self.admittance_force_bias_tau + dt)
        bias_update = alpha * (axial_force - self.admittance_force_bias)
        if (
            self.admittance_use_loading_rate_gate
            and self.admittance_freeze_bias_during_gate
        ):
            self.admittance_force_bias += torch.where(
                gate_active,
                torch.zeros_like(bias_update),
                bias_update,
            )
        else:
            self.admittance_force_bias += bias_update

        delta_l_ddot = (
            transient_force
            - self.admittance_damping * self.admittance_delta_l_dot
            - self.admittance_stiffness * self.admittance_delta_l
        ) / self.admittance_mass

        # Semi-implicit Euler is more stable than explicit Euler for the virtual
        # spring at the 200 Hz low-level update rate.
        self.admittance_delta_l_dot += delta_l_ddot * dt
        self.admittance_delta_l_dot.clamp_(
            -self.admittance_max_compression_vel,
            self.admittance_max_compression_vel,
        )
        self.admittance_delta_l += self.admittance_delta_l_dot * dt
        self.admittance_delta_l.clamp_(0.0, self.admittance_max_compression)

        # Do not let the integrator keep pushing outside either displacement
        # bound; this also prevents a delayed rebound after saturation.
        self.admittance_delta_l_dot = torch.where(
            (self.admittance_delta_l <= 0.0)
            & (self.admittance_delta_l_dot < 0.0),
            torch.zeros_like(self.admittance_delta_l_dot),
            self.admittance_delta_l_dot,
        )
        self.admittance_delta_l_dot = torch.where(
            (self.admittance_delta_l >= self.admittance_max_compression)
            & (self.admittance_delta_l_dot > 0.0),
            torch.zeros_like(self.admittance_delta_l_dot),
            self.admittance_delta_l_dot,
        )

        self.admittance_axial_force.copy_(axial_force)
        self.admittance_force_input.copy_(transient_force)

    def _joint_offsets_from_compression(self, actions_scaled):
        """Map axial Cartesian compression to HIP/KNEE offsets by DLS Jacobian."""
        q_nom = self.default_dof_pos + actions_scaled
        qh = q_nom[:, self.hip_indices]
        qk = q_nom[:, self.knee_indices]

        l1 = self.admittance_l1
        l2 = self.admittance_l2
        qhk = qh + qk

        # Sagittal two-link FK from the URDF joint layout.
        x = -l1 * torch.sin(qh) - l2 * torch.sin(qhk)
        z = -l1 * torch.cos(qh) - l2 * torch.cos(qhk)
        leg_len = torch.clamp(torch.sqrt(x * x + z * z), min=1.0e-6)

        # Compression moves the wheel toward the hip along the nominal leg axis.
        dp_x = -self.admittance_delta_l * (x / leg_len)
        dp_z = -self.admittance_delta_l * (z / leg_len)

        j11 = -l1 * torch.cos(qh) - l2 * torch.cos(qhk)
        j12 = -l2 * torch.cos(qhk)
        j21 = l1 * torch.sin(qh) + l2 * torch.sin(qhk)
        j22 = l2 * torch.sin(qhk)

        # Damped least-squares pseudoinverse:
        # dq = J^T (J J^T + lambda^2 I)^-1 dp
        lam2 = self.admittance_jacobian_damping ** 2
        a11 = j11 * j11 + j12 * j12 + lam2
        a12 = j11 * j21 + j12 * j22
        a22 = j21 * j21 + j22 * j22 + lam2
        det = torch.clamp(a11 * a22 - a12 * a12, min=1.0e-8)

        y1 = (a22 * dp_x - a12 * dp_z) / det
        y2 = (-a12 * dp_x + a11 * dp_z) / det
        dqh = j11 * y1 + j21 * y2
        dqk = j12 * y1 + j22 * y2

        offsets = torch.stack((dqh, dqk), dim=-1)
        offsets.clamp_(
            -self.admittance_max_joint_offset,
            self.admittance_max_joint_offset,
        )
        self.admittance_joint_offset.copy_(offsets)
        return offsets

    def _compute_torques(self, actions):
        """Original mixed MC controller with an admittance-modified leg target."""
        self._update_admittance_state()

        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled = actions_scaled.clone()
        actions_scaled[:, self.wheel_indices] = 0.0

        q_target = self.default_dof_pos + actions_scaled
        offsets = self._joint_offsets_from_compression(actions_scaled)
        q_target[:, self.hip_indices] += offsets[:, :, 0]
        q_target[:, self.knee_indices] += offsets[:, :, 1]

        # Respect the configured soft joint limits after adding compliance.
        q_target[:, self.hip_indices] = torch.maximum(
            torch.minimum(
                q_target[:, self.hip_indices],
                self.dof_pos_limits[self.hip_indices, 1].unsqueeze(0),
            ),
            self.dof_pos_limits[self.hip_indices, 0].unsqueeze(0),
        )
        q_target[:, self.knee_indices] = torch.maximum(
            torch.minimum(
                q_target[:, self.knee_indices],
                self.dof_pos_limits[self.knee_indices, 1].unsqueeze(0),
            ),
            self.dof_pos_limits[self.knee_indices, 0].unsqueeze(0),
        )

        pos_err = q_target - self.dof_pos
        pos_err[:, self.wheel_indices] = 0.0

        vel_ref = torch.zeros_like(actions_scaled)
        vel_tmp = actions * self.cfg.control.vel_scale
        vel_ref[:, self.wheel_indices] = vel_tmp[:, self.wheel_indices]

        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = (
                self.p_gains * self.Kp_factors * pos_err
                + self.d_gains * self.Kd_factors * (vel_ref - self.dof_vel)
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
        """Run admittance/contact-state refresh at every 200 Hz physics substep."""
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
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

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
        if len(env_ids) == 0 or not hasattr(self, "admittance_delta_l"):
            return
        self.admittance_delta_l[env_ids] = 0.0
        self.admittance_delta_l_dot[env_ids] = 0.0
        self.admittance_force_bias[env_ids] = 0.0
        self.admittance_axial_force[env_ids] = 0.0
        self.admittance_force_input[env_ids] = 0.0
        self.admittance_loading_rate[env_ids] = 0.0
        self.admittance_prev_axial_force[env_ids] = 0.0
        self.admittance_gate_timer[env_ids] = 0.0
        self.admittance_gate_active[env_ids] = False
        self.admittance_joint_offset[env_ids] = 0.0
