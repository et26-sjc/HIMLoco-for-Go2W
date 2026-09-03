"""Learned, sensorless per-leg admittance controller for the MC robot.

This module deliberately contains no simulator contact-force reads. It accepts
only the contact state predicted by the policy-side estimator plus the four RL
compliance actions. Consequently the same controller can be used in Isaac Gym,
MuJoCo and on hardware without a force sensor.

Per leg, the virtual dynamics are

    M x_ddot + D(alpha) x_dot + K(alpha) x = alpha * g(dF) * F_transient

where ``alpha`` is the extra RL compliance action in [0, 1], ``F_transient`` is
computed from the *estimated* contact force after subtracting a slow support
force baseline, and ``g(dF)`` is a smooth loading-rate gate. The resulting axial
leg compression is mapped to HIP/KNEE target offsets through a damped
least-squares Jacobian.
"""

import torch


class MCLearnedAdmittance:
    """Stateful four-leg virtual admittance driven by estimated contact state."""

    def __init__(self, cfg, num_envs, device):
        self.device = device
        self.num_envs = int(num_envs)
        self.num_legs = 4

        self.mass = float(cfg.virtual_mass_kg)
        self.zeta = float(cfg.damping_ratio)
        self.k_min = float(cfg.min_stiffness_n_per_m)
        self.k_max = float(cfg.max_stiffness_n_per_m)
        self.force_bias_tau = float(cfg.force_bias_time_constant_s)
        self.force_deadband = float(cfg.force_deadband_n)
        self.force_scale = float(cfg.contact_force_scale_n)
        self.loading_rate_scale = float(cfg.contact_loading_rate_scale_nps)
        self.loading_rate_threshold = float(cfg.loading_rate_gate_nps)
        self.loading_rate_softness = float(cfg.loading_rate_gate_softness_nps)

        self.max_force_input = float(cfg.max_force_input_n)
        self.max_compression = float(cfg.max_compression_m)
        self.max_compression_vel = float(cfg.max_compression_velocity_mps)
        self.max_joint_offset = float(cfg.max_joint_offset_rad)

        self.l1 = float(cfg.upper_leg_length_m)
        self.l2 = float(cfg.lower_leg_length_m)
        self.jacobian_damping = float(cfg.jacobian_damping)

        shape = (self.num_envs, self.num_legs)
        self.delta_l = torch.zeros(shape, device=device)
        self.delta_l_dot = torch.zeros_like(self.delta_l)
        self.force_bias = torch.zeros_like(self.delta_l)
        self.alpha = torch.zeros_like(self.delta_l)
        self.estimated_force = torch.zeros_like(self.delta_l)
        self.estimated_loading_rate = torch.zeros_like(self.delta_l)
        self.transient_force = torch.zeros_like(self.delta_l)

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        self.delta_l[env_ids] = 0.0
        self.delta_l_dot[env_ids] = 0.0
        self.force_bias[env_ids] = 0.0
        self.alpha[env_ids] = 0.0
        self.estimated_force[env_ids] = 0.0
        self.estimated_loading_rate[env_ids] = 0.0
        self.transient_force[env_ids] = 0.0

    def state(self):
        """Deployable controller state exposed to the adaptive policy.

        Returns [delta_l(4), delta_l_dot(4), alpha(4)] = 12 dimensions. These are
        controller-internal quantities and therefore remain available on hardware.
        """
        return torch.cat((self.delta_l, self.delta_l_dot, self.alpha), dim=-1)

    def _decode_contact_estimate(self, estimated_contact):
        if estimated_contact.shape[-1] != 8:
            raise RuntimeError(
                f"Expected 8-D contact estimate [F(4), dF(4)], got "
                f"{tuple(estimated_contact.shape)}"
            )
        force = torch.clamp(estimated_contact[:, :4], min=0.0) * self.force_scale
        loading = (
            torch.clamp(estimated_contact[:, 4:8], min=0.0)
            * self.loading_rate_scale
        )
        return force, loading

    def _loading_gate(self, loading_rate):
        softness = max(self.loading_rate_softness, 1.0)
        return torch.sigmoid(
            (loading_rate - self.loading_rate_threshold) / softness
        )

    def _joint_offsets_from_compression(self, q_nom, hip_indices, knee_indices):
        qh = q_nom[:, hip_indices]
        qk = q_nom[:, knee_indices]
        qhk = qh + qk

        x = -self.l1 * torch.sin(qh) - self.l2 * torch.sin(qhk)
        z = -self.l1 * torch.cos(qh) - self.l2 * torch.cos(qhk)
        leg_len = torch.clamp(torch.sqrt(x * x + z * z), min=1.0e-6)

        dp_x = -self.delta_l * (x / leg_len)
        dp_z = -self.delta_l * (z / leg_len)

        j11 = -self.l1 * torch.cos(qh) - self.l2 * torch.cos(qhk)
        j12 = -self.l2 * torch.cos(qhk)
        j21 = self.l1 * torch.sin(qh) + self.l2 * torch.sin(qhk)
        j22 = self.l2 * torch.sin(qhk)

        lam2 = self.jacobian_damping ** 2
        a11 = j11 * j11 + j12 * j12 + lam2
        a12 = j11 * j21 + j12 * j22
        a22 = j21 * j21 + j22 * j22 + lam2
        det = torch.clamp(a11 * a22 - a12 * a12, min=1.0e-8)

        y1 = (a22 * dp_x - a12 * dp_z) / det
        y2 = (-a12 * dp_x + a11 * dp_z) / det
        dqh = j11 * y1 + j21 * y2
        dqk = j12 * y1 + j22 * y2

        offsets = torch.stack((dqh, dqk), dim=-1)
        return torch.clamp(
            offsets, -self.max_joint_offset, self.max_joint_offset
        )

    def step(
        self,
        compliance_action,
        estimated_contact,
        q_nom,
        hip_indices,
        knee_indices,
        dt,
    ):
        """Advance the admittance and return per-leg [HIP, KNEE] offsets."""
        if compliance_action.shape[-1] != 4:
            raise RuntimeError(
                f"Expected 4-D compliance action, got {tuple(compliance_action.shape)}"
            )

        # Zero compliance reproduces the original HIMLoco controller exactly.
        alpha = torch.clamp(compliance_action, 0.0, 1.0)
        force, loading_rate = self._decode_contact_estimate(estimated_contact)

        gate = self._loading_gate(loading_rate)
        bias_alpha = float(dt) / max(float(dt) + self.force_bias_tau, 1.0e-6)
        bias_update = bias_alpha * (force - self.force_bias)
        self.force_bias += (1.0 - gate) * bias_update

        transient = torch.clamp(
            force - self.force_bias - self.force_deadband,
            min=0.0,
            max=self.max_force_input,
        )

        stiffness = self.k_max - alpha * (self.k_max - self.k_min)
        stiffness = torch.clamp(stiffness, min=self.k_min, max=self.k_max)
        damping = 2.0 * self.zeta * torch.sqrt(
            torch.clamp(self.mass * stiffness, min=1.0e-6)
        )

        drive = alpha * gate * transient
        delta_l_ddot = (
            drive - damping * self.delta_l_dot - stiffness * self.delta_l
        ) / self.mass

        self.delta_l_dot += delta_l_ddot * float(dt)
        self.delta_l_dot.clamp_(
            -self.max_compression_vel, self.max_compression_vel
        )
        self.delta_l += self.delta_l_dot * float(dt)
        self.delta_l.clamp_(0.0, self.max_compression)

        self.delta_l_dot = torch.where(
            (self.delta_l <= 0.0) & (self.delta_l_dot < 0.0),
            torch.zeros_like(self.delta_l_dot),
            self.delta_l_dot,
        )
        self.delta_l_dot = torch.where(
            (self.delta_l >= self.max_compression) & (self.delta_l_dot > 0.0),
            torch.zeros_like(self.delta_l_dot),
            self.delta_l_dot,
        )

        self.alpha.copy_(alpha)
        self.estimated_force.copy_(force)
        self.estimated_loading_rate.copy_(loading_rate)
        self.transient_force.copy_(transient)

        return self._joint_offsets_from_compression(
            q_nom, hip_indices, knee_indices
        )
