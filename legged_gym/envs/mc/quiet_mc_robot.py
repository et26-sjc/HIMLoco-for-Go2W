"""Physics-rate quiet-motion instrumentation for the MC wheeled quadruped.

The class deliberately leaves the trained MC policy, reward, observations and
controller unchanged.  It only refreshes simulator tensors after every physics
substep and records impact / vibration signals for deterministic evaluation.
"""

from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse
import torch

from .mc_robot import MC


class QuietMC(MC):
    """MC environment with passive physics-rate quiet metrics."""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self._quiet_initialized = False
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._init_quiet_metrics()
        self._quiet_initialized = True

    def _zeros_wheels(self):
        return torch.zeros(
            self.num_envs,
            int(self.feet_indices.numel()),
            device=self.device,
            dtype=torch.float,
        )

    def _to_base_frame(self, vectors):
        """Rotate [num_envs, num_wheels, 3] world vectors into base frame."""
        num_wheels = vectors.shape[1]
        quats = self.root_states[:, 3:7].unsqueeze(1).expand(-1, num_wheels, -1)
        rotated = quat_rotate_inverse(
            quats.reshape(-1, 4), vectors.reshape(-1, 3)
        )
        return rotated.reshape(self.num_envs, num_wheels, 3)

    def _wheel_kinematics(self):
        states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        wheel_pos_world = states[:, self.feet_indices, 0:3]
        wheel_vel_world = states[:, self.feet_indices, 7:10]
        wheel_vel_body = self._to_base_frame(wheel_vel_world)
        relative_pos_world = wheel_pos_world - self.root_states[:, None, 0:3]
        wheel_rel_pos_body = self._to_base_frame(relative_pos_world)
        return wheel_vel_world, wheel_vel_body, wheel_rel_pos_body[:, :, 2]

    def _base_vel_z(self):
        return quat_rotate_inverse(
            self.root_states[:, 3:7], self.root_states[:, 7:10]
        )[:, 2]

    def _init_quiet_metrics(self):
        quiet_cfg = getattr(self.cfg, "quiet_metrics", None)
        self.quiet_metrics_enabled = bool(
            quiet_cfg is not None and getattr(quiet_cfg, "enabled", True)
        )
        self.quiet_contact_on = float(
            getattr(quiet_cfg, "contact_on_threshold", 5.0)
        )
        self.quiet_contact_off = float(
            getattr(quiet_cfg, "contact_off_threshold", 2.0)
        )
        self.quiet_wheel_radius = float(getattr(quiet_cfg, "wheel_radius", 0.075))

        num_wheels = int(self.feet_indices.numel())
        if num_wheels != 4 or int(self.wheel_indices.numel()) != 4:
            raise RuntimeError(
                "QuietMC expects four wheel links and four wheel joints, got "
                f"feet={num_wheels}, wheel_dofs={int(self.wheel_indices.numel())}."
            )

        zeros = self._zeros_wheels
        wheel_vel_world, _, wheel_rel_z = self._wheel_kinematics()
        force_vec = self.contact_forces[:, self.feet_indices, :]
        force_z = torch.clamp(force_vec[:, :, 2], min=0.0)
        force_norm = torch.norm(force_vec, dim=-1)

        # Previous physics-substep values.
        self.quiet_contact_state = force_norm >= self.quiet_contact_on
        self.quiet_prev_wheel_vel_world = wheel_vel_world.clone()
        self.quiet_prev_force_z = force_z.clone()
        self.quiet_prev_force_norm = force_norm.clone()
        self.quiet_prev_base_vel_z = self._base_vel_z().clone()
        self.quiet_prev_base_acc_z = torch.zeros(self.num_envs, device=self.device)
        self.quiet_prev_torques = self.torques.clone()
        self.quiet_prev_wheel_omega = self.dof_vel[:, self.wheel_indices].clone()
        self.quiet_touchdown_rel_z = wheel_rel_z.clone()

        # Current contact-event accumulators.  Fz versions preserve direct
        # comparability with the previous quadruped QuietGo evaluation; force
        # norm versions additionally capture wheel impacts against stair risers.
        self.quiet_event_peak_force_z = zeros()
        self.quiet_event_peak_force_norm = zeros()
        self.quiet_event_peak_loading_rate_z = zeros()
        self.quiet_event_peak_loading_rate_norm = zeros()
        self.quiet_event_normal_impulse = zeros()
        self.quiet_event_contact_impulse_norm = zeros()
        self.quiet_event_duration = zeros()
        self.quiet_event_peak_leg_compression = zeros()
        self.quiet_event_peak_base_acc = zeros()
        self.quiet_event_peak_base_jerk = zeros()
        self.quiet_event_peak_torque_rate = zeros()

        # Cumulative event statistics.
        self.quiet_touchdown_count = zeros()
        self.quiet_completed_event_count = zeros()
        self.quiet_sum_touchdown_vertical_speed = zeros()
        self.quiet_max_touchdown_vertical_speed = zeros()
        self.quiet_sum_touchdown_speed_3d = zeros()
        self.quiet_max_touchdown_speed_3d = zeros()

        cumulative_metric_names = [
            "peak_force_z",
            "peak_force_norm",
            "peak_loading_rate_z",
            "peak_loading_rate_norm",
            "normal_impulse",
            "contact_impulse_norm",
            "event_duration",
            "peak_leg_compression",
            "peak_base_acc",
            "peak_base_jerk",
            "peak_torque_rate",
        ]
        for metric in cumulative_metric_names:
            setattr(self, f"quiet_sum_{metric}", zeros())
            setattr(self, f"quiet_max_{metric}", zeros())

        # Pulses / completed-event values exposed once per policy step.
        self.quiet_step_touchdown = torch.zeros(
            self.num_envs, num_wheels, device=self.device, dtype=torch.bool
        )
        self.quiet_step_event_finished = torch.zeros_like(self.quiet_step_touchdown)
        for name in [
            "touchdown_vertical_speed",
            "touchdown_speed_3d",
            "completed_peak_force_z",
            "completed_peak_force_norm",
            "completed_peak_loading_rate_z",
            "completed_peak_loading_rate_norm",
            "completed_normal_impulse",
            "completed_contact_impulse_norm",
            "completed_peak_leg_compression",
        ]:
            setattr(self, f"quiet_step_{name}", zeros())

        # All-environment physics-rate traces.  For the current baseline these
        # contain 4 substeps per 50 Hz policy step = 200 Hz samples.  The same
        # code automatically follows future 100/1000 Hz configurations.
        decimation = int(self.cfg.control.decimation)
        wheel_trace_shape = (decimation, self.num_envs, num_wheels)
        env_trace_shape = (decimation, self.num_envs)
        self.quiet_trace_force_z = torch.zeros(wheel_trace_shape, device=self.device)
        self.quiet_trace_force_norm = torch.zeros(wheel_trace_shape, device=self.device)
        self.quiet_trace_loading_rate_z = torch.zeros(wheel_trace_shape, device=self.device)
        self.quiet_trace_loading_rate_norm = torch.zeros(wheel_trace_shape, device=self.device)
        self.quiet_trace_wheel_vel_z = torch.zeros(wheel_trace_shape, device=self.device)
        self.quiet_trace_wheel_lateral_speed = torch.zeros(wheel_trace_shape, device=self.device)
        self.quiet_trace_wheel_omega = torch.zeros(wheel_trace_shape, device=self.device)
        self.quiet_trace_wheel_alpha = torch.zeros(wheel_trace_shape, device=self.device)
        self.quiet_trace_leg_compression = torch.zeros(wheel_trace_shape, device=self.device)
        self.quiet_trace_contact = torch.zeros(
            wheel_trace_shape, device=self.device, dtype=torch.bool
        )
        self.quiet_trace_base_acc_z = torch.zeros(env_trace_shape, device=self.device)
        self.quiet_trace_base_jerk_z = torch.zeros(env_trace_shape, device=self.device)
        self.quiet_trace_max_torque_rate = torch.zeros(env_trace_shape, device=self.device)

    def _clear_quiet_step_buffers(self):
        self.quiet_step_touchdown.zero_()
        self.quiet_step_event_finished.zero_()
        for name in [
            "touchdown_vertical_speed",
            "touchdown_speed_3d",
            "completed_peak_force_z",
            "completed_peak_force_norm",
            "completed_peak_loading_rate_z",
            "completed_peak_loading_rate_norm",
            "completed_normal_impulse",
            "completed_contact_impulse_norm",
            "completed_peak_leg_compression",
        ]:
            getattr(self, f"quiet_step_{name}").zero_()

    def _clear_current_events(self):
        for name in [
            "peak_force_z",
            "peak_force_norm",
            "peak_loading_rate_z",
            "peak_loading_rate_norm",
            "normal_impulse",
            "contact_impulse_norm",
            "duration",
            "peak_leg_compression",
            "peak_base_acc",
            "peak_base_jerk",
            "peak_torque_rate",
        ]:
            getattr(self, f"quiet_event_{name}").zero_()

    def step(self, actions):
        """Run the unchanged MC controller while sampling every physics step."""
        if not self.quiet_metrics_enabled:
            return super().step(actions)

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
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

        self._clear_quiet_step_buffers()
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
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self._update_quiet_metrics_substep(substep)

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

    def _update_quiet_metrics_substep(self, substep):
        physics_dt = float(self.sim_params.dt)
        wheel_vel_world, wheel_vel_body, wheel_rel_z = self._wheel_kinematics()
        force_vec = self.contact_forces[:, self.feet_indices, :]
        force_z = torch.clamp(force_vec[:, :, 2], min=0.0)
        force_norm = torch.norm(force_vec, dim=-1)

        loading_rate_z = torch.clamp(
            (force_z - self.quiet_prev_force_z) / physics_dt, min=0.0
        )
        loading_rate_norm = torch.clamp(
            (force_norm - self.quiet_prev_force_norm) / physics_dt, min=0.0
        )

        base_vel_z = self._base_vel_z()
        base_acc_z = (base_vel_z - self.quiet_prev_base_vel_z) / physics_dt
        base_jerk_z = (base_acc_z - self.quiet_prev_base_acc_z) / physics_dt

        torque_rate = torch.abs(self.torques - self.quiet_prev_torques) / physics_dt
        max_torque_rate = torch.max(torque_rate, dim=1).values
        wheel_omega = self.dof_vel[:, self.wheel_indices]
        wheel_alpha = torch.abs(
            (wheel_omega - self.quiet_prev_wheel_omega) / physics_dt
        )

        previous_contact = self.quiet_contact_state
        next_contact = torch.where(
            previous_contact,
            force_norm > self.quiet_contact_off,
            force_norm >= self.quiet_contact_on,
        )
        touchdown = torch.logical_and(~previous_contact, next_contact)
        liftoff = torch.logical_and(previous_contact, ~next_contact)

        prev_vel_z = self.quiet_prev_wheel_vel_world[:, :, 2]
        touchdown_vertical_speed = (
            torch.clamp(-prev_vel_z, min=0.0) * touchdown.float()
        )
        touchdown_speed_3d = (
            torch.norm(self.quiet_prev_wheel_vel_world, dim=-1) * touchdown.float()
        )
        self.quiet_step_touchdown |= touchdown
        self.quiet_step_touchdown_vertical_speed = torch.maximum(
            self.quiet_step_touchdown_vertical_speed, touchdown_vertical_speed
        )
        self.quiet_step_touchdown_speed_3d = torch.maximum(
            self.quiet_step_touchdown_speed_3d, touchdown_speed_3d
        )
        self.quiet_touchdown_count += touchdown.float()
        self.quiet_sum_touchdown_vertical_speed += touchdown_vertical_speed
        self.quiet_max_touchdown_vertical_speed = torch.maximum(
            self.quiet_max_touchdown_vertical_speed, touchdown_vertical_speed
        )
        self.quiet_sum_touchdown_speed_3d += touchdown_speed_3d
        self.quiet_max_touchdown_speed_3d = torch.maximum(
            self.quiet_max_touchdown_speed_3d, touchdown_speed_3d
        )

        self.quiet_touchdown_rel_z = torch.where(
            touchdown, wheel_rel_z, self.quiet_touchdown_rel_z
        )

        # Fresh contact-event accumulators at touchdown.
        for event_name, value in [
            ("peak_force_z", force_z),
            ("peak_force_norm", force_norm),
            ("peak_loading_rate_z", loading_rate_z),
            ("peak_loading_rate_norm", loading_rate_norm),
        ]:
            current = getattr(self, f"quiet_event_{event_name}")
            setattr(self, f"quiet_event_{event_name}", torch.where(touchdown, value, current))

        for event_name in [
            "normal_impulse",
            "contact_impulse_norm",
            "duration",
            "peak_leg_compression",
            "peak_base_acc",
            "peak_base_jerk",
            "peak_torque_rate",
        ]:
            current = getattr(self, f"quiet_event_{event_name}")
            setattr(
                self,
                f"quiet_event_{event_name}",
                torch.where(touchdown, torch.zeros_like(current), current),
            )

        active = next_contact
        active_float = active.float()
        leg_compression = torch.clamp(
            wheel_rel_z - self.quiet_touchdown_rel_z, min=0.0
        ) * active_float
        base_acc_abs = torch.abs(base_acc_z).unsqueeze(1).expand_as(force_z)
        base_jerk_abs = torch.abs(base_jerk_z).unsqueeze(1).expand_as(force_z)
        torque_rate_expanded = max_torque_rate.unsqueeze(1).expand_as(force_z)

        for event_name, value in [
            ("peak_force_z", force_z),
            ("peak_force_norm", force_norm),
            ("peak_loading_rate_z", loading_rate_z),
            ("peak_loading_rate_norm", loading_rate_norm),
            ("peak_leg_compression", leg_compression),
            ("peak_base_acc", base_acc_abs),
            ("peak_base_jerk", base_jerk_abs),
            ("peak_torque_rate", torque_rate_expanded),
        ]:
            current = getattr(self, f"quiet_event_{event_name}")
            updated = torch.where(active, torch.maximum(current, value), current)
            setattr(self, f"quiet_event_{event_name}", updated)

        self.quiet_event_normal_impulse += force_z * physics_dt * active_float
        self.quiet_event_contact_impulse_norm += (
            force_norm * physics_dt * active_float
        )
        self.quiet_event_duration += physics_dt * active_float

        # Finalize contact events at liftoff.
        finished = liftoff.float()
        self.quiet_step_event_finished |= liftoff
        completed_pairs = [
            ("peak_force_z", "completed_peak_force_z"),
            ("peak_force_norm", "completed_peak_force_norm"),
            ("peak_loading_rate_z", "completed_peak_loading_rate_z"),
            ("peak_loading_rate_norm", "completed_peak_loading_rate_norm"),
            ("normal_impulse", "completed_normal_impulse"),
            ("contact_impulse_norm", "completed_contact_impulse_norm"),
            ("peak_leg_compression", "completed_peak_leg_compression"),
        ]
        for event_name, step_name in completed_pairs:
            event_value = getattr(self, f"quiet_event_{event_name}")
            step_value = getattr(self, f"quiet_step_{step_name}")
            setattr(
                self,
                f"quiet_step_{step_name}",
                torch.where(liftoff, event_value, step_value),
            )

        self.quiet_completed_event_count += finished
        for metric in [
            "peak_force_z",
            "peak_force_norm",
            "peak_loading_rate_z",
            "peak_loading_rate_norm",
            "normal_impulse",
            "contact_impulse_norm",
            "event_duration",
            "peak_leg_compression",
            "peak_base_acc",
            "peak_base_jerk",
            "peak_torque_rate",
        ]:
            event_attr = "duration" if metric == "event_duration" else metric
            event_value = getattr(self, f"quiet_event_{event_attr}")
            sum_tensor = getattr(self, f"quiet_sum_{metric}")
            max_tensor = getattr(self, f"quiet_max_{metric}")
            sum_tensor += event_value * finished
            torch.maximum(max_tensor, event_value * finished, out=max_tensor)

        keep = (~liftoff).float()
        for event_name in [
            "peak_force_z",
            "peak_force_norm",
            "peak_loading_rate_z",
            "peak_loading_rate_norm",
            "normal_impulse",
            "contact_impulse_norm",
            "duration",
            "peak_leg_compression",
            "peak_base_acc",
            "peak_base_jerk",
            "peak_torque_rate",
        ]:
            getattr(self, f"quiet_event_{event_name}").mul_(keep)
        self.quiet_contact_state = next_contact

        # Physics-rate traces across every evaluation environment.
        self.quiet_trace_force_z[substep].copy_(force_z)
        self.quiet_trace_force_norm[substep].copy_(force_norm)
        self.quiet_trace_loading_rate_z[substep].copy_(loading_rate_z)
        self.quiet_trace_loading_rate_norm[substep].copy_(loading_rate_norm)
        self.quiet_trace_wheel_vel_z[substep].copy_(wheel_vel_world[:, :, 2])
        self.quiet_trace_wheel_lateral_speed[substep].copy_(
            torch.abs(wheel_vel_body[:, :, 1])
        )
        self.quiet_trace_wheel_omega[substep].copy_(wheel_omega)
        self.quiet_trace_wheel_alpha[substep].copy_(wheel_alpha)
        self.quiet_trace_leg_compression[substep].copy_(leg_compression)
        self.quiet_trace_contact[substep].copy_(next_contact)
        self.quiet_trace_base_acc_z[substep].copy_(base_acc_z)
        self.quiet_trace_base_jerk_z[substep].copy_(base_jerk_z)
        self.quiet_trace_max_torque_rate[substep].copy_(max_torque_rate)

        self.quiet_prev_wheel_vel_world.copy_(wheel_vel_world)
        self.quiet_prev_force_z.copy_(force_z)
        self.quiet_prev_force_norm.copy_(force_norm)
        self.quiet_prev_base_vel_z.copy_(base_vel_z)
        self.quiet_prev_base_acc_z.copy_(base_acc_z)
        self.quiet_prev_torques.copy_(self.torques)
        self.quiet_prev_wheel_omega.copy_(wheel_omega)

    def _clear_quiet_transient(self, env_ids):
        if not self._quiet_initialized or len(env_ids) == 0:
            return
        self.quiet_contact_state[env_ids] = False
        self.quiet_prev_wheel_vel_world[env_ids] = 0.0
        self.quiet_prev_force_z[env_ids] = 0.0
        self.quiet_prev_force_norm[env_ids] = 0.0
        self.quiet_prev_base_vel_z[env_ids] = 0.0
        self.quiet_prev_base_acc_z[env_ids] = 0.0
        self.quiet_prev_torques[env_ids] = 0.0
        self.quiet_prev_wheel_omega[env_ids] = 0.0
        for event_name in [
            "peak_force_z",
            "peak_force_norm",
            "peak_loading_rate_z",
            "peak_loading_rate_norm",
            "normal_impulse",
            "contact_impulse_norm",
            "duration",
            "peak_leg_compression",
            "peak_base_acc",
            "peak_base_jerk",
            "peak_torque_rate",
        ]:
            getattr(self, f"quiet_event_{event_name}")[env_ids] = 0.0

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._clear_quiet_transient(env_ids)

    def reset_quiet_metrics(self):
        """Start a clean measurement window without resetting the robot."""
        if not self._quiet_initialized:
            return

        self.quiet_touchdown_count.zero_()
        self.quiet_completed_event_count.zero_()
        self.quiet_sum_touchdown_vertical_speed.zero_()
        self.quiet_max_touchdown_vertical_speed.zero_()
        self.quiet_sum_touchdown_speed_3d.zero_()
        self.quiet_max_touchdown_speed_3d.zero_()
        for metric in [
            "peak_force_z",
            "peak_force_norm",
            "peak_loading_rate_z",
            "peak_loading_rate_norm",
            "normal_impulse",
            "contact_impulse_norm",
            "event_duration",
            "peak_leg_compression",
            "peak_base_acc",
            "peak_base_jerk",
            "peak_torque_rate",
        ]:
            getattr(self, f"quiet_sum_{metric}").zero_()
            getattr(self, f"quiet_max_{metric}").zero_()
        self._clear_quiet_step_buffers()
        self._clear_current_events()

        # Align derivative/contact state with the current physics state so the
        # first measured sample does not contain a warmup-boundary impulse.
        wheel_vel_world, _, wheel_rel_z = self._wheel_kinematics()
        force_vec = self.contact_forces[:, self.feet_indices, :]
        force_z = torch.clamp(force_vec[:, :, 2], min=0.0)
        force_norm = torch.norm(force_vec, dim=-1)
        self.quiet_contact_state = force_norm >= self.quiet_contact_on
        self.quiet_prev_wheel_vel_world.copy_(wheel_vel_world)
        self.quiet_prev_force_z.copy_(force_z)
        self.quiet_prev_force_norm.copy_(force_norm)
        self.quiet_prev_base_vel_z.copy_(self._base_vel_z())
        self.quiet_prev_base_acc_z.zero_()
        self.quiet_prev_torques.copy_(self.torques)
        self.quiet_prev_wheel_omega.copy_(self.dof_vel[:, self.wheel_indices])
        self.quiet_touchdown_rel_z.copy_(wheel_rel_z)

    @staticmethod
    def _safe_mean(total, count):
        denominator = torch.clamp(torch.sum(count), min=1.0)
        return float((torch.sum(total) / denominator).item())

    def get_quiet_metrics(self):
        touchdowns = self.quiet_touchdown_count
        completed = self.quiet_completed_event_count
        result = {
            "physics_dt_s": float(self.sim_params.dt),
            "policy_dt_s": float(self.dt),
            "touchdown_count": int(torch.sum(touchdowns).item()),
            "completed_event_count": int(torch.sum(completed).item()),
            "mean_touchdown_speed_mps": self._safe_mean(
                self.quiet_sum_touchdown_vertical_speed, touchdowns
            ),
            "max_touchdown_speed_mps": float(
                torch.max(self.quiet_max_touchdown_vertical_speed).item()
            ),
            "mean_touchdown_speed_3d_mps": self._safe_mean(
                self.quiet_sum_touchdown_speed_3d, touchdowns
            ),
            "max_touchdown_speed_3d_mps": float(
                torch.max(self.quiet_max_touchdown_speed_3d).item()
            ),
        }

        output_names = {
            "peak_force_z": "peak_force_n",
            "peak_force_norm": "peak_contact_force_norm_n",
            "peak_loading_rate_z": "peak_loading_rate_nps",
            "peak_loading_rate_norm": "peak_contact_loading_rate_norm_nps",
            "normal_impulse": "normal_impulse_ns",
            "contact_impulse_norm": "contact_impulse_norm_ns",
            "event_duration": "contact_duration_s",
            "peak_leg_compression": "event_peak_leg_compression_m",
            "peak_base_acc": "event_peak_base_acc_mps2",
            "peak_base_jerk": "event_peak_base_jerk_mps3",
            "peak_torque_rate": "event_peak_torque_rate_nmps",
        }
        for metric, suffix in output_names.items():
            result[f"mean_{suffix}"] = self._safe_mean(
                getattr(self, f"quiet_sum_{metric}"), completed
            )
            result[f"max_{suffix}"] = float(
                torch.max(getattr(self, f"quiet_max_{metric}")).item()
            )
        return result
