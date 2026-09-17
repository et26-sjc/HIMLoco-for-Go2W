"""Physics-rate quiet evaluation for an adaptive MC checkpoint.

This is an evaluation-only wrapper.  It reuses the existing ``QuietMC``
instrumentation, while executing the unchanged 20-D learned-admittance policy
and its 16-D internal controller state.  No training, reward, estimator loss,
or action semantics are changed.
"""

import os
import sys

import isaacgym  # noqa: F401; must be imported before torch
from isaacgym import gymtorch
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.envs.mc.mc_learned_admittance_100hz_config import (
    MCLearnedAdmittance100HzCfg,
    MCLearnedAdmittance100HzCfgPPO,
)
from legged_gym.envs.mc.mc_learned_admittance import MCLearnedAdmittance
from legged_gym.envs.mc.mc_learned_admittance_100hz_robot import (
    MCLearnedAdmittance100Hz,
)
from legged_gym.envs.mc.quiet_mc_robot import QuietMC
from legged_gym.scripts.evaluate_mc_quiet import (
    _add_continuous_metrics,
    _append_event_samples,
    _append_traces,
    _configure_environment,
    _extract_custom_args,
    _pack,
    _place_on_fixed_terrain,
    _save_results,
    _set_fixed_commands,
    _synchronize_observation_history,
)
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import set_seed


TASK_NAME = "mc_impact_quiet_eval"


class _CounterfactualAdmittance(MCLearnedAdmittance):
    """Evaluation-only state/drive interventions used by archived diagnostics."""

    def __init__(self, cfg, num_envs, device):
        super().__init__(cfg, num_envs, device)
        self.counterfactual_mode = None
        self.counterfactual_mask = None

    def set_counterfactual(self, mode, mask=None):
        if mode not in (
            None,
            "transient",
            "residual",
            "compression_velocity_reset",
            "compression_state_reset",
            "compression_state_clamp",
        ):
            raise ValueError("Unknown counterfactual mode: " + str(mode))
        self.counterfactual_mode = mode
        self.counterfactual_mask = mask

    def step(
        self,
        compliance_action,
        estimated_contact,
        q_nom,
        hip_indices,
        knee_indices,
        dt,
    ):
        mode = self.counterfactual_mode
        mask = self.counterfactual_mask
        if mode is None or mask is None:
            return super().step(
                compliance_action,
                estimated_contact,
                q_nom,
                hip_indices,
                knee_indices,
                dt,
            )

        if compliance_action.shape[-1] != 4:
            raise RuntimeError(
                f"Expected 4-D compliance action, got {tuple(compliance_action.shape)}"
            )
        alpha = torch.clamp(compliance_action, 0.0, 1.0)
        beta = self._effective_compliance(alpha)
        force, impact_logits, impact_probability = self._decode_contact_estimate(
            estimated_contact
        )
        bias_alpha = float(dt) / max(float(dt) + self.force_bias_tau, 1.0e-6)
        bias_update = bias_alpha * (force - self.force_bias)
        self.force_bias += (1.0 - impact_probability) * bias_update
        transient = torch.clamp(
            force - self.force_bias - self.force_deadband,
            min=0.0,
            max=self.max_force_input,
        )
        stiffness = self.k_max - beta * (self.k_max - self.k_min)
        stiffness = torch.clamp(stiffness, min=self.k_min, max=self.k_max)
        damping = 2.0 * self.zeta * torch.sqrt(
            torch.clamp(self.mass * stiffness, min=1.0e-6)
        )
        drive = beta * (1.0 + self.impact_gain * impact_probability) * transient

        if mode == "compression_velocity_reset":
            self.delta_l_dot = torch.where(
                mask, torch.zeros_like(self.delta_l_dot), self.delta_l_dot
            )
        elif mode in ("compression_state_reset", "compression_state_clamp"):
            self.delta_l = torch.where(
                mask, torch.zeros_like(self.delta_l), self.delta_l
            )
            self.delta_l_dot = torch.where(
                mask, torch.zeros_like(self.delta_l_dot), self.delta_l_dot
            )
        if mode == "transient":
            drive = torch.where(mask, torch.zeros_like(drive), drive)

        acceleration = (
            drive - damping * self.delta_l_dot - stiffness * self.delta_l
        ) / self.mass
        self.delta_l_dot += acceleration * float(dt)
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
        if mode == "compression_state_clamp":
            self.delta_l = torch.where(
                mask, torch.zeros_like(self.delta_l), self.delta_l
            )
            self.delta_l_dot = torch.where(
                mask, torch.zeros_like(self.delta_l_dot), self.delta_l_dot
            )

        offsets = self._joint_offsets_from_compression(
            q_nom, hip_indices, knee_indices
        )
        if mode == "residual":
            offsets = torch.where(
                mask.unsqueeze(-1), torch.zeros_like(offsets), offsets
            )

        self.alpha.copy_(alpha)
        self.effective_alpha.copy_(beta)
        self.estimated_force.copy_(force)
        self.impact_logits.copy_(impact_logits)
        self.impact_probability.copy_(impact_probability)
        self.transient_force.copy_(transient)
        self.drive_force.copy_(drive)
        self.stiffness.copy_(stiffness)
        self.last_joint_offsets.copy_(offsets)
        return offsets


class _QuietMetricsMixin:
    """Reuse QuietMC methods without creating an incompatible MC MRO."""

    _zeros_wheels = QuietMC._zeros_wheels
    _to_base_frame = QuietMC._to_base_frame
    _wheel_kinematics = QuietMC._wheel_kinematics
    _wheel_geometry = QuietMC._wheel_geometry
    _base_vel_z = QuietMC._base_vel_z
    _init_quiet_metrics = QuietMC._init_quiet_metrics
    _clear_quiet_step_buffers = QuietMC._clear_quiet_step_buffers
    _clear_current_events = QuietMC._clear_current_events
    _update_quiet_metrics_substep = QuietMC._update_quiet_metrics_substep
    _clear_quiet_transient = QuietMC._clear_quiet_transient
    _safe_mean = staticmethod(QuietMC._safe_mean)
    reset_quiet_metrics = QuietMC.reset_quiet_metrics
    get_quiet_metrics = QuietMC.get_quiet_metrics


class QuietMCLearnedAdmittance100Hz(
    _QuietMetricsMixin, MCLearnedAdmittance100Hz
):
    """Adaptive MC environment with passive physics-rate instrumentation."""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # Counterfactual state belongs exclusively to this evaluator. The
        # production environment always constructs the plain controller.
        self.admittance = _CounterfactualAdmittance(
            self.cfg.learned_admittance, self.num_envs, self.device
        )
        self._init_quiet_metrics()
        # QuietMC follows rigid-body asset order (FR, FL, RR, RL), while the
        # learned admittance state is semantic (FL, FR, RR, RL). Event-level
        # diagnostics must explicitly align the two; aggregate means happened
        # to be permutation-invariant, but per-contact trajectories are not.
        quiet_to_adm = []
        for foot_index in self.feet_indices.tolist():
            matches = torch.nonzero(
                self.adm_feet_indices == int(foot_index), as_tuple=False
            ).flatten()
            if matches.numel() != 1:
                raise RuntimeError(
                    "Unable to align quiet/admittance leg order for foot "
                    + str(foot_index)
                )
            quiet_to_adm.append(int(matches.item()))
        self.quiet_to_adm_leg = torch.tensor(
            quiet_to_adm, dtype=torch.long, device=self.device
        )
        self.adm_to_quiet_leg = torch.argsort(self.quiet_to_adm_leg)
        trace_shape = (
            int(self.cfg.control.decimation),
            self.num_envs,
            self.num_compliance_actions,
        )
        for name in (
            "impact_probability",
            "alpha",
            "beta",
            "drive_force_n",
            "transient_force_n",
            "admittance_compression_m",
            "admittance_compression_velocity_mps",
        ):
            setattr(
                self,
                "quiet_trace_" + name,
                torch.zeros(trace_shape, device=self.device),
            )
        self.quiet_trace_baseline_torque_residual = torch.zeros(
            int(self.cfg.control.decimation),
            self.num_envs,
            self.num_dof,
            device=self.device,
        )
        self.quiet_trace_compliance_power = torch.zeros(
            int(self.cfg.control.decimation), self.num_envs,
            self.num_compliance_actions, device=self.device
        )
        self.quiet_trace_counterfactual_active = torch.zeros(
            int(self.cfg.control.decimation), self.num_envs,
            self.num_compliance_actions, dtype=torch.bool, device=self.device
        )
        self.quiet_trace_late_authority_scale = torch.ones(trace_shape, device=self.device)
        self.eval_late_authority_scale = None
        self.eval_late_warmup_steps = 0
        self.eval_late_policy_step = 0
        self.eval_counterfactual_mode = None
        self.eval_counterfactual_window_steps = 0
        self.eval_contact_state = torch.zeros(
            self.num_envs, self.num_compliance_actions,
            dtype=torch.bool, device=self.device
        )
        self.eval_release_age = torch.zeros(
            self.num_envs, self.num_compliance_actions,
            dtype=torch.long, device=self.device
        )
        self.eval_had_contact = torch.zeros(
            self.num_envs, self.num_compliance_actions,
            dtype=torch.bool, device=self.device
        )
        self._quiet_initialized = True

    def set_eval_counterfactual(self, mode, window_ms=50.0):
        if mode not in (
            None,
            "transient",
            "residual",
            "compression_velocity_reset",
            "compression_state_reset",
            "compression_state_clamp",
        ):
            raise ValueError("unknown evaluation counterfactual mode: " + str(mode))
        self.eval_counterfactual_mode = mode
        self.eval_counterfactual_window_steps = max(
            1, int(round(float(window_ms) * 1.0e-3 / float(self.sim_params.dt)))
        ) if mode is not None else 0

    def set_eval_late_authority(self, scale, warmup_steps):
        """Evaluation-only authority schedule; no GT contact enters the env."""
        if scale.shape[1:] != (self.num_envs, self.num_compliance_actions):
            raise ValueError("Offline authority schedule has incompatible shape")
        if not torch.all((scale >= 0.0) & (scale <= 1.0)):
            raise ValueError("Offline authority scale must be in [0, 1]")
        self.eval_late_authority_scale = scale.to(self.device)
        self.eval_late_warmup_steps = int(warmup_steps)
        self.eval_late_policy_step = 0

    def _update_quiet_metrics_substep(self, substep):
        _QuietMetricsMixin._update_quiet_metrics_substep(self, substep)
        values = {
            "impact_probability": self.admittance.impact_probability,
            "alpha": self.admittance.alpha,
            "beta": self.admittance.effective_alpha,
            "drive_force_n": self.admittance.drive_force,
            "transient_force_n": self.admittance.transient_force,
            "admittance_compression_m": self.admittance.delta_l,
            "admittance_compression_velocity_mps": self.admittance.delta_l_dot,
        }
        for name, value in values.items():
            getattr(self, "quiet_trace_" + name)[substep].copy_(
                value[:, self.quiet_to_adm_leg]
            )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._clear_quiet_transient(env_ids)
        self.eval_contact_state[env_ids] = False
        self.eval_release_age[env_ids] = 0
        self.eval_had_contact[env_ids] = False

    def step(self, policy_actions, contact_estimate=None):
        motion_actions, compliance_actions = self._split_policy_action(
            policy_actions
        )
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(motion_actions, -clip_actions, clip_actions).to(
            self.device
        )
        self.compliance_actions = torch.clamp(
            compliance_actions.to(self.device), 0.0, 1.0
        )
        self.policy_actions = torch.cat(
            (self.actions, self.compliance_actions), dim=-1
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
            for substep in range(self.cfg.control.decimation):
                self.delayed_actions[:, substep] = self.last_actions + (
                    self.actions - self.last_actions
                ) * (substep >= delay_steps)
        self._clear_quiet_step_buffers()
        self._begin_gt_impact_step()
        self.render()
        for substep in range(self.cfg.control.decimation):
            late_index = (
                (self.eval_late_policy_step - self.eval_late_warmup_steps)
                * self.cfg.control.decimation + substep
            )
            if self.eval_late_authority_scale is not None and late_index >= 0:
                if late_index >= len(self.eval_late_authority_scale):
                    raise RuntimeError("Offline authority schedule ended before evaluation")
                late_scale = self.eval_late_authority_scale[late_index]
            else:
                late_scale = torch.ones_like(self.compliance_actions)
            self.quiet_trace_late_authority_scale[substep].copy_(late_scale)
            release_window = (
                (self.eval_counterfactual_mode is not None)
                & self.eval_had_contact
                & (~self.eval_contact_state)
                & (self.eval_release_age > 0)
                & (self.eval_release_age <= self.eval_counterfactual_window_steps)
            )
            # Reset interventions act exactly once on the first controllable
            # physics substep after confirmed GT release. Clamp/transient/
            # residual modes cover the configured diagnostic window.
            if self.eval_counterfactual_mode in (
                "compression_velocity_reset", "compression_state_reset"
            ):
                cf_active = release_window & (self.eval_release_age == 1)
            else:
                cf_active = release_window
            self.quiet_trace_counterfactual_active[substep].copy_(cf_active)
            self.admittance.set_counterfactual(
                self.eval_counterfactual_mode,
                cf_active[:, self.adm_to_quiet_leg],
            )
            baseline_torques = self._compute_torques(
                self.delayed_actions[:, substep]
            ).view(self.torques.shape)
            self.torques = self._compute_adaptive_torques(
                self.delayed_actions[:, substep],
                self.compliance_actions,
                self.estimated_contact,
            ).view(self.torques.shape)
            if self.eval_late_authority_scale is not None:
                # Convex torque interpolation changes only the evaluated
                # hip/knee compliance residual. Alpha=0 stays exactly PD;
                # policy, estimator, and internal M/D/K state are unchanged.
                for indices in (self.adm_hip_indices, self.adm_knee_indices):
                    self.torques[:, indices] = (
                        baseline_torques[:, indices]
                        + late_scale[:, self.adm_to_quiet_leg]
                        * (self.torques[:, indices] - baseline_torques[:, indices])
                    )
            # Evaluation-only same-state/action counterfactual. Only adaptive
            # torques are applied to PhysX; this records exactly where the
            # compliance path departs from the original fixed-PD controller.
            self.quiet_trace_baseline_torque_residual[substep].copy_(
                self.torques - baseline_torques
            )
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
            torque_residual = self.quiet_trace_baseline_torque_residual[substep]
            hip_power = (
                torque_residual[:, self.adm_hip_indices]
                * self.dof_vel[:, self.adm_hip_indices]
            )
            knee_power = (
                torque_residual[:, self.adm_knee_indices]
                * self.dof_vel[:, self.adm_knee_indices]
            )
            # Signed mechanical power injected by the compliance residual.
            # This passive trace is never observed by the policy/controller.
            self.quiet_trace_compliance_power[substep].copy_(
                (hip_power + knee_power)[:, self.quiet_to_adm_leg]
            )
            self._update_quiet_metrics_substep(substep)
            self._update_gt_impact_substep()
            next_contact = self.quiet_trace_contact[substep]
            was_contact = self.eval_contact_state
            released = was_contact & (~next_contact)
            self.eval_release_age = torch.where(
                next_contact,
                torch.zeros_like(self.eval_release_age),
                torch.where(
                    released,
                    torch.ones_like(self.eval_release_age),
                    torch.where(
                        self.eval_had_contact,
                        torch.clamp(self.eval_release_age + 1, max=1000000),
                        torch.zeros_like(self.eval_release_age),
                    ),
                ),
            )
            self.eval_had_contact |= next_contact
            self.eval_contact_state.copy_(next_contact)

        self.eval_late_policy_step += 1

        self._finish_gt_impact_step()
        self.transition_contact_estimator_target.copy_(
            self.contact_estimator_target
        )
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


class QuietMCLearnedAdmittance100HzCfg(MCLearnedAdmittance100HzCfg):
    class quiet_metrics:
        enabled = True
        contact_on_threshold = 5.0
        contact_off_threshold = 2.0
        wheel_radius = 0.075


class QuietMCLearnedAdmittance100HzCfgPPO(MCLearnedAdmittance100HzCfgPPO):
    pass


def _extract_checkpoint_path():
    checkpoint = None
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--checkpoint_path":
            if i + 1 >= len(sys.argv):
                raise ValueError("Missing value after --checkpoint_path")
            checkpoint = sys.argv[i + 1]
            del sys.argv[i : i + 2]
            continue
        prefix = "--checkpoint_path="
        if arg.startswith(prefix):
            checkpoint = arg[len(prefix) :]
            del sys.argv[i]
            continue
        i += 1
    if not checkpoint:
        raise ValueError("--checkpoint_path is required")
    return os.path.abspath(checkpoint)


def evaluate(checkpoint_path, args, options):
    if TASK_NAME not in task_registry.task_classes:
        task_registry.register(
            TASK_NAME,
            QuietMCLearnedAdmittance100Hz,
            QuietMCLearnedAdmittance100HzCfg(),
            QuietMCLearnedAdmittance100HzCfgPPO(),
        )
    args.task = TASK_NAME
    env_cfg, train_cfg = task_registry.get_cfgs(TASK_NAME)
    _configure_environment(env_cfg, args, options)
    env, _ = task_registry.make_env(name=TASK_NAME, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = False
    train_cfg.runner.wandb_enabled = False
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=TASK_NAME,
        args=args,
        train_cfg=train_cfg,
        log_root="default",
    )
    runner.load(checkpoint_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    # Match the baseline evaluator's randomized reset exactly despite the
    # adaptive runner consuming additional RNG state during initialization.
    set_seed(options["eval_seed"])
    terrain_level, actual_difficulty = _place_on_fixed_terrain(
        env, options["eval_scenario"], options["stair_difficulty"]
    )
    _set_fixed_commands(env, options["eval_command_x"])
    _synchronize_observation_history(env)
    env.set_eval_counterfactual(
        options["counterfactual_mode"], options["counterfactual_window_ms"]
    )
    observations = env.get_observations()
    controller_state = env.get_controller_state().to(env.device)
    warmup_steps = int(options["warmup_seconds"] / env.dt)
    evaluation_steps = int(options["eval_seconds"] / env.dt)
    schedule_path = options.get("late_authority_schedule")
    if schedule_path:
        with np.load(schedule_path) as schedule:
            scale = torch.from_numpy(schedule[options["late_authority_mode"]].copy())
            reference_contact_shape = schedule["reference_contact"].shape
            if scale.shape != reference_contact_shape:
                raise ValueError("Offline reference contact shape differs from authority scale")
            if scale.shape[0] != evaluation_steps * env.cfg.control.decimation:
                raise ValueError("Offline schedule duration must match evaluation duration")
            if int(schedule["eval_seed"]) != options["eval_seed"]:
                raise ValueError("Offline schedule seed differs from evaluation seed")
            if not np.isclose(float(schedule["command_x_mps"]), options["eval_command_x"]):
                raise ValueError("Offline schedule command differs from evaluation command")
            if not np.isclose(float(schedule["stair_difficulty"]), options["stair_difficulty"]):
                raise ValueError("Offline schedule terrain differs from evaluation terrain")
            if os.path.realpath(str(schedule["reference_checkpoint"])) != os.path.realpath(checkpoint_path):
                raise ValueError("Offline schedule was generated from a different checkpoint")
        env.set_eval_late_authority(scale, warmup_steps)

    trace_keys = [
        "force_z_n", "force_norm_n", "loading_rate_z_nps",
        "loading_rate_norm_nps", "wheel_vel_z_mps",
        "wheel_pos_z_world_m", "wheel_pos_z_base_m", "leg_length_m",
        "wheel_lateral_speed_mps", "wheel_omega_radps",
        "wheel_alpha_radps2", "leg_compression_m", "contact",
        "base_acc_z_mps2", "base_jerk_z_mps3", "max_torque_rate_nmps",
        "motion_action", "torque", "dof_pos", "dof_vel", "root_state",
        "impact_probability", "alpha", "beta", "drive_force_n",
        "transient_force_n", "admittance_compression_m",
        "admittance_compression_velocity_mps", "baseline_torque_residual_nm",
        "compliance_power_w",
        "counterfactual_active",
        "late_authority_scale",
    ]
    traces = {key: [] for key in trace_keys}
    event_keys = [
        "touchdown_speed_mps", "touchdown_speed_3d_mps", "peak_force_n",
        "peak_contact_force_norm_n", "peak_loading_rate_nps",
        "peak_contact_loading_rate_norm_nps", "normal_impulse_ns",
        "contact_impulse_norm_ns", "peak_leg_compression_m",
    ]
    event_samples = {key: [] for key in event_keys}
    base_speed_samples = []
    tracking_error_samples = []
    orientation_error_samples = []
    tracking_lin_reward_samples = []
    tracking_ang_reward_samples = []
    reset_count = 0

    for step_index in range(warmup_steps + evaluation_steps):
        _set_fixed_commands(env, options["eval_command_x"])
        with torch.no_grad():
            controller_state = env.get_controller_state().to(env.device)
            actions = policy(observations.detach(), controller_state)
            if options["eval_zero_compliance"]:
                actions = actions.clone()
                actions[:, env.num_actions :] = 0.0
            contact_estimate = runner.alg.actor_critic.last_contact_estimate
            observations, _, _, dones, _, _, _ = env.step(
                actions.detach(), contact_estimate.detach()
            )
        if step_index == warmup_steps - 1:
            env.reset_quiet_metrics()
            reset_count = 0
        elif step_index >= warmup_steps:
            _append_traces(traces, env)
            for name in (
                "impact_probability",
                "alpha",
                "beta",
                "drive_force_n",
                "transient_force_n",
                "admittance_compression_m",
                "admittance_compression_velocity_mps",
            ):
                traces[name].append(
                    getattr(env, "quiet_trace_" + name).detach().cpu().numpy()
                )
            traces["baseline_torque_residual_nm"].append(
                env.quiet_trace_baseline_torque_residual.detach().cpu().numpy()
            )
            traces["compliance_power_w"].append(
                env.quiet_trace_compliance_power.detach().cpu().numpy()
            )
            traces["counterfactual_active"].append(
                env.quiet_trace_counterfactual_active.detach().cpu().numpy()
            )
            traces["late_authority_scale"].append(
                env.quiet_trace_late_authority_scale.detach().cpu().numpy()
            )
            _append_event_samples(event_samples, env)
            speeds = env.base_lin_vel[:, 0].detach().cpu().numpy()
            base_speed_samples.append(speeds)
            tracking_error_samples.append(
                np.abs(speeds - options["eval_command_x"])
            )
            orientation_error_samples.append(
                torch.norm(env.projected_gravity[:, :2], dim=1)
                .detach()
                .cpu()
                .numpy()
            )
            tracking_lin_reward_samples.append(
                (
                    env._reward_tracking_lin_vel()
                    * env.reward_scales["tracking_lin_vel"]
                )
                .detach()
                .cpu()
                .numpy()
            )
            tracking_ang_reward_samples.append(
                (
                    env._reward_tracking_ang_vel()
                    * env.reward_scales["tracking_ang_vel"]
                )
                .detach()
                .cpu()
                .numpy()
            )
            reset_count += int(torch.sum(dones).detach().cpu().item())

    packed_traces = {key: _pack(chunks) for key, chunks in traces.items()}
    packed_events = {key: _pack(chunks) for key, chunks in event_samples.items()}
    summary = env.get_quiet_metrics()
    _add_continuous_metrics(summary, packed_traces)
    for key, values in packed_events.items():
        from legged_gym.scripts.evaluate_mc_quiet import _add_distribution

        _add_distribution(summary, key, values)
    speed_values = _pack(base_speed_samples)
    tracking_values = _pack(tracking_error_samples)
    orientation_values = _pack(orientation_error_samples)
    scenario = options["eval_scenario"]
    step_height = 0.05 + 0.18 * actual_difficulty if scenario != "flat" else 0.0
    summary.update(
        {
            "task": TASK_NAME,
            "robot": "mc",
            "baseline": "MC_ImpactAware_Admittance_replay_0.25",
            "checkpoint_path": checkpoint_path,
            "scenario": scenario,
            "evaluation_seconds": float(options["eval_seconds"]),
            "warmup_seconds": float(options["warmup_seconds"]),
            "num_envs": int(env.num_envs),
            "eval_seed": int(options["eval_seed"]),
            "eval_zero_compliance": bool(options["eval_zero_compliance"]),
            "counterfactual_mode": options["counterfactual_mode"],
            "counterfactual_window_ms": float(options["counterfactual_window_ms"]),
            "late_authority_mode": options.get("late_authority_mode") if schedule_path else None,
            "late_authority_schedule": schedule_path,
            "command_x_mps": float(options["eval_command_x"]),
            "mean_actual_base_speed_x_mps": float(np.mean(speed_values)) if speed_values.size else 0.0,
            "mean_abs_tracking_error_x_mps": float(np.mean(tracking_values)) if tracking_values.size else 0.0,
            "mean_tracking_lin_reward": float(np.mean(_pack(tracking_lin_reward_samples))) if tracking_lin_reward_samples else 0.0,
            "mean_tracking_ang_reward": float(np.mean(_pack(tracking_ang_reward_samples))) if tracking_ang_reward_samples else 0.0,
            "mean_orientation_error": float(np.mean(orientation_values)) if orientation_values.size else 0.0,
            "reset_count": int(reset_count),
            "resets_per_robot_second": float(reset_count) / max(1.0, float(env.num_envs) * float(options["eval_seconds"])),
            "terrain_level": int(terrain_level),
            "terrain_difficulty": float(actual_difficulty),
            "stair_step_height_m": float(step_height),
            "stair_step_width_m": 0.30 if scenario != "flat" else 0.0,
            "sample_rate_hz": int(round(1.0 / float(env.sim_params.dt))),
        }
    )
    output = _save_results(
        summary, packed_traces, packed_events, scenario, summary["sample_rate_hz"]
    )
    print(f"Adaptive quiet evaluation summary: {output[1]}")
    print(f"Adaptive quiet evaluation output: {output[0]}")
    return summary


if __name__ == "__main__":
    checkpoint_path = _extract_checkpoint_path()
    # Parse private evaluation options before the shared HIM argument parser.
    late_options = {"late_authority_schedule": None, "late_authority_mode": "half"}
    for name in tuple(late_options):
        for arg in list(sys.argv[1:]):
            if arg.startswith("--" + name + "="):
                late_options[name] = arg.split("=", 1)[1]
                sys.argv.remove(arg)
                break
    if late_options["late_authority_mode"] not in ("half", "decay"):
        raise ValueError("--late_authority_mode must be half or decay")
    options = _extract_custom_args()
    options.update(late_options)
    evaluate(checkpoint_path, get_args(), options)
