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
)
from legged_gym.utils import get_args, task_registry


TASK_NAME = "mc_impact_quiet_eval"


class _QuietMetricsMixin:
    """Reuse QuietMC methods without creating an incompatible MC MRO."""

    _zeros_wheels = QuietMC._zeros_wheels
    _to_base_frame = QuietMC._to_base_frame
    _wheel_kinematics = QuietMC._wheel_kinematics
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
        self._init_quiet_metrics()
        self._quiet_initialized = True

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._clear_quiet_transient(env_ids)

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
        self._clear_quiet_step_buffers()
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
            self._update_quiet_metrics_substep(substep)
            self._update_gt_impact_substep()

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

    terrain_level, actual_difficulty = _place_on_fixed_terrain(
        env, options["eval_scenario"], options["stair_difficulty"]
    )
    _set_fixed_commands(env, options["eval_command_x"])
    observations = env.get_observations()
    controller_state = env.get_controller_state().to(env.device)
    warmup_steps = int(options["warmup_seconds"] / env.dt)
    evaluation_steps = int(options["eval_seconds"] / env.dt)

    trace_keys = [
        "force_z_n", "force_norm_n", "loading_rate_z_nps",
        "loading_rate_norm_nps", "wheel_vel_z_mps",
        "wheel_lateral_speed_mps", "wheel_omega_radps",
        "wheel_alpha_radps2", "leg_compression_m", "contact",
        "base_acc_z_mps2", "base_jerk_z_mps3", "max_torque_rate_nmps",
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
            contact_estimate = runner.alg.actor_critic.last_contact_estimate
            observations, _, _, dones, _, _, _ = env.step(
                actions.detach(), contact_estimate.detach()
            )
        if step_index == warmup_steps - 1:
            env.reset_quiet_metrics()
            reset_count = 0
        elif step_index >= warmup_steps:
            _append_traces(traces, env)
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
    options = _extract_custom_args()
    evaluate(checkpoint_path, get_args(), options)
