"""Evaluate MC-HIM-100Hz checkpoints with the same quiet protocol v2.

The evaluator also records optional controller diagnostics when the environment
exposes them.  This keeps fixed-PD and admittance runs on exactly the same
terrain, command and quiet-metric protocol.
"""

import os

import isaacgym  # noqa: F401
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import get_load_path
from legged_gym.scripts.evaluate_mc_quiet import (
    _add_continuous_metrics,
    _add_distribution,
    _append_event_samples,
    _append_traces,
    _pack,
    _set_fixed_commands,
)
from legged_gym.scripts.evaluate_mc_quiet_v2 import (
    _extract_custom_args,
    _configure_environment,
    _place_and_reset,
    _save_results,
)


def evaluate(args, options):
    args.task = "quiet_mc_100hz_v2"
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    _configure_environment(env_cfg, args, options)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    train_cfg.runner.resume = True
    train_cfg.runner.wandb_enabled = False
    load_run = args.load_run if args.load_run is not None else train_cfg.runner.load_run
    checkpoint = args.checkpoint if args.checkpoint is not None else train_cfg.runner.checkpoint
    log_root = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name
    )
    checkpoint_path = get_load_path(log_root, load_run=load_run, checkpoint=checkpoint)

    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root="default",
    )
    policy = runner.get_inference_policy(device=env.device)

    terrain_level, actual_difficulty = _place_and_reset(env, options)
    observations = env.get_observations()
    warmup_steps = int(round(options["warmup_seconds"] / env.dt))
    evaluation_steps = int(round(options["eval_seconds"] / env.dt))

    trace_keys = [
        "force_z_n",
        "force_norm_n",
        "loading_rate_z_nps",
        "loading_rate_norm_nps",
        "wheel_vel_z_mps",
        "wheel_lateral_speed_mps",
        "wheel_omega_radps",
        "wheel_alpha_radps2",
        "leg_compression_m",
        "contact",
        "base_acc_z_mps2",
        "base_jerk_z_mps3",
        "max_torque_rate_nmps",
    ]
    has_admittance = hasattr(env, "admittance_delta_l")
    if has_admittance:
        trace_keys += [
            "admittance_delta_l_m",
            "admittance_delta_l_dot_mps",
            "admittance_axial_force_n",
            "admittance_force_input_n",
            "admittance_joint_offset_rad",
        ]
    traces = {key: [] for key in trace_keys}

    event_keys = [
        "touchdown_speed_mps",
        "touchdown_speed_3d_mps",
        "peak_force_n",
        "peak_contact_force_norm_n",
        "peak_loading_rate_nps",
        "peak_contact_loading_rate_norm_nps",
        "normal_impulse_ns",
        "contact_impulse_norm_ns",
        "peak_leg_compression_m",
    ]
    event_samples = {key: [] for key in event_keys}

    base_speed_samples = []
    tracking_error_samples = []
    orientation_error_samples = []
    reset_count_per_env = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    measurement_start_x = None
    max_progress = torch.zeros(env.num_envs, device=env.device)

    total_steps = warmup_steps + evaluation_steps
    for step_index in range(total_steps):
        _set_fixed_commands(env, options["eval_command_x"])
        with torch.no_grad():
            actions = policy(observations.detach())
            observations, _, _, dones, _, _, _ = env.step(actions.detach())

        if step_index == warmup_steps - 1:
            env.reset_quiet_metrics()
            measurement_start_x = env.root_states[:, 0].clone()
            max_progress.zero_()
            reset_count_per_env.zero_()
            continue

        if step_index >= warmup_steps:
            _append_traces(traces, env)
            if has_admittance:
                traces["admittance_delta_l_m"].append(
                    env.admittance_delta_l.detach().cpu().numpy()
                )
                traces["admittance_delta_l_dot_mps"].append(
                    env.admittance_delta_l_dot.detach().cpu().numpy()
                )
                traces["admittance_axial_force_n"].append(
                    env.admittance_axial_force.detach().cpu().numpy()
                )
                traces["admittance_force_input_n"].append(
                    env.admittance_force_input.detach().cpu().numpy()
                )
                traces["admittance_joint_offset_rad"].append(
                    env.admittance_joint_offset.detach().cpu().numpy()
                )

            _append_event_samples(event_samples, env)
            speeds = env.base_lin_vel[:, 0].detach().cpu().numpy()
            base_speed_samples.append(speeds)
            tracking_error_samples.append(
                np.abs(speeds - options["eval_command_x"])
            )
            orientation_error_samples.append(
                torch.norm(env.projected_gravity[:, :2], dim=1)
                .detach().cpu().numpy()
            )
            reset_count_per_env += dones.to(torch.long)
            progress = env.root_states[:, 0] - measurement_start_x
            max_progress = torch.maximum(max_progress, progress)

    final_progress = env.root_states[:, 0] - measurement_start_x
    expected_progress = max(
        1.0e-6,
        float(options["eval_command_x"]) * float(options["eval_seconds"]),
    )
    max_progress_ratio = max_progress / expected_progress
    final_progress_ratio = final_progress / expected_progress
    success = (
        (max_progress_ratio >= options["success_progress_ratio"])
        & (reset_count_per_env == 0)
    )

    packed_traces = {key: _pack(chunks) for key, chunks in traces.items()}
    packed_events = {key: _pack(chunks) for key, chunks in event_samples.items()}
    summary = env.get_quiet_metrics()
    _add_continuous_metrics(summary, packed_traces)
    for key, values in packed_events.items():
        _add_distribution(summary, key, values)

    if has_admittance:
        _add_distribution(
            summary, "admittance_compression_m", packed_traces["admittance_delta_l_m"]
        )
        _add_distribution(
            summary,
            "admittance_compression_speed_abs_mps",
            np.abs(packed_traces["admittance_delta_l_dot_mps"]),
        )
        _add_distribution(
            summary, "admittance_axial_force_n", packed_traces["admittance_axial_force_n"]
        )
        _add_distribution(
            summary, "admittance_force_input_n", packed_traces["admittance_force_input_n"]
        )
        _add_distribution(
            summary,
            "admittance_joint_offset_abs_rad",
            np.abs(packed_traces["admittance_joint_offset_rad"]),
        )
        compression_values = np.asarray(packed_traces["admittance_delta_l_m"])
        summary["admittance_active_ratio"] = (
            float(np.mean(compression_values > 1.0e-4))
            if compression_values.size
            else 0.0
        )

    speed_values = _pack(base_speed_samples)
    tracking_values = _pack(tracking_error_samples)
    orientation_values = _pack(orientation_error_samples)
    max_progress_np = max_progress.detach().cpu().numpy()
    final_progress_np = final_progress.detach().cpu().numpy()
    max_progress_ratio_np = max_progress_ratio.detach().cpu().numpy()
    final_progress_ratio_np = final_progress_ratio.detach().cpu().numpy()
    reset_per_env_np = reset_count_per_env.detach().cpu().numpy()
    success_np = success.detach().cpu().numpy().astype(np.uint8)

    scenario = options["eval_scenario"]
    stair_step_height = (
        0.05 + 0.18 * actual_difficulty if scenario != "flat" else 0.0
    )
    sample_rate_hz = int(round(1.0 / float(env.sim_params.dt)))
    reset_count = int(np.sum(reset_per_env_np))
    mean_speed = float(np.mean(speed_values)) if speed_values.size else 0.0

    baseline_name = getattr(
        env, "quiet_baseline_name", "MC_HIM_100Hz_fixed_pd_original_reward"
    )
    controller_name = getattr(env, "quiet_controller_name", "fixed_pd")

    summary.update(
        {
            "eval_protocol_version": 2,
            "task": "quiet_mc_100hz_v2",
            "robot": "mc",
            "baseline": baseline_name,
            "controller": controller_name,
            "checkpoint_path": checkpoint_path,
            "scenario": scenario,
            "evaluation_seconds": float(options["eval_seconds"]),
            "warmup_seconds": float(options["warmup_seconds"]),
            "num_envs": int(env.num_envs),
            "command_x_mps": float(options["eval_command_x"]),
            "mean_actual_base_speed_x_mps": mean_speed,
            "mean_speed_command_ratio": mean_speed / max(1.0e-6, float(options["eval_command_x"])),
            "mean_abs_tracking_error_x_mps": float(np.mean(tracking_values)) if tracking_values.size else 0.0,
            "mean_orientation_error": float(np.mean(orientation_values)) if orientation_values.size else 0.0,
            "mean_final_forward_progress_m": float(np.mean(final_progress_np)),
            "mean_max_forward_progress_m": float(np.mean(max_progress_np)),
            "mean_final_progress_ratio": float(np.mean(final_progress_ratio_np)),
            "mean_max_progress_ratio": float(np.mean(max_progress_ratio_np)),
            "traversal_success_progress_ratio_threshold": float(options["success_progress_ratio"]),
            "traversal_success_rate": float(np.mean(success_np)),
            "successful_env_count": int(np.sum(success_np)),
            "reset_count": reset_count,
            "resets_per_robot_second": float(reset_count) / max(
                1.0, float(env.num_envs) * float(options["eval_seconds"])
            ),
            "sample_rate_hz": sample_rate_hz,
            "policy_rate_hz": int(round(1.0 / float(env.dt))),
            "terrain_level": int(terrain_level),
            "terrain_difficulty": float(actual_difficulty),
            "stair_step_height_m": float(stair_step_height),
            "stair_step_width_m": 0.30 if scenario != "flat" else 0.0,
            "wheel_radius_m": float(env.quiet_wheel_radius),
            "leg_compression_definition": "hip_to_wheel_geometric_length_reduction",
        }
    )

    progress_arrays = {
        "final_forward_progress_m": final_progress_np,
        "max_forward_progress_m": max_progress_np,
        "final_progress_ratio": final_progress_ratio_np,
        "max_progress_ratio": max_progress_ratio_np,
        "reset_count": reset_per_env_np,
        "success": success_np,
    }
    output = _save_results(
        summary,
        packed_traces,
        packed_events,
        progress_arrays,
        scenario,
        sample_rate_hz,
    )
    print("\n========== MC-HIM-100Hz Quiet Evaluation v2 ==========")
    for key in sorted(summary.keys()):
        print(f"{key}: {summary[key]}")
    print("=======================================================")
    print(f"Output directory: {output[0]}")


if __name__ == "__main__":
    options = _extract_custom_args()
    args = get_args()
    evaluate(args, options)
