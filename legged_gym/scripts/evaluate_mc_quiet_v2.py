"""MC quiet-evaluation protocol v2.

Compared with ``evaluate_mc_quiet.py`` this protocol adds:

* deterministic nominal resets via ``quiet_mc_v2``;
* true plane geometry for the flat test;
* short fixed-difficulty stair windows that stay inside one 8 m terrain tile;
* forward-progress and traversal-success gates;
* geometric hip-to-wheel leg-compression instrumentation.

The policy, reward and controller are not modified.

Recommended protocol:

    flat:        0.50 m/s, 2 s warmup + 20 s measurement
    stairs_up:   0.50 m/s, 1 s warmup + 6 s measurement
    stairs_down: 0.35 m/s, 1 s warmup + 6 s measurement

Examples:
    python legged_gym/scripts/evaluate_mc_quiet_v2.py \
        --load_run=<run> --checkpoint=6000 --headless \
        --eval_scenario=stairs_up

    python legged_gym/scripts/evaluate_mc_quiet_v2.py \
        --load_run=<run> --checkpoint=6000 --headless \
        --eval_scenario=stairs_down
"""

import csv
import json
import os
import sys
from datetime import datetime

import isaacgym  # noqa: F401; Isaac Gym must be imported before torch
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403; registers quiet_mc_v2
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import get_load_path

# Reuse the already validated signal packing/statistics helpers from eval v1.
from legged_gym.scripts.evaluate_mc_quiet import (
    _add_continuous_metrics,
    _add_distribution,
    _append_event_samples,
    _append_traces,
    _pack,
    _scenario_proportions,
    _set_fixed_commands,
)


TERRAIN_ROWS = 10
DEFAULT_STAIR_DIFFICULTY = 0.5
DEFAULT_SUCCESS_PROGRESS_RATIO = 0.8

SCENARIO_DEFAULTS = {
    "flat": {"command_x": 0.50, "warmup_s": 2.0, "eval_s": 20.0},
    "stairs_up": {"command_x": 0.50, "warmup_s": 1.0, "eval_s": 6.0},
    "stairs_down": {"command_x": 0.35, "warmup_s": 1.0, "eval_s": 6.0},
}


def _extract_custom_args():
    specs = {
        "eval_scenario": (str, "flat"),
        "eval_seconds": (float, None),
        "warmup_seconds": (float, None),
        "eval_command_x": (float, None),
        "stair_difficulty": (float, DEFAULT_STAIR_DIFFICULTY),
        "success_progress_ratio": (float, DEFAULT_SUCCESS_PROGRESS_RATIO),
    }
    values = {name: default for name, (_, default) in specs.items()}

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        matched = False
        for name, (cast, _) in specs.items():
            flag = "--" + name
            if arg == flag:
                if i + 1 >= len(sys.argv):
                    raise ValueError(f"Missing value after {flag}")
                values[name] = cast(sys.argv[i + 1])
                del sys.argv[i:i + 2]
                matched = True
                break
            prefix = flag + "="
            if arg.startswith(prefix):
                values[name] = cast(arg[len(prefix):])
                del sys.argv[i]
                matched = True
                break
        if not matched:
            i += 1

    scenario = values["eval_scenario"]
    if scenario not in SCENARIO_DEFAULTS:
        raise ValueError("--eval_scenario must be flat, stairs_up or stairs_down")

    defaults = SCENARIO_DEFAULTS[scenario]
    if values["eval_seconds"] is None:
        values["eval_seconds"] = defaults["eval_s"]
    if values["warmup_seconds"] is None:
        values["warmup_seconds"] = defaults["warmup_s"]
    if values["eval_command_x"] is None:
        values["eval_command_x"] = defaults["command_x"]

    values["stair_difficulty"] = float(
        np.clip(values["stair_difficulty"], 0.0, 0.9)
    )
    values["success_progress_ratio"] = float(
        np.clip(values["success_progress_ratio"], 0.0, 1.5)
    )
    return values


def _configure_environment(env_cfg, args, options):
    env_cfg.env.num_envs = args.num_envs if args.num_envs is not None else 16
    env_cfg.env.episode_length_s = (
        options["warmup_seconds"] + options["eval_seconds"] + 10.0
    )

    scenario = options["eval_scenario"]
    if scenario == "flat":
        # A true infinite plane avoids the v1 problem where a long run could
        # leave terrain row 0 and enter a non-flat curriculum row.
        env_cfg.terrain.mesh_type = "plane"
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.selected = False
        env_cfg.terrain.measure_heights = True
    else:
        # Keep the training terrain generator so policy height observations are
        # unchanged.  The measurement window is deliberately short enough that
        # the robot remains in one 8 m terrain tile.
        env_cfg.terrain.mesh_type = "trimesh"
        env_cfg.terrain.curriculum = True
        env_cfg.terrain.selected = False
        env_cfg.terrain.num_rows = TERRAIN_ROWS
        env_cfg.terrain.num_cols = 1
        env_cfg.terrain.max_init_terrain_level = 0
        env_cfg.terrain.terrain_proportions = _scenario_proportions(scenario)
        env_cfg.terrain.measure_heights = True

    command_x = options["eval_command_x"]
    env_cfg.commands.curriculum = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 1.0e9
    env_cfg.commands.ranges.lin_vel_x = [command_x, command_x]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0.0, 0.0]

    # Deterministic dynamics. QuietMCV2 also removes root/joint reset randomness
    # that the base environment applies unconditionally.
    env_cfg.noise.add_noise = False
    for field in [
        "randomize_friction",
        "randomize_restitution",
        "randomize_payload_mass",
        "randomize_com_displacement",
        "randomize_link_mass",
        "randomize_motor_strength",
        "randomize_kp",
        "randomize_kd",
        "randomize_initial_joint_pos",
        "push_robots",
        "disturbance",
        "delay",
    ]:
        if hasattr(env_cfg.domain_rand, field):
            setattr(env_cfg.domain_rand, field, False)


def _place_and_reset(env, options):
    scenario = options["eval_scenario"]
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    if scenario == "flat":
        terrain_level = 0
        actual_difficulty = 0.0
    else:
        level = int(round(options["stair_difficulty"] * TERRAIN_ROWS))
        level = max(0, min(level, TERRAIN_ROWS - 1))
        env.cfg.terrain.curriculum = False
        env.terrain_levels[:] = level
        env.terrain_types[:] = 0
        env.env_origins[:] = env.terrain_origins[level, 0]
        terrain_level = level
        actual_difficulty = float(level) / float(TERRAIN_ROWS)

    env.reset_idx(env_ids)
    _set_fixed_commands(env, options["eval_command_x"])

    # Discard any observation history left from runner construction/reset and
    # start a clean warmup history at the fixed command.
    env.obs_buf.zero_()
    if env.privileged_obs_buf is not None:
        env.privileged_obs_buf.zero_()
    env.compute_observations()
    return terrain_level, actual_difficulty


def _save_results(summary, traces, event_samples, progress_arrays, scenario, sample_rate_hz):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", "mc_quiet_eval_v2", scenario, timestamp
    )
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "summary.json")
    with open(json_path, "w") as file:
        json.dump(summary, file, indent=2, sort_keys=True)

    csv_path = os.path.join(output_dir, "summary.csv")
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    trace_path = os.path.join(output_dir, f"raw_trace_{sample_rate_hz}hz.npz")
    np.savez_compressed(trace_path, **traces)
    event_path = os.path.join(output_dir, "event_samples.npz")
    np.savez_compressed(event_path, **event_samples)
    progress_path = os.path.join(output_dir, "progress_per_env.npz")
    np.savez_compressed(progress_path, **progress_arrays)
    return output_dir, json_path, csv_path, trace_path, event_path, progress_path


def evaluate(args, options):
    args.task = "quiet_mc_v2"
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

        # Start measurement after warmup and use the robot position at exactly
        # that instant as progress zero.
        if step_index == warmup_steps - 1:
            env.reset_quiet_metrics()
            measurement_start_x = env.root_states[:, 0].clone()
            max_progress.zero_()
            reset_count_per_env.zero_()
            continue

        if step_index >= warmup_steps:
            _append_traces(traces, env)
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

    # Warn if the commanded total motion would leave the selected 8 m stair
    # tile.  The recommended defaults stay below the ~4 m half-tile distance.
    total_commanded_distance = float(options["eval_command_x"]) * (
        float(options["warmup_seconds"]) + float(options["eval_seconds"])
    )
    stair_tile_warning = bool(
        scenario != "flat" and total_commanded_distance > 3.6
    )

    summary.update(
        {
            "eval_protocol_version": 2,
            "task": "quiet_mc_v2",
            "robot": "mc",
            "baseline": "MC_HIM_fixed_pd_original_reward",
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
            "contact_on_threshold_n": float(env.quiet_contact_on),
            "contact_off_threshold_n": float(env.quiet_contact_off),
            "sample_rate_hz": sample_rate_hz,
            "terrain_level": int(terrain_level),
            "terrain_difficulty": float(actual_difficulty),
            "stair_step_height_m": float(stair_step_height),
            "stair_step_width_m": 0.30 if scenario != "flat" else 0.0,
            "wheel_radius_m": float(env.quiet_wheel_radius),
            "leg_compression_definition": "hip_to_wheel_geometric_length_reduction",
            "stair_tile_boundary_warning": stair_tile_warning,
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
    output_dir, json_path, csv_path, trace_path, event_path, progress_path = output

    print("\n========== MC Quiet Evaluation v2 ==========")
    for key in sorted(summary.keys()):
        print(f"{key}: {summary[key]}")
    print("============================================")
    print(f"Output directory: {output_dir}")
    print(f"Summary JSON: {json_path}")
    print(f"Summary CSV: {csv_path}")
    print(f"Physics-rate raw trace: {trace_path}")
    print(f"Event samples: {event_path}")
    print(f"Per-env progress: {progress_path}")


if __name__ == "__main__":
    custom_options = _extract_custom_args()
    evaluate(get_args(), custom_options)
