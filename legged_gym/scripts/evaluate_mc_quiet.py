"""Evaluate an MC HIMLoco checkpoint with physics-rate quiet-motion metrics.

The evaluation is passive: it does not change the trained policy, reward or
controller.  Three deterministic scenarios are supported:

    flat         common quiet baseline, comparable with the quadruped B0 test
    stairs_up    wheel-legged primary task, fixed staircase
    stairs_down  wheel-legged primary task, fixed staircase

Examples:
    python legged_gym/scripts/evaluate_mc_quiet.py --load_run=<run> \
        --checkpoint=6000 --headless --eval_scenario=flat

    python legged_gym/scripts/evaluate_mc_quiet.py --load_run=<run> \
        --checkpoint=6000 --headless --eval_scenario=stairs_down \
        --stair_difficulty=0.5 --eval_command_x=0.35

Custom evaluation arguments are removed from argv before the normal Isaac Gym
argument parser runs, so no global training CLI changes are required.
"""

import csv
import json
import os
import sys
from datetime import datetime

import isaacgym  # noqa: F401; must be imported before torch
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403; registers quiet_mc
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import get_load_path


DEFAULT_EVAL_SECONDS = 20.0
DEFAULT_WARMUP_SECONDS = 2.0
DEFAULT_NUM_ENVS = 16
DEFAULT_COMMAND_X = 0.5
DEFAULT_STAIR_DIFFICULTY = 0.5
TERRAIN_ROWS = 10


def _extract_custom_args():
    """Extract eval-only arguments without modifying the global HIM CLI."""
    specs = {
        "eval_scenario": (str, "flat"),
        "eval_seconds": (float, DEFAULT_EVAL_SECONDS),
        "warmup_seconds": (float, DEFAULT_WARMUP_SECONDS),
        "eval_command_x": (float, DEFAULT_COMMAND_X),
        "stair_difficulty": (float, DEFAULT_STAIR_DIFFICULTY),
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

    if values["eval_scenario"] not in ("flat", "stairs_up", "stairs_down"):
        raise ValueError(
            "--eval_scenario must be one of: flat, stairs_up, stairs_down"
        )
    values["stair_difficulty"] = float(
        np.clip(values["stair_difficulty"], 0.0, 0.9)
    )
    return values


def _scenario_proportions(scenario):
    if scenario == "flat":
        return [1.0, 0.0, 0.0, 0.0, 0.0]
    if scenario == "stairs_up":
        # Matches the repository terrain convention: third bucket uses
        # negative step_height and is labelled stairs-up in mc_config.py.
        return [0.0, 0.0, 1.0, 0.0, 0.0]
    return [0.0, 0.0, 0.0, 1.0, 0.0]


def _configure_environment(env_cfg, args, options):
    num_envs = args.num_envs if args.num_envs is not None else DEFAULT_NUM_ENVS
    env_cfg.env.num_envs = num_envs
    env_cfg.env.episode_length_s = (
        options["warmup_seconds"] + options["eval_seconds"] + 10.0
    )

    # Keep trimesh and height observations so the checkpoint observation shape
    # remains exactly identical to training.  A 10-row/1-column curriculum map
    # gives known difficulty per row; after construction all robots are moved to
    # one selected row and curriculum updates are disabled.
    env_cfg.terrain.mesh_type = "trimesh"
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.selected = False
    env_cfg.terrain.num_rows = TERRAIN_ROWS
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.max_init_terrain_level = 0
    env_cfg.terrain.terrain_proportions = _scenario_proportions(
        options["eval_scenario"]
    )
    env_cfg.terrain.measure_heights = True

    command_x = options["eval_command_x"]
    env_cfg.commands.curriculum = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 1.0e9
    # Fixed ranges also keep commands deterministic after any fall/reset.
    env_cfg.commands.ranges.lin_vel_x = [command_x, command_x]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0.0, 0.0]

    # Deterministic evaluation: disable every training randomization.
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


def _place_on_fixed_terrain(env, scenario, requested_difficulty):
    """Move every evaluation robot to one deterministic terrain row."""
    if scenario == "flat":
        level = 0
    else:
        # Terrain.make_terrain uses difficulty = row / num_rows.
        level = int(round(requested_difficulty * TERRAIN_ROWS))
        level = max(0, min(level, TERRAIN_ROWS - 1))

    env.cfg.terrain.curriculum = False
    env.terrain_levels[:] = level
    env.terrain_types[:] = 0
    env.env_origins[:] = env.terrain_origins[level, 0]
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.reset_idx(env_ids)
    return level, float(level) / float(TERRAIN_ROWS)


def _set_fixed_commands(env, command_x):
    env.commands[:, 0] = command_x
    env.commands[:, 1] = 0.0
    env.commands[:, 2] = 0.0
    if env.commands.shape[1] > 3:
        env.commands[:, 3] = 0.0


def _append_traces(store, env):
    mapping = {
        "force_z_n": env.quiet_trace_force_z,
        "force_norm_n": env.quiet_trace_force_norm,
        "loading_rate_z_nps": env.quiet_trace_loading_rate_z,
        "loading_rate_norm_nps": env.quiet_trace_loading_rate_norm,
        "wheel_vel_z_mps": env.quiet_trace_wheel_vel_z,
        "wheel_lateral_speed_mps": env.quiet_trace_wheel_lateral_speed,
        "wheel_omega_radps": env.quiet_trace_wheel_omega,
        "wheel_alpha_radps2": env.quiet_trace_wheel_alpha,
        "leg_compression_m": env.quiet_trace_leg_compression,
        "contact": env.quiet_trace_contact,
        "base_acc_z_mps2": env.quiet_trace_base_acc_z,
        "base_jerk_z_mps3": env.quiet_trace_base_jerk_z,
        "max_torque_rate_nmps": env.quiet_trace_max_torque_rate,
    }
    for key, tensor in mapping.items():
        array = tensor.detach().cpu().numpy()
        if key == "contact":
            array = array.astype(np.uint8)
        store[key].append(array)


def _append_event_samples(store, env):
    touchdown_mask = env.quiet_step_touchdown.detach().cpu().numpy().astype(bool)
    if np.any(touchdown_mask):
        for key, tensor in [
            ("touchdown_speed_mps", env.quiet_step_touchdown_vertical_speed),
            ("touchdown_speed_3d_mps", env.quiet_step_touchdown_speed_3d),
        ]:
            values = tensor.detach().cpu().numpy()[touchdown_mask]
            store[key].append(values)

    finished_mask = env.quiet_step_event_finished.detach().cpu().numpy().astype(bool)
    if np.any(finished_mask):
        mapping = {
            "peak_force_n": env.quiet_step_completed_peak_force_z,
            "peak_contact_force_norm_n": env.quiet_step_completed_peak_force_norm,
            "peak_loading_rate_nps": env.quiet_step_completed_peak_loading_rate_z,
            "peak_contact_loading_rate_norm_nps": env.quiet_step_completed_peak_loading_rate_norm,
            "normal_impulse_ns": env.quiet_step_completed_normal_impulse,
            "contact_impulse_norm_ns": env.quiet_step_completed_contact_impulse_norm,
            "peak_leg_compression_m": env.quiet_step_completed_peak_leg_compression,
        }
        for key, tensor in mapping.items():
            store[key].append(tensor.detach().cpu().numpy()[finished_mask])


def _pack(chunks):
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0,))


def _finite(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def _percentile(values, q):
    values = _finite(values)
    return float(np.percentile(values, q)) if values.size else 0.0


def _rms(values):
    values = _finite(values)
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


def _add_distribution(summary, prefix, values):
    values = _finite(values)
    summary[prefix + "_sample_count"] = int(values.size)
    summary[prefix + "_p50"] = _percentile(values, 50)
    summary[prefix + "_p75"] = _percentile(values, 75)
    summary[prefix + "_p90"] = _percentile(values, 90)
    summary[prefix + "_p95"] = _percentile(values, 95)
    summary[prefix + "_p99"] = _percentile(values, 99)
    summary[prefix + "_max"] = float(np.max(values)) if values.size else 0.0


def _add_continuous_metrics(summary, traces):
    contact = traces["contact"].astype(bool)
    contact_force_norm = traces["force_norm_n"][contact]
    contact_force_z = traces["force_z_n"][contact]
    contact_lateral_speed = traces["wheel_lateral_speed_mps"][contact]
    contact_compression = traces["leg_compression_m"][contact]

    positive_load_norm = traces["loading_rate_norm_nps"]
    positive_load_norm = positive_load_norm[positive_load_norm > 0.0]
    positive_load_z = traces["loading_rate_z_nps"]
    positive_load_z = positive_load_z[positive_load_z > 0.0]

    summary["wheel_contact_duty_ratio"] = float(np.mean(contact)) if contact.size else 0.0
    _add_distribution(summary, "contact_force_norm_n", contact_force_norm)
    _add_distribution(summary, "contact_force_z_n", contact_force_z)
    _add_distribution(summary, "positive_loading_rate_norm_nps", positive_load_norm)
    _add_distribution(summary, "positive_loading_rate_z_nps", positive_load_z)
    _add_distribution(summary, "contact_wheel_lateral_speed_mps", contact_lateral_speed)
    _add_distribution(summary, "contact_leg_compression_m", contact_compression)
    _add_distribution(summary, "wheel_alpha_radps2", traces["wheel_alpha_radps2"])
    _add_distribution(summary, "base_acc_z_mps2", np.abs(traces["base_acc_z_mps2"]))
    _add_distribution(summary, "base_jerk_z_mps3", np.abs(traces["base_jerk_z_mps3"]))
    _add_distribution(summary, "max_torque_rate_nmps", traces["max_torque_rate_nmps"])

    summary["base_acc_z_rms_mps2"] = _rms(traces["base_acc_z_mps2"])
    summary["base_jerk_z_rms_mps3"] = _rms(traces["base_jerk_z_mps3"])
    summary["wheel_alpha_rms_radps2"] = _rms(traces["wheel_alpha_radps2"])
    summary["max_torque_rate_rms_nmps"] = _rms(traces["max_torque_rate_nmps"])


def _save_results(summary, traces, event_samples, scenario, sample_rate_hz):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", "mc_quiet_eval", scenario, timestamp
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
    return output_dir, json_path, csv_path, trace_path, event_path


def evaluate(args, options):
    args.task = "quiet_mc"
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    _configure_environment(env_cfg, args, options)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    train_cfg.runner.resume = True
    train_cfg.runner.wandb_enabled = False

    load_run = args.load_run if args.load_run is not None else train_cfg.runner.load_run
    checkpoint = (
        args.checkpoint if args.checkpoint is not None else train_cfg.runner.checkpoint
    )
    log_root = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name
    )
    checkpoint_path = get_load_path(
        log_root, load_run=load_run, checkpoint=checkpoint
    )

    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root="default",
    )
    policy = runner.get_inference_policy(device=env.device)

    terrain_level, actual_difficulty = _place_on_fixed_terrain(
        env, options["eval_scenario"], options["stair_difficulty"]
    )
    _set_fixed_commands(env, options["eval_command_x"])
    observations = env.get_observations()

    warmup_steps = int(options["warmup_seconds"] / env.dt)
    evaluation_steps = int(options["eval_seconds"] / env.dt)

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
    reset_count = 0

    total_steps = warmup_steps + evaluation_steps
    for step_index in range(total_steps):
        _set_fixed_commands(env, options["eval_command_x"])
        with torch.no_grad():
            actions = policy(observations.detach())
            observations, _, _, dones, _, _, _ = env.step(actions.detach())

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
                .detach().cpu().numpy()
            )
            reset_count += int(torch.sum(dones).detach().cpu().item())

    packed_traces = {key: _pack(chunks) for key, chunks in traces.items()}
    packed_events = {key: _pack(chunks) for key, chunks in event_samples.items()}

    summary = env.get_quiet_metrics()
    _add_continuous_metrics(summary, packed_traces)
    for key, values in packed_events.items():
        _add_distribution(summary, key, values)

    speed_values = _pack(base_speed_samples)
    tracking_values = _pack(tracking_error_samples)
    orientation_values = _pack(orientation_error_samples)

    scenario = options["eval_scenario"]
    stair_step_height = (
        0.05 + 0.18 * actual_difficulty if scenario != "flat" else 0.0
    )
    sample_rate_hz = int(round(1.0 / float(env.sim_params.dt)))
    summary.update(
        {
            "task": "quiet_mc",
            "robot": "mc",
            "baseline": "MC_HIM_fixed_pd_original_reward",
            "checkpoint_path": checkpoint_path,
            "scenario": scenario,
            "evaluation_seconds": float(options["eval_seconds"]),
            "warmup_seconds": float(options["warmup_seconds"]),
            "num_envs": int(env.num_envs),
            "command_x_mps": float(options["eval_command_x"]),
            "mean_actual_base_speed_x_mps": float(np.mean(speed_values)) if speed_values.size else 0.0,
            "mean_abs_tracking_error_x_mps": float(np.mean(tracking_values)) if tracking_values.size else 0.0,
            "mean_orientation_error": float(np.mean(orientation_values)) if orientation_values.size else 0.0,
            "reset_count": int(reset_count),
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
        }
    )

    output = _save_results(
        summary, packed_traces, packed_events, scenario, sample_rate_hz
    )
    output_dir, json_path, csv_path, trace_path, event_path = output

    print("\n========== MC Quiet Evaluation ==========")
    for key in sorted(summary.keys()):
        print(f"{key}: {summary[key]}")
    print("=========================================")
    print(f"Output directory: {output_dir}")
    print(f"Summary JSON: {json_path}")
    print(f"Summary CSV: {csv_path}")
    print(f"Physics-rate raw trace: {trace_path}")
    print(f"Event samples: {event_path}")


if __name__ == "__main__":
    custom_options = _extract_custom_args()
    evaluate(get_args(), custom_options)
