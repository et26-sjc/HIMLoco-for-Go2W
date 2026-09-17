"""Render one learned-admittance MC robot walking down a single staircase.

This is a viewer-only script.  It does not train, does not touch the reward,
the estimator loss, the admittance law or any policy dimension.  It reuses the
registered ``mc_learned_admittance_100hz`` task and drives a trained adaptive
checkpoint exactly the way ``evaluate_mc_impact_quiet.py`` does:

    controller_state = env.get_controller_state()
    policy_action    = policy(obs, controller_state)
    env.step(policy_action, policy.last_contact_estimate)

Differences from the evaluators are limited to the scene: one environment, a
two-cell terrain map and the Isaac Gym viewer left on, so the script fits in a
4 GiB GPU and can be watched interactively.

Terrain
-------
The scene is built through the regular curriculum path with a proportions
vector that selects one terrain type, exactly as ``evaluate_mc_quiet.py`` does
for its ``stairs_down`` scenario:

    terrain_proportions = [0, 0, 0, 1, 0]   # fourth bucket -> stairs, positive
                                            # step_height -> descending

``make_terrain`` picks the riser height from the row difficulty
(``difficulty = row / num_rows``, ``step_height = 0.05 + 0.18 * difficulty``),
so two rows give an 8 m x 16 m map whose second row is the documented
difficulty-0.5 staircase with 0.14 m risers.  The robot is placed on that row
and walks down it towards +x.

``pyramid_stairs_terrain`` raises the height field by one riser on each inward
ring, so the cell centre is the highest point, the ground descends in every
direction, and the outermost ring stays at z = 0 where it merges with the flat
border.  The robot therefore starts on the top platform, steps down, and
continues onto flat ground.

(The ``terrain.selected`` + ``terrain_kwargs`` shortcut is not usable here: the
``selected_terrain`` branch of ``legged_gym/utils/terrain.py`` reads
``self.vertical_scale`` / ``self.horizontal_scale``, which ``Terrain`` never
defines, so it raises ``AttributeError`` before any cell is built.)

Usage
-----
    python legged_gym/scripts/play_mc_impact_stairs.py \
        --checkpoint_path=logs/.../model_500.pt

    # slower descent, no automatic restart, stop after 2000 policy steps
    python legged_gym/scripts/play_mc_impact_stairs.py \
        --checkpoint_path=logs/.../model_500.pt \
        --command_x=0.35 --loop_reset=0 --steps=2000

Press ESC in the viewer (or Ctrl-C in the terminal) to quit; ``V`` toggles
viewer synchronisation.  Closing the window with the window-manager button can
end the process with a native SIGSEGV inside the Isaac Gym viewer teardown on
this machine; that happens after the loop has already stopped and is harmless,
but ``--steps=<n>`` is the clean way to bound a run.
"""

import os
import sys

import numpy as np

import isaacgym  # noqa: F401; must be imported before torch
import torch

from legged_gym.envs import *  # noqa: F401,F403; registers the adaptive MC task
from legged_gym.utils import get_args, task_registry


TASK_NAME = "mc_learned_admittance_100hz"


# ``make_terrain`` hard-codes the tread at 0.30 m and floors it to whole
# horizontal pixels, so the physical tread is int(0.30 / 0.1) * 0.1 = 0.2 m.
TERRAIN_TREAD_M = 0.2
TERRAIN_PLATFORM_M = 3.0


def _riser_height(row, num_rows):
    """Riser used by ``make_terrain`` for a given curriculum row."""
    difficulty = float(row) / float(num_rows)
    return 0.05 + 0.18 * difficulty


def _terrain_origin(env_cfg, row):
    """World x/y of a curriculum cell, matching ``add_terrain_to_map``."""
    return (
        (row + 0.5) * float(env_cfg.terrain.terrain_length),
        0.5 * float(env_cfg.terrain.terrain_width),
    )


def _extract_custom_args():
    """Remove play-only arguments from argv before the Isaac Gym parser runs.

    ``get_args`` forwards argv to the Isaac Gym argument parser, so custom
    flags are consumed here first.  This mirrors the pattern already used by
    ``evaluate_mc_quiet.py`` and leaves the global CLI unchanged.
    """
    specs = {
        "checkpoint_path": (str, None),
        "command_x": (float, 0.5),
        "loop_reset": (int, 1),
        "steps": (int, 0),
        "print_every": (int, 25),
        "num_rows": (int, 2),
        "terrain_row": (int, 1),
    }
    values = {name: default for name, (_, default) in specs.items()}

    index = 1
    while index < len(sys.argv):
        arg = sys.argv[index]
        matched = False
        for name, (cast, _) in specs.items():
            flag = "--" + name
            if arg == flag:
                if index + 1 >= len(sys.argv):
                    raise ValueError(f"Missing value after {flag}")
                values[name] = cast(sys.argv[index + 1])
                del sys.argv[index:index + 2]
                matched = True
                break
            prefix = flag + "="
            if arg.startswith(prefix):
                values[name] = cast(arg[len(prefix):])
                del sys.argv[index]
                matched = True
                break
        if not matched:
            index += 1

    if values["command_x"] <= 0.0:
        raise ValueError("--command_x must be positive")
    if not values["checkpoint_path"]:
        raise ValueError("--checkpoint_path is required")
    if values["num_rows"] < 1:
        raise ValueError("--num_rows must be at least 1")
    if not 0 <= values["terrain_row"] < values["num_rows"]:
        raise ValueError("--terrain_row must satisfy 0 <= row < num_rows")
    return values


def _configure_single_stair_scene(env_cfg, options):
    """One staircase map, deterministic commands, no randomization."""
    # The observation layout must stay byte-identical to training, so the
    # terrain type, the height measurements and the observation noise flags
    # keep their trained semantics; only the scene extent changes.
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 1.0e6  # the demo must not time out

    # Terrain through the regular curriculum path, exactly as
    # evaluate_mc_quiet.py builds its stairs_down scenario: the proportions
    # vector puts all the weight on the fourth bucket, whose positive
    # ``step_height`` makes make_terrain build a descending pyramid staircase.
    # ``curriculum`` must stay True *while the map is built*, because the
    # alternative paths either build a random map or are broken; the flag is
    # switched off after construction so the levels are frozen (see main()).
    env_cfg.terrain.mesh_type = "trimesh"
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.selected = False
    env_cfg.terrain.num_rows = int(options["num_rows"])
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.max_init_terrain_level = 0
    env_cfg.terrain.terrain_proportions = [0.0, 0.0, 0.0, 1.0, 0.0]
    env_cfg.terrain.measure_heights = True

    command_x = float(options["command_x"])
    env_cfg.commands.curriculum = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 1.0e9
    env_cfg.commands.ranges.lin_vel_x = [command_x, command_x]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0.0, 0.0]

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

    # Side view of the descent path.  The robot spawns on the cell-centre
    # plateau and walks towards +x; the staircase hugs the +x half of that cell,
    # so the camera sits off to the -y side and looks across the steps.
    terrain_row = int(options["terrain_row"])
    origin_x = _terrain_origin(env_cfg, terrain_row)[0]
    origin_y = 0.5 * float(env_cfg.terrain.terrain_width)
    env_cfg.viewer.ref_env = 0
    env_cfg.viewer.pos = [origin_x + 0.6, origin_y - 4.0, 2.6]
    env_cfg.viewer.lookat = [origin_x + 2.4, origin_y, 0.9]


def _set_fixed_command(env, command_x):
    env.commands[:, 0] = command_x
    env.commands[:, 1] = 0.0
    env.commands[:, 2] = 0.0
    if env.commands.shape[1] > 3:
        env.commands[:, 3] = 0.0


def _synchronize_observation_history(env):
    """Restart the 6-frame history from the current state.

    The history buffer is filled at construction time, before the checkpoint is
    loaded.  Repeating the current one-step observation avoids pairing the
    first policy call with stale pre-load frames.
    """
    env.compute_observations()
    one_step = int(env.num_one_step_obs)
    history_steps = int(env.obs_buf.shape[1] // one_step)
    env.obs_buf.copy_(env.obs_buf[:, :one_step].clone().repeat(1, history_steps))


def _status_line(env, step_index, descents):
    admittance = env.admittance
    position = env.root_states[0, :3].detach().cpu().numpy()
    velocity_x = float(env.base_lin_vel[0, 0].item())
    alpha = float(admittance.alpha[0].mean().item())
    beta = float(admittance.effective_alpha[0].mean().item())
    compression_mm = float(admittance.delta_l[0].mean().item()) * 1000.0
    force_n = float(admittance.estimated_force[0].mean().item())
    probability = float(admittance.impact_probability[0].mean().item())
    return (
        f"t={step_index * env.dt:6.2f}s | "
        f"pos=({position[0]:5.2f},{position[1]:5.2f},{position[2]:5.2f}) | "
        f"vx={velocity_x:5.2f} | "
        f"alpha={alpha:5.3f} beta={beta:5.3f} | "
        f"comp={compression_mm:5.2f}mm | "
        f"F_hat={force_n:6.1f}N | "
        f"p_impact={probability:4.2f} | "
        f"descents={descents}"
    )


def main():
    options = _extract_custom_args()

    args = get_args()
    args.task = TASK_NAME
    args.num_envs = 1  # one cell, one robot
    if not args.headless:
        print("[play] press ESC in the viewer (or Ctrl-C here) to quit\n")

    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK_NAME)
    _configure_single_stair_scene(env_cfg, options)

    env, env_cfg = task_registry.make_env(
        name=TASK_NAME, args=args, env_cfg=env_cfg
    )
    print(
        f"[play] task={TASK_NAME} num_envs={env.num_envs} "
        f"num_actions={env.num_actions} num_policy_actions={env.num_policy_actions} "
        f"controller_state={env.controller_state_dim} "
        f"contact_estimate={env.contact_estimate_dim}"
    )
    riser = _riser_height(options["terrain_row"], options["num_rows"])
    print(
        f"[play] terrain {options['num_rows']} x 1 cells of "
        f"{env.cfg.terrain.terrain_length:.0f} m x "
        f"{env.cfg.terrain.terrain_width:.0f} m, staircase row "
        f"{options['terrain_row']} (riser {riser:.3f} m, "
        f"tread {TERRAIN_TREAD_M:.2f} m, platform {TERRAIN_PLATFORM_M:.1f} m)"
    )

    # Freeze the curriculum and place the single robot on the chosen row.
    # ``terrain_origins`` already accounts for the staircase base height, so the
    # robot spawns on the top plateau.  This mirrors
    # evaluate_mc_quiet.py::_place_on_fixed_terrain.
    env.cfg.terrain.curriculum = False
    terrain_row = int(options["terrain_row"])
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.terrain_levels[:] = terrain_row
    env.terrain_types[:] = 0
    env.env_origins[:] = env.terrain_origins[terrain_row, 0]
    env.reset_idx(env_ids)

    # Same construction order as evaluate_mc_impact_quiet.py: build the runner
    # without let it resolve a checkpoint, then load the explicit path.  No
    # Stage-0 warm-up or PPO update is executed here.
    train_cfg.runner.resume = False
    train_cfg.runner.wandb_enabled = False
    runner, _ = task_registry.make_alg_runner(
        env=env, name=TASK_NAME, args=args, train_cfg=train_cfg, log_root=None
    )

    checkpoint_path = os.path.abspath(options["checkpoint_path"])
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Pass --checkpoint_path=<model_*.pt> pointing at an adaptive run."
        )
    runner.load(checkpoint_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    print(f"[play] loaded checkpoint: {checkpoint_path}")

    # Clean start: reset the single robot on the top plateau and rebuild the
    # 6-frame observation history from that state.
    env.reset_idx(env_ids)
    _synchronize_observation_history(env)
    observations = env.get_observations()

    # The staircase ends one tread before the cell edge; past it the robot is on
    # the flat border at ground level.  This only drives the demo restart.
    stair_exit_x = _terrain_origin(env.cfg, terrain_row)[0] + 0.5 * float(
        env.cfg.terrain.terrain_length
    ) - TERRAIN_TREAD_M

    command_x = float(options["command_x"])
    loop_reset = bool(options["loop_reset"])
    max_steps = int(options["steps"])  # 0 = run until the viewer is closed
    print_every = max(1, int(options["print_every"]))

    step_index = 0
    descents = 0
    falls = 0
    try:
        while max_steps == 0 or step_index < max_steps:
            _set_fixed_command(env, command_x)
            with torch.no_grad():
                controller_state = env.get_controller_state().to(env.device)
                actions = policy(observations.detach(), controller_state)
                contact_estimate = runner.alg.actor_critic.last_contact_estimate
                observations, _, _, dones, _, _, _ = env.step(
                    actions.detach(), contact_estimate.detach()
                )

            if bool(dones.any()):
                # The environment already reset terminated robots internally;
                # just make the event visible in the console log.
                falls += 1
                print(f"[play] step {step_index}: robot fell and was reset")

            if loop_reset and float(env.root_states[0, 0].item()) > stair_exit_x:
                env.reset_idx(env_ids)
                _synchronize_observation_history(env)
                observations = env.get_observations()
                descents += 1
                print(
                    f"[play] descent {descents} finished "
                    f"(falls so far: {falls}); restarting at the top"
                )

            step_index += 1
            if step_index % print_every == 0:
                print(_status_line(env, step_index, descents))
    except KeyboardInterrupt:
        print("\n[play] interrupted by user")

    print("[play] done")


if __name__ == "__main__":
    main()
