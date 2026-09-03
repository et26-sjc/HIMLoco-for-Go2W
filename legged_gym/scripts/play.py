# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args, x_vel=1.0, y_vel=0.0, yaw_vel=0.0):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 8
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = 9
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.commands.heading_command = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.commands[:, 0] = x_vel
    env.commands[:, 1] = y_vel
    env.commands[:, 2] = yaw_vel

    obs = env.get_observations()
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    adaptive = hasattr(env, "get_controller_state") and hasattr(
        ppo_runner.alg.actor_critic, "contact_estimator"
    )

    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            'logs',
            train_cfg.runner.experiment_name,
            'exported',
            'policies',
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0
    joint_index = 1
    stop_state_log = 100
    stop_rew_log = env.max_episode_length + 1
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    for i in range(10 * int(env.max_episode_length)):
        if adaptive:
            controller_state = env.get_controller_state().detach()
            actions = policy(obs.detach(), controller_state)
            contact_estimate = ppo_runner.alg.actor_critic.last_contact_estimate
        else:
            actions = policy(obs.detach())
            contact_estimate = None

        env.commands[:, 0] = x_vel
        env.commands[:, 1] = y_vel
        env.commands[:, 2] = yaw_vel
        if adaptive:
            obs, _, rews, dones, infos, _, _ = env.step(
                actions.detach(), contact_estimate.detach()
            )
        else:
            obs, _, rews, dones, infos, _, _ = env.step(actions.detach())

        if RECORD_FRAMES and i % 2:
            filename = os.path.join(
                LEGGED_GYM_ROOT_DIR,
                'logs',
                train_cfg.runner.experiment_name,
                'exported',
                'frames',
                f"{img_idx}.png",
            )
            env.gym.write_viewer_image_to_file(env.viewer, filename)
            img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            states = {
                'dof_pos_target': actions[robot_index, joint_index].item()
                * env.cfg.control.action_scale
                + env.default_dof_pos[robot_index, joint_index].item(),
                'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                'dof_torque': env.torques[robot_index, joint_index].item(),
                'command_x': env.commands[robot_index, 0].item(),
                'command_y': env.commands[robot_index, 1].item(),
                'command_yaw': env.commands[robot_index, 2].item(),
                'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                'contact_forces_z': env.contact_forces[
                    robot_index, env.feet_indices, 2
                ].cpu().numpy(),
            }
            if adaptive:
                states.update(
                    {
                        'compliance_mean': env.admittance.alpha[
                            robot_index
                        ].mean().item(),
                        'admittance_compression_mean': env.admittance.delta_l[
                            robot_index
                        ].mean().item(),
                        'estimated_force_mean': env.admittance.estimated_force[
                            robot_index
                        ].mean().item(),
                    }
                )
            logger.log_states(states)
        elif i == stop_state_log:
            logger.plot_states()

        if 0 < i < stop_rew_log and infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes > 0:
                logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args, x_vel=1.0, y_vel=0.0, yaw_vel=0.0)
