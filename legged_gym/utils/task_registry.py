# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner, HIMOnPolicyRunner, AdaptiveHIMOnPolicyRunner

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]

    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg

    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        if args is None:
            args = get_args()
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            env_cfg, _ = self.get_cfgs(name)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(
            cfg=env_cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            sim_device=args.sim_device,
            headless=args.headless,
        )
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        if args is None:
            args = get_args()
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            _, train_cfg = self.get_cfgs(name)
        elif name is not None:
            print(f"'train_cfg' provided -> Ignoring 'name={name}'")

        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        if log_root == "default":
            log_root = os.path.join(
                LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name
            )
            log_dir = os.path.join(
                log_root,
                datetime.now().strftime('%b%d_%H-%M-%S')
                + '_'
                + train_cfg.runner.run_name,
            )
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(
                log_root,
                datetime.now().strftime('%b%d_%H-%M-%S')
                + '_'
                + train_cfg.runner.run_name,
            )

        train_cfg_dict = class_to_dict(train_cfg)
        runner_name = getattr(
            train_cfg.runner, 'runner_class_name', 'HIMOnPolicyRunner'
        )
        runner_classes = {
            'OnPolicyRunner': OnPolicyRunner,
            'HIMOnPolicyRunner': HIMOnPolicyRunner,
            'AdaptiveHIMOnPolicyRunner': AdaptiveHIMOnPolicyRunner,
        }
        if runner_name not in runner_classes:
            raise ValueError(
                f"Unknown runner_class_name={runner_name}. Available: "
                f"{list(runner_classes.keys())}"
            )
        runner = runner_classes[runner_name](
            env, train_cfg_dict, log_dir, device=args.rl_device
        )

        resume = train_cfg.runner.resume
        if resume:
            resume_path = get_load_path(
                log_root,
                load_run=train_cfg.runner.load_run,
                checkpoint=train_cfg.runner.checkpoint,
            )
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
        else:
            # Adaptive tasks can start from a separately trained locomotion
            # baseline without reusing that run's optimizer or iteration count.
            init_experiment_name = getattr(
                train_cfg.runner, 'init_experiment_name', None
            )
            if init_experiment_name:
                init_root = os.path.join(
                    LEGGED_GYM_ROOT_DIR, 'logs', init_experiment_name
                )
                init_path = get_load_path(
                    init_root,
                    load_run=getattr(train_cfg.runner, 'init_load_run', -1),
                    checkpoint=getattr(train_cfg.runner, 'init_checkpoint', -1),
                )
                print(f"Initializing adaptive policy from baseline: {init_path}")
                runner.load(init_path, load_optimizer=False)
                # This is a new adaptive experiment, not a continuation of the
                # baseline iteration counter.
                runner.current_learning_iteration = 0

        return runner, train_cfg


# make global task registry
task_registry = TaskRegistry()
