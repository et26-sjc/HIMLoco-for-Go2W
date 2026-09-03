# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil
import torch.nn.functional as F

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_sim_params(args, cfg):
    sim_params = gymapi.SimParams()
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads
    return sim_params


def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        runs.sort()
        if 'exported' in runs:
            runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except Exception:
        raise ValueError("No runs in this directory: " + root)
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)
    return os.path.join(load_run, model)


def update_cfg_from_args(env_cfg, cfg_train, args):
    if env_cfg is not None:
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        if args.seed is not None:
            env_cfg.seed = args.seed
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint
    return env_cfg, cfg_train


def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "aliengo", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False, "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str, "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str, "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str, "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int, "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
    ]
    args = gymutil.parse_arguments(
        description="RL Policy", custom_parameters=custom_parameters
    )
    args.sim_device = args.rl_device
    return args


def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'contact_estimator'):
        exporter = PolicyExporterAdaptiveHIM(actor_critic)
        exporter.export(path)
    elif hasattr(actor_critic, 'estimator'):
        exporter = PolicyExporterHIM(actor_critic)
        exporter.export(path)
    else:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class PolicyExporterHIM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder)

    def forward(self, obs_history):
        parts = self.estimator(obs_history).squeeze(0)[0:19]
        vel, z = parts[:3], parts[3:]
        z = F.normalize(z, dim=-1, p=2.0)
        obs_curr = obs_history.squeeze(0)[:57]
        actor_in = torch.cat([obs_curr, vel, z], dim=0)
        return self.actor(actor_in.unsqueeze(0)).squeeze(0)

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy.pt')
        self.to('cpu')
        torch.jit.script(self).save(path)


class PolicyExporterAdaptiveHIM(torch.nn.Module):
    """Export deployable policy inference without any privileged force input.

    Forward inputs are the original flattened 342-D HIM history and the 12-D
    internal admittance state. It returns a tuple ``(policy_action20,
    contact_estimate8)``. The low-level deployment code must feed the latter into
    the same sensorless admittance dynamics used during training.
    """

    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.him_estimator = copy.deepcopy(actor_critic.estimator.encoder)
        self.contact_estimator = copy.deepcopy(
            actor_critic.contact_estimator.encoder
        )
        self.motion_adapter = copy.deepcopy(actor_critic.motion_adapter)
        self.compliance_head = copy.deepcopy(actor_critic.compliance_head)
        self.num_one_step_obs = int(actor_critic.num_one_step_obs)
        self.motion_adapter_scale = float(actor_critic.motion_adapter_scale)

    def forward(self, obs_history, controller_state):
        # Deployment currently uses one robot at a time, matching the old
        # PolicyExporterHIM convention of one flattened history vector.
        obs_b = obs_history.unsqueeze(0)
        ctrl_b = controller_state.unsqueeze(0)

        parts = self.him_estimator(obs_b).squeeze(0)
        vel, z = parts[:3], parts[3:]
        z = F.normalize(z, dim=-1, p=2.0)

        contact_raw = self.contact_estimator(
            torch.cat((obs_b, ctrl_b), dim=-1)
        ).squeeze(0)
        contact = F.softplus(contact_raw)

        obs_curr = obs_history[: self.num_one_step_obs]
        baseline_input = torch.cat((obs_curr, vel, z), dim=0)
        augmented = torch.cat(
            (baseline_input, contact, controller_state), dim=0
        )

        base_motion = self.actor(baseline_input.unsqueeze(0)).squeeze(0)
        motion_delta = self.motion_adapter(augmented.unsqueeze(0)).squeeze(0)
        motion = base_motion + self.motion_adapter_scale * torch.tanh(
            motion_delta
        )
        compliance = self.compliance_head(
            augmented.unsqueeze(0)
        ).squeeze(0)
        policy_action = torch.cat((motion, compliance), dim=0)
        return policy_action, contact

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_adaptive.pt')
        self.to('cpu')
        torch.jit.script(self).save(path)
