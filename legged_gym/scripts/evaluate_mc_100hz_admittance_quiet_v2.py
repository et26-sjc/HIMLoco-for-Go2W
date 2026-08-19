"""Quiet evaluation for policies trained with the 100 Hz axial admittance task."""

import isaacgym  # noqa: F401

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry
from legged_gym.scripts.evaluate_mc_quiet_v2 import _extract_custom_args
from legged_gym.scripts.evaluate_mc_100hz_quiet_v2 import evaluate


task_registry.task_classes["quiet_mc_100hz_v2"] = task_registry.task_classes[
    "quiet_mc_100hz_admittance_v2"
]
task_registry.env_cfgs["quiet_mc_100hz_v2"] = task_registry.env_cfgs[
    "quiet_mc_100hz_admittance_v2"
]
task_registry.train_cfgs["quiet_mc_100hz_v2"] = task_registry.train_cfgs[
    "quiet_mc_100hz_admittance_v2"
]


if __name__ == "__main__":
    options = _extract_custom_args()
    args = get_args()
    evaluate(args, options)
