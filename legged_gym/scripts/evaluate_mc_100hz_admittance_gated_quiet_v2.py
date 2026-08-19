"""Quiet evaluation of policies trained with A0.1 gated axial admittance."""

import isaacgym  # noqa: F401; Isaac Gym must be imported before torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry
from legged_gym.scripts.evaluate_mc_quiet_v2 import _extract_custom_args
from legged_gym.scripts.evaluate_mc_100hz_quiet_v2 import evaluate


# Reuse the shared 100 Hz protocol while loading checkpoints from the dedicated
# MC100HzAdmittanceGated experiment.
task_registry.task_classes["quiet_mc_100hz_v2"] = task_registry.task_classes[
    "quiet_mc_100hz_admittance_gated_v2"
]
task_registry.env_cfgs["quiet_mc_100hz_v2"] = task_registry.env_cfgs[
    "quiet_mc_100hz_admittance_gated_v2"
]
task_registry.train_cfgs["quiet_mc_100hz_v2"] = task_registry.train_cfgs[
    "quiet_mc_100hz_admittance_gated_v2"
]


if __name__ == "__main__":
    options = _extract_custom_args()
    args = get_args()
    evaluate(args, options)
