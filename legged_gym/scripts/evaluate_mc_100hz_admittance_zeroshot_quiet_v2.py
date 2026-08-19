"""Zero-shot quiet evaluation of axial admittance on MC100HzMinimal weights.

The policy checkpoint is loaded from ``logs/MC100HzMinimal``.  Only the low-level
controller is changed from fixed PD target tracking to the axial-admittance outer
loop plus the same fixed PD inner loop.
"""

import isaacgym  # noqa: F401; Isaac Gym must be imported before torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry
from legged_gym.scripts.evaluate_mc_quiet_v2 import _extract_custom_args
from legged_gym.scripts.evaluate_mc_100hz_quiet_v2 import evaluate


# Reuse the shared 100 Hz evaluator without duplicating the evaluation protocol.
task_registry.task_classes["quiet_mc_100hz_v2"] = task_registry.task_classes[
    "quiet_mc_100hz_admittance_zeroshot_v2"
]
task_registry.env_cfgs["quiet_mc_100hz_v2"] = task_registry.env_cfgs[
    "quiet_mc_100hz_admittance_zeroshot_v2"
]
task_registry.train_cfgs["quiet_mc_100hz_v2"] = task_registry.train_cfgs[
    "quiet_mc_100hz_admittance_zeroshot_v2"
]


if __name__ == "__main__":
    options = _extract_custom_args()
    args = get_args()
    evaluate(args, options)
