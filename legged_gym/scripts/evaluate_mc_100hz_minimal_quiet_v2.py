"""Evaluate minimal MC-HIM-100Hz checkpoints with quiet protocol v2.

This wrapper reuses the validated 100 Hz v2 evaluator while redirecting its
registered evaluation task to the minimal 100 Hz configuration (6-frame HIM
history / 342 estimator inputs).  This prevents accidentally constructing the
12-frame/684-input model used by the earlier time-matched experiment.
"""

import isaacgym  # noqa: F401; Isaac Gym must be imported before torch

from legged_gym.envs import *  # noqa: F401,F403; registers both eval tasks
from legged_gym.utils import get_args, task_registry
from legged_gym.scripts.evaluate_mc_quiet_v2 import _extract_custom_args
from legged_gym.scripts.evaluate_mc_100hz_quiet_v2 import evaluate


# The shared evaluator currently uses the generic ``quiet_mc_100hz_v2`` task
# name internally.  Redirect that registry entry to the minimal-frequency
# evaluation config before constructing the environment/runner.  The registry
# is a process-local singleton, so this affects only this evaluation process.
task_registry.task_classes["quiet_mc_100hz_v2"] = task_registry.task_classes[
    "quiet_mc_100hz_minimal_v2"
]
task_registry.env_cfgs["quiet_mc_100hz_v2"] = task_registry.env_cfgs[
    "quiet_mc_100hz_minimal_v2"
]
task_registry.train_cfgs["quiet_mc_100hz_v2"] = task_registry.train_cfgs[
    "quiet_mc_100hz_minimal_v2"
]


if __name__ == "__main__":
    # Custom evaluation flags (e.g. --eval_scenario) are not known to the
    # Isaac Gym argparse helper.  Remove them from sys.argv first, then parse
    # the standard legged-gym arguments.
    options = _extract_custom_args()
    args = get_args()
    evaluate(args, options)
