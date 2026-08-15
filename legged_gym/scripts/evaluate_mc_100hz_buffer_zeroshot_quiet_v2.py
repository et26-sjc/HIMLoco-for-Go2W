"""Zero-shot quiet evaluation for the contact-buffered 100 Hz Minimal policy.

Loads an existing MC100HzMinimal checkpoint (six-frame HIM history / 342
estimator inputs) and evaluates the same policy with the contact-aware HIP/KNEE
gain schedule enabled.  No retraining is performed, so comparison against the
fixed-PD evaluator isolates the controller effect.
"""

import isaacgym  # noqa: F401; Isaac Gym must be imported before torch

from legged_gym.envs import *  # noqa: F401,F403; registers eval tasks
from legged_gym.utils import get_args, task_registry
import legged_gym.scripts.evaluate_mc_quiet_v2 as base_eval


# Reuse the validated v2 evaluator, but redirect its process-local generic task
# entry to the buffered zero-shot environment/config.  base_eval.evaluate()
# internally requests "quiet_mc_v2", so replacing these three registry entries
# is sufficient and does not affect any other process.
task_registry.task_classes["quiet_mc_v2"] = task_registry.task_classes[
    "quiet_mc_100hz_buffer_zeroshot_v2"
]
task_registry.env_cfgs["quiet_mc_v2"] = task_registry.env_cfgs[
    "quiet_mc_100hz_buffer_zeroshot_v2"
]
task_registry.train_cfgs["quiet_mc_v2"] = task_registry.train_cfgs[
    "quiet_mc_100hz_buffer_zeroshot_v2"
]


# The shared evaluator labels its default controller as fixed PD.  Patch only
# the metadata immediately before files are written; all metric computation and
# the deterministic v2 protocol remain unchanged.
_original_save_results = base_eval._save_results


def _save_buffer_results(summary, traces, event_samples, progress_arrays,
                         scenario, sample_rate_hz):
    summary["task"] = "quiet_mc_100hz_buffer_zeroshot_v2"
    summary["baseline"] = "MC_HIM_100Hz_Minimal_contact_buffer_zeroshot"
    summary["controller"] = "contact_aware_hip_knee_gain_schedule"
    summary["policy_source"] = "MC100HzMinimal"
    summary["buffer_loading_rate_threshold_nps"] = 20000.0
    summary["buffer_contact_on_threshold_n"] = 5.0
    summary["buffer_hold_time_s"] = 0.030
    summary["buffer_hip_knee_kp_scale"] = 0.60
    summary["buffer_hip_knee_kd_scale"] = 1.50
    return _original_save_results(
        summary,
        traces,
        event_samples,
        progress_arrays,
        scenario,
        sample_rate_hz,
    )


base_eval._save_results = _save_buffer_results


if __name__ == "__main__":
    options = base_eval._extract_custom_args()
    args = get_args()
    base_eval.evaluate(args, options)
