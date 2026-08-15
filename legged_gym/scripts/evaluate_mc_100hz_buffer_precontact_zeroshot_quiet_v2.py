"""Zero-shot v2 quiet evaluation for the pre-contact 100 Hz buffer variant."""

import isaacgym  # noqa: F401

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry
import legged_gym.scripts.evaluate_mc_quiet_v2 as base_eval

TASK = "quiet_mc_100hz_buffer_precontact_zeroshot_v2"

task_registry.task_classes["quiet_mc_v2"] = task_registry.task_classes[TASK]
task_registry.env_cfgs["quiet_mc_v2"] = task_registry.env_cfgs[TASK]
task_registry.train_cfgs["quiet_mc_v2"] = task_registry.train_cfgs[TASK]

_original_save_results = base_eval._save_results


def _save_results(summary, traces, event_samples, progress_arrays,
                  scenario, sample_rate_hz):
    summary["task"] = TASK
    summary["baseline"] = "MC_HIM_100Hz_Minimal_buffer_precontact_zeroshot"
    summary["controller"] = "precontact_damped_hip_knee_gain_schedule"
    summary["policy_source"] = "MC100HzMinimal"
    summary["buffer_loading_rate_threshold_nps"] = 30000.0
    summary["buffer_contact_on_threshold_n"] = 5.0
    summary["buffer_hold_time_s"] = 0.020
    summary["buffer_hip_knee_kp_scale"] = 0.75
    summary["buffer_hip_knee_kd_scale"] = 2.00
    summary["buffer_trigger_on_new_contact"] = False
    summary["buffer_precontact_enabled"] = True
    summary["buffer_precontact_downward_speed_threshold_mps"] = 0.25
    return _original_save_results(
        summary, traces, event_samples, progress_arrays, scenario, sample_rate_hz
    )


base_eval._save_results = _save_results


if __name__ == "__main__":
    options = base_eval._extract_custom_args()
    args = get_args()
    base_eval.evaluate(args, options)
