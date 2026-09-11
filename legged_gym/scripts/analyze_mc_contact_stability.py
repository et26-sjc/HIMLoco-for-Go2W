"""Summarize paired ContactEstimator stability TensorBoard runs as JSON."""

import argparse
import json

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


WINDOWS = ((0, 19), (20, 39), (40, 59), (60, 79), (80, 99))
ONLINE_TAGS = (
    "Estimator/impact_precision",
    "Estimator/impact_recall",
    "Estimator/impact_f1",
    "Estimator/raw_logit_separation",
    "Estimator/raw_probability_separation",
    "Estimator/impact_gt_ratio",
    "Estimator/impact_pred_ratio",
    "Admittance/compression_after_impact_mm",
    "Admittance/compression_after_noimpact_mm",
    "Admittance/compression_p95_mm",
    "Admittance/compression_max_mm",
    "Admittance/compression_saturation_ratio",
    "Impact/gt_3d_force_peak_mean_n",
    "Impact/gt_3d_loading_peak_mean_nps",
    "Impact/gt_base_acc_peak_mean_mps2",
    "Episode/rew_tracking_lin_vel",
    "Episode/rew_tracking_ang_vel",
    "Episode/rew_base_height",
    "Train/mean_reward",
)


def _load_scalars(log_dir):
    accumulator = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    accumulator.Reload()
    return {
        tag: {event.step: event.value for event in accumulator.Scalars(tag)}
        for tag in accumulator.Tags()["scalars"]
    }


def _window_mean(values, start, stop):
    selected = [value for step, value in values.items() if start <= step <= stop]
    return sum(selected) / len(selected) if selected else None


def summarize(log_dir):
    scalars = _load_scalars(log_dir)
    result = {"log_dir": log_dir, "windows": {}, "validation": {}}
    for start, stop in WINDOWS:
        row = {
            tag: _window_mean(scalars.get(tag, {}), start, stop)
            for tag in ONLINE_TAGS
        }
        impact = row["Admittance/compression_after_impact_mm"]
        no_impact = row["Admittance/compression_after_noimpact_mm"]
        row["Admittance/compression_selectivity_ratio"] = (
            impact / (no_impact + 1.0e-8)
            if impact is not None and no_impact is not None
            else None
        )
        result["windows"][f"{start}-{stop}"] = row
    result["validation"] = {
        tag: values
        for tag, values in scalars.items()
        if tag.startswith("Validation/")
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--continual", required=True)
    parser.add_argument("--frozen", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "continual": summarize(args.continual),
                "frozen": summarize(args.frozen),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
