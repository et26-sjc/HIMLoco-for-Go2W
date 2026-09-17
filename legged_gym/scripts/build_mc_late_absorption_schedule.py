"""Make a fixed, evaluation-only unloading schedule from an actual MC trace.

The reference rollout's *completed* episodes supply a future release time
solely to restrict the diagnostic to their final 40 ms. This offline marker
is NOT deployable; the branched controller never queries new-trajectory GT.
The resulting NPZ belongs in ignored logs/, never in training or deployment.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def build_schedule(contact, force, loading, compression, decay_steps=4, last_steps=8):
    if contact.shape != force.shape or force.shape != loading.shape or force.shape != compression.shape:
        raise ValueError("Contact/force/loading/compression dimensions must match")
    if decay_steps < 2:
        raise ValueError("decay_steps must be >= 2")
    if last_steps < decay_steps:
        raise ValueError("last_steps must accommodate the authority decay")
    samples, envs, legs = contact.shape
    until_release = np.zeros(contact.shape, dtype=np.int32)
    for t in range(samples - 2, -1, -1):
        until_release[t] = np.where(contact[t], 1 + until_release[t + 1], 0)
    half = np.ones((samples, envs, legs), dtype=np.float32)
    decay = np.ones_like(half)
    eligible = np.zeros_like(contact, dtype=bool)
    peak_force = np.zeros((envs, legs), dtype=np.float32)
    peak_loading = np.zeros_like(peak_force)
    peak_force_age = np.zeros((envs, legs), dtype=np.int32)
    peak_loading_age = np.zeros_like(peak_force_age)
    contact_age = np.zeros_like(peak_force_age)
    latched = np.zeros_like(peak_force, dtype=bool)
    decay_age = np.zeros_like(peak_force_age)
    # The first observed GT frame cannot alter the currently executing frame.
    for t in range(1, samples):
        previous = contact[t - 1]
        contact_age = np.where(previous, contact_age + 1, 0)
        new_peak_force = previous & (force[t - 1] > peak_force)
        new_peak_loading = previous & (loading[t - 1] > peak_loading)
        peak_force_age = np.where(previous, peak_force_age + 1, 0)
        peak_loading_age = np.where(previous, peak_loading_age + 1, 0)
        peak_force = np.where(previous, np.maximum(peak_force, force[t - 1]), 0)
        peak_loading = np.where(previous, np.maximum(peak_loading, loading[t - 1]), 0)
        peak_force_age = np.where(new_peak_force, 0, peak_force_age)
        peak_loading_age = np.where(new_peak_loading, 0, peak_loading_age)
        # Per-contact *past* peaks, significant GT impact and a still-compressed
        # actuator; falling force AND falling loading avoid attenuating onset.
        unloading = (
            previous & (contact_age >= 4) & (compression[t - 1] > 0)
            & (until_release[t - 1] <= last_steps)
            & (peak_loading >= 5000.0)
            & (peak_force_age >= 2) & (peak_loading_age >= 2)
            & (force[t - 1] <= 0.8 * peak_force)
            & (loading[t - 1] <= 0.8 * peak_loading)
        )
        started = unloading & ~latched
        latched = previous & (latched | unloading)
        decay_age = np.where(started, 0, np.where(latched, decay_age + 1, 0))
        eligible[t] = latched
        half[t] = np.where(latched, 0.5, 1.0)
        decay[t] = np.where(latched, np.maximum(0., 1. - (decay_age + 1) / decay_steps), 1.)
    return half, decay, eligible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Unmodified adaptive raw_trace_200hz.npz")
    parser.add_argument("--output", required=True, help="Ignored logs/ output .npz")
    parser.add_argument("--decay_ms", type=float, default=20.0)
    parser.add_argument("--last_contact_ms", type=float, default=40.0)
    args = parser.parse_args()
    reference = Path(args.reference).resolve()
    summary = json.loads((reference.parent / "summary.json").read_text())
    if int(summary["sample_rate_hz"]) != 200 or summary["scenario"] != "stairs_down":
        raise ValueError("Reference must be a 200 Hz down-stair adaptive run")
    if summary.get("counterfactual_mode") or summary.get("eval_zero_compliance") or summary.get("late_authority_schedule"):
        raise ValueError("Reference must be an unmodified adaptive run")
    with np.load(reference) as archive:
        contact = archive["contact"].astype(bool)
        half, decay, mask = build_schedule(
            contact, archive["force_norm_n"], archive["loading_rate_norm_nps"],
            archive["admittance_compression_m"],
            decay_steps=int(round(args.decay_ms * 0.2)),
            last_steps=int(round(args.last_contact_ms * 0.2)),
        )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, half=half, decay=decay, eligible=mask,
        reference_contact=contact, eval_seed=int(summary["eval_seed"]),
        command_x_mps=float(summary["command_x_mps"]),
        stair_difficulty=float(summary["terrain_difficulty"]),
        reference_checkpoint=str(summary["checkpoint_path"]),
        reference_trace=str(reference),
    )
    print(json.dumps({
        "output": str(output), "qualified_samples": int(mask.sum()),
        "fraction_of_contact_samples": float(mask.sum() / max(contact.sum(), 1)),
        "half_torque_samples": int((half < 1).sum()),
        "decay_ms": args.decay_ms,
        "last_contact_ms": args.last_contact_ms,
    }, indent=2))


if __name__ == "__main__":
    main()
