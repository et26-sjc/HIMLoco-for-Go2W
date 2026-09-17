"""Analyze offline unloading-authority counterfactuals without simulator access."""

import argparse
import json
from pathlib import Path

import numpy as np

from legged_gym.scripts.analyze_mc_prerelease_kinematics import _episodes


def _bands(acc, fs=200):
    demeaned = acc - acc.mean(axis=0, keepdims=True)
    spectrum = np.fft.rfft(demeaned, axis=0)
    freqs = np.fft.rfftfreq(len(acc), 1.0 / fs)
    bands = {}
    for low, high in ((5, 20), (20, 50), (50, 100)):
        filtered = np.fft.irfft(
            spectrum * ((freqs >= low) & (freqs < high))[:, None],
            n=len(acc), axis=0,
        )
        bands[f"{low}-{high}hz_rms_mps2"] = float(
            np.sqrt(np.mean(filtered * filtered, axis=0)).mean()
        )
    return bands


def measure(summary_path, schedule=None):
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text())
    trace_path = summary_path.parent / f"raw_trace_{summary['sample_rate_hz']}hz.npz"
    labels, features, trajectories, episodes = _episodes(trace_path, True)
    with np.load(trace_path) as archive:
        contact = archive["contact"].astype(bool)
        acc = archive["base_acc_z_mps2"]
        authority = archive["late_authority_scale"] if "late_authority_scale" in archive else np.ones_like(contact, dtype=np.float32)
        compression = archive["admittance_compression_m"]
        residual = archive["baseline_torque_residual_nm"]
        force = archive["force_norm_n"]
        loading = archive["loading_rate_norm_nps"]
        work = archive["compliance_power_w"]
    events = contact[1:] & ~contact[:-1]
    any_release = contact[:-1] & ~contact[1:]
    # release[t] = contact[t] & ~contact[t+1], touchdown[t+g] =
    # contact[t+g+1] & ~contact[t+g], with g=1..4 representing <=20 ms.
    comeback = np.zeros_like(any_release)
    for gap in range(1, 5):
        valid = events[gap:]
        comeback[:len(valid)] |= valid
    rec_20ms = int(np.count_nonzero(any_release & comeback))
    out = {
        "summary_path": str(summary_path.resolve()),
        "raw_touchdowns": int(events.sum()),
        "release_count": int(any_release.sum()),
        "recontacts_within_20ms": rec_20ms,
        "recontact_per_release": rec_20ms / max(1, int(any_release.sum())),
        "filtered_short_recontact_count": int(labels.sum()),
        "filtered_short_recontact_fraction": float(labels.mean()) if len(labels) else None,
        "filtered_event_count": int(len(labels)),
        "release_vz_short_mps": float(features["wheel_vz_release_mps"][labels].mean()) if labels.any() else None,
        "release_vz_stable_mps": float(features["wheel_vz_release_mps"][~labels].mean()) if (~labels).any() else None,
        "release_leg_length_short_m": float(features["leg_length_release_m"][labels].mean()) if labels.any() else None,
        "release_leg_length_stable_m": float(features["leg_length_release_m"][~labels].mean()) if (~labels).any() else None,
        "pre_release_leg_length_short_trajectory_m": trajectories["leg_length_m"][labels].mean(axis=0).tolist() if labels.any() else None,
        "pre_release_leg_length_stable_trajectory_m": trajectories["leg_length_m"][~labels].mean(axis=0).tolist() if (~labels).any() else None,
        "work_short_mean_j": float(features["compliance_work_net_j"][labels].mean()) if labels.any() else None,
        "work_stable_mean_j": float(features["compliance_work_net_j"][~labels].mean()) if (~labels).any() else None,
        "compliance_net_work_total_j": float(work.sum() / 200.),
        "release_compression_short_mm": float(features["admittance_compression_release_mm"][labels].mean()) if labels.any() else None,
        "residual_abs_mean_nm": float(np.abs(residual).mean()),
        "mean_compression_mm": float(compression.mean() * 1000.),
        "force_p95_n": float(summary["peak_contact_force_norm_n_p95"]),
        "force_p99_n": float(summary["peak_contact_force_norm_n_p99"]),
        "loading_p95_nps": float(summary["peak_contact_loading_rate_norm_nps_p95"]),
        "loading_p99_nps": float(summary["peak_contact_loading_rate_norm_nps_p99"]),
        "force_peak_while_attenuated_p95_n": float(np.percentile(force[authority < 1.], 95)) if (authority < 1.).any() else None,
        "loading_peak_while_attenuated_p95_nps": float(np.percentile(loading[authority < 1.], 95)) if (authority < 1.).any() else None,
        "base_acc_rms_mps2": float(np.sqrt(np.mean(acc * acc))),
        "base_acc_p95_abs_mps2": float(np.percentile(np.abs(acc), 95)),
        "mean_abs_tracking_error_x_mps": float(summary["mean_abs_tracking_error_x_mps"]),
        "mean_tracking_lin_reward": float(summary["mean_tracking_lin_reward"]),
        "reset_count": int(summary["reset_count"]),
        "fraction_of_samples_attenuated": float((authority < 1.).mean()),
    }
    out.update(_bands(acc))
    if schedule is not None:
        with np.load(schedule) as archive:
            reference_contact = archive["reference_contact"].astype(bool)
            if reference_contact.shape != contact.shape:
                raise ValueError("Reference GT contact and branch contact shape mismatch")
            eligible = archive["eligible"].astype(bool)
        active = authority < 1.
        out.update({
            "contact_agreement_fraction": float((contact == reference_contact).mean()),
            "active_when_branch_contact_false_fraction": float((active & ~contact).sum() / max(1, active.sum())),
            "active_reference_window_branch_contact_fraction": float((eligible & contact).sum() / max(1, eligible.sum())),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for mode in ("actual", "half", "decay"):
        parser.add_argument("--" + mode, required=True, help="Evaluation summary.json")
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {mode: measure(getattr(args, mode), args.schedule) for mode in ("actual", "half", "decay")}
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2, sort_keys=True))
    for name, metrics in result.items():
        print(name, {key: metrics[key] for key in (
            "recontacts_within_20ms", "raw_touchdowns", "release_vz_short_mps",
            "force_p95_n", "loading_p95_nps", "base_acc_rms_mps2",
            "5-20hz_rms_mps2", "20-50hz_rms_mps2", "50-100hz_rms_mps2",
            "contact_agreement_fraction", "active_when_branch_contact_false_fraction",
        )})
    print(dest.resolve())


if __name__ == "__main__":
    main()
