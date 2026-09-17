"""Offline pre-release kinematics analysis for paired MC quiet traces.

GT contact is used only to label completed contact episodes and whether release
is followed by a touchdown within 20 ms. No simulator, policy, or optimizer is
created by this script.
"""

import argparse
import json
from pathlib import Path

import numpy as np


DT = 0.005
SHORT_STEPS = 4
PHASES = np.linspace(0.0, 1.0, 5)
LEG_DOF = ((1, 2), (5, 6), (9, 10), (13, 14))  # quiet order FR, FL, RR, RL


def _rank_auc(values, labels):
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    pos, neg = int(labels.sum()), int((~labels).sum())
    if not pos or not neg:
        return 0.5
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    auc = (ranks[labels].sum() - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def _effect(stable, short):
    stable = np.asarray(stable, dtype=np.float64)
    short = np.asarray(short, dtype=np.float64)
    pooled = np.sqrt(0.5 * (stable.var() + short.var()))
    return float((short.mean() - stable.mean()) / max(pooled, 1.0e-12))


def _interp(values, start, end):
    indices = np.rint(start + PHASES * (end - start)).astype(int)
    return values[indices]


def _leg_torque_magnitude(torque_residual):
    return np.stack(
        [np.max(np.abs(torque_residual[:, :, dofs]), axis=-1) for dofs in LEG_DOF],
        axis=-1,
    )


def _episodes(trace_path, adaptive):
    # NPZ access is lazy and decompresses on every lookup. Materialize once so
    # the per-event loop remains linear in the number of contact episodes.
    with np.load(trace_path) as archive:
        trace = {name: archive[name] for name in archive.files}
    contact = trace["contact"].astype(bool)
    touchdown = contact.copy()
    touchdown[0] = False
    touchdown[1:] &= ~contact[:-1]
    release = np.zeros_like(contact)
    release[1:] = contact[:-1] & ~contact[1:]

    force = trace["force_norm_n"]
    total_force = np.maximum(force.sum(axis=2), 1.0e-6)
    load_fraction = force / total_force[:, :, None]
    torque_mag = (
        _leg_torque_magnitude(trace["baseline_torque_residual_nm"])
        if adaptive else np.zeros_like(force)
    )
    power = trace["compliance_power_w"] if adaptive else np.zeros_like(force)
    adm_comp = (
        trace["admittance_compression_m"] if adaptive else np.zeros_like(force)
    )
    adm_vel = (
        trace["admittance_compression_velocity_mps"]
        if adaptive else np.zeros_like(force)
    )

    feature_names = [
        "contact_duration_ms",
        "wheel_dz_impact_to_release_mm",
        "wheel_vz_impact_mps",
        "wheel_vz_release_mps",
        "wheel_vz_last20ms_mean_mps",
        "leg_length_impact_m",
        "leg_length_release_m",
        "leg_length_change_mm",
        "physical_compression_max_mm",
        "physical_compression_release_mm",
        "load_fraction_mean",
        "load_fraction_release",
        "admittance_compression_max_mm",
        "admittance_compression_release_mm",
        "admittance_velocity_positive_max_mps",
        "admittance_velocity_release_mps",
        "torque_residual_mean_nm",
        "torque_residual_release_nm",
        "torque_residual_max_nm",
        "compliance_work_net_j",
        "compliance_work_positive_j",
        "compliance_work_abs_j",
    ]
    features = {name: [] for name in feature_names}
    labels = []
    trajectories = {
        key: [] for key in (
            "wheel_pos_z_base_m", "wheel_vel_z_mps", "leg_length_m",
            "leg_compression_m", "force_norm_n", "load_fraction",
            "admittance_compression_m",
            "admittance_compression_velocity_mps",
            "torque_residual_nm", "compliance_power_w",
        )
    }

    time_steps, num_envs, num_legs = contact.shape
    paired_episode_count = 0
    single_sample_episode_count = 0
    boundary_excluded_count = 0
    for env in range(num_envs):
        for leg in range(num_legs):
            starts = np.flatnonzero(touchdown[:, env, leg])
            ends = np.flatnonzero(release[:, env, leg])
            for start in starts:
                end_index = int(np.searchsorted(ends, start + 1))
                if end_index >= ends.size:
                    boundary_excluded_count += 1
                    continue
                end = int(ends[end_index])  # first off-contact sample
                paired_episode_count += 1
                if end - start < 2:
                    single_sample_episode_count += 1
                    continue
                if end + SHORT_STEPS >= time_steps:
                    boundary_excluded_count += 1
                    continue
                is_short = bool(touchdown[end : end + SHORT_STEPS + 1, env, leg].any())
                impact_stop = min(end, start + 5)
                impact = start + int(
                    np.argmax(trace["loading_rate_norm_nps"][start:impact_stop, env, leg])
                )
                last = end - 1
                segment = slice(impact, end)
                last20 = slice(max(impact, end - SHORT_STEPS), end)

                wheel_z = trace["wheel_pos_z_world_m"][:, env, leg]
                wheel_vz = trace["wheel_vel_z_mps"][:, env, leg]
                length = trace["leg_length_m"][:, env, leg]
                physical_comp = trace["leg_compression_m"][:, env, leg]
                frac = load_fraction[:, env, leg]
                comp = adm_comp[:, env, leg]
                comp_vel = adm_vel[:, env, leg]
                torque = torque_mag[:, env, leg]
                leg_power = power[:, env, leg]

                values = {
                    "contact_duration_ms": (end - start) * DT * 1000.0,
                    "wheel_dz_impact_to_release_mm": (wheel_z[last] - wheel_z[impact]) * 1000.0,
                    "wheel_vz_impact_mps": wheel_vz[impact],
                    "wheel_vz_release_mps": wheel_vz[last],
                    "wheel_vz_last20ms_mean_mps": wheel_vz[last20].mean(),
                    "leg_length_impact_m": length[impact],
                    "leg_length_release_m": length[last],
                    "leg_length_change_mm": (length[last] - length[impact]) * 1000.0,
                    "physical_compression_max_mm": physical_comp[segment].max() * 1000.0,
                    "physical_compression_release_mm": physical_comp[last] * 1000.0,
                    "load_fraction_mean": frac[segment].mean(),
                    "load_fraction_release": frac[last],
                    "admittance_compression_max_mm": comp[segment].max() * 1000.0,
                    "admittance_compression_release_mm": comp[last] * 1000.0,
                    "admittance_velocity_positive_max_mps": max(0.0, float(comp_vel[segment].max())),
                    "admittance_velocity_release_mps": comp_vel[last],
                    "torque_residual_mean_nm": torque[segment].mean(),
                    "torque_residual_release_nm": torque[last],
                    "torque_residual_max_nm": torque[segment].max(),
                    "compliance_work_net_j": leg_power[segment].sum() * DT,
                    "compliance_work_positive_j": np.maximum(leg_power[segment], 0.0).sum() * DT,
                    "compliance_work_abs_j": np.abs(leg_power[segment]).sum() * DT,
                }
                for name, value in values.items():
                    features[name].append(float(value))
                labels.append(is_short)

                phase_data = {
                    "wheel_pos_z_base_m": trace["wheel_pos_z_base_m"][:, env, leg],
                    "wheel_vel_z_mps": wheel_vz,
                    "leg_length_m": length,
                    "leg_compression_m": physical_comp,
                    "force_norm_n": force[:, env, leg],
                    "load_fraction": frac,
                    "admittance_compression_m": comp,
                    "admittance_compression_velocity_mps": comp_vel,
                    "torque_residual_nm": torque,
                    "compliance_power_w": leg_power,
                }
                for name, values_by_time in phase_data.items():
                    trajectories[name].append(_interp(values_by_time, impact, last))

    labels = np.asarray(labels, dtype=bool)
    features = {key: np.asarray(value) for key, value in features.items()}
    trajectories = {key: np.asarray(value) for key, value in trajectories.items()}
    metadata = {
        "raw_touchdown_count": int(touchdown.sum()),
        "paired_episode_count": paired_episode_count,
        "single_sample_episode_count": single_sample_episode_count,
        "single_sample_episode_ratio": (
            float(single_sample_episode_count) / max(1, paired_episode_count)
        ),
        "boundary_excluded_count": boundary_excluded_count,
    }
    return labels, features, trajectories, metadata


def _summarize(labels, features):
    result = {}
    for name, values in features.items():
        stable, short = values[~labels], values[labels]
        auc = _rank_auc(values, labels)
        result[name] = {
            "stable_mean": float(stable.mean()),
            "short_mean": float(short.mean()),
            "difference": float(short.mean() - stable.mean()),
            "standardized_effect": _effect(stable, short),
            "auc": auc,
            "directionless_auc": max(auc, 1.0 - auc),
        }
    return result


def _trajectory_summary(labels, trajectories):
    return {
        name: {
            "stable_mean": values[~labels].mean(axis=0).tolist(),
            "short_mean": values[labels].mean(axis=0).tolist(),
        }
        for name, values in trajectories.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_trace", required=True)
    parser.add_argument("--adaptive_trace", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    output = {"dt_s": DT, "short_recontact_window_ms": SHORT_STEPS * DT * 1000.0}
    for name, path, adaptive in (
        ("baseline", args.baseline_trace, False),
        ("adaptive", args.adaptive_trace, True),
    ):
        labels, features, trajectories, metadata = _episodes(path, adaptive)
        output[name] = {
            "event_count": int(labels.size),
            "stable_count": int((~labels).sum()),
            "short_recontact_count": int(labels.sum()),
            "short_recontact_ratio": float(labels.mean()),
            "features": _summarize(labels, features),
            "phase_trajectories": _trajectory_summary(labels, trajectories),
            "episode_filtering": metadata,
        }

    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(destination.resolve())


if __name__ == "__main__":
    main()
