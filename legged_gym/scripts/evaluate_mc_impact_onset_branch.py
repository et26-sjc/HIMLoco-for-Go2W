"""Impact-onset same-state branching for the MC admittance controller.

The source rollout captures simulator/controller state immediately before a
wheel's first >=5 N contact sample. Branches replay the source policy action
and ContactEstimator output open-loop. Only the target leg's physical
compliance torque-residual scale differs between branches.

This is evaluation-only causal instrumentation. GT contact selects and scores
events; it is never exposed to the policy, estimator, or production control
law.
"""

import json
import os
import sys
from datetime import datetime

import isaacgym  # noqa: F401; must precede torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.scripts.evaluate_mc_impact_quiet import (
    QuietMCLearnedAdmittance100HzCfg,
    QuietMCLearnedAdmittance100HzCfgPPO,
)
from legged_gym.scripts.evaluate_mc_quiet import (
    _configure_environment,
    _place_on_fixed_terrain,
    _set_fixed_commands,
    _synchronize_observation_history,
)
from legged_gym.scripts.evaluate_mc_short_horizon_branch import (
    ShortHorizonBranchEnv,
)
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import set_seed


TASK_NAME = "mc_impact_onset_branch"
AUTHORITY_ARMS = (
    ("a1", 1.0),
    ("a2", 1.0),
    ("half", 0.5),
    ("zero", 0.0),
    ("double", 2.0),
)
POSTURE_ARMS = (
    ("a1", 0.0),
    ("a2", 0.0),
    ("retract_5mm", 0.005),
    ("extend_5mm", -0.005),
)


def _extract_private_args():
    specs = {
        "checkpoint_path": (str, None),
        "branch_experiment": (str, "authority"),
        "eval_seed": (int, 123),
        "warmup_seconds": (float, 2.0),
        "collection_seconds": (float, 20.0),
        "horizon_ms": (float, 300.0),
        "posture_pulse_ms": (float, 50.0),
        "max_events": (int, 96),
        "eval_command_x": (float, 0.5),
        "stair_difficulty": (float, 0.5),
        "min_onset_loading_nps": (float, 5000.0),
    }
    values = {name: default for name, (_, default) in specs.items()}
    index = 1
    while index < len(sys.argv):
        argument = sys.argv[index]
        matched = False
        for name, (cast, _) in specs.items():
            flag = "--" + name
            if argument == flag:
                if index + 1 >= len(sys.argv):
                    raise ValueError("Missing value after " + flag)
                values[name] = cast(sys.argv[index + 1])
                del sys.argv[index:index + 2]
                matched = True
                break
            prefix = flag + "="
            if argument.startswith(prefix):
                values[name] = cast(argument[len(prefix):])
                del sys.argv[index]
                matched = True
                break
        if not matched:
            index += 1
    if not values["checkpoint_path"]:
        raise ValueError("--checkpoint_path is required")
    values["checkpoint_path"] = os.path.abspath(values["checkpoint_path"])
    if values["branch_experiment"] not in ("authority", "posture"):
        raise ValueError("--branch_experiment must be authority or posture")
    if not 50.0 <= values["horizon_ms"] <= 500.0:
        raise ValueError("--horizon_ms must be in [50, 500]")
    if not 10.0 <= values["posture_pulse_ms"] <= values["horizon_ms"]:
        raise ValueError("posture pulse must be within the branch horizon")
    if values["max_events"] <= 0:
        raise ValueError("--max_events must be positive")
    return values


class ImpactOnsetBranchEnv(ShortHorizonBranchEnv):
    """Exact pre-touchdown capture and one-physics-step branch execution."""

    def source_policy_step(
        self,
        policy_actions,
        contact_estimate,
        action_schedule,
        estimate_schedule,
        candidates,
        max_candidates,
        min_onset_loading_nps,
    ):
        self.control_aligned_gt_impact.copy_(
            self.transition_contact_estimator_target[:, 4:] >= 0.5
        )
        motion_actions, compliance_actions = self._split_policy_action(
            policy_actions
        )
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clamp(
            motion_actions, -clip_actions, clip_actions
        ).to(self.device)
        self.compliance_actions = torch.clamp(
            compliance_actions.to(self.device), 0.0, 1.0
        )
        self.policy_actions = torch.cat(
            (self.actions, self.compliance_actions), dim=-1
        )
        self.estimated_contact.copy_(contact_estimate.to(self.device))
        self.delayed_actions = self.actions.clone().view(
            self.num_envs, 1, self.num_actions
        ).repeat(1, self.cfg.control.decimation, 1)

        self._clear_quiet_step_buffers()
        self._begin_gt_impact_step()
        self.render()
        for substep in range(int(self.cfg.control.decimation)):
            schedule_index = len(action_schedule)
            action_schedule.append(self.policy_actions.detach().cpu().clone())
            estimate_schedule.append(
                self.estimated_contact.detach().cpu().clone()
            )
            state_before = None
            if len(candidates) < max_candidates:
                state_before = self.capture_branch_state()
            previous_contact = self.quiet_contact_state.clone()

            baseline = self._compute_torques(
                self.delayed_actions[:, substep]
            ).view(self.torques.shape)
            self.torques = self._compute_adaptive_torques(
                self.delayed_actions[:, substep],
                self.compliance_actions,
                self.estimated_contact,
            ).view(self.torques.shape)
            source_residual = self.torques - baseline
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self._update_quiet_metrics_substep(substep)
            self._update_gt_impact_substep()

            if state_before is not None:
                contact = self.quiet_trace_contact[substep]
                touchdown = (~previous_contact) & contact
                loading = self.quiet_trace_loading_rate_norm[substep]
                valid = touchdown & (loading >= min_onset_loading_nps)
                for env_id, quiet_leg in torch.nonzero(
                    valid, as_tuple=False
                ).tolist():
                    adm_leg = int(self.quiet_to_adm_leg[quiet_leg].item())
                    hip_index = int(self.adm_hip_indices[adm_leg].item())
                    knee_index = int(self.adm_knee_indices[adm_leg].item())
                    target_residual = source_residual[
                        env_id, [hip_index, knee_index]
                    ]
                    candidates.append(
                        {
                            "schedule_index": schedule_index,
                            "source_env": env_id,
                            "quiet_leg": quiet_leg,
                            "adm_leg": adm_leg,
                            "onset_loading_nps": float(
                                loading[env_id, quiet_leg].item()
                            ),
                            "onset_force_n": float(
                                self.quiet_trace_force_norm[
                                    substep, env_id, quiet_leg
                                ].item()
                            ),
                            "source_residual_norm_nm": float(
                                torch.norm(target_residual).item()
                            ),
                            "state": self.slice_branch_state(
                                state_before, env_id
                            ),
                        }
                    )
                    if len(candidates) >= max_candidates:
                        break

        self._finish_gt_impact_step()
        self.transition_contact_estimator_target.copy_(
            self.contact_estimator_target
        )
        self.transition_gt_axial_loading_rate.copy_(
            self.gt_step_peak_axial_loading_rate
        )
        self._cache_admittance_diagnostics()
        termination_ids, termination_privileged_obs = self.post_physics_step()
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clamp(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clamp(
                self.privileged_obs_buf, -clip_obs, clip_obs
            )
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            termination_ids,
            termination_privileged_obs,
        )

    def _signed_leg_offsets(self, q_nominal, signed_retraction_m):
        qh = q_nominal[:, self.adm_hip_indices]
        qk = q_nominal[:, self.adm_knee_indices]
        qhk = qh + qk
        x = -self.admittance.l1 * torch.sin(qh) - self.admittance.l2 * torch.sin(qhk)
        z = -self.admittance.l1 * torch.cos(qh) - self.admittance.l2 * torch.cos(qhk)
        leg_length = torch.sqrt(torch.clamp(x * x + z * z, min=1.0e-8))
        dp_x = -signed_retraction_m * (x / leg_length)
        dp_z = -signed_retraction_m * (z / leg_length)
        j11 = -self.admittance.l1 * torch.cos(qh) - self.admittance.l2 * torch.cos(qhk)
        j12 = -self.admittance.l2 * torch.cos(qhk)
        j21 = self.admittance.l1 * torch.sin(qh) + self.admittance.l2 * torch.sin(qhk)
        j22 = self.admittance.l2 * torch.sin(qhk)
        lam2 = self.admittance.jacobian_damping ** 2
        a11 = j11 * j11 + j12 * j12 + lam2
        a12 = j11 * j21 + j12 * j22
        a22 = j21 * j21 + j22 * j22 + lam2
        determinant = torch.clamp(a11 * a22 - a12 * a12, min=1.0e-8)
        y1 = (a22 * dp_x - a12 * dp_z) / determinant
        y2 = (-a12 * dp_x + a11 * dp_z) / determinant
        dqh = j11 * y1 + j21 * y2
        dqk = j12 * y1 + j22 * y2
        return torch.stack((dqh, dqk), dim=-1)

    def branch_physics_substep(
        self,
        policy_actions,
        contact_estimate,
        authority_scale_adm,
        signed_posture_m_adm=None,
    ):
        motion_actions, compliance_actions = self._split_policy_action(
            policy_actions
        )
        clip_actions = self.cfg.normalization.clip_actions
        motion_actions = torch.clamp(
            motion_actions.to(self.device), -clip_actions, clip_actions
        )
        compliance_actions = torch.clamp(
            compliance_actions.to(self.device), 0.0, 1.0
        )
        contact_estimate = contact_estimate.to(self.device)
        baseline = self._compute_torques(motion_actions).view(self.torques.shape)
        adaptive = self._compute_adaptive_torques(
            motion_actions, compliance_actions, contact_estimate
        ).view(self.torques.shape)
        applied = adaptive.clone()
        scale = authority_scale_adm.to(self.device)
        for indices in (self.adm_hip_indices, self.adm_knee_indices):
            applied[:, indices] = (
                baseline[:, indices]
                + scale * (adaptive[:, indices] - baseline[:, indices])
            )

        if signed_posture_m_adm is not None:
            actions_scaled = motion_actions * self.cfg.control.action_scale
            actions_scaled = actions_scaled.clone()
            actions_scaled[:, self.wheel_indices] = 0.0
            q_nominal = self.default_dof_pos + actions_scaled
            posture_offsets = self._signed_leg_offsets(
                q_nominal, signed_posture_m_adm.to(self.device)
            )
            for indices, offset_index in (
                (self.adm_hip_indices, 0),
                (self.adm_knee_indices, 1),
            ):
                nominal = q_nominal[:, indices]
                lower = self.dof_pos_limits[indices, 0].unsqueeze(0)
                upper = self.dof_pos_limits[indices, 1].unsqueeze(0)
                residual_lower = torch.where(
                    nominal < lower,
                    torch.zeros_like(nominal),
                    lower - nominal,
                )
                residual_upper = torch.where(
                    nominal > upper,
                    torch.zeros_like(nominal),
                    upper - nominal,
                )
                safe_offset = torch.maximum(
                    torch.minimum(
                        posture_offsets[:, :, offset_index], residual_upper
                    ),
                    residual_lower,
                )
                applied[:, indices] += (
                    self.p_gains[indices].unsqueeze(0)
                    * self.Kp_factors
                    * safe_offset
                )

        self.torques = torch.clamp(
            applied, -self.torque_limits, self.torque_limits
        )
        torque_residual = self.torques - baseline
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(self.torques)
        )
        self.gym.simulate(self.sim)
        if self.device == "cpu":
            self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self._update_quiet_metrics_substep(0)
        hip_power = (
            torque_residual[:, self.adm_hip_indices]
            * self.dof_vel[:, self.adm_hip_indices]
        )
        knee_power = (
            torque_residual[:, self.adm_knee_indices]
            * self.dof_vel[:, self.adm_knee_indices]
        )
        base_velocity = quat_rotate_inverse(
            self.root_states[:, 3:7], self.root_states[:, 7:10]
        )
        return {
            "contact": self.quiet_trace_contact[0].clone(),
            "wheel_vz": self.quiet_trace_wheel_vel_z[0].clone(),
            "wheel_z": self.quiet_trace_wheel_pos_z_world[0].clone(),
            "leg_length": self.quiet_trace_leg_length[0].clone(),
            "force": self.quiet_trace_force_norm[0].clone(),
            "loading": self.quiet_trace_loading_rate_norm[0].clone(),
            "base_acc": self.quiet_trace_base_acc_z[0].clone(),
            "compression": self.admittance.delta_l[
                :, self.quiet_to_adm_leg
            ].clone(),
            "compression_velocity": self.admittance.delta_l_dot[
                :, self.quiet_to_adm_leg
            ].clone(),
            "compliance_power": (hip_power + knee_power)[
                :, self.quiet_to_adm_leg
            ].clone(),
            "base_vx": base_velocity[:, 0].clone(),
        }


class ImpactOnsetBranchCfg(QuietMCLearnedAdmittance100HzCfg):
    pass


class ImpactOnsetBranchCfgPPO(QuietMCLearnedAdmittance100HzCfgPPO):
    pass


def _environment_options(private):
    return {
        "eval_scenario": "stairs_down",
        "eval_seconds": private["collection_seconds"],
        "warmup_seconds": private["warmup_seconds"],
        "eval_command_x": private["eval_command_x"],
        "stair_difficulty": private["stair_difficulty"],
    }


def _collect_source(env, runner, policy, private):
    observations = env.get_observations()
    warmup_steps = int(round(private["warmup_seconds"] / env.dt))
    collection_steps = int(round(private["collection_seconds"] / env.dt))
    horizon_substeps = int(
        round(private["horizon_ms"] * 1.0e-3 / env.sim_params.dt)
    )
    action_schedule = []
    estimate_schedule = []
    candidates = []
    resets = []

    for _ in range(warmup_steps):
        _set_fixed_commands(env, private["eval_command_x"])
        with torch.no_grad():
            controller_state = env.get_controller_state().to(env.device)
            actions = policy(observations.detach(), controller_state)
            estimate = runner.alg.actor_critic.last_contact_estimate
            observations, _, _, _, _, _, _ = env.step(
                actions.detach(), estimate.detach()
            )
    env.reset_quiet_metrics()

    max_candidates = private["max_events"] * 4
    for _ in range(collection_steps):
        _set_fixed_commands(env, private["eval_command_x"])
        with torch.no_grad():
            controller_state = env.get_controller_state().to(env.device)
            actions = policy(observations.detach(), controller_state)
            estimate = runner.alg.actor_critic.last_contact_estimate
            observations, _, _, dones, _, _, _ = env.source_policy_step(
                actions.detach(),
                estimate.detach(),
                action_schedule,
                estimate_schedule,
                candidates,
                max_candidates,
                private["min_onset_loading_nps"],
            )
        resets.append(dones.detach().cpu().clone())

    latest_start = len(action_schedule) - horizon_substeps
    valid = []
    decimation = int(env.cfg.control.decimation)
    for candidate in candidates:
        if candidate["schedule_index"] > latest_start:
            continue
        start_policy = candidate["schedule_index"] // decimation
        end_policy = int(
            np.ceil(
                (candidate["schedule_index"] + horizon_substeps) / decimation
            )
        )
        source_env = candidate["source_env"]
        reset_during_branch = any(
            bool(reset[source_env])
            for reset in resets[start_policy:min(end_policy, len(resets))]
        )
        if not reset_during_branch:
            valid.append(candidate)
    if len(valid) > private["max_events"]:
        rng = np.random.default_rng(private["eval_seed"])
        indices = np.sort(
            rng.choice(len(valid), private["max_events"], replace=False)
        )
        valid = [valid[index] for index in indices]
    return valid, action_schedule, estimate_schedule, int(
        sum(int(torch.sum(reset).item()) for reset in resets)
    )


def _event_metrics(trace, target_leg, physics_dt, command_x):
    def stack(name):
        return np.stack(
            [item[name][:, target_leg].cpu().numpy() for item in trace]
        ).T

    contact = stack("contact").astype(bool)
    wheel_vz = stack("wheel_vz")
    wheel_z = stack("wheel_z")
    leg_length = stack("leg_length")
    force = stack("force")
    loading = stack("loading")
    compression = stack("compression")
    compression_velocity = stack("compression_velocity")
    power = stack("compliance_power")
    base_acc = np.stack(
        [item["base_acc"].cpu().numpy() for item in trace]
    ).T
    base_vx = np.stack(
        [item["base_vx"].cpu().numpy() for item in trace]
    ).T
    short_steps = max(1, int(round(0.020 / physics_dt)))
    results = []
    for env_id in range(contact.shape[0]):
        onset_candidates = np.flatnonzero(contact[env_id])
        touched = onset_candidates.size > 0
        onset = int(onset_candidates[0]) if touched else -1
        release = -1
        if touched:
            release_candidates = np.flatnonzero(~contact[env_id, onset + 1:])
            if release_candidates.size:
                release = onset + 1 + int(release_candidates[0])
        released = release >= 0
        short_recontact = np.nan
        release_vz = np.nan
        release_leg_length = np.nan
        contact_duration = np.nan
        if released:
            stop = min(contact.shape[1], release + short_steps + 1)
            short_recontact = float(np.any(contact[env_id, release + 1:stop]))
            release_vz = float(wheel_vz[env_id, release])
            release_leg_length = float(leg_length[env_id, release])
            contact_duration = float((release - onset) * physics_dt)
        onset_leg_length = (
            float(leg_length[env_id, onset]) if touched else np.nan
        )
        results.append(
            {
                "touchdown": float(touched),
                "released": float(released),
                "short_recontact": short_recontact,
                "contact_duration_s": contact_duration,
                "onset_wheel_vz_mps": float(wheel_vz[env_id, onset]) if touched else np.nan,
                "release_wheel_vz_mps": release_vz,
                "wheel_vz_peak_mps": float(np.max(wheel_vz[env_id])),
                "onset_leg_length_m": onset_leg_length,
                "release_leg_length_m": release_leg_length,
                "leg_length_delta_mm": (
                    1000.0 * (release_leg_length - onset_leg_length)
                    if released else np.nan
                ),
                "leg_length_min_m": float(np.min(leg_length[env_id])),
                "leg_length_max_m": float(np.max(leg_length[env_id])),
                "wheel_height_delta_mm": float(
                    1000.0 * (wheel_z[env_id, -1] - wheel_z[env_id, 0])
                ),
                "force_peak_n": float(np.max(force[env_id])),
                "loading_peak_nps": float(np.max(loading[env_id])),
                "base_acc_rms_mps2": float(
                    np.sqrt(np.mean(np.square(base_acc[env_id])))
                ),
                "base_acc_peak_mps2": float(np.max(np.abs(base_acc[env_id]))),
                "compression_peak_mm": float(1000.0 * np.max(compression[env_id])),
                "compression_velocity_peak_mps": float(
                    np.max(compression_velocity[env_id])
                ),
                "compliance_work_j": float(
                    np.sum(power[env_id]) * physics_dt
                ),
                "tracking_error_x_mps": float(
                    np.mean(np.abs(base_vx[env_id] - command_x))
                ),
            }
        )
    return results


def _mean(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(values)) if np.any(np.isfinite(values)) else None


def _aggregate(rows):
    metric_names = [
        key for key in rows[0] if key not in ("arm", "event")
    ]
    return {
        metric: _mean([row[metric] for row in rows])
        for metric in metric_names
    }


def _run_branches(env, candidates, actions, estimates, private):
    authority = private["branch_experiment"] == "authority"
    arm_specs = AUTHORITY_ARMS if authority else POSTURE_ARMS
    horizon_substeps = int(
        round(private["horizon_ms"] * 1.0e-3 / env.sim_params.dt)
    )
    pulse_substeps = int(
        round(private["posture_pulse_ms"] * 1.0e-3 / env.sim_params.dt)
    )
    rng = np.random.default_rng(private["eval_seed"] + 991)
    rows = []
    event_summaries = []
    for event_index, candidate in enumerate(candidates):
        env.restore_broadcast_state(candidate["state"])
        indices = np.arange(env.num_envs) % len(arm_specs)
        rng.shuffle(indices)
        labels = np.asarray([arm_specs[index][0] for index in indices])
        values = np.asarray([arm_specs[index][1] for index in indices])
        authority_scale = torch.ones(
            env.num_envs, env.num_compliance_actions, device=env.device
        )
        if authority:
            authority_scale[:, candidate["adm_leg"]] = torch.as_tensor(
                values, device=env.device, dtype=authority_scale.dtype
            )
        trace = []
        for offset in range(horizon_substeps):
            source_index = candidate["schedule_index"] + offset
            source_env = candidate["source_env"]
            action = actions[source_index][source_env].to(env.device)
            estimate = estimates[source_index][source_env].to(env.device)
            signed_posture = None
            if not authority:
                phase = min(1.0, float(offset + 1) / max(pulse_substeps, 1))
                pulse = np.sin(np.pi * phase) if offset < pulse_substeps else 0.0
                signed_posture = torch.zeros_like(authority_scale)
                signed_posture[:, candidate["adm_leg"]] = torch.as_tensor(
                    values * pulse,
                    device=env.device,
                    dtype=authority_scale.dtype,
                )
            trace.append(
                env.branch_physics_substep(
                    action.unsqueeze(0).expand(env.num_envs, -1),
                    estimate.unsqueeze(0).expand(env.num_envs, -1),
                    authority_scale,
                    signed_posture,
                )
            )
        metrics = _event_metrics(
            trace,
            candidate["quiet_leg"],
            float(env.sim_params.dt),
            private["eval_command_x"],
        )
        for env_id, metric in enumerate(metrics):
            metric.update({"event": event_index, "arm": str(labels[env_id])})
            rows.append(metric)
        summary = {
            "event": event_index,
            "source_env": candidate["source_env"],
            "source_schedule_index": candidate["schedule_index"],
            "quiet_leg": candidate["quiet_leg"],
            "adm_leg": candidate["adm_leg"],
            "source_onset_loading_nps": candidate["onset_loading_nps"],
            "source_onset_force_n": candidate["onset_force_n"],
            "source_residual_norm_nm": candidate["source_residual_norm_nm"],
            "arms": {},
        }
        for arm_name, _ in arm_specs:
            arm_rows = [
                row for row in rows
                if row["event"] == event_index and row["arm"] == arm_name
            ]
            summary["arms"][arm_name] = _aggregate(arm_rows)
        event_summaries.append(summary)
    return rows, event_summaries, arm_specs


def _paired_effects(event_summaries, arm, reference="a1"):
    metrics = tuple(event_summaries[0]["arms"][reference])
    effects = {}
    for metric in metrics:
        values = []
        for event in event_summaries:
            candidate = event["arms"][arm][metric]
            baseline = event["arms"][reference][metric]
            if candidate is not None and baseline is not None:
                values.append(candidate - baseline)
        if values:
            array = np.asarray(values, dtype=np.float64)
            rng = np.random.default_rng(20260917)
            bootstrap = np.mean(
                rng.choice(array, size=(20000, len(array)), replace=True),
                axis=1,
            )
            effects[metric] = {
                "mean": float(np.mean(array)),
                "ci95": [
                    float(np.quantile(bootstrap, 0.025)),
                    float(np.quantile(bootstrap, 0.975)),
                ],
                "n": len(array),
            }
        else:
            effects[metric] = {
                "mean": None, "ci95": [None, None], "n": 0
            }
    return effects


def _analyze(rows, event_summaries, arm_specs):
    arms = {}
    for arm_name, value in arm_specs:
        arm_rows = [row for row in rows if row["arm"] == arm_name]
        arms[arm_name] = {"value": value, **_aggregate(arm_rows)}
    effects = {
        arm_name: _paired_effects(event_summaries, arm_name)
        for arm_name, _ in arm_specs if arm_name != "a1"
    }
    aa = effects["a2"]
    a1_rms = arms["a1"]["base_acc_rms_mps2"] or 0.0
    aa_rms = abs(aa["base_acc_rms_mps2"]["mean"] or 0.0)
    a1_force = arms["a1"]["force_peak_n"] or 0.0
    a1_loading = arms["a1"]["loading_peak_nps"] or 0.0
    aa_force = abs(aa["force_peak_n"]["mean"] or 0.0)
    aa_loading = abs(aa["loading_peak_nps"]["mean"] or 0.0)
    aa_gate = {
        "touchdown_rate_abs_diff": abs(aa["touchdown"]["mean"] or 0.0),
        "release_rate_abs_diff": abs(aa["released"]["mean"] or 0.0),
        "short_rate_abs_diff": abs(aa["short_recontact"]["mean"] or 0.0),
        "release_vz_abs_diff_mps": abs(
            aa["release_wheel_vz_mps"]["mean"] or 0.0
        ),
        "base_rms_relative_diff": aa_rms / max(abs(a1_rms), 1.0e-9),
        "force_relative_diff": aa_force / max(abs(a1_force), 1.0e-9),
        "loading_relative_diff": aa_loading / max(abs(a1_loading), 1.0e-9),
    }
    aa_gate["pass"] = bool(
        aa_gate["touchdown_rate_abs_diff"] <= 0.02
        and aa_gate["release_rate_abs_diff"] <= 0.02
        and aa_gate["short_rate_abs_diff"] <= 0.02
        and aa_gate["release_vz_abs_diff_mps"] <= 0.01
        and aa_gate["base_rms_relative_diff"] <= 0.05
        and aa_gate["force_relative_diff"] <= 0.01
        and aa_gate["loading_relative_diff"] <= 0.01
    )
    return {"arms": arms, "paired_effects_vs_a1": effects, "aa_gate": aa_gate}


def _format(value, digits=4):
    return "NA" if value is None else f"{value:.{digits}f}"


def _write_report(path, result):
    analysis = result["analysis"]
    experiment = result["branch_experiment"]
    unit = "scale" if experiment == "authority" else "signed retraction (m)"
    lines = [
        "# MC Impact-Onset Identical-State Branch",
        "",
        "## Scope",
        "",
        f"Experiment: **{experiment}**. State is captured immediately before the "
        "first >=5 N contact sample. Policy action, ContactEstimator output, and "
        "admittance state are identical at the branch point and replayed open-loop.",
        "",
        "GT contact selects and scores events only. It does not enter the policy, "
        "estimator, production controller, reward, or training.",
        "",
        "## Configuration",
        "",
        f"- checkpoint: `{result['checkpoint_path']}`",
        f"- evaluation seed: `{result['eval_seed']}`",
        f"- environments/events: `{result['num_envs']}` / `{result['event_count']}`",
        f"- source collection / branch horizon: `{result['collection_seconds']:.1f} s` / `{result['horizon_ms']:.1f} ms`",
        f"- minimum onset loading: `{result['min_onset_loading_nps']:.1f} N/s`",
        f"- source resets: `{result['source_reset_count']}`",
        "",
        "## Aggregate result",
        "",
        f"| Arm | {unit} | Force peak (N) | Loading peak (N/s) | Release vz (m/s) | Leg delta (mm) | Duration (s) | Short/release | Base RMS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm_name, _ in result["arm_specs"]:
        arm = analysis["arms"][arm_name]
        lines.append(
            "| {name} | {value:.4f} | {force} | {loading} | {vz} | {leg} | {duration} | {short} | {rms} |".format(
                name=arm_name,
                value=arm["value"],
                force=_format(arm["force_peak_n"], 2),
                loading=_format(arm["loading_peak_nps"], 1),
                vz=_format(arm["release_wheel_vz_mps"], 4),
                leg=_format(arm["leg_length_delta_mm"], 3),
                duration=_format(arm["contact_duration_s"], 4),
                short=_format(arm["short_recontact"], 4),
                rms=_format(arm["base_acc_rms_mps2"], 4),
            )
        )
    gate = analysis["aa_gate"]
    lines.extend([
        "",
        "## A/A validity",
        "",
        f"**{'PASS' if gate['pass'] else 'FAIL'}**",
        "",
        f"- touchdown-rate difference: `{gate['touchdown_rate_abs_diff']:.6f}`",
        f"- release-rate difference: `{gate['release_rate_abs_diff']:.6f}`",
        f"- short-rate difference: `{gate['short_rate_abs_diff']:.6f}`",
        f"- release-vz difference: `{gate['release_vz_abs_diff_mps']:.8f} m/s`",
        f"- base-RMS relative difference: `{100.0 * gate['base_rms_relative_diff']:.5f}%`",
        f"- force/loading relative difference: `{100.0 * gate['force_relative_diff']:.5f}%` / `{100.0 * gate['loading_relative_diff']:.5f}%`",
        "",
        "## Paired effects versus A1",
        "",
        "| Arm | dForce [95% CI] | dLoading [95% CI] | dRelease vz [95% CI] | dLeg delta [95% CI] | dShort [95% CI] |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for arm_name, _ in result["arm_specs"]:
        if arm_name == "a1":
            continue
        effect = analysis["paired_effects_vs_a1"][arm_name]
        pieces = []
        for metric, digits in (
            ("force_peak_n", 2),
            ("loading_peak_nps", 1),
            ("release_wheel_vz_mps", 5),
            ("leg_length_delta_mm", 3),
            ("short_recontact", 4),
        ):
            item = effect[metric]
            if item["mean"] is None:
                pieces.append("NA")
            else:
                pieces.append(
                    f"{item['mean']:+.{digits}f} "
                    f"[{item['ci95'][0]:+.{digits}f}, {item['ci95'][1]:+.{digits}f}]"
                )
        lines.append("| " + arm_name + " | " + " | ".join(pieces) + " |")
    lines.extend([
        "",
        "A/A must pass before intervention arms receive a causal interpretation. "
        "Generated JSON retains event-level metrics for cross-seed bootstrap.",
    ])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    private = _extract_private_args()
    args = get_args()
    if TASK_NAME not in task_registry.task_classes:
        task_registry.register(
            TASK_NAME,
            ImpactOnsetBranchEnv,
            ImpactOnsetBranchCfg(),
            ImpactOnsetBranchCfgPPO(),
        )
    args.task = TASK_NAME
    env_cfg, train_cfg = task_registry.get_cfgs(TASK_NAME)
    _configure_environment(env_cfg, args, _environment_options(private))
    env, _ = task_registry.make_env(name=TASK_NAME, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = False
    train_cfg.runner.wandb_enabled = False
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=TASK_NAME,
        args=args,
        train_cfg=train_cfg,
        log_root="default",
    )
    runner.load(private["checkpoint_path"], load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    set_seed(private["eval_seed"])
    _place_on_fixed_terrain(env, "stairs_down", private["stair_difficulty"])
    _set_fixed_commands(env, private["eval_command_x"])
    _synchronize_observation_history(env)
    candidates, actions, estimates, reset_count = _collect_source(
        env, runner, policy, private
    )
    if not candidates:
        raise RuntimeError("No eligible impact-onset events were collected")
    rows, event_summaries, arm_specs = _run_branches(
        env, candidates, actions, estimates, private
    )
    result = {
        "checkpoint_path": private["checkpoint_path"],
        "branch_experiment": private["branch_experiment"],
        "eval_seed": private["eval_seed"],
        "num_envs": env.num_envs,
        "event_count": len(candidates),
        "collection_seconds": private["collection_seconds"],
        "horizon_ms": private["horizon_ms"],
        "posture_pulse_ms": private["posture_pulse_ms"],
        "min_onset_loading_nps": private["min_onset_loading_nps"],
        "source_reset_count": reset_count,
        "arm_specs": list(arm_specs),
        "analysis": _analyze(rows, event_summaries, arm_specs),
        "event_summaries": event_summaries,
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        "mc_impact_onset_branch",
        private["branch_experiment"],
        stamp,
    )
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "results.json")
    report_path = os.path.join(output_dir, "report.md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    _write_report(report_path, result)
    print("Impact-onset branch JSON: " + json_path)
    print("Impact-onset branch report: " + report_path)
    print("A/A gate: " + ("PASS" if result["analysis"]["aa_gate"]["pass"] else "FAIL"))


if __name__ == "__main__":
    main()
