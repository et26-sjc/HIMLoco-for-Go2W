"""Same-state short-horizon authority branching for MC admittance.

This script is evaluation-only. It first collects pre-release states from an
unchanged adaptive rollout, then broadcasts each state across simulator
environments and replays the same policy/contact-estimator outputs for every
branch. The only branch variable is the physical hip/knee compliance torque
residual scale on the releasing leg.

GPU PhysX does not expose contact-manifold or solver warm-start snapshots.
Consequently two independent scale-1 arms are mandatory: intervention results
are interpreted only when their A/A discrepancy stays below the declared
reproducibility gates.
"""

import json
import os
import sys
from datetime import datetime

import isaacgym  # noqa: F401; must be imported before torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.scripts.evaluate_mc_impact_quiet import (
    QuietMCLearnedAdmittance100Hz,
    QuietMCLearnedAdmittance100HzCfg,
    QuietMCLearnedAdmittance100HzCfgPPO,
)
from legged_gym.scripts.evaluate_mc_quiet import (
    _configure_environment,
    _place_on_fixed_terrain,
    _set_fixed_commands,
    _synchronize_observation_history,
)
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import set_seed


TASK_NAME = "mc_impact_short_horizon_branch"
ARM_SPECS = (
    ("a1", 1.0),
    ("a2", 1.0),
    ("half", 0.5),
    ("zero", 0.0),
    ("double", 2.0),
)


def _extract_private_args():
    specs = {
        "checkpoint_path": (str, None),
        "eval_seed": (int, 123),
        "warmup_seconds": (float, 2.0),
        "collection_seconds": (float, 20.0),
        "horizon_ms": (float, 50.0),
        "max_events": (int, 96),
        "eval_command_x": (float, 0.5),
        "stair_difficulty": (float, 0.5),
        "min_event_loading_nps": (float, 5000.0),
        "min_snapshot_compression_mm": (float, 0.01),
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
    if values["horizon_ms"] < 20.0 or values["horizon_ms"] > 100.0:
        raise ValueError("--horizon_ms must be in [20, 100]")
    if values["max_events"] <= 0:
        raise ValueError("--max_events must be positive")
    return values


class ShortHorizonBranchEnv(QuietMCLearnedAdmittance100Hz):
    """Adds snapshot/restore and open-loop residual scaling to the evaluator."""

    _ADM_STATE_NAMES = (
        "delta_l",
        "delta_l_dot",
        "force_bias",
        "alpha",
        "effective_alpha",
        "estimated_force",
        "impact_logits",
        "impact_probability",
        "transient_force",
        "drive_force",
        "stiffness",
        "last_joint_offsets",
    )
    _ENV_STATE_NAMES = (
        "obs_buf",
        "actions",
        "last_actions",
        "last_last_actions",
        "last_dof_vel",
        "last_root_vel",
        "commands",
        "torques",
        "policy_actions",
        "compliance_actions",
        "estimated_contact",
    )
    _QUIET_STATE_NAMES = (
        "quiet_contact_state",
        "quiet_prev_wheel_vel_world",
        "quiet_prev_force_z",
        "quiet_prev_force_norm",
        "quiet_prev_base_vel_z",
        "quiet_prev_base_acc_z",
        "quiet_prev_torques",
        "quiet_prev_wheel_omega",
        "quiet_touchdown_rel_z",
    )
    _QUIET_EVENT_NAMES = (
        "peak_force_z",
        "peak_force_norm",
        "peak_loading_rate_z",
        "peak_loading_rate_norm",
        "normal_impulse",
        "contact_impulse_norm",
        "duration",
        "peak_leg_compression",
        "peak_base_acc",
        "peak_base_jerk",
        "peak_torque_rate",
    )

    def capture_branch_state(self):
        state = {
            "root_states": self.root_states.clone(),
            "dof_state": self.dof_state.view(
                self.num_envs, self.num_dof, 2
            ).clone(),
        }
        for name in self._ENV_STATE_NAMES:
            if hasattr(self, name):
                state["env." + name] = getattr(self, name).clone()
        for name in self._ADM_STATE_NAMES:
            state["adm." + name] = getattr(self.admittance, name).clone()
        for name in self._QUIET_STATE_NAMES:
            state["quiet." + name] = getattr(self, name).clone()
        for name in self._QUIET_EVENT_NAMES:
            state["event." + name] = getattr(
                self, "quiet_event_" + name
            ).clone()
        return state

    @staticmethod
    def slice_branch_state(state, env_id):
        return {
            name: value[env_id].detach().cpu().clone()
            for name, value in state.items()
        }

    def restore_broadcast_state(self, state):
        def broadcast(value, target):
            source = value.to(device=self.device, dtype=target.dtype)
            target.copy_(source.unsqueeze(0).expand_as(target))

        broadcast(state["root_states"], self.root_states)
        dof_view = self.dof_state.view(self.num_envs, self.num_dof, 2)
        broadcast(state["dof_state"], dof_view)
        env_ids = torch.arange(
            self.num_envs, device=self.device, dtype=torch.int32
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids),
            int(self.num_envs),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids),
            int(self.num_envs),
        )
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        for name in self._ENV_STATE_NAMES:
            key = "env." + name
            if key in state and hasattr(self, name):
                broadcast(state[key], getattr(self, name))
        for name in self._ADM_STATE_NAMES:
            broadcast(state["adm." + name], getattr(self.admittance, name))
        for name in self._QUIET_STATE_NAMES:
            broadcast(state["quiet." + name], getattr(self, name))
        for name in self._QUIET_EVENT_NAMES:
            broadcast(
                state["event." + name], getattr(self, "quiet_event_" + name)
            )
        self._clear_quiet_step_buffers()

    def branch_policy_step(
        self, policy_actions, contact_estimate, residual_scale_adm
    ):
        """Advance one policy period without policy/reward/observation updates."""
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
        scale = residual_scale_adm.to(self.device)
        if scale.shape != (self.num_envs, self.num_compliance_actions):
            raise ValueError("residual scale must have shape [num_envs, 4]")
        self._clear_quiet_step_buffers()
        traces = []
        for substep in range(int(self.cfg.control.decimation)):
            baseline = self._compute_torques(motion_actions).view(
                self.torques.shape
            )
            adaptive = self._compute_adaptive_torques(
                motion_actions, compliance_actions, contact_estimate
            ).view(self.torques.shape)
            applied = adaptive.clone()
            for indices in (self.adm_hip_indices, self.adm_knee_indices):
                applied[:, indices] = (
                    baseline[:, indices]
                    + scale * (adaptive[:, indices] - baseline[:, indices])
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
            self._update_quiet_metrics_substep(substep)

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
            traces.append(
                {
                    "contact": self.quiet_trace_contact[substep].clone(),
                    "wheel_vz": self.quiet_trace_wheel_vel_z[substep].clone(),
                    "wheel_z": self.quiet_trace_wheel_pos_z_world[substep].clone(),
                    "leg_length": self.quiet_trace_leg_length[substep].clone(),
                    "force": self.quiet_trace_force_norm[substep].clone(),
                    "loading": self.quiet_trace_loading_rate_norm[substep].clone(),
                    "base_acc": self.quiet_trace_base_acc_z[substep].clone(),
                    "compression": self.admittance.delta_l[
                        :, self.quiet_to_adm_leg
                    ].clone(),
                    "compression_velocity": self.admittance.delta_l_dot[
                        :, self.quiet_to_adm_leg
                    ].clone(),
                    "torque_residual": torque_residual.clone(),
                    "torque_residual_leg": torch.stack(
                        (
                            torque_residual[:, self.adm_hip_indices],
                            torque_residual[:, self.adm_knee_indices],
                        ),
                        dim=-1,
                    )[:, self.quiet_to_adm_leg].clone(),
                    "compliance_power": (hip_power + knee_power)[
                        :, self.quiet_to_adm_leg
                    ].clone(),
                    "base_vx": base_velocity[:, 0].clone(),
                }
            )
        return traces


class ShortHorizonBranchCfg(QuietMCLearnedAdmittance100HzCfg):
    pass


class ShortHorizonBranchCfgPPO(QuietMCLearnedAdmittance100HzCfgPPO):
    pass


def _options_for_environment(private, args):
    return {
        "eval_scenario": "stairs_down",
        "eval_seconds": private["collection_seconds"],
        "warmup_seconds": private["warmup_seconds"],
        "eval_command_x": private["eval_command_x"],
        "stair_difficulty": private["stair_difficulty"],
    }


def _source_rollout(env, runner, policy, private):
    observations = env.get_observations()
    warmup_steps = int(round(private["warmup_seconds"] / env.dt))
    horizon_policy_steps = int(
        np.ceil(private["horizon_ms"] * 1.0e-3 / env.dt)
    )
    collection_steps = int(round(private["collection_seconds"] / env.dt))
    action_trace = []
    estimate_trace = []
    candidates = []
    reset_count = 0

    for _ in range(warmup_steps):
        _set_fixed_commands(env, private["eval_command_x"])
        with torch.no_grad():
            controller_state = env.get_controller_state().to(env.device)
            actions = policy(observations.detach(), controller_state)
            estimate = runner.alg.actor_critic.last_contact_estimate
            observations, _, _, dones, _, _, _ = env.step(
                actions.detach(), estimate.detach()
            )
        reset_count += int(torch.sum(dones).item())

    env.reset_quiet_metrics()
    for step_index in range(collection_steps):
        _set_fixed_commands(env, private["eval_command_x"])
        with torch.no_grad():
            controller_state = env.get_controller_state().to(env.device)
            actions = policy(observations.detach(), controller_state)
            estimate = runner.alg.actor_critic.last_contact_estimate
        action_trace.append(actions.detach().cpu().clone())
        estimate_trace.append(estimate.detach().cpu().clone())
        state_before = env.capture_branch_state()
        with torch.no_grad():
            observations, _, _, dones, _, _, _ = env.step(
                actions.detach(), estimate.detach()
            )
        reset_count += int(torch.sum(dones).item())

        if len(candidates) >= private["max_events"] * 4:
            continue
        finished = env.quiet_step_event_finished
        loading = env.quiet_step_completed_peak_loading_rate_norm
        valid = finished & (loading >= private["min_event_loading_nps"])
        valid &= ~dones.unsqueeze(1)
        for env_id, quiet_leg in torch.nonzero(valid, as_tuple=False).tolist():
            adm_leg = int(env.quiet_to_adm_leg[quiet_leg].item())
            compression = float(
                state_before["adm.delta_l"][env_id, adm_leg].item()
            )
            if compression < private["min_snapshot_compression_mm"] * 1.0e-3:
                continue
            if not bool(
                state_before["quiet.quiet_contact_state"][env_id, quiet_leg]
            ):
                continue
            candidates.append(
                {
                    "step": step_index,
                    "source_env": env_id,
                    "quiet_leg": quiet_leg,
                    "adm_leg": adm_leg,
                    "source_peak_loading_nps": float(loading[env_id, quiet_leg]),
                    "snapshot_compression_m": compression,
                    "state": env.slice_branch_state(state_before, env_id),
                }
            )

    latest_start = len(action_trace) - horizon_policy_steps
    candidates = [item for item in candidates if item["step"] <= latest_start]
    if len(candidates) > private["max_events"]:
        rng = np.random.default_rng(private["eval_seed"])
        selected = np.sort(
            rng.choice(
                len(candidates), size=private["max_events"], replace=False
            )
        )
        candidates = [candidates[index] for index in selected]
    return candidates, action_trace, estimate_trace, reset_count


def _branch_metrics(trace, target_leg, target_adm_leg, physics_dt, command_x):
    contact = np.stack(
        [item["contact"][:, target_leg].cpu().numpy() for item in trace]
    ).T.astype(bool)
    wheel_vz = np.stack(
        [item["wheel_vz"][:, target_leg].cpu().numpy() for item in trace]
    ).T
    force = np.stack(
        [item["force"][:, target_leg].cpu().numpy() for item in trace]
    ).T
    loading = np.stack(
        [item["loading"][:, target_leg].cpu().numpy() for item in trace]
    ).T
    base_acc = np.stack(
        [item["base_acc"].cpu().numpy() for item in trace]
    ).T
    compression = np.stack(
        [item["compression"][:, target_leg].cpu().numpy() for item in trace]
    ).T
    compression_velocity = np.stack(
        [
            item["compression_velocity"][:, target_leg].cpu().numpy()
            for item in trace
        ]
    ).T
    power = np.stack(
        [item["compliance_power"][:, target_leg].cpu().numpy() for item in trace]
    ).T
    base_vx = np.stack(
        [item["base_vx"].cpu().numpy() for item in trace]
    ).T
    residual = np.stack(
        [
            item["torque_residual_leg"][:, target_leg].cpu().numpy()
            for item in trace
        ]
    ).transpose(1, 0, 2)

    results = []
    short_steps = max(1, int(round(0.020 / physics_dt)))
    for env_id in range(contact.shape[0]):
        release_indices = np.flatnonzero(~contact[env_id])
        released = release_indices.size > 0
        release_index = int(release_indices[0]) if released else -1
        recontact = False
        release_vz = np.nan
        if released:
            release_vz = float(wheel_vz[env_id, release_index])
            stop = min(contact.shape[1], release_index + short_steps + 1)
            recontact = bool(np.any(contact[env_id, release_index + 1:stop]))
        results.append(
            {
                "released": float(released),
                "short_recontact": float(recontact) if released else np.nan,
                "release_wheel_vz_mps": release_vz,
                "force_peak_n": float(np.max(force[env_id])),
                "loading_peak_nps": float(np.max(loading[env_id])),
                "base_acc_rms_mps2": float(
                    np.sqrt(np.mean(np.square(base_acc[env_id])))
                ),
                "base_acc_peak_mps2": float(np.max(np.abs(base_acc[env_id]))),
                "compression_peak_mm": float(
                    1000.0 * np.max(compression[env_id])
                ),
                "compression_velocity_peak_mps": float(
                    np.max(compression_velocity[env_id])
                ),
                "compliance_work_j": float(
                    np.sum(power[env_id]) * physics_dt
                ),
                "torque_residual_rms_nm": float(
                    np.sqrt(np.mean(np.square(residual[env_id])))
                ),
                "tracking_error_x_mps": float(
                    np.mean(np.abs(base_vx[env_id] - command_x))
                ),
            }
        )
    return results


def _nanmean(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(values)) if np.any(np.isfinite(values)) else None


def _aggregate(rows):
    metrics = [key for key in rows[0] if key not in ("arm", "event")]
    return {metric: _nanmean([row[metric] for row in rows]) for metric in metrics}


def _run_branches(env, candidates, actions, estimates, private):
    horizon_policy_steps = int(
        np.ceil(private["horizon_ms"] * 1.0e-3 / env.dt)
    )
    arm_count = len(ARM_SPECS)
    rng = np.random.default_rng(private["eval_seed"] + 991)
    rows = []
    event_summaries = []
    for event_index, candidate in enumerate(candidates):
        env.restore_broadcast_state(candidate["state"])
        arm_indices = np.arange(env.num_envs) % arm_count
        rng.shuffle(arm_indices)
        arm_labels = np.asarray([ARM_SPECS[index][0] for index in arm_indices])
        arm_scales = np.asarray([ARM_SPECS[index][1] for index in arm_indices])
        scales = torch.ones(
            env.num_envs,
            env.num_compliance_actions,
            device=env.device,
        )
        scales[:, candidate["adm_leg"]] = torch.as_tensor(
            arm_scales, device=env.device, dtype=scales.dtype
        )

        trace = []
        start = candidate["step"]
        for offset in range(horizon_policy_steps):
            source_env = candidate["source_env"]
            action = actions[start + offset][source_env].to(env.device)
            estimate = estimates[start + offset][source_env].to(env.device)
            trace.extend(
                env.branch_policy_step(
                    action.unsqueeze(0).expand(env.num_envs, -1),
                    estimate.unsqueeze(0).expand(env.num_envs, -1),
                    scales,
                )
            )

        metrics = _branch_metrics(
            trace,
            candidate["quiet_leg"],
            candidate["adm_leg"],
            float(env.sim_params.dt),
            private["eval_command_x"],
        )
        for env_id, metric in enumerate(metrics):
            metric.update({"event": event_index, "arm": str(arm_labels[env_id])})
            rows.append(metric)
        summary = {
            "event": event_index,
            "source_env": candidate["source_env"],
            "source_step": candidate["step"],
            "quiet_leg": candidate["quiet_leg"],
            "source_peak_loading_nps": candidate["source_peak_loading_nps"],
            "snapshot_compression_mm": 1000.0 * candidate["snapshot_compression_m"],
            "arms": {},
        }
        for arm_name, _ in ARM_SPECS:
            arm_rows = [row for row in rows if row["event"] == event_index and row["arm"] == arm_name]
            summary["arms"][arm_name] = _aggregate(arm_rows)
        event_summaries.append(summary)
    return rows, event_summaries


def _paired_event_effects(event_summaries, arm, reference="a1"):
    metric_names = tuple(event_summaries[0]["arms"][reference])
    effects = {}
    for metric in metric_names:
        values = []
        for event in event_summaries:
            lhs = event["arms"][arm][metric]
            rhs = event["arms"][reference][metric]
            if lhs is not None and rhs is not None:
                values.append(lhs - rhs)
        if values:
            values_array = np.asarray(values, dtype=np.float64)
            rng = np.random.default_rng(20260916)
            bootstrap = np.mean(
                rng.choice(
                    values_array,
                    size=(20000, len(values_array)),
                    replace=True,
                ),
                axis=1,
            )
            effects[metric] = {
                "mean": float(np.mean(values_array)),
                "std": float(np.std(values_array, ddof=1)) if len(values) > 1 else 0.0,
                "ci95": [
                    float(np.quantile(bootstrap, 0.025)),
                    float(np.quantile(bootstrap, 0.975)),
                ],
                "n": len(values),
            }
        else:
            effects[metric] = {
                "mean": None,
                "std": None,
                "ci95": [None, None],
                "n": 0,
            }
    return effects


def _analyze(rows, event_summaries):
    arms = {}
    for arm_name, scale in ARM_SPECS:
        arm_rows = [row for row in rows if row["arm"] == arm_name]
        arms[arm_name] = {"scale": scale, **_aggregate(arm_rows)}
    effects = {
        arm: _paired_event_effects(event_summaries, arm)
        for arm in ("a2", "half", "zero", "double")
    }
    aa = effects["a2"]
    aa_release = abs(aa["released"]["mean"] or 0.0)
    aa_short = abs(aa["short_recontact"]["mean"] or 0.0)
    aa_vz = abs(aa["release_wheel_vz_mps"]["mean"] or 0.0)
    a1_rms = arms["a1"]["base_acc_rms_mps2"] or 0.0
    aa_rms = abs(aa["base_acc_rms_mps2"]["mean"] or 0.0)
    aa_rms_relative = aa_rms / max(abs(a1_rms), 1.0e-9)
    aa_pass = (
        aa_release <= 0.02
        and aa_short <= 0.02
        and aa_vz <= 0.01
        and aa_rms_relative <= 0.05
    )
    return {
        "arms": arms,
        "paired_effects_vs_a1": effects,
        "aa_gate": {
            "pass": bool(aa_pass),
            "release_rate_abs_diff": aa_release,
            "short_rate_abs_diff": aa_short,
            "release_vz_abs_diff_mps": aa_vz,
            "base_acc_rms_abs_diff_mps2": aa_rms,
            "base_acc_rms_relative_diff": aa_rms_relative,
            "limits": {
                "release_rate": 0.02,
                "short_rate": 0.02,
                "release_vz_mps": 0.01,
                "base_acc_rms_relative": 0.05,
            },
        },
    }


def _write_report(path, result):
    analysis = result["analysis"]
    lines = [
        "# MC Identical-State Short-Horizon Authority Branch",
        "",
        "## Scope",
        "",
        "Evaluation-only 50 ms branches replay the same policy and ContactEstimator "
        "outputs. The only branch variable is the target leg's physical compliance "
        "torque-residual scale. No training, reward, estimator, PPO, M/D/K, impact "
        "mapping, action, observation, or production-controller definition changed.",
        "",
        "GPU PhysX contact solver caches are not snapshot-accessible, so A/A is a "
        "hard validity gate rather than an assumed property.",
        "",
        "## Configuration",
        "",
        f"- Checkpoint: `{result['checkpoint_path']}`",
        f"- Evaluation seed: `{result['eval_seed']}`",
        f"- Environments: `{result['num_envs']}`",
        f"- Eligible branch events: `{result['event_count']}`",
        f"- Horizon: `{result['horizon_ms']:.1f} ms`",
        f"- Collection duration: `{result['collection_seconds']:.1f} s`",
        f"- Minimum completed-event loading: `{result['min_event_loading_nps']:.1f} N/s`",
        f"- Minimum snapshot compression: `{result['min_snapshot_compression_mm']:.3f} mm`",
        f"- Source rollout resets: `{result['source_reset_count']}`",
        "",
        "## Aggregate branch result",
        "",
        "| Arm | Scale | Release | Short/release | Release vz (m/s) | Force peak (N) | Loading peak (N/s) | Base RMS | Work (J) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm_name, _ in ARM_SPECS:
        arm = analysis["arms"][arm_name]
        lines.append(
            "| {name} | {scale:.2f} | {released:.4f} | {short:.4f} | "
            "{vz:.4f} | {force:.2f} | {loading:.1f} | {rms:.4f} | {work:.5f} |".format(
                name=arm_name,
                scale=arm["scale"],
                released=arm["released"],
                short=arm["short_recontact"],
                vz=arm["release_wheel_vz_mps"],
                force=arm["force_peak_n"],
                loading=arm["loading_peak_nps"],
                rms=arm["base_acc_rms_mps2"],
                work=arm["compliance_work_j"],
            )
        )
    gate = analysis["aa_gate"]
    lines.extend(
        [
            "",
            "## A/A validity",
            "",
            f"**{'PASS' if gate['pass'] else 'FAIL'}**",
            "",
            f"- release-rate difference: `{gate['release_rate_abs_diff']:.5f}`",
            f"- short-rate difference: `{gate['short_rate_abs_diff']:.5f}`",
            f"- release-vz difference: `{gate['release_vz_abs_diff_mps']:.6f} m/s`",
            f"- base-RMS relative difference: `{100.0 * gate['base_acc_rms_relative_diff']:.3f}%`",
            "",
            "Intervention arms must not be interpreted causally when this gate fails.",
            "",
            "## Paired mean effects versus A1",
            "",
            "| Arm | dShort [95% CI] | dRelease vz [95% CI] (m/s) | dForce (N) | dLoading (N/s) | dBase RMS |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm_name in ("a2", "half", "zero", "double"):
        effect = analysis["paired_effects_vs_a1"][arm_name]
        short_ci = effect["short_recontact"]["ci95"]
        vz_ci = effect["release_wheel_vz_mps"]["ci95"]
        lines.append(
            "| {name} | {short:+.5f} [{short_lo:+.5f}, {short_hi:+.5f}] | "
            "{vz:+.6f} [{vz_lo:+.6f}, {vz_hi:+.6f}] | {force:+.3f} | "
            "{loading:+.1f} | {rms:+.5f} |".format(
                name=arm_name,
                short=effect["short_recontact"]["mean"] or 0.0,
                short_lo=short_ci[0] or 0.0,
                short_hi=short_ci[1] or 0.0,
                vz=effect["release_wheel_vz_mps"]["mean"] or 0.0,
                vz_lo=vz_ci[0] or 0.0,
                vz_hi=vz_ci[1] or 0.0,
                force=effect["force_peak_n"]["mean"] or 0.0,
                loading=effect["loading_peak_nps"]["mean"] or 0.0,
                rms=effect["base_acc_rms_mps2"]["mean"] or 0.0,
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "- Lower authority reducing release vz/re-contact supports a late lifecycle mechanism.",
            "- Double authority reducing both impact severity and re-contact supports insufficient scale.",
            "- Double authority reducing force but worsening release/re-contact supports a structural axial-action trade-off.",
            "- No effect after a valid A/A points to earlier absorption geometry or missing action-space expressivity.",
            "",
            "GT contact is used only to select and score diagnostic release events. It is never fed to the policy, estimator, or production controller.",
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    private = _extract_private_args()
    args = get_args()
    if TASK_NAME not in task_registry.task_classes:
        task_registry.register(
            TASK_NAME,
            ShortHorizonBranchEnv,
            ShortHorizonBranchCfg(),
            ShortHorizonBranchCfgPPO(),
        )
    args.task = TASK_NAME
    env_cfg, train_cfg = task_registry.get_cfgs(TASK_NAME)
    options = _options_for_environment(private, args)
    _configure_environment(env_cfg, args, options)
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
    _place_on_fixed_terrain(
        env, "stairs_down", private["stair_difficulty"]
    )
    _set_fixed_commands(env, private["eval_command_x"])
    _synchronize_observation_history(env)
    candidates, actions, estimates, reset_count = _source_rollout(
        env, runner, policy, private
    )
    if not candidates:
        raise RuntimeError(
            "No eligible release events collected; extend collection or lower "
            "the diagnostic compression filter"
        )
    rows, event_summaries = _run_branches(
        env, candidates, actions, estimates, private
    )
    result = {
        "checkpoint_path": private["checkpoint_path"],
        "eval_seed": private["eval_seed"],
        "num_envs": env.num_envs,
        "event_count": len(candidates),
        "horizon_ms": private["horizon_ms"],
        "collection_seconds": private["collection_seconds"],
        "min_event_loading_nps": private["min_event_loading_nps"],
        "min_snapshot_compression_mm": private["min_snapshot_compression_mm"],
        "source_reset_count": reset_count,
        "arm_specs": list(ARM_SPECS),
        "analysis": _analyze(rows, event_summaries),
        "event_summaries": event_summaries,
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", "mc_short_horizon_branch", stamp
    )
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "results.json")
    report_path = os.path.join(output_dir, "report.md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    _write_report(report_path, result)
    print("Short-horizon branch JSON: " + json_path)
    print("Short-horizon branch report: " + report_path)
    print("A/A gate: " + ("PASS" if result["analysis"]["aa_gate"]["pass"] else "FAIL"))


if __name__ == "__main__":
    main()
