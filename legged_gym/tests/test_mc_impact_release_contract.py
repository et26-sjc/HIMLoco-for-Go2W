"""Release-contract tests for the frozen MC impact-admittance method."""

import isaacgym  # noqa: F401; must precede torch
import torch

from legged_gym.envs.mc.mc_learned_admittance_100hz_config import (
    MCLearnedAdmittance100HzCfg,
    MCLearnedAdmittance100HzCfgPPO,
)
from legged_gym.envs.mc.mc_learned_admittance import MCLearnedAdmittance
from legged_gym.scripts.evaluate_mc_impact_quiet import (
    _CounterfactualAdmittance,
)
from legged_gym.utils.helpers import PolicyExporterAdaptiveHIM
from rsl_rl.modules import AdaptiveHIMActorCritic
from rsl_rl.storage.adaptive_him_rollout_storage import AdaptiveHIMRolloutStorage


def test_frozen_release_configuration():
    env_cfg = MCLearnedAdmittance100HzCfg()
    train_cfg = MCLearnedAdmittance100HzCfgPPO()
    assert (
        env_cfg.env.num_actions,
        env_cfg.env.num_motion_actions,
        env_cfg.env.num_compliance_actions,
        env_cfg.env.num_policy_actions,
        env_cfg.env.controller_state_dim,
        env_cfg.env.contact_estimate_dim,
    ) == (16, 16, 4, 20, 16, 8)
    assert env_cfg.learned_admittance.impact_gain == 2.0
    assert env_cfg.learned_admittance.impact_logit_correction_scale == 0.10
    assert train_cfg.policy.contact_estimator_warmup_lr == 1.0e-3
    assert train_cfg.policy.contact_estimator_online_lr == 3.0e-4
    assert train_cfg.policy.contact_estimator_replay_ratio == 0.25
    assert train_cfg.runner.init_checkpoint == 9000
    assert not hasattr(
        env_cfg.learned_admittance, "compliance_alpha_rise_rate_per_s"
    )
    assert not hasattr(env_cfg.rewards.scales, "quiet_short_recontact")


def test_rollout_storage_scales_with_runtime_environment_count():
    storage = AdaptiveHIMRolloutStorage(
        num_envs=1024,
        num_transitions_per_env=2,
        obs_shape=[342],
        controller_state_shape=[16],
        privileged_obs_shape=[262],
        actions_shape=[20],
        contact_target_shape=[8],
        device="cpu",
    )
    assert storage.observations.shape == (2, 1024, 342)
    assert storage.controller_states.shape == (2, 1024, 16)
    assert storage.actions.shape == (2, 1024, 20)
    assert storage.contact_targets.shape == (2, 1024, 8)


def test_jit_export_contract_preserves_raw_impact_logits():
    actor_critic = AdaptiveHIMActorCritic(
        num_actor_obs=342,
        num_critic_obs=262,
        num_one_step_obs=57,
        num_actions=16,
        num_policy_actions=20,
        controller_state_dim=16,
        contact_estimate_dim=8,
        motion_adapter_scale=0.0,
    )
    exporter = torch.jit.script(PolicyExporterAdaptiveHIM(actor_critic))
    policy_action, contact_estimate = exporter(
        torch.zeros(342), torch.zeros(16)
    )
    assert policy_action.shape == (20,)
    assert contact_estimate.shape == (8,)
    assert torch.all(contact_estimate[:4] >= 0.0)
    # The initialized estimator bias is -3. A sigmoid exporter would return
    # approximately 0.047 instead, violating the controller calibration path.
    torch.testing.assert_close(
        contact_estimate[4:], torch.full((4,), -3.0)
    )


def test_evaluator_default_admittance_matches_production():
    cfg = MCLearnedAdmittance100HzCfg().learned_admittance
    production = MCLearnedAdmittance(cfg, 3, "cpu")
    evaluator = _CounterfactualAdmittance(cfg, 3, "cpu")
    compliance = torch.tensor(
        [[0.0, 0.1, 0.4, 0.8], [0.2, 0.0, 0.7, 0.3], [1.0, 0.5, 0.0, 0.1]]
    )
    estimate = torch.linspace(-2.0, 3.0, 24).reshape(3, 8)
    q_nominal = torch.linspace(-1.0, 1.0, 48).reshape(3, 16)
    hips = torch.tensor([5, 1, 9, 13])
    knees = torch.tensor([6, 2, 10, 14])

    expected = production.step(
        compliance, estimate, q_nominal, hips, knees, 0.005
    )
    actual = evaluator.step(
        compliance, estimate, q_nominal, hips, knees, 0.005
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    for name in (
        "delta_l",
        "delta_l_dot",
        "force_bias",
        "alpha",
        "effective_alpha",
        "impact_probability",
        "drive_force",
    ):
        torch.testing.assert_close(
            getattr(evaluator, name), getattr(production, name),
            rtol=0.0, atol=0.0,
        )
