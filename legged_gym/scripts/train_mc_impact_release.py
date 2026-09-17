"""Train the frozen replay-stabilized MC impact-admittance release method."""

import argparse
import os
import sys

import isaacgym  # noqa: F401; must precede torch

from legged_gym import LEGGED_GYM_ROOT_DIR


TASK_NAME = "mc_learned_admittance_100hz"
CONFIG_PATH = "legged_gym/envs/mc/mc_learned_admittance_100hz_config.py"


def _release_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--baseline_run", default="Aug08_18-20-03_baseline"
    )
    parser.add_argument("--baseline_checkpoint", type=int, default=9000)
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument(
        "--log_root",
        default=os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            "MC_ImpactClassifier_Admittance_100Hz",
        ),
    )
    parser.add_argument(
        "--wandb_mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    if known.baseline_checkpoint < 0:
        raise ValueError("--baseline_checkpoint must be non-negative")
    if known.save_interval <= 0:
        raise ValueError("--save_interval must be positive")
    return known


def _assert_frozen_configuration(env_cfg, train_cfg):
    expected = {
        "physical_action": (env_cfg.env.num_actions, 16),
        "motion_action": (env_cfg.env.num_motion_actions, 16),
        "compliance_action": (env_cfg.env.num_compliance_actions, 4),
        "policy_action": (env_cfg.env.num_policy_actions, 20),
        "controller_state": (env_cfg.env.controller_state_dim, 16),
        "contact_estimate": (env_cfg.env.contact_estimate_dim, 8),
        "impact_gain": (env_cfg.learned_admittance.impact_gain, 2.0),
        "correction_scale": (
            env_cfg.learned_admittance.impact_logit_correction_scale,
            0.10,
        ),
        "warmup_lr": (train_cfg.policy.contact_estimator_warmup_lr, 1.0e-3),
        "online_lr": (train_cfg.policy.contact_estimator_online_lr, 3.0e-4),
        "replay_ratio": (
            train_cfg.policy.contact_estimator_replay_ratio,
            0.25,
        ),
        "motion_adapter_scale": (train_cfg.policy.motion_adapter_scale, 0.0),
        "base_actor_lr_scale": (train_cfg.algorithm.base_actor_lr_scale, 0.0),
        "action_std_lr_scale": (train_cfg.algorithm.action_std_lr_scale, 0.0),
        "update_him_estimator": (
            train_cfg.algorithm.update_him_estimator,
            False,
        ),
    }
    mismatches = [
        f"{name}={actual!r}, expected {wanted!r}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise RuntimeError(
            "Frozen MC impact-admittance configuration mismatch: "
            + "; ".join(mismatches)
        )


def main():
    release = _release_args()

    import legged_gym.envs  # noqa: F401; registers tasks after Isaac Gym
    from legged_gym.utils import get_args, task_registry

    args = get_args()
    if args.task != TASK_NAME:
        raise ValueError(f"This launcher requires --task={TASK_NAME}")
    if args.resume:
        raise ValueError(
            "Scale validation must start fresh from the HIMLoco baseline; "
            "use train.py directly for an intentional adaptive resume."
        )

    env_cfg, train_cfg = task_registry.get_cfgs(TASK_NAME)
    _assert_frozen_configuration(env_cfg, train_cfg)
    train_cfg.runner.init_load_run = release.baseline_run
    train_cfg.runner.init_checkpoint = release.baseline_checkpoint
    train_cfg.runner.save_interval = release.save_interval
    train_cfg.runner.wandb_mode = release.wandb_mode

    baseline_path = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        train_cfg.runner.init_experiment_name,
        release.baseline_run,
        f"model_{release.baseline_checkpoint}.pt",
    )
    if not os.path.isfile(baseline_path):
        raise FileNotFoundError(
            "Required HIMLoco baseline checkpoint not found: " + baseline_path
        )

    print("Frozen config: " + os.path.join(LEGGED_GYM_ROOT_DIR, CONFIG_PATH))
    print("Baseline checkpoint: " + baseline_path)
    print("Log root: " + os.path.abspath(release.log_root))

    env, _ = task_registry.make_env(
        name=TASK_NAME, args=args, env_cfg=env_cfg
    )
    runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        args=args,
        train_cfg=train_cfg,
        log_root=os.path.abspath(release.log_root),
    )
    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    main()
