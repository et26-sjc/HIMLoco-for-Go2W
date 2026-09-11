"""Run one fresh MC ContactEstimator stability diagnostic experiment."""

import argparse
import os
import sys

import isaacgym  # must be imported before torch


def _parse_stability_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--freeze_contact_estimator_after_warmup",
        type=int,
        choices=(0, 1),
        required=True,
    )
    parser.add_argument("--validation_steps", type=int, default=200)
    parser.add_argument("--validation_interval", type=int, default=20)
    parser.add_argument("--contact_estimator_lr", type=float, default=None)
    parser.add_argument("--contact_estimator_warmup_lr", type=float, default=None)
    parser.add_argument("--contact_estimator_online_lr", type=float, default=None)
    parser.add_argument("--contact_estimator_replay_ratio", type=float, default=0.0)
    parser.add_argument(
        "--contact_estimator_replay_buffer_size", type=int, default=8192
    )
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return known


def main():
    diagnostic = _parse_stability_args()

    import legged_gym.envs  # noqa: F401
    from legged_gym import LEGGED_GYM_ROOT_DIR
    from legged_gym.utils import get_args, task_registry

    args = get_args()
    if args.task != "mc_learned_admittance_100hz":
        raise ValueError("This script only supports mc_learned_admittance_100hz")

    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    seed = train_cfg.seed if args.seed is None else args.seed
    num_envs = env_cfg.env.num_envs if args.num_envs is None else args.num_envs
    iterations = (
        train_cfg.runner.max_iterations
        if args.max_iterations is None
        else args.max_iterations
    )
    frozen = bool(diagnostic.freeze_contact_estimator_after_warmup)
    mode = "frozen" if frozen else "continual"
    if diagnostic.contact_estimator_lr is not None and (
        diagnostic.contact_estimator_warmup_lr is not None
        or diagnostic.contact_estimator_online_lr is not None
    ):
        raise ValueError(
            "Use either legacy --contact_estimator_lr or the stage-specific "
            "warmup/online arguments, not both."
        )
    legacy_lr = (
        float(train_cfg.policy.contact_estimator_lr)
        if diagnostic.contact_estimator_lr is None
        else float(diagnostic.contact_estimator_lr)
    )
    warmup_lr = (
        legacy_lr
        if diagnostic.contact_estimator_warmup_lr is None
        else float(diagnostic.contact_estimator_warmup_lr)
    )
    online_lr = (
        legacy_lr
        if diagnostic.contact_estimator_online_lr is None
        else float(diagnostic.contact_estimator_online_lr)
    )
    if warmup_lr <= 0.0 or online_lr <= 0.0:
        raise ValueError("ContactEstimator warmup/online LRs must be positive")
    replay_ratio = float(diagnostic.contact_estimator_replay_ratio)
    replay_buffer_size = int(diagnostic.contact_estimator_replay_buffer_size)
    if replay_ratio < 0.0 or replay_ratio > 1.0:
        raise ValueError("ContactEstimator replay ratio must be in [0, 1]")
    if replay_buffer_size <= 0:
        raise ValueError("ContactEstimator replay buffer size must be positive")

    # Stability experiments keep the selected classifier-to-control mapping.
    env_cfg.learned_admittance.impact_logit_correction_scale = 0.10
    env_cfg.learned_admittance.impact_gain = 2.0
    train_cfg.policy.contact_estimator_impact_pos_weight = 3.0
    train_cfg.policy.contact_estimator_lr = warmup_lr
    train_cfg.policy.contact_estimator_warmup_lr = warmup_lr
    train_cfg.policy.contact_estimator_online_lr = online_lr
    train_cfg.policy.contact_estimator_replay_ratio = replay_ratio
    train_cfg.policy.contact_estimator_replay_buffer_size = replay_buffer_size
    train_cfg.algorithm.freeze_contact_estimator_after_warmup = frozen
    train_cfg.runner.contact_validation_steps = int(diagnostic.validation_steps)
    train_cfg.runner.contact_validation_interval = int(
        diagnostic.validation_interval
    )
    train_cfg.runner.contact_validation_path = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        "validation",
        (
            f"mc_contact_validation_seed_{seed}_env_{num_envs}_"
            f"steps_{diagnostic.validation_steps}.pt"
        ),
    )
    train_cfg.runner.contact_replay_path = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        "replay",
        (
            f"mc_contact_replay_seed_{seed}_env_{num_envs}_"
            f"steps_{train_cfg.runner.contact_pretrain_steps}_"
            f"size_{replay_buffer_size}.pt"
        ),
    )
    warmup_lr_tag = f"{warmup_lr:.0e}".replace("-", "m").replace("+", "p")
    online_lr_tag = f"{online_lr:.0e}".replace("-", "m").replace("+", "p")
    replay_name = ""
    if replay_ratio > 0.0:
        replay_tag = f"{replay_ratio:.2f}".replace(".", "p")
        replay_name = f"replay_{replay_tag}_"
    train_cfg.runner.run_name = (
        f"contact_stability_warm_{warmup_lr_tag}_online_{online_lr_tag}_"
        f"{replay_name}{mode}_seed_{seed}_env_{num_envs}_iter_{iterations}"
    )
    train_cfg.runner.resume = False
    train_cfg.runner.wandb_mode = "disabled"
    # Pin the initialization checkpoint instead of relying on latest-run logic.
    train_cfg.runner.init_load_run = "Aug08_18-20-03_baseline"
    train_cfg.runner.init_checkpoint = 9000

    env, _ = task_registry.make_env(
        name=args.task, args=args, env_cfg=env_cfg
    )
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, args=args, train_cfg=train_cfg
    )
    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    main()
