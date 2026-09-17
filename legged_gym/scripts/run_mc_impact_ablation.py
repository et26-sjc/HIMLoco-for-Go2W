"""Run one fresh MC impact-control mapping ablation from HIMLoco baseline."""

import argparse
import sys

import isaacgym  # must be imported before torch


def _parse_ablation_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--correction_scale", type=float, required=True)
    parser.add_argument("--impact_gain", type=float, default=2.0)
    parser.add_argument("--online_lr", type=float, default=None)
    parser.add_argument("--replay_ratio", type=float, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return known


def main():
    ablation = _parse_ablation_args()
    if ablation.correction_scale < 0.0:
        raise ValueError("correction_scale must be non-negative")
    if ablation.online_lr is not None and ablation.online_lr <= 0.0:
        raise ValueError("online_lr must be positive")
    if ablation.replay_ratio is not None and not 0.0 <= ablation.replay_ratio <= 1.0:
        raise ValueError("replay_ratio must be in [0, 1]")
    if ablation.save_interval is not None and ablation.save_interval <= 0:
        raise ValueError("save_interval must be positive")

    import legged_gym.envs  # noqa: F401
    from legged_gym.utils import get_args, task_registry

    args = get_args()
    if args.task != "mc_learned_admittance_100hz":
        raise ValueError("This script only supports mc_learned_admittance_100hz")

    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.learned_admittance.impact_logit_correction_scale = float(
        ablation.correction_scale
    )
    env_cfg.learned_admittance.impact_gain = float(ablation.impact_gain)
    if ablation.online_lr is not None:
        train_cfg.policy.contact_estimator_online_lr = float(ablation.online_lr)
    if ablation.replay_ratio is not None:
        train_cfg.policy.contact_estimator_replay_ratio = float(ablation.replay_ratio)
    if ablation.save_interval is not None:
        train_cfg.runner.save_interval = int(ablation.save_interval)

    seed = train_cfg.seed if args.seed is None else args.seed
    num_envs = env_cfg.env.num_envs if args.num_envs is None else args.num_envs
    iterations = (
        train_cfg.runner.max_iterations
        if args.max_iterations is None
        else args.max_iterations
    )
    scale_tag = f"{ablation.correction_scale:.3f}".replace(".", "p")
    gain_tag = f"{ablation.impact_gain:.2f}".replace(".", "p")
    lr_tag = "cfg" if ablation.online_lr is None else f"{ablation.online_lr:.0e}".replace("-", "m")
    replay_tag = "cfg" if ablation.replay_ratio is None else f"{ablation.replay_ratio:.2f}".replace(".", "p")
    train_cfg.runner.run_name = (
        f"impact_map_corr_{scale_tag}_gain_{gain_tag}_"
        f"online_{lr_tag}_replay_{replay_tag}_seed_{seed}_"
        f"env_{num_envs}_iter_{iterations}"
    )
    train_cfg.runner.resume = False
    train_cfg.runner.wandb_mode = "disabled"

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
