"""Minimal 100 Hz training configuration for the MC HIMLoco baseline.

This variant intentionally changes only the policy/control frequency relative to
``mc``.  Isaac Gym physics remains at 200 Hz (sim.dt = 0.005 s), while
``decimation = 2`` yields a 100 Hz policy/control loop.  HIM history length,
PPO rollout length, gamma/lambda, rewards, observations, and controller gains
are inherited unchanged so this task isolates the frequency change.
"""

from .mc_config import MCRoughCfg, MCRoughCfgPPO


class MC100HzCfg(MCRoughCfg):
    class control(MCRoughCfg.control):
        # Physics: 200 Hz (dt=0.005); policy/control: 100 Hz.
        decimation = 2


class MC100HzCfgPPO(MCRoughCfgPPO):
    class runner(MCRoughCfgPPO.runner):
        experiment_name = "MC_100Hz"
        run_name = "minimal"
        wandb_project = "MC-HIMLoco"
        wandb_group = "mc-100hz-minimal"
        wandb_tags = ["MC", "HIMLoco", "wheel-legged", "100Hz", "minimal"]
