"""Passive quiet-motion evaluation for the 100 Hz MC policy.

The policy/control loop runs at 100 Hz (decimation=2 with 200 Hz physics), while
quiet metrics are sampled after every physics substep, i.e. at 200 Hz.  This task
is intended to evaluate checkpoints trained with ``mc_100hz`` without changing
their observations, rewards, controller, or timing.
"""

from .mc_100hz_config import MC100HzCfg, MC100HzCfgPPO


class QuietMC100HzCfg(MC100HzCfg):
    class quiet_metrics:
        enabled = True
        contact_on_threshold = 5.0
        contact_off_threshold = 2.0
        wheel_radius = 0.075


class QuietMC100HzCfgPPO(MC100HzCfgPPO):
    class runner(MC100HzCfgPPO.runner):
        # Reuse checkpoints trained by mc_100hz.
        experiment_name = "MC_100Hz"
        run_name = ""
        wandb_enabled = False
