"""Passive quiet-evaluation config for MC-HIM-100Hz checkpoints."""

from .mc_100hz_config import MC100HzCfg, MC100HzCfgPPO


class QuietMC100HzCfg(MC100HzCfg):
    class quiet_metrics:
        enabled = True
        contact_on_threshold = 5.0
        contact_off_threshold = 2.0
        wheel_radius = 0.075


class QuietMC100HzCfgPPO(MC100HzCfgPPO):
    class runner(MC100HzCfgPPO.runner):
        experiment_name = "MC100Hz"
        run_name = ""
        wandb_enabled = False
