"""Passive quiet-evaluation config for minimal MC-HIM-100Hz checkpoints."""

from .mc_100hz_minimal_config import MC100HzMinimalCfg, MC100HzMinimalCfgPPO


class QuietMC100HzMinimalCfg(MC100HzMinimalCfg):
    class quiet_metrics:
        enabled = True
        contact_on_threshold = 5.0
        contact_off_threshold = 2.0
        wheel_radius = 0.075


class QuietMC100HzMinimalCfgPPO(MC100HzMinimalCfgPPO):
    class runner(MC100HzMinimalCfgPPO.runner):
        experiment_name = "MC100HzMinimal"
        run_name = ""
        wandb_enabled = False
