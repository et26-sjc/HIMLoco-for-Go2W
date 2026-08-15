"""Quiet-evaluation configs for 100 Hz buffer variants."""

from .mc_buffered_100hz_variants_config import (
    MCBuffered100HzMildCfg,
    MCBuffered100HzMildCfgPPO,
    MCBuffered100HzPrecontactCfg,
    MCBuffered100HzPrecontactCfgPPO,
)


class _QuietMetrics:
    enabled = True
    contact_on_threshold = 5.0
    contact_off_threshold = 2.0
    wheel_radius = 0.075


class QuietMCBuffered100HzMildCfg(MCBuffered100HzMildCfg):
    class quiet_metrics(_QuietMetrics):
        pass


class QuietMCBuffered100HzMildZeroShotCfgPPO(MCBuffered100HzMildCfgPPO):
    class runner(MCBuffered100HzMildCfgPPO.runner):
        experiment_name = "MC100HzMinimal"
        run_name = ""
        wandb_enabled = False


class QuietMCBuffered100HzPrecontactCfg(MCBuffered100HzPrecontactCfg):
    class quiet_metrics(_QuietMetrics):
        pass


class QuietMCBuffered100HzPrecontactZeroShotCfgPPO(MCBuffered100HzPrecontactCfgPPO):
    class runner(MCBuffered100HzPrecontactCfgPPO.runner):
        experiment_name = "MC100HzMinimal"
        run_name = ""
        wandb_enabled = False
