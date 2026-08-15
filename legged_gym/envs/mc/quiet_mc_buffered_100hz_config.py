"""Quiet evaluation configs for contact-buffered MC-HIM-100Hz Minimal."""

from .mc_buffered_100hz_config import (
    MCBuffered100HzCfg,
    MCBuffered100HzCfgPPO,
)


class QuietMCBuffered100HzCfg(MCBuffered100HzCfg):
    class quiet_metrics:
        enabled = True
        contact_on_threshold = 5.0
        contact_off_threshold = 2.0
        wheel_radius = 0.075


class QuietMCBuffered100HzZeroShotCfgPPO(MCBuffered100HzCfgPPO):
    class runner(MCBuffered100HzCfgPPO.runner):
        # Apply the buffered controller to the successful ordinary
        # MC-HIM-100Hz Minimal checkpoint without retraining.  Because the
        # buffered env now inherits the same six-frame/342-input architecture,
        # the model weights are shape-compatible and only the controller differs.
        experiment_name = "MC100HzMinimal"
        run_name = ""
        wandb_enabled = False


class QuietMCBuffered100HzCfgPPO(MCBuffered100HzCfgPPO):
    class runner(MCBuffered100HzCfgPPO.runner):
        # Evaluate a policy trained from scratch with the buffered controller.
        experiment_name = "MC100HzBufferMinimal"
        run_name = ""
        wandb_enabled = False
