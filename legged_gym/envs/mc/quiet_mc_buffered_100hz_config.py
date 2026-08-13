"""Quiet evaluation configs for contact-buffered MC-HIM-100Hz."""

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
        # Apply the buffered controller to an ordinary MC-HIM-100Hz checkpoint
        # without retraining. This isolates the controller's zero-shot effect.
        experiment_name = "MC100Hz"
        run_name = ""
        wandb_enabled = False


class QuietMCBuffered100HzCfgPPO(MCBuffered100HzCfgPPO):
    class runner(MCBuffered100HzCfgPPO.runner):
        # Evaluate a policy trained with the buffered controller.
        experiment_name = "MC100HzBuffer"
        run_name = ""
        wandb_enabled = False
