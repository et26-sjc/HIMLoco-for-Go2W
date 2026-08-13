"""Passive quiet-motion evaluation configuration for the MC robot.

This task reuses checkpoints trained by ``mc``.  It does not change policy
observations, rewards, actions, or the fixed-PD / wheel-velocity controller;
it only enables physics-rate instrumentation in :class:`QuietMC`.
"""

from .mc_config import MCRoughCfg, MCRoughCfgPPO


class QuietMCCfg(MCRoughCfg):
    class quiet_metrics:
        enabled = True
        # Contact hysteresis is based on total wheel contact-force magnitude.
        contact_on_threshold = 5.0
        contact_off_threshold = 2.0
        wheel_radius = 0.075


class QuietMCCfgPPO(MCRoughCfgPPO):
    class runner(MCRoughCfgPPO.runner):
        # Reuse checkpoints trained by the normal MC baseline task.
        experiment_name = "MC"
        run_name = ""
        wandb_enabled = False
