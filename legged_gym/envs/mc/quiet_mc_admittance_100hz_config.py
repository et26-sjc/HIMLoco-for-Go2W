"""Quiet-evaluation configs for 100 Hz MC axial admittance."""

from .mc_admittance_100hz_config import (
    MCAdmittance100HzCfg,
    MCAdmittance100HzCfgPPO,
    MCAdmittanceGated100HzCfg,
    MCAdmittanceGated100HzCfgPPO,
)


class QuietMCAdmittance100HzCfg(MCAdmittance100HzCfg):
    class quiet_metrics:
        enabled = True
        contact_on_threshold = 5.0
        contact_off_threshold = 2.0
        wheel_radius = 0.075


class QuietMCAdmittanceGated100HzCfg(MCAdmittanceGated100HzCfg):
    class quiet_metrics:
        enabled = True
        contact_on_threshold = 5.0
        contact_off_threshold = 2.0
        wheel_radius = 0.075


class QuietMCAdmittance100HzZeroShotCfgPPO(MCAdmittance100HzCfgPPO):
    """Load the already-trained MC100HzMinimal policy for controller-only eval."""

    class runner(MCAdmittance100HzCfgPPO.runner):
        experiment_name = "MC100HzMinimal"
        run_name = ""
        wandb_enabled = False


class QuietMCAdmittance100HzCfgPPO(MCAdmittance100HzCfgPPO):
    """Load a policy trained together with the A0 admittance controller."""

    class runner(MCAdmittance100HzCfgPPO.runner):
        experiment_name = "MC100HzAdmittance"
        run_name = ""
        wandb_enabled = False


class QuietMCAdmittanceGated100HzZeroShotCfgPPO(MCAdmittanceGated100HzCfgPPO):
    """Load MC100HzMinimal weights and change only the A0.1 low-level controller."""

    class runner(MCAdmittanceGated100HzCfgPPO.runner):
        experiment_name = "MC100HzMinimal"
        run_name = ""
        wandb_enabled = False


class QuietMCAdmittanceGated100HzCfgPPO(MCAdmittanceGated100HzCfgPPO):
    """Load a policy trained together with the A0.1 gated admittance controller."""

    class runner(MCAdmittanceGated100HzCfgPPO.runner):
        experiment_name = "MC100HzAdmittanceGated"
        run_name = ""
        wandb_enabled = False
