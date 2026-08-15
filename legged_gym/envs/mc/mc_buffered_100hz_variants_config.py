"""Zero-shot buffer variants for isolating contact-control behavior.

Both variants inherit the successful MC100HzMinimal setup and differ only in
low-level HIP/KNEE gain scheduling. They are intended for zero-shot evaluation
before committing to a full retraining run.
"""

from .mc_buffered_100hz_config import MCBuffered100HzCfg, MCBuffered100HzCfgPPO


class MCBuffered100HzMildCfg(MCBuffered100HzCfg):
    """Conservative post-impact buffer.

    Removes the unconditional new-contact trigger, waits for a clearly abnormal
    force rise, uses milder stiffness reduction, and shortens the hold window.
    """

    class buffer_control(MCBuffered100HzCfg.buffer_control):
        loading_rate_threshold_nps = 30000.0
        contact_on_threshold_n = 5.0
        hold_time_s = 0.015
        hip_knee_kp_scale = 0.80
        hip_knee_kd_scale = 1.50
        trigger_on_new_contact = False
        precontact_enabled = False
        precontact_downward_speed_threshold_mps = 0.25


class MCBuffered100HzMildCfgPPO(MCBuffered100HzCfgPPO):
    class runner(MCBuffered100HzCfgPPO.runner):
        experiment_name = "MC100HzBufferMild"
        run_name = "buffer_mild"
        wandb_group = "mc-buffer-ablation"
        wandb_tags = [
            "MC", "HIMLoco", "wheel-legged", "100Hz", "history-6",
            "buffer-mild", "post-impact", "zero-shot-candidate",
        ]


class MCBuffered100HzPrecontactCfg(MCBuffered100HzCfg):
    """Pre-touchdown damping buffer for downstairs impacts.

    Adds a pre-contact trigger when an unloaded wheel is moving downward quickly,
    so the leg can soften before the first force spike.  The post-impact force
    trigger is retained as a fallback, but ordinary low-force contact alone does
    not trigger the buffer.
    """

    class buffer_control(MCBuffered100HzCfg.buffer_control):
        loading_rate_threshold_nps = 30000.0
        contact_on_threshold_n = 5.0
        hold_time_s = 0.020
        hip_knee_kp_scale = 0.75
        hip_knee_kd_scale = 2.00
        trigger_on_new_contact = False
        precontact_enabled = True
        precontact_downward_speed_threshold_mps = 0.25


class MCBuffered100HzPrecontactCfgPPO(MCBuffered100HzCfgPPO):
    class runner(MCBuffered100HzCfgPPO.runner):
        experiment_name = "MC100HzBufferPrecontact"
        run_name = "buffer_precontact"
        wandb_group = "mc-buffer-ablation"
        wandb_tags = [
            "MC", "HIMLoco", "wheel-legged", "100Hz", "history-6",
            "buffer-precontact", "downstairs", "zero-shot-candidate",
        ]
