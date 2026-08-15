"""Minimal 100 Hz MC HIMLoco frequency ablation base.

This is copied into the buffer branch so buffered experiments use exactly the
same learning setup as the successful MC100HzMinimal baseline: only control
frequency changes relative to the validated 50 Hz MC baseline.
"""

from .mc_config import MCRoughCfg, MCRoughCfgPPO


class MC100HzMinimalCfg(MCRoughCfg):
    class control(MCRoughCfg.control):
        # 200 Hz physics / 2 = 100 Hz policy/action update rate.
        decimation = 2


class MC100HzMinimalCfgPPO(MCRoughCfgPPO):
    class runner(MCRoughCfgPPO.runner):
        save_interval = 500
        max_iterations = 20000
        experiment_name = "MC100HzMinimal"
        run_name = "minimal_frequency_ablation"
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

        wandb_enabled = True
        wandb_project = "MC-HIMLoco"
        wandb_entity = None
        wandb_group = "mc-frequency-ablation"
        wandb_tags = [
            "MC",
            "HIMLoco",
            "wheel-legged",
            "100Hz",
            "fixed-PD",
            "minimal-frequency-ablation",
            "history-6",
        ]
        wandb_mode = "online"
