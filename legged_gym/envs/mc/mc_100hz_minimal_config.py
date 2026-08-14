"""Minimal 100 Hz MC HIMLoco frequency ablation.

This task deliberately changes ONLY the policy/control update frequency relative
 to the validated 50 Hz MC baseline:

* physics remains 200 Hz (sim.dt = 0.005 s)
* control decimation changes 4 -> 2 (policy 50 -> 100 Hz)

Everything else is inherited unchanged from the validated baseline, including:
* six-frame HIM history (342 estimator inputs)
* 48 rollout steps per environment
* gamma=0.99 and lambda=0.95
* policy/critic/estimator architecture
* exploration noise
* fixed PD gains, rewards, terrain and domain randomization

The purpose is diagnostic: establish whether 100 Hz itself trains normally
before introducing any time-matched history/rollout changes.
"""

from .mc_config import MCRoughCfg, MCRoughCfgPPO


class MC100HzMinimalCfg(MCRoughCfg):
    class control(MCRoughCfg.control):
        # 200 Hz physics / 2 = 100 Hz policy/action update rate.
        decimation = 2


class MC100HzMinimalCfgPPO(MCRoughCfgPPO):
    class runner(MCRoughCfgPPO.runner):
        save_interval = 500
        max_iterations = 8000
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
