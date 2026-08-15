"""100 Hz MC experiment with minimal contact-triggered leg buffering.

This experiment is intentionally identical to the successful MC100HzMinimal
training setup except for the low-level contact-aware HIP/KNEE gain schedule.
"""

from .mc_100hz_minimal_config import MC100HzMinimalCfg, MC100HzMinimalCfgPPO


class MCBuffered100HzCfg(MC100HzMinimalCfg):
    class buffer_control:
        enabled = True
        # Flat positive loading-rate p99 is about 14-15 kN/s, while the
        # downstairs tail is much larger (~31 kN/s for B1 model_10000).
        # Start above normal rolling so the controller mainly reacts to impacts.
        loading_rate_threshold_nps = 20000.0
        contact_on_threshold_n = 5.0
        # 30 ms = 6 physics steps at 200 Hz.
        hold_time_s = 0.030
        # Only HIP/KNEE are softened; ABAD and wheel gains stay unchanged.
        hip_knee_kp_scale = 0.60
        hip_knee_kd_scale = 1.50


class MCBuffered100HzCfgPPO(MC100HzMinimalCfgPPO):
    class runner(MC100HzMinimalCfgPPO.runner):
        experiment_name = "MC100HzBufferMinimal"
        run_name = "contact_buffer"
        save_interval = 500
        max_iterations = 20000
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

        wandb_enabled = True
        wandb_project = "MC-HIMLoco"
        wandb_entity = None
        wandb_group = "mc-buffer-ablation"
        wandb_tags = [
            "MC",
            "HIMLoco",
            "wheel-legged",
            "100Hz",
            "history-6",
            "contact-buffer",
            "gain-scheduling",
            "minimal-frequency-base",
        ]
        wandb_mode = "online"
