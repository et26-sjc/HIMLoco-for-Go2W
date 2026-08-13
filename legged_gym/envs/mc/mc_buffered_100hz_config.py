"""100 Hz MC experiment with minimal contact-triggered leg buffering."""

from .mc_100hz_config import MC100HzCfg, MC100HzCfgPPO


class MCBuffered100HzCfg(MC100HzCfg):
    class buffer_control:
        enabled = True
        # Flat baseline positive loading-rate p99 is about 14 kN/s; stair tails
        # are much larger.  Start above the flat tail to avoid softening during
        # ordinary rolling.
        loading_rate_threshold_nps = 20000.0
        contact_on_threshold_n = 5.0
        # 30 ms = 6 physics steps at 200 Hz.
        hold_time_s = 0.030
        # Only HIP/KNEE are softened; ABAD and wheel gains stay unchanged.
        hip_knee_kp_scale = 0.60
        hip_knee_kd_scale = 1.50


class MCBuffered100HzCfgPPO(MC100HzCfgPPO):
    class runner(MC100HzCfgPPO.runner):
        experiment_name = "MC100HzBuffer"
        run_name = "contact_buffer"
        save_interval = 500
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
            "contact-buffer",
            "gain-scheduling",
        ]
        wandb_mode = "online"
