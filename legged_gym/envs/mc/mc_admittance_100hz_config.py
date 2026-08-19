"""100 Hz MC HIMLoco with per-leg axial admittance.

The successful MC100HzMinimal task is kept unchanged at the policy/HIM/PPO
level.  The only new element is a 200 Hz low-level outer admittance loop that
turns transient wheel contact force into a small virtual leg compression.  The
resulting Cartesian displacement is mapped to HIP/KNEE target offsets and is
then executed by the original fixed PD controller.
"""

from .mc_100hz_minimal_config import MC100HzMinimalCfg, MC100HzMinimalCfgPPO


class MCAdmittance100HzCfg(MC100HzMinimalCfg):
    class admittance_control:
        enabled = True

        # Virtual axial mass-spring-damper:
        #   M * d2x + D * dx + K * x = F_transient
        virtual_mass_kg = 1.9
        virtual_damping_ns_per_m = 210.0
        virtual_stiffness_n_per_m = 6000.0

        # Only fast excess force above a slowly varying support-force baseline
        # drives the admittance.  This keeps flat rolling close to the original
        # fixed-PD policy.
        force_bias_time_constant_s = 0.10
        force_deadband_n = 15.0
        max_force_input_n = 250.0

        # Safety bounds for the virtual compliance state.
        max_compression_m = 0.020
        max_compression_velocity_mps = 0.40
        max_joint_offset_rad = 0.25

        # MC sagittal leg geometry from bot_mc.urdf:
        # HIP_JOINT -> KNEE_JOINT = 0.20 m
        # KNEE_JOINT -> FOOT_JOINT = 0.22 m
        upper_leg_length_m = 0.20
        lower_leg_length_m = 0.22

        # Damped least-squares Jacobian inverse regularization.
        jacobian_damping = 0.02


class MCAdmittance100HzCfgPPO(MC100HzMinimalCfgPPO):
    class runner(MC100HzMinimalCfgPPO.runner):
        save_interval = 500
        max_iterations = 20000
        experiment_name = "MC100HzAdmittance"
        run_name = "axial_admittance"
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

        wandb_enabled = True
        wandb_project = "MC-HIMLoco"
        wandb_entity = None
        wandb_group = "mc-admittance-ablation"
        wandb_tags = [
            "MC",
            "HIMLoco",
            "wheel-legged",
            "100Hz",
            "admittance",
            "axial-leg-compliance",
            "history-6",
        ]
        wandb_mode = "online"
