"""100 Hz MC HIMLoco task with learned sensorless admittance.

The physical robot/controller remains 16-DOF. The policy has four additional
outputs that modulate per-leg admittance, while simulator contact force remains
training-only privileged information.
"""

from .mc_100hz_config import MC100HzCfg, MC100HzCfgPPO


class MCLearnedAdmittance100HzCfg(MC100HzCfg):
    class env(MC100HzCfg.env):
        num_actions = 16
        num_motion_actions = 16
        num_compliance_actions = 4
        num_policy_actions = 20
        controller_state_dim = 12
        contact_estimate_dim = 8

    class learned_admittance:
        enabled = True
        contact_force_scale_n = 100.0
        contact_loading_rate_scale_nps = 10000.0
        contact_target_clip = 5.0

        virtual_mass_kg = 1.9
        damping_ratio = 1.0
        min_stiffness_n_per_m = 2500.0
        max_stiffness_n_per_m = 8000.0

        force_bias_time_constant_s = 0.10
        force_deadband_n = 10.0
        loading_rate_gate_nps = 5000.0
        loading_rate_gate_softness_nps = 2500.0
        max_force_input_n = 250.0

        max_compression_m = 0.020
        max_compression_velocity_mps = 0.15
        max_joint_offset_rad = 0.20

        upper_leg_length_m = 0.20
        lower_leg_length_m = 0.22
        jacobian_damping = 0.02

    class quiet_training:
        force_threshold_n = 60.0
        force_normalizer_n = 100.0
        loading_rate_threshold_nps = 5000.0
        loading_rate_normalizer_nps = 10000.0
        base_acc_normalizer_mps2 = 9.81

    class rewards(MC100HzCfg.rewards):
        class scales(MC100HzCfg.rewards.scales):
            quiet_impact_force = -0.08
            quiet_loading_rate = -0.10
            quiet_base_acc = -0.03
            compliance_usage = -0.02
            admittance_displacement = -0.02


class MCLearnedAdmittance100HzCfgPPO(MC100HzCfgPPO):
    class policy(MC100HzCfgPPO.policy):
        contact_estimator_hidden_dims = [128, 64]
        contact_estimator_lr = 1.0e-3
        contact_estimator_loss_force = 1.0
        contact_estimator_loss_loading = 0.5
        motion_adapter_scale = 0.05
        # Keep initial compliance exploration small. At deterministic inference
        # the zero-initialized compliance head gives exactly alpha=0.
        compliance_init_std = 0.05

    class runner(MC100HzCfgPPO.runner):
        policy_class_name = "AdaptiveHIMActorCritic"
        algorithm_class_name = "AdaptiveHIMPPO"
        runner_class_name = "AdaptiveHIMOnPolicyRunner"
        experiment_name = "MC_LearnedAdmittance_100Hz"
        run_name = "sensorless_admittance_v1"

        # Initialize the locomotion/HIM/critic tensors from the already trained
        # 100 Hz baseline. The adaptive runner performs shape-aware migration and
        # leaves new contact/compliance modules at their safe initialization.
        init_experiment_name = "MC_100Hz"
        init_load_run = -1
        init_checkpoint = -1

        wandb_group = "mc-learned-admittance"
        wandb_tags = [
            "MC",
            "HIMLoco",
            "wheel-legged",
            "100Hz",
            "sensorless-contact-estimator",
            "learned-admittance",
            "policy20-physical16",
        ]
