"""100 Hz MC HIMLoco task with impact-aware sensorless admittance.

The physical robot/controller remains 16-DOF. The policy has four additional
outputs that modulate per-leg admittance. Contact force remains training-only
privileged information. The estimator learns current impact state rather than
predicting future terrain interaction.
"""

from .mc_100hz_config import MC100HzCfg, MC100HzCfgPPO


class MCLearnedAdmittance100HzCfg(MC100HzCfg):
    class env(MC100HzCfg.env):
        num_actions = 16
        num_motion_actions = 16
        num_compliance_actions = 4
        num_policy_actions = 20
        controller_state_dim = 16
        contact_estimate_dim = 8

    class learned_admittance:
        enabled = True
        contact_force_scale_n = 100.0
        contact_target_clip = 5.0

        virtual_mass_kg = 1.9
        damping_ratio = 1.0
        min_stiffness_n_per_m = 2500.0
        max_stiffness_n_per_m = 8000.0

        compliance_activation_gain = 6.0

        # Impact-aware gate replaces loading-rate gate. The estimator outputs
        # current impact probability (not future prediction).
        impact_gate_gain = 2.0
        max_force_input_n = 250.0

        max_compression_m = 0.020
        max_compression_velocity_mps = 0.15
        max_joint_offset_rad = 0.20

        upper_leg_length_m = 0.20
        lower_leg_length_m = 0.22
        jacobian_damping = 0.02

        diagnostic_alpha_active_threshold = 0.05
        diagnostic_gate_active_threshold = 0.50
        diagnostic_force_event_threshold_n = 60.0
        diagnostic_impact_threshold = 0.50

    class quiet_training(MC100HzCfg.quiet_training):
        pass

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
        contact_estimator_loss_impact = 1.0
        motion_adapter_scale = 0.0
        compliance_init_std = 0.15

    class algorithm(MC100HzCfgPPO.algorithm):
        base_actor_lr_scale = 0.0
        action_std_lr_scale = 0.0
        update_him_estimator = False

    class runner(MC100HzCfgPPO.runner):
        policy_class_name = "AdaptiveHIMActorCritic"
        algorithm_class_name = "AdaptiveHIMPPO"
        runner_class_name = "AdaptiveHIMOnPolicyRunner"
        experiment_name = "MC_ImpactAwareAdmittance_100Hz"
        run_name = "impact_classifier_v1"

        init_experiment_name = "MC_100Hz"
        init_load_run = -1
        init_checkpoint = -1

        contact_pretrain_steps = 500
        contact_pretrain_log_interval = 50

        wandb_group = "mc-impact-aware-admittance"
        wandb_tags = [
            "MC",
            "HIMLoco",
            "impact-aware",
            "sensorless-contact-estimator",
            "classification-gate",
            "policy20-physical16",
        ]
