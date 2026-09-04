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
        controller_state_dim = 16
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

        # Raw RL alpha stays in [0, 1], but a concave authority mapping gives
        # small non-zero compliance intents enough physical authority to be
        # observable during Stage-1 learning while preserving alpha=0 -> exact
        # baseline behaviour:
        #     beta = 1 - exp(-gain * alpha)
        compliance_activation_gain = 6.0

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

        # Diagnostic thresholds only. These values are never fed back into the
        # actor/critic/reward path; they only define W&B summary counters.
        diagnostic_alpha_active_threshold = 0.05
        diagnostic_gate_active_threshold = 0.50
        diagnostic_force_event_threshold_n = 60.0
        diagnostic_loading_event_threshold_nps = 5000.0

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
        motion_adapter_scale = 0.0
        # 0.05 was too conservative: after clipping negative samples to zero the
        # admittance almost never experienced a physically meaningful response.
        # 0.15 keeps Stage-1 exploration small while making compliance observable.
        compliance_init_std = 0.15

    class algorithm(MC100HzCfgPPO.algorithm):
        # Stage 1 freezes every baseline-policy degree of freedom. The four new
        # compliance dimensions use a fixed small exploration std=0.15.
        base_actor_lr_scale = 0.0
        action_std_lr_scale = 0.0
        update_him_estimator = False

    class runner(MC100HzCfgPPO.runner):
        policy_class_name = "AdaptiveHIMActorCritic"
        algorithm_class_name = "AdaptiveHIMPPO"
        runner_class_name = "AdaptiveHIMOnPolicyRunner"
        experiment_name = "MC_LearnedAdmittance_100Hz"
        run_name = "sensorless_admittance_v1"

        init_experiment_name = "MC_100Hz"
        init_load_run = -1
        init_checkpoint = -1

        contact_pretrain_steps = 500
        contact_pretrain_log_interval = 50

        wandb_group = "mc-learned-admittance"
        wandb_tags = [
            "MC",
            "HIMLoco",
            "wheel-legged",
            "100Hz",
            "sensorless-contact-estimator",
            "learned-admittance",
            "policy20-physical16",
            "stage0-contact-warmup",
            "stage1-frozen-locomotion",
            "stage1-compliance-only",
            "compliance-std-0.15",
            "effective-alpha-gain-6",
            "raw-admittance-diagnostics",
        ]
