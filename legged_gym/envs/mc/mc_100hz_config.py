"""100 Hz policy-frequency MC HIMLoco experiment.

This configuration changes only temporal resolution relative to MCRoughCfg:

* physics remains 200 Hz (sim.dt = 0.005 s);
* policy/control decimation changes 4 -> 2 (50 -> 100 Hz);
* actor history changes 6 -> 12 frames, preserving ~120 ms context;
* PPO rollout changes 48 -> 96 steps, preserving ~0.96 s physical horizon;
* gamma/lambda are square-root rescaled so their real-time decay matches the
  50 Hz baseline approximately.

Robot, terrain, commands, action semantics, reward definitions and fixed PD
parameters are inherited unchanged.
"""

import math

from .mc_config import MCRoughCfg, MCRoughCfgPPO


class MC100HzCfg(MCRoughCfg):
    class env(MCRoughCfg.env):
        num_one_step_observations = MCRoughCfg.env.num_one_step_observations
        num_observations = num_one_step_observations * 12
        num_one_step_privileged_obs = MCRoughCfg.env.num_one_step_privileged_obs
        num_privileged_obs = num_one_step_privileged_obs
        num_actions = MCRoughCfg.env.num_actions

    class control(MCRoughCfg.control):
        # 200 Hz physics / 2 = 100 Hz policy and action update rate.
        decimation = 2


class MC100HzCfgPPO(MCRoughCfgPPO):
    class algorithm(MCRoughCfgPPO.algorithm):
        # Preserve approximately the same discount/GAE decay per second:
        # gamma_100^2 = gamma_50 and lambda_100^2 = lambda_50.
        gamma = math.sqrt(0.99)
        lam = math.sqrt(0.95)

    class runner(MCRoughCfgPPO.runner):
        # 96 * 0.01 s = 0.96 s, equal to 48 * 0.02 s in the 50 Hz baseline.
        num_steps_per_env = 96
        save_interval = 500
        max_iterations = 20000
        experiment_name = "MC100Hz"
        run_name = "fixed_pd_100hz"
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
            "frequency-ablation",
        ]
        wandb_mode = "online"
