from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot

from legged_gym.envs.go2w.go2w_config import GO2WRoughCfg, GO2WRoughCfgPPO
from .go2w.go2w_robot import Go2w

from legged_gym.envs.mc.mc_config import MCRoughCfg, MCRoughCfgPPO
from .mc.mc_robot import MC
from legged_gym.envs.mc.quiet_mc_config import QuietMCCfg, QuietMCCfgPPO
from .mc.quiet_mc_robot import QuietMC
from .mc.quiet_mc_v2_robot import QuietMCV2

import os

from legged_gym.utils.task_registry import task_registry

task_registry.register("go2w", Go2w, GO2WRoughCfg(), GO2WRoughCfgPPO())
task_registry.register("mc", MC, MCRoughCfg(), MCRoughCfgPPO())
task_registry.register("quiet_mc", QuietMC, QuietMCCfg(), QuietMCCfgPPO())
task_registry.register("quiet_mc_v2", QuietMCV2, QuietMCCfg(), QuietMCCfgPPO())
