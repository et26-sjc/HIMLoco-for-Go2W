from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot

from legged_gym.envs.go2w.go2w_config import GO2WRoughCfg, GO2WRoughCfgPPO
from .go2w.go2w_robot import Go2w

from legged_gym.envs.mc.mc_config import MCRoughCfg, MCRoughCfgPPO
from .mc.mc_robot import MC
from legged_gym.envs.mc.quiet_mc_config import QuietMCCfg, QuietMCCfgPPO
from .mc.quiet_mc_robot import QuietMC
from .mc.quiet_mc_v2_robot import QuietMCV2
from .mc.mc_100hz_config import MC100HzCfg, MC100HzCfgPPO
from .mc.mc_100hz_minimal_config import MC100HzMinimalCfg, MC100HzMinimalCfgPPO
from .mc.quiet_mc_100hz_config import QuietMC100HzCfg, QuietMC100HzCfgPPO
from .mc.quiet_mc_100hz_minimal_config import (
    QuietMC100HzMinimalCfg,
    QuietMC100HzMinimalCfgPPO,
)
from .mc.mc_admittance_100hz_robot import MCAdmittance100Hz
from .mc.mc_admittance_100hz_config import (
    MCAdmittance100HzCfg,
    MCAdmittance100HzCfgPPO,
)
from .mc.quiet_mc_admittance_100hz_robot import QuietMCAdmittance100HzV2
from .mc.quiet_mc_admittance_100hz_config import (
    QuietMCAdmittance100HzCfg,
    QuietMCAdmittance100HzCfgPPO,
    QuietMCAdmittance100HzZeroShotCfgPPO,
)

import os

from legged_gym.utils.task_registry import task_registry

task_registry.register("go2w", Go2w, GO2WRoughCfg(), GO2WRoughCfgPPO())
task_registry.register("mc", MC, MCRoughCfg(), MCRoughCfgPPO())
task_registry.register("quiet_mc", QuietMC, QuietMCCfg(), QuietMCCfgPPO())
task_registry.register("quiet_mc_v2", QuietMCV2, QuietMCCfg(), QuietMCCfgPPO())
task_registry.register("mc_100hz", MC, MC100HzCfg(), MC100HzCfgPPO())
task_registry.register(
    "mc_100hz_minimal",
    MC,
    MC100HzMinimalCfg(),
    MC100HzMinimalCfgPPO(),
)
task_registry.register(
    "quiet_mc_100hz_v2",
    QuietMCV2,
    QuietMC100HzCfg(),
    QuietMC100HzCfgPPO(),
)
task_registry.register(
    "quiet_mc_100hz_minimal_v2",
    QuietMCV2,
    QuietMC100HzMinimalCfg(),
    QuietMC100HzMinimalCfgPPO(),
)
task_registry.register(
    "mc_100hz_admittance",
    MCAdmittance100Hz,
    MCAdmittance100HzCfg(),
    MCAdmittance100HzCfgPPO(),
)
task_registry.register(
    "quiet_mc_100hz_admittance_zeroshot_v2",
    QuietMCAdmittance100HzV2,
    QuietMCAdmittance100HzCfg(),
    QuietMCAdmittance100HzZeroShotCfgPPO(),
)
task_registry.register(
    "quiet_mc_100hz_admittance_v2",
    QuietMCAdmittance100HzV2,
    QuietMCAdmittance100HzCfg(),
    QuietMCAdmittance100HzCfgPPO(),
)
