# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# -------------------------------- 上游原始注册 ---------------------------------
from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR, WHEEL_LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from .wheel_legged.wheel_legged_config import WheelLeggedCfg, WheelLeggedCfgPPO
from .wheel_legged_vmc.wheel_legged_vmc import LeggedRobotVMC
from .wheel_legged_vmc.wheel_legged_vmc_config import (
    WheelLeggedVMCCfg,
    WheelLeggedVMCCfgPPO,
)
from .wheel_legged_vmc_flat.wheel_legged_vmc_flat_config import (
    WheelLeggedVMCFlatCfg,
    WheelLeggedVMCFlatCfgPPO,
)

# -------------------------------- 残差 RL (本项目新增) ---------------------------
from .wheel_legged_vmc_flat.wheel_legged_residual_flat import LeggedRobotResidual
from .wheel_legged_vmc_flat.wheel_legged_residual_flat_config import (
    WheelLeggedResidualFlatCfg,
    WheelLeggedResidualFlatCfgPPO,
)

import os
from wheel_legged_gym.utils.task_registry import task_registry

task_registry.register(
    "wheel_legged", LeggedRobot, WheelLeggedCfg(), WheelLeggedCfgPPO()
)
task_registry.register(
    "wheel_legged_vmc", LeggedRobotVMC, WheelLeggedVMCCfg(), WheelLeggedVMCCfgPPO()
)
task_registry.register(
    "wheel_legged_vmc_flat",
    LeggedRobotVMC,
    WheelLeggedVMCFlatCfg(),
    WheelLeggedVMCFlatCfgPPO(),
)
task_registry.register(
    "wheel_legged_residual_flat",
    LeggedRobotResidual,
    WheelLeggedResidualFlatCfg(),
    WheelLeggedResidualFlatCfgPPO(),
)
