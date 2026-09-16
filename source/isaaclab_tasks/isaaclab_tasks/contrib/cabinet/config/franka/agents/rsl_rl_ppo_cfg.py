# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any

from isaaclab.utils import config_field

from isaaclab_tasks.core.cabinet.config.franka.agents.rsl_rl_ppo_cfg import CabinetPPORunnerCfg


@dataclass
class FrankaCabinetIKAbsPPORunnerCfg(CabinetPPORunnerCfg):
    experiment_name: Any = config_field("franka_open_drawer_ik_abs")


@dataclass
class FrankaCabinetIKRelPPORunnerCfg(CabinetPPORunnerCfg):
    experiment_name: Any = config_field("franka_open_drawer_ik_rel")
