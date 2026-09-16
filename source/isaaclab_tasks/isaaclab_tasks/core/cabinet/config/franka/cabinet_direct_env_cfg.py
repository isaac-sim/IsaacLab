# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from isaaclab.utils import config_field, replace_config

from isaaclab_tasks.core.cabinet.cabinet_direct_env_cfg import CabinetDirectEnvCfg, CabinetDirectSceneCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


@dataclass
class FrankaCabinetDirectSceneCfg(CabinetDirectSceneCfg):
    """Direct-workflow cabinet scene configured for the Franka robot."""

    robot: Any = config_field(replace_config(FRANKA_PANDA_CFG, prim_path="{ENV_REGEX_NS}/Robot"))


@dataclass
class FrankaCabinetDirectEnvCfg(CabinetDirectEnvCfg):
    """Direct-workflow cabinet task with a Franka Panda arm."""

    scene: FrankaCabinetDirectSceneCfg = config_field(FrankaCabinetDirectSceneCfg(num_envs=4096, env_spacing=2.0))

    arm_joint_names: str | list[str] = config_field("panda_joint.*")
    finger_joint_names: str | list[str] = config_field("panda_finger_joint.*")
    ee_body_name: str = config_field("panda_hand")
    left_finger_body_name: str = config_field("panda_leftfinger")
    right_finger_body_name: str = config_field("panda_rightfinger")
    ee_pos_offset: tuple[float, float, float] = config_field((0.0, 0.0, 0.1034))
    finger_pos_offset: tuple[float, float, float] = config_field((0.0, 0.0, 0.046))

    gripper_open_command: float = config_field(0.04)
    gripper_close_command: float = config_field(0.0)
    approach_gripper_handle_offset: float = config_field(0.04)
