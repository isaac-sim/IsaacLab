# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_tasks.core.cabinet.cabinet_direct_env_cfg import CabinetDirectEnvCfg, CabinetDirectSceneCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


@configclass
class FrankaCabinetDirectSceneCfg(CabinetDirectSceneCfg):
    """Direct-workflow cabinet scene configured for the Franka robot."""

    robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class FrankaCabinetDirectEnvCfg(CabinetDirectEnvCfg):
    """Direct-workflow cabinet task with a Franka Panda arm."""

    scene: FrankaCabinetDirectSceneCfg = FrankaCabinetDirectSceneCfg(num_envs=4096, env_spacing=2.0)

    arm_joint_names: str | list[str] = "panda_joint.*"
    finger_joint_names: str | list[str] = "panda_finger_joint.*"
    ee_body_name: str = "panda_hand"
    left_finger_body_name: str = "panda_leftfinger"
    right_finger_body_name: str = "panda_rightfinger"
    ee_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.1034)
    finger_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.046)

    gripper_open_command: float = 0.04
    gripper_close_command: float = 0.0
    approach_gripper_handle_offset: float = 0.04
