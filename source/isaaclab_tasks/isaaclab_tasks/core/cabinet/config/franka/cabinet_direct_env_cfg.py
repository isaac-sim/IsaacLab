# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the direct-workflow Franka cabinet-opening environment."""

from __future__ import annotations

from isaaclab.envs.mdp.actions import EMAJointPositionToLimitsActionCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.utils import preset

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

from ...cabinet_direct_env_cfg import CabinetDirectEnvCfg, CabinetDirectSceneCfg


@configclass
class FrankaCabinetDirectSceneCfg(CabinetDirectSceneCfg):
    """Direct-workflow cabinet scene configured for the Franka robot."""

    robot = replace(FRANKA_PANDA_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.variants["Physics"] = preset(default="mujoco", isaacsim_physx="physx", physx="physx", ovphysx="physx")


@configclass
class FrankaCabinetDirectEnvCfg(CabinetDirectEnvCfg):
    """Direct-workflow cabinet task with a Franka Panda arm."""

    scene: FrankaCabinetDirectSceneCfg = FrankaCabinetDirectSceneCfg(num_envs=4096, env_spacing=2.0)

    arm_action = EMAJointPositionToLimitsActionCfg(
        asset_name="robot", joint_names=["panda_joint.*"], scale=1.0, rescale_to_limits=True, alpha=0.01
    )
    finger_joint_names: str | list[str] = "panda_finger_joint.*"
    ee_body_name: str = "panda_hand"
    left_finger_body_name: str = "panda_leftfinger"
    right_finger_body_name: str = "panda_rightfinger"
    ee_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.1034)
    finger_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.046)

    gripper_open_command: float = 0.04
    gripper_close_command: float = 0.0
    approach_gripper_handle_offset: float = 0.04

    episode_length_s = 16.0
    action_rate_reward_scale = -0.05
    joint_vel_reward_scale = -5.0
    joint_pos_limits_reward_scale = -10.0
    joint_vel_limits_reward_scale = -100.0

    def __post_init__(self):
        # Match the manager-based Franka task's reset clearance and arm control.
        self.scene.robot.soft_joint_pos_limit_factor = 0.9
        self.scene.robot.init_state.joint_pos["panda_joint2"] = -1.1
        self.scene.robot.actuators["panda_arm"].damping = 80.0
        # MJWarp does not enforce this limit; the velocity-limit reward penalizes excess speed.
        self.scene.robot.actuators["panda_arm"].joint_velocity_limit = 0.3
