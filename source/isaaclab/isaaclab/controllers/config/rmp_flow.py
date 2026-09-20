# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab.controllers.rmp_flow_cfg import RmpFlowControllerCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# Directory on Nucleus Server for RMP-Flow assets (URDFs, collision models, etc.)
ISAACLAB_NUCLEUS_RMPFLOW_DIR = os.path.join(ISAACLAB_NUCLEUS_DIR, "Controllers", "RmpFlowAssets")

# Sentinel prefix for paths inside the isaacsim.robot_motion.motion_generation extension.
# RmpFlowController resolves these at init time (after enabling the extension) so that
# this cfg file stays free of any isaacsim/Kit imports.
_EXT_MOTION_CFG = "rmpflow_ext:motion_policy_configs/"

# Path to current directory
_CUR_DIR = os.path.dirname(os.path.realpath(__file__))

FRANKA_RMPFLOW_CFG = RmpFlowControllerCfg(
    config_file=_EXT_MOTION_CFG + "franka/rmpflow/franka_rmpflow_common.yaml",
    urdf_file=os.path.join(_CUR_DIR, "data", "lula_franka_gen.urdf"),
    collision_file=_EXT_MOTION_CFG + "franka/rmpflow/robot_descriptor.yaml",
    frame_name="panda_end_effector",
    evaluations_per_frame=5,
)
"""Configuration of RMPFlow for Franka arm (default from `isaacsim.robot_motion.motion_generation`)."""


UR10_RMPFLOW_CFG = RmpFlowControllerCfg(
    config_file=_EXT_MOTION_CFG + "ur10/rmpflow/ur10_rmpflow_config.yaml",
    urdf_file=_EXT_MOTION_CFG + "ur10/ur10_robot.urdf",
    collision_file=_EXT_MOTION_CFG + "ur10/rmpflow/ur10_robot_description.yaml",
    frame_name="ee_link",
    evaluations_per_frame=5,
)
"""Configuration of RMPFlow for UR10 arm (default from `isaacsim.robot_motion.motion_generation`)."""

GALBOT_LEFT_ARM_RMPFLOW_CFG = RmpFlowControllerCfg(
    config_file=os.path.join(
        ISAACLAB_NUCLEUS_RMPFLOW_DIR,
        "galbot_one_charlie",
        "rmpflow",
        "galbot_one_charlie_left_arm_rmpflow_config.yaml",
    ),
    urdf_file=os.path.join(ISAACLAB_NUCLEUS_RMPFLOW_DIR, "galbot_one_charlie", "galbot_one_charlie.urdf"),
    collision_file=os.path.join(
        ISAACLAB_NUCLEUS_RMPFLOW_DIR, "galbot_one_charlie", "rmpflow", "galbot_one_charlie_left_arm_gripper.yaml"
    ),
    frame_name="left_gripper_tcp_link",
    evaluations_per_frame=5,
    ignore_robot_state_updates=True,
)

GALBOT_RIGHT_ARM_RMPFLOW_CFG = RmpFlowControllerCfg(
    config_file=os.path.join(
        ISAACLAB_NUCLEUS_RMPFLOW_DIR,
        "galbot_one_charlie",
        "rmpflow",
        "galbot_one_charlie_right_arm_rmpflow_config.yaml",
    ),
    urdf_file=os.path.join(ISAACLAB_NUCLEUS_RMPFLOW_DIR, "galbot_one_charlie", "galbot_one_charlie.urdf"),
    collision_file=os.path.join(
        ISAACLAB_NUCLEUS_RMPFLOW_DIR, "galbot_one_charlie", "rmpflow", "galbot_one_charlie_right_arm_suction.yaml"
    ),
    frame_name="right_suction_cup_tcp_link",
    evaluations_per_frame=5,
    ignore_robot_state_updates=True,
)

"""Configuration of RMPFlow for Galbot humanoid."""

AGIBOT_LEFT_ARM_RMPFLOW_CFG = RmpFlowControllerCfg(
    config_file=os.path.join(ISAACLAB_NUCLEUS_RMPFLOW_DIR, "agibot", "rmpflow", "agibot_left_arm_rmpflow_config.yaml"),
    urdf_file=os.path.join(ISAACLAB_NUCLEUS_RMPFLOW_DIR, "agibot", "agibot.urdf"),
    collision_file=os.path.join(ISAACLAB_NUCLEUS_RMPFLOW_DIR, "agibot", "rmpflow", "agibot_left_arm_gripper.yaml"),
    frame_name="gripper_center",
    evaluations_per_frame=5,
    ignore_robot_state_updates=True,
)

AGIBOT_RIGHT_ARM_RMPFLOW_CFG = RmpFlowControllerCfg(
    config_file=os.path.join(ISAACLAB_NUCLEUS_RMPFLOW_DIR, "agibot", "rmpflow", "agibot_right_arm_rmpflow_config.yaml"),
    urdf_file=os.path.join(ISAACLAB_NUCLEUS_RMPFLOW_DIR, "agibot", "agibot.urdf"),
    collision_file=os.path.join(ISAACLAB_NUCLEUS_RMPFLOW_DIR, "agibot", "rmpflow", "agibot_right_arm_gripper.yaml"),
    frame_name="right_gripper_center",
    evaluations_per_frame=5,
    ignore_robot_state_updates=True,
)

"""Configuration of RMPFlow for Agibot humanoid."""
