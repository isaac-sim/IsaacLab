# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaacsim.core.utils.extensions import get_extension_path_from_name

from isaaclab.controllers.rmp_flow import RmpFlowControllerCfg

# Note: RMP-Flow config files for supported robots are stored in the motion_generation extension
_RMP_CONFIG_DIR = os.path.join(
    get_extension_path_from_name("isaacsim.robot_motion.motion_generation"), "motion_policy_configs"
)

# Path to current directory
_CUR_DIR = os.path.dirname(os.path.realpath(__file__))

FRANKA_RMPFLOW_CFG = RmpFlowControllerCfg(
    config_file=os.path.join(_RMP_CONFIG_DIR, "franka", "rmpflow", "franka_rmpflow_common.yaml"),
    urdf_file=os.path.join(_CUR_DIR, "data", "lula_franka_gen.urdf"),
    collision_file=os.path.join(_RMP_CONFIG_DIR, "franka", "rmpflow", "robot_descriptor.yaml"),
    frame_name="panda_end_effector",
    evaluations_per_frame=5,
)
"""Configuration of RMPFlow for Franka arm (default from `isaacsim.robot_motion.motion_generation`)."""


UR10_RMPFLOW_CFG = RmpFlowControllerCfg(
    config_file=os.path.join(_RMP_CONFIG_DIR, "ur10", "rmpflow", "ur10_rmpflow_config.yaml"),
    urdf_file=os.path.join(_RMP_CONFIG_DIR, "ur10", "ur10_robot.urdf"),
    collision_file=os.path.join(_RMP_CONFIG_DIR, "ur10", "rmpflow", "ur10_robot_description.yaml"),
    frame_name="ee_link",
    evaluations_per_frame=5,
)
"""Configuration of RMPFlow for UR10 arm (default from `isaacsim.robot_motion.motion_generation`)."""
