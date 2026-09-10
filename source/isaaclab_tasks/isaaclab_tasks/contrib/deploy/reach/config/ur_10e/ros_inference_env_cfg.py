# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .joint_pos_env_cfg import UR10eReachEnvCfg


@configclass
class UR10eReachROSInferenceEnvCfg(UR10eReachEnvCfg):
    """Exposing variables for ROS inferences"""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Variables used by Isaac Manipuulator for on robot inference
        # TODO: @ashwinvk: Remove these from env cfg once the generic inference node has been implemented
        self.obs_order = ["arm_dof_pos", "arm_dof_vel", "target_pos", "target_quat"]
        self.policy_action_space = "joint"
        self.arm_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.action_space = 6
        self.state_space = 19
        self.observation_space = 19

        # Set joint_action_scale from the existing arm_action.scale
        self.joint_action_scale = self.actions.arm_action.scale

        # Dynamically generate action_scale_joint_space based on action_space
        self.action_scale_joint_space = [self.joint_action_scale] * self.action_space

        # Extract initial joint positions from robot configuration
        # Convert joint_pos dict to list in the order specified by arm_joint_names
        self.initial_joint_pos = [
            self.scene.robot.init_state.joint_pos[joint_name] for joint_name in self.arm_joint_names
        ]
