# Copyright (c) 2026-2027, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .joint_pos_env_cfg import Rizon4sReachEnvCfg


@configclass
class Rizon4sReachROSInferenceEnvCfg(Rizon4sReachEnvCfg):
    """Exposing variables for ROS inferences"""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Variables used by Isaac Manipulator for on robot inference
        # TODO: @ashwinvk: Remove these from env cfg once the generic inference node has been implemented
        self.obs_order = ["arm_dof_pos", "arm_dof_vel", "target_pos", "target_quat"]
        self.policy_action_space = "joint"
        self.arm_joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        self.policy_action_space = "joint"
        self.action_space = 7
        self.state_space = 21
        self.observation_space = 21

        # Set joint_action_scale from the existing arm_action.scale
        self.joint_action_scale = self.actions.arm_action.scale

        self.action_scale_joint_space = [
            self.joint_action_scale,
            self.joint_action_scale,
            self.joint_action_scale,
            self.joint_action_scale,
            self.joint_action_scale,
            self.joint_action_scale,
            self.joint_action_scale,
        ]

        # Extract initial joint positions from robot configuration
        self.initial_joint_pos = [
            self.scene.robot.init_state.joint_pos[joint_name] for joint_name in self.arm_joint_names
        ]
