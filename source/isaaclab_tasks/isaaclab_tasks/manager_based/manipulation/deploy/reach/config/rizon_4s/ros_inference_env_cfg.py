# Copyright (c) 2026-2027, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

from .joint_pos_env_cfg import Rizon4sReachEnvCfg


@configclass
class Rizon4sReachROSInferenceEnvCfg(Rizon4sReachEnvCfg):
    """ROS / Isaac Manipulator inference fields plus deployment alignment for NVIDIA Hubble Lab.

    The Hubble-specific block in this config matches how the Flexiv Rizon 4s is mounted there.
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # --- NVIDIA Hubble Lab: Flexiv Rizon 4s mount and workspace ---
        # Remove vertical mount stand since Hubble deployment does not use the sim stand asset
        self.scene.table = None

        # Lab home joint pose (radians); aligns sim defaults / reset with the physical stand
        self.scene.robot.init_state.joint_pos = {
            "joint1": math.radians(-90.0),
            "joint2": math.radians(90.0),
            "joint3": 0.0,
            "joint4": math.radians(90.0),
            "joint5": 0.0,
            "joint6": 0.0,
            "joint7": 0.0,
        }

        # Orientation of robot is based on the Flexiv Rizon 4s mount in the Hubble Lab
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.0)
        self.scene.robot.init_state.rot = (0.5, 0.5, 0.5, 0.5)

        # end-effector is along z-direction for Rizon 4s
        # target_pos_centre and target_rot_centre are approximately the end effector pose when
        # the robot is in the self.scene.robot.init_state.joint_pos pose
        self.target_pos_centre = (0.0, 0.3, 0.9)
        self.target_pos_range = (0.4, 0.4, 0.35)
        self.commands.ee_pose.body_name = "flange"
        self.commands.ee_pose.ranges.pos_x = (
            self.target_pos_centre[0] - self.target_pos_range[0],
            self.target_pos_centre[0] + self.target_pos_range[0],
        )
        self.commands.ee_pose.ranges.pos_y = (
            self.target_pos_centre[1] - self.target_pos_range[1],
            self.target_pos_centre[1] + self.target_pos_range[1],
        )
        self.commands.ee_pose.ranges.pos_z = (
            self.target_pos_centre[2] - self.target_pos_range[2],
            self.target_pos_centre[2] + self.target_pos_range[2],
        )

        self.target_rot_centre = (math.pi / 2, math.pi / 2, 0.0)  # end-effector facing down
        self.target_rot_range = (math.pi / 2, math.pi / 2, math.pi)
        self.commands.ee_pose.ranges.roll = (
            self.target_rot_centre[0] - self.target_rot_range[0],
            self.target_rot_centre[0] + self.target_rot_range[0],
        )
        self.commands.ee_pose.ranges.pitch = (
            self.target_rot_centre[1] - self.target_rot_range[1],
            self.target_rot_centre[1] + self.target_rot_range[1],
        )
        self.commands.ee_pose.ranges.yaw = (
            self.target_rot_centre[2] - self.target_rot_range[2],
            self.target_rot_centre[2] + self.target_rot_range[2],
        )

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
