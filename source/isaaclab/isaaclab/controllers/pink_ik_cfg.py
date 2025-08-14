# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Pink IK controller."""

from dataclasses import MISSING

from pink.tasks import FrameTask

from isaaclab.utils import configclass


@configclass
class PinkIKControllerCfg:
    """Configuration settings for the Pink IK Controller.

    The Pink IK controller can be found at: https://github.com/stephane-caron/pink
    """

    urdf_path: str | None = None
    """Path to the robot's URDF file. This file is used by Pinocchio's `robot_wrapper.BuildFromURDF` to load the robot model."""

    mesh_path: str | None = None
    """Path to the mesh files associated with the robot. These files are also loaded by Pinocchio's `robot_wrapper.BuildFromURDF`."""

    num_hand_joints: int = 0
    """The number of hand joints in the robot. The action space for the controller contains the pose_dim(7)*num_controlled_frames + num_hand_joints.
    The last num_hand_joints values of the action are the hand joint angles."""

    variable_input_tasks: list[FrameTask] = MISSING
    """
    A list of tasks for the Pink IK controller. These tasks are controllable by the env action.

    These tasks can be used to control the pose of a frame or the angles of joints.
    For more details, visit: https://github.com/stephane-caron/pink
    """

    fixed_input_tasks: list[FrameTask] = MISSING
    """
    A list of tasks for the Pink IK controller. These tasks are fixed and not controllable by the env action.

    These tasks can be used to fix the pose of a frame or the angles of joints to a desired configuration.
    For more details, visit: https://github.com/stephane-caron/pink
    """

    joint_names: list[str] | None = None
    """A list of joint names in the USD asset. This is required because the joint naming conventions differ between USD and URDF files.
    This value is currently designed to be automatically populated by the action term in a manager based environment."""

    articulation_name: str = "robot"
    """The name of the articulation USD asset in the scene."""

    base_link_name: str = "base_link"
    """The name of the base link in the USD asset."""

    show_ik_warnings: bool = True
    """Show warning if IK solver fails to find a solution."""
