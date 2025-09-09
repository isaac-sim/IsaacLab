# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.controllers.pink_ik import PinkIKControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import pink_task_space_actions


@configclass
class PinkInverseKinematicsActionCfg(ActionTermCfg):
    """Configuration for Pink inverse kinematics action term.

    This configuration is used to define settings for the Pink inverse kinematics action term,
    which is a inverse kinematics framework.
    """

    class_type: type[ActionTerm] = pink_task_space_actions.PinkInverseKinematicsAction
    """Specifies the action term class type for Pink inverse kinematics action."""

    pink_controlled_joint_names: list[str] = MISSING
    """List of joint names or regular expression patterns that specify the joints controlled by pink IK."""

    ik_urdf_fixed_joint_names: list[str] = MISSING
    """List of joint names that specify the joints to be locked in URDF."""

    hand_joint_names: list[str] = MISSING
    """List of joint names or regular expression patterns that specify the joints controlled by hand retargeting."""

    controller: PinkIKControllerCfg = MISSING
    """Configuration for the Pink IK controller that will be used to solve the inverse kinematics."""

    target_eef_link_names: dict[str, str] = MISSING
    """Dictionary mapping task names to controlled link names for the Pink IK controller.

    This dictionary should map the task names (e.g., 'left_wrist', 'right_wrist') to the
    corresponding link names in the URDF that will be controlled by the IK solver.
    """
