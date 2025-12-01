# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from dataclasses import MISSING

from isaaclab.controllers.rmp_flow import RmpFlowControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import rmpflow_task_space_actions


@configclass
class RMPFlowActionCfg(ActionTermCfg):
    @configclass
    class OffsetCfg:
        """The offset pose from parent frame to child frame.

        On many robots, end-effector frames are fictitious frames that do not have a corresponding
        rigid body. In such cases, it is easier to define this transform w.r.t. their parent rigid body.
        For instance, for the Franka Emika arm, the end-effector is defined at an offset to the the
        "panda_hand" frame.
        """

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    class_type: type[ActionTerm] = rmpflow_task_space_actions.RMPFlowAction

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    body_name: str = MISSING
    """Name of the body or frame for which IK is performed."""
    body_offset: OffsetCfg | None = None
    """Offset of target frame w.r.t. to the body frame. Defaults to None, in which case no offset is applied."""
    scale: float | tuple[float, ...] = 1.0

    controller: RmpFlowControllerCfg = MISSING

    articulation_prim_expr: str = MISSING  # The expression to find the articulation prim paths.
    """The configuration for the RMPFlow controller."""

    use_relative_mode: bool = False
    """
    Defaults to False.
    If True, then the controller treats the input command as a delta change in the position/pose.
    Otherwise, the controller treats the input command as the absolute position/pose.
    """
