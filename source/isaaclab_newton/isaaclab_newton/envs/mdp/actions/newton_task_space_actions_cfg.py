# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils import configclass

from isaaclab_newton.controllers.differential_ik_cfg import NewtonDifferentialIKControllerCfg
from isaaclab_newton.controllers.operational_space_cfg import NewtonOperationalSpaceControllerCfg

if TYPE_CHECKING:
    from .newton_task_space_actions import (
        NewtonDifferentialInverseKinematicsAction,
        NewtonOperationalSpaceControllerAction,
    )


@configclass
class NewtonDifferentialInverseKinematicsActionCfg(ActionTermCfg):
    """Configuration for :class:`NewtonDifferentialInverseKinematicsAction`.

    Commands and the Jacobian are expressed in the robot root frame. The action works with any physics backend.
    """

    class_type: type[NewtonDifferentialInverseKinematicsAction] | str = (
        "{DIR}.newton_task_space_actions:NewtonDifferentialInverseKinematicsAction"
    )

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""

    body_name: str = MISSING
    """Name of the body for which IK is performed."""

    body_offset: DifferentialInverseKinematicsActionCfg.OffsetCfg | None = None
    """Offset of the target frame from the body frame. Defaults to None, in which case no offset is applied."""

    command_type: Literal["position", "pose"] = MISSING
    """Whether the action commands the target position or pose.

    With ``"position"``, a controller :attr:`~NewtonDifferentialIKControllerCfg.axis_weight` left at ``None``
    solves for position only.
    """

    use_relative_mode: bool = False
    """Whether the action is a delta from the current pose (position and axis-angle) instead of an absolute pose.

    Absolute poses are ``(x, y, z, qx, qy, qz, qw)``.
    """

    scale: float | tuple[float, ...] = 1.0
    """Scale factor for the action. Defaults to 1.0."""

    null_space_joint_pos_target: Literal["default", "center"] = "default"
    """Posture target for null-space posture control: default joint positions or the soft-limit center."""

    controller: NewtonDifferentialIKControllerCfg = MISSING
    """The configuration for the Newton differential IK controller."""


@configclass
class NewtonOperationalSpaceControllerActionCfg(ActionTermCfg):
    """Configuration for :class:`NewtonOperationalSpaceControllerAction`.

    The action is the target pose, followed by the target wrench when wrench control is enabled, then the
    motion stiffness and damping when the controller leaves them live (``None``). Poses, twists, and the Jacobian
    are expressed in the robot root frame, and the controller's operational frame is relative to it. The action
    works with any physics backend.
    """

    class_type: type[NewtonOperationalSpaceControllerAction] | str = (
        "{DIR}.newton_task_space_actions:NewtonOperationalSpaceControllerAction"
    )

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""

    body_name: str = MISSING
    """Name of the body for which operational-space control is performed."""

    body_offset: DifferentialInverseKinematicsActionCfg.OffsetCfg | None = None
    """Offset of the target frame from the body frame. Defaults to None, in which case no offset is applied."""

    target_type: Literal["pose_abs", "pose_rel"] = "pose_abs"
    """Whether the pose target is absolute ``(x, y, z, qx, qy, qz, qw)`` or a delta ``(x, y, z, rx, ry, rz)``
    from the current pose, in the operational frame."""

    position_scale: float = 1.0
    """Scale factor for the position targets. Defaults to 1.0."""

    orientation_scale: float = 1.0
    """Scale factor for the orientation targets. Defaults to 1.0."""

    wrench_scale: float = 1.0
    """Scale factor for the wrench targets. Defaults to 1.0."""

    stiffness_scale: float = 1.0
    """Scale factor for the stiffness commands. Defaults to 1.0."""

    damping_scale: float = 1.0
    """Scale factor for the damping commands. Defaults to 1.0."""

    null_space_joint_pos_target: Literal["default", "center", "zero"] = "default"
    """Posture target for null-space control: default joint positions, the soft-limit center, or zero."""

    controller: NewtonOperationalSpaceControllerCfg = MISSING
    """The configuration for the Newton operational-space controller."""
