# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_newton.ik.newton_ik_manager_cfg import NewtonIKManagerCfg

if TYPE_CHECKING:
    from .newton_ik_actions import NewtonInverseKinematicsAction


@configclass
class NewtonInverseKinematicsActionCfg(ActionTermCfg):
    """Configuration for a Newton inverse-kinematics action term.

    The action currently supports fixed-base articulations only. The configured
    joints and body must resolve both in Isaac Lab and in the registered Newton
    prototype model for the controlled asset.
    """

    @configclass
    class OffsetCfg:
        """Offset pose from the controlled body frame to the IK target frame."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation [m] w.r.t. the parent body frame."""

        rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
        """Quaternion rotation ``(x, y, z, w)`` w.r.t. the parent body frame."""

    class_type: type[NewtonInverseKinematicsAction] | str = (
        "isaaclab_newton.envs.mdp.actions.newton_ik_actions:NewtonInverseKinematicsAction"
    )

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions controlled by the action."""

    body_name: str = MISSING
    """Name of the body for which IK is performed."""

    body_offset: OffsetCfg | None = None
    """Offset of the target frame w.r.t. the body frame."""

    scale: float | tuple[float, ...] = 1.0
    """Scale factor applied to the raw action.

    For position coordinates this is in meters. For relative rotation coordinates
    this is in radians. For quaternions this is dimensionless.
    """

    controller: NewtonIKManagerCfg = MISSING
    """Configuration for the Newton IK manager."""
