# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for task-specific cube-stacking actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, JointActionCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .actions import (
        ResetBufferedGripperAction,
        WorkspaceBoundedRelativeJointPositionAction,
    )


@configclass
class ResetBufferedGripperActionCfg(BinaryJointPositionActionCfg):
    """Configuration for :class:`ResetBufferedGripperAction`."""

    class_type: type[ResetBufferedGripperAction] | str = "{DIR}.actions:ResetBufferedGripperAction"

    force_close_steps: int = 5
    """Number of initial policy steps for which a reset-supplied grasp is protected."""


@configclass
class WorkspaceBoundedRelativeJointPositionActionCfg(JointActionCfg):
    """Configuration for bounded, gravity-compensated relative joint control."""

    class_type: type[WorkspaceBoundedRelativeJointPositionAction] | str = (
        "{DIR}.actions:WorkspaceBoundedRelativeJointPositionAction"
    )

    joint_limit_margin: float = 0.02
    """Distance kept from each soft joint-position limit [rad]."""

    max_delta: float = 0.10
    """Maximum position-target change from one policy step [rad]."""

    workspace_lower: tuple[float, ...] = (-2.0,) * 7
    """Lower joint-space boundary of the task manipulation workspace [rad]."""

    workspace_upper: tuple[float, ...] = (2.0,) * 7
    """Upper joint-space boundary of the task manipulation workspace [rad]."""

    gravity_compensation: bool = False
    """Whether to add model-based gravity feedforward to the controlled joints."""
