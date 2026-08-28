# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action configurations for conveyor transfer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, JointActionCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .actions import ConveyorRelativeJointPositionAction, ResetBufferedGripperAction


@configclass
class ConveyorRelativeJointPositionActionCfg(JointActionCfg):
    """Configuration for measured-state relative Franka joint control."""

    class_type: type[ConveyorRelativeJointPositionAction] | str = "{DIR}.actions:ConveyorRelativeJointPositionAction"

    joint_limit_margin: float = 0.02
    """Distance kept from each soft joint limit [rad]."""

    max_delta: float = 0.12
    """Maximum target change per policy step [rad]."""

    workspace_lower: tuple[float, ...] = (-0.75, -0.45, -0.55, -2.75, -0.45, 1.85, -0.10)
    """Lower boundary of the validated transfer workspace [rad]."""

    workspace_upper: tuple[float, ...] = (0.85, 0.85, 0.35, -1.75, 0.45, 3.05, 1.65)
    """Upper boundary of the validated transfer workspace [rad]."""


@configclass
class ResetBufferedGripperActionCfg(BinaryJointPositionActionCfg):
    """Configuration for reset-grasp protection."""

    class_type: type[ResetBufferedGripperAction] | str = "{DIR}.actions:ResetBufferedGripperAction"

    force_close_steps: int = 5
    """Initial policy steps that preserve a reset-authored grasp."""

    command_name: str = "transfer"
    """Command term that owns reset-authored held-cube state."""
