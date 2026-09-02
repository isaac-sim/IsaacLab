# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for contributed selection-aware joint actions."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import ActionTermCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .actions import SelectedBinaryJointPositionAction, SelectedJointPositionAction


@configclass
class SelectedJointPositionActionCfg(ActionTermCfg):
    """Configuration for a selection-aware joint-position action."""

    class_type: type[SelectedJointPositionAction] | str = "{DIR}.actions:SelectedJointPositionAction"
    joint_names: list[str] = MISSING
    """Joint-name expressions resolved on the selected articulation."""
    scale: float = 1.0
    """Multiplicative action scale [m or rad, depending on joint type]."""
    relative: bool = False
    """Whether actions are offsets from the current joint positions."""
    joint_limit_margin: float | None = None
    """Margin inside the soft joint-position limits [m or rad, depending on joint type].

    When set, applied position targets are clamped to the soft limits reduced by this margin.
    """


@configclass
class SelectedBinaryJointPositionActionCfg(ActionTermCfg):
    """Configuration for a selection-aware binary joint-position action."""

    class_type: type[SelectedBinaryJointPositionAction] | str = "{DIR}.actions:SelectedBinaryJointPositionAction"
    joint_names: list[str] = MISSING
    """Joint-name expressions resolved on the selected articulation."""
    open_command: float = MISSING
    """Open joint target [m or rad, depending on joint type]."""
    close_command: float = MISSING
    """Closed joint target [m or rad, depending on joint type]."""
