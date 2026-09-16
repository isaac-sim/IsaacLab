# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for contributed selection-aware joint actions."""

from __future__ import annotations

from dataclasses import MISSING, dataclass
from typing import TYPE_CHECKING

from isaaclab.managers import ActionTermCfg
from isaaclab.utils import config_field

if TYPE_CHECKING:
    from .actions import SelectedBinaryJointPositionAction, SelectedJointPositionAction


@dataclass
class SelectedJointPositionActionCfg(ActionTermCfg):
    """Configuration for a selection-aware joint-position action."""

    class_type: type[SelectedJointPositionAction] | str = config_field("{DIR}.actions:SelectedJointPositionAction")
    joint_names: list[str] = config_field(MISSING)
    """Joint-name expressions resolved on the selected articulation."""
    scale: float = config_field(1.0)
    """Multiplicative action scale [m or rad, depending on joint type]."""
    relative: bool = config_field(False)
    """Whether actions are offsets from the current joint positions."""
    joint_limit_margin: float | None = config_field(None)
    """Margin inside the soft joint-position limits [m or rad, depending on joint type].

    When set, applied position targets are clamped to the soft limits reduced by this margin.
    """


@dataclass
class SelectedBinaryJointPositionActionCfg(ActionTermCfg):
    """Configuration for a selection-aware binary joint-position action."""

    class_type: type[SelectedBinaryJointPositionAction] | str = config_field(
        "{DIR}.actions:SelectedBinaryJointPositionAction"
    )
    joint_names: list[str] = config_field(MISSING)
    """Joint-name expressions resolved on the selected articulation."""
    open_command: float = config_field(MISSING)
    """Open joint target [m or rad, depending on joint type]."""
    close_command: float = config_field(MISSING)
    """Closed joint target [m or rad, depending on joint type]."""
