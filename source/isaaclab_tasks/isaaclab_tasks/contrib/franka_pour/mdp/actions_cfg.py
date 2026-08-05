# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Franka Pour action terms."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions import BinaryJointPositionActionCfg, RelativeJointPositionActionCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .actions import (
        CurriculumGripperPositionAction,
        EMARelativeJointPositionAction,
    )


@configclass
class EMARelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Configuration for :class:`EMARelativeJointPositionAction`."""

    alpha: float = 1.0
    """Weight of the newest relative joint delta; one preserves the unfiltered action."""

    class_type: type[EMARelativeJointPositionAction] | str = "{DIR}.actions:EMARelativeJointPositionAction"


@configclass
class CurriculumGripperPositionActionCfg(BinaryJointPositionActionCfg):
    """Configuration for :class:`CurriculumGripperPositionAction`."""

    open_command_expr: dict[str, float] = field(default_factory=dict)
    """Populated from :attr:`neutral_position` when the action term is constructed."""
    close_command_expr: dict[str, float] = field(default_factory=dict)
    """Populated from :attr:`close_position` when the action term is constructed."""
    alpha: float = 0.2
    """Interpolation weight applied to the selected finger target."""
    close_position: float = 0.0
    """Per-finger closed command [m]."""
    neutral_position: float = 0.025
    """Largest per-finger command accepted from the action [m]."""
    default_position: float | None = None
    """Initial filtered target [m]. ``None`` uses :attr:`close_position`."""
    contact_min_deflection: float = 0.001
    """Minimum settled position-drive deflection required on each finger [m]."""
    class_type: type[CurriculumGripperPositionAction] | str = "{DIR}.actions:CurriculumGripperPositionAction"
