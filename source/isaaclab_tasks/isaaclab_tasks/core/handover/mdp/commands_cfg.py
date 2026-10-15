# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the manager-based handover task's goal-pose command."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass, replace

from ..handover_common import GOAL_MARKER_CFG, GOAL_POSITION_OFFSET
from .commands import HandoverCommand


@configclass
class HandoverCommandCfg(CommandTermCfg):
    """Configuration for :class:`HandoverCommand`."""

    class_type: type[HandoverCommand] = HandoverCommand
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    asset_name: str = MISSING
    right_hand_name: str = "right_hand"
    left_hand_name: str = "left_hand"
    position_offset: tuple[float, float, float] = GOAL_POSITION_OFFSET
    """Shared goal offset in each hand's local root frame [m], resolved once at initialization."""
    success_distance_threshold: float = 0.1
    """Object-to-goal distance below which success steps accumulate [m]."""
    success_steps_required: int = 20
    """Cumulative steps inside the success distance required to switch goals; must be positive."""
    goal_visualizer_cfg: VisualizationMarkersCfg = replace(GOAL_MARKER_CFG, prim_path="/Visuals/Command/goal_marker")
