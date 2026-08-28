# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the manager-based handover task's goal-pose command."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.handover.handover_common import GOAL_MARKER_CFG, GOAL_POSITION_OFFSET

from .commands import HandoverCommand


@configclass
class HandoverCommandCfg(CommandTermCfg):
    """Configuration for :class:`HandoverCommand`."""

    class_type: type[HandoverCommand] = HandoverCommand
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    asset_name: str = MISSING
    position_offset: tuple[float, float, float] = GOAL_POSITION_OFFSET
    """Goal-position offset from the object's default position [m]."""
    success_distance_threshold: float = 0.1
    """Object-to-goal distance below which an episode counts as successful [m]."""
    goal_visualizer_cfg: VisualizationMarkersCfg = GOAL_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_marker")
