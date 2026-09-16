# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the manager-based handover task's goal-pose command."""

from __future__ import annotations

from dataclasses import MISSING, dataclass

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import config_field, replace_config

from isaaclab_tasks.core.handover.handover_common import GOAL_MARKER_CFG, GOAL_POSITION_OFFSET

from .commands import HandoverCommand


@dataclass
class HandoverCommandCfg(CommandTermCfg):
    """Configuration for :class:`HandoverCommand`."""

    class_type: type[HandoverCommand] = config_field(HandoverCommand)
    resampling_time_range: tuple[float, float] = config_field((1.0e6, 1.0e6))
    asset_name: str = config_field(MISSING)
    position_offset: tuple[float, float, float] = config_field(GOAL_POSITION_OFFSET)
    """Goal-position offset from the object's default position [m]."""
    success_distance_threshold: float = config_field(0.1)
    """Object-to-goal distance below which an episode counts as successful [m]."""
    goal_visualizer_cfg: VisualizationMarkersCfg = config_field(
        replace_config(GOAL_MARKER_CFG, prim_path="/Visuals/Command/goal_marker")
    )
