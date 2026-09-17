# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for selection-aware pose commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkersCfg
from isaaclab.utils import REQUIRED, replace_config

from ..selection_utils import SceneEntitySelectionCfg

if TYPE_CHECKING:
    from .commands import SelectedUniformPoseCommand


@dataclass
class SelectedUniformPoseCommandCfg(CommandTermCfg):
    """Configuration for a selection-aware uniform pose command."""

    @dataclass
    class Ranges:
        """Uniform pose sampling ranges."""

        pos_x: tuple[float, float] = REQUIRED
        """Position range along x [m]."""
        pos_y: tuple[float, float] = REQUIRED
        """Position range along y [m]."""
        pos_z: tuple[float, float] = REQUIRED
        """Position range along z [m]."""
        roll: tuple[float, float] = REQUIRED
        """Roll range [rad]."""
        pitch: tuple[float, float] = REQUIRED
        """Pitch range [rad]."""
        yaw: tuple[float, float] = REQUIRED
        """Yaw range [rad]."""

    class_type: type[SelectedUniformPoseCommand] | str = (
        "isaaclab_tasks.contrib.multitask_manipulation.mdp.commands:SelectedUniformPoseCommand"
    )
    reference_cfg: SceneEntitySelectionCfg = REQUIRED
    """Entity whose root frame contains the sampled command."""
    tracked_cfg: SceneEntitySelectionCfg = REQUIRED
    """Entity root or selected body whose current pose is visualized."""
    ranges: Ranges = REQUIRED
    """Pose sampling ranges."""
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = field(
        default_factory=lambda: replace_config(FRAME_MARKER_CFG, prim_path="/Visuals/Command/goal_pose")
    )
    """Goal-pose frame marker configuration."""
    current_pose_visualizer_cfg: VisualizationMarkersCfg = field(
        default_factory=lambda: replace_config(FRAME_MARKER_CFG, prim_path="/Visuals/Command/current_pose")
    )
    """Tracked-pose frame marker configuration."""

    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
