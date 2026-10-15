# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG
from isaaclab.utils.configclass import configclass

from .velocity_command import UniformLevelVelocityCommand, UniformVelocityYawCommand


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = UniformLevelVelocityCommand

    goal_linvel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""
    goal_angvel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/angvel_goal"
    )
    """The configuration for the goal angvel visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_linvel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""
    current_angvel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/angvel_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_linvel_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)
    goal_angvel_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)
    current_linvel_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)
    current_angvel_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)


@configclass
class UniformVelocityYawCommandCfg(UniformLevelVelocityCommandCfg):
    class_type: type = UniformVelocityYawCommand
    goal_heading_visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/heading_goal"
    )
    """The configuration for the goal heading visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""
