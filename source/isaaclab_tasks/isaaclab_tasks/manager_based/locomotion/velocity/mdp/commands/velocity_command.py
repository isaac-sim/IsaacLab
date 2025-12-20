# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING
import torch

import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.envs.mdp import UniformVelocityCommandCfg, UniformVelocityCommand
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    
    
class UniformLevelVelocityCommand(UniformVelocityCommand):
    """
    This class inherits from `UniformVelocityCommand` to 
    - apply curriclum to lin/ang velocity sampling range
    - provide debug vis for both linear and angular velocity commands
    """
    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_linvel_visualizer"):
                # -- goal
                self.goal_linvel_visualizer = VisualizationMarkers(self.cfg.goal_linvel_visualizer_cfg)
                self.goal_angvel_visualizer = VisualizationMarkers(self.cfg.goal_angvel_visualizer_cfg)
                # -- current
                self.curr_linvel_visualizer = VisualizationMarkers(self.cfg.current_linvel_visualizer_cfg)
                self.curr_angvel_visualizer = VisualizationMarkers(self.cfg.current_angvel_visualizer_cfg)
            # set their visibility to true
            self.goal_linvel_visualizer.set_visibility(True)
            self.goal_angvel_visualizer.set_visibility(True)
            self.curr_linvel_visualizer.set_visibility(True)
            self.curr_angvel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_linvel_visualizer.set_visibility(False)
                self.goal_angvel_visualizer.set_visibility(False)
                self.curr_linvel_visualizer.set_visibility(False)
                self.curr_angvel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        # -- resolve linear velocity arrows (in xy plane)
        linvel_des_scale, linvel_des_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        linvel_curr_scale, linvel_curr_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])

        # -- resolve angular velocity arrows (around z axis)
        angvel_des_scale, angvel_des_quat = self._resolve_z_angvel_to_arrow(self.command[:, 2])
        angvel_curr_scale, angvel_curr_quat = self._resolve_z_angvel_to_arrow(self.robot.data.root_ang_vel_b[:, 2])

        # display markers
        self.goal_linvel_visualizer.visualize(base_pos_w, linvel_des_quat, linvel_des_scale)
        self.curr_linvel_visualizer.visualize(base_pos_w, linvel_curr_quat, linvel_curr_scale)

        # offset angular velocity arrows slightly higher
        angvel_pos_w = base_pos_w.clone()
        angvel_pos_w[:, 2] += 0.5
        self.goal_angvel_visualizer.visualize(angvel_pos_w, angvel_des_quat, angvel_des_scale)
        self.curr_angvel_visualizer.visualize(angvel_pos_w, angvel_curr_quat, angvel_curr_scale)
        
    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_linvel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

    def _resolve_z_angvel_to_arrow(self, z_angvel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the Z angular velocity to arrow pointing in +/- z direction."""
        # obtain default scale of the marker
        default_scale = self.goal_angvel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale based on angular velocity magnitude
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(z_angvel.shape[0], 1)
        arrow_scale[:, 0] *= torch.abs(z_angvel) * 2.0
        # arrow direction: point up (+z) for positive angvel (ccw), down (-z) for negative angvel (cw)
        # rotate around y-axis: +90 degrees for +z, -90 degrees for -z
        pitch_angle = torch.where(z_angvel < 0, torch.full_like(z_angvel, torch.pi / 2), torch.full_like(z_angvel, -torch.pi / 2))
        zeros = torch.zeros_like(pitch_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, pitch_angle, zeros)
        # convert from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = UniformLevelVelocityCommand
    
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    
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
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_linvel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    goal_angvel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_linvel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_angvel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)