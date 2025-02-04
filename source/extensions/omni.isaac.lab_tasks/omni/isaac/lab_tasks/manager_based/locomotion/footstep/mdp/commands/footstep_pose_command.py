# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for 3D orientation goals for objects."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import RigidObject, Articulation
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.markers.visualization_markers import VisualizationMarkers

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

    from .commands_cfg import FootstepPoseCommandCfg


class FootstepPoseCommand(CommandTerm):
    """
    """

    cfg: FootstepPoseCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: FootstepPoseCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # object
        # self.object: RigidObject = env.scene[cfg.asset_name]
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # create buffers to store the command
        # -- footstep position: (x, y)
        # init_pos_offset = torch.tensor(cfg.init_pos_offset, dtype=torch.float, device=self.device)
        # self.pos_command_e = self.object.data.default_root_state[:, :3] + init_pos_offset
        # self.pos_command_w = self.pos_command_e + self._env.scene.env_origins
        self.feet_id = self.robot.find_bodies(".*foot")
        self.feet_pos = self.robot.data.body_pos_w[:, self.feet_id, :]

        self.footstep_pose_command = torch.zeros(self.num_envs, 3, device=self.device)

        # -- footstep yaw orientation: (w, x, y, z)
        self.quat_command_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.quat_command_w[:, 0] = 1.0  # set the scalar component to 1.0

        # -- unit vectors
        self._X_UNIT_VEC = torch.tensor([1.0, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self._Y_UNIT_VEC = torch.tensor([0, 1.0, 0], device=self.device).repeat((self.num_envs, 1))
        self._Z_UNIT_VEC = torch.tensor([0, 0, 1.0], device=self.device).repeat((self.num_envs, 1))

        # -- metrics
        self.metrics["angle_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["consecutive_success"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "FootstepPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        #TODO
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired goal pose in the environment frame. Shape is (num_envs, 7)."""
        # return torch.cat((self.pos_command_e, self.quat_command_w), dim=-1)
        return self.footstep_pose_command

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        # -- compute the orientation error
        self.metrics["angle_error"] = math_utils.quat_error_magnitude(
            self.object.data.root_link_quat_w, self.quat_command_w
        )
        # -- compute the position error
        self.metrics["position_error"] = torch.norm(self.object.data.root_link_pos_w - self.pos_command_w, dim=1)
        # -- compute the number of consecutive successes
        successes_angle = self.metrics["angle_error"] < self.cfg.angle_success_threshold
        successes_position = self.metrics["position_error"] < self.cfg.position_success_threshold
        successes = successes_angle & successes_position
        self.metrics["consecutive_success"] += successes.float()

    def _resample_command(self, env_ids: Sequence[int]):
        # sample foot pose commands
        # Called at the begining of the episode and when resample time is reached.
        # Resample the footstep pose command in the frame of each foot.
        # We only want to change the footstep pose of each foot at the beginning of the episode. 
        # It stays the same for the rest of the episode (in the frame of the foot).
        command = torch.empty(len(env_ids), device=self.device)
        self.feet_pos = self.robot.data.body_pos_w[:, self.feet_id, :]

        # -- foot position - x direction
        self.footstep_pose_command[env_ids, 0] = command.uniform_(*self.cfg.ranges.pos_x)
        # -- foot position - y direction
        self.footstep_pose_command[env_ids, 1] = command.uniform_(*self.cfg.ranges.pos_y)
        # -- ang foot pose yaw - rotation around z
        self.footstep_pose_command[env_ids, 2] = command.uniform_(*self.cfg.ranges.ang_z)

    def _update_command(self):
        # Called at each step of the environment.

        # Update the footstep pose command if goal is reached
        goal_resets_angle = self.metrics["angle_error"] < self.cfg.angle_success_threshold
        goal_resets_position = self.metrics["position_error"] < self.cfg.position_success_threshold
        goal_resets = goal_resets_angle & goal_resets_position
        goal_reset_ids = goal_resets.nonzero(as_tuple=False).squeeze(-1)
        


        # if self.cfg.update_goal_on_success:
        #     # compute the goal resets
        #     goal_resets = self.metrics["orientation_error"] < self.cfg.orientation_success_threshold
        #     goal_reset_ids = goal_resets.nonzero(as_tuple=False).squeeze(-1)
        #     # resample the goals

    def _set_debug_vis_impl(self, debug_vis: TYPE_CHECKING):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_pose_left_visualizer") or not hasattr(self, "goal_pose_right_visualizer"):
                self.goal_pose_left_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_left_cfg)
                self.goal_pose_right_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_right_cfg)
                
            # set visibility
            self.goal_pose_left_visualizer.set_visibility(True)
            self.goal_pose_right_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer") and hasattr(self, "goal_pose_right_visualizer"):
                self.goal_pose_left_visualizer.set_visibility(False)
                self.goal_pose_right_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # add an offset to the marker position to visualize the goal
        if not self.robot.is_initialized:
            return
        
        marker_pos_left = self.pos_command_w + torch.tensor(self.cfg.marker_pos_offset, device=self.device)
        marker_quat_left = self.quat_command_w
        marker_pos_right = self.pos_command_w + torch.tensor(self.cfg.marker_pos_offset, device=self.device)
        marker_quat_right = self.quat_command_w
        # visualize the goal marker
        self.goal_pose_left_visualizer.visualize(translations=marker_pos_left, orientations=marker_quat_left)
        self.goal_pose_left_visualizer.visualize(translations=marker_pos_right, orientations=marker_quat_right)

