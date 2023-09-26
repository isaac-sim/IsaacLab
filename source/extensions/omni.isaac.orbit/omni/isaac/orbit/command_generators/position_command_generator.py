# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the position-based locomotion task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.markers import VisualizationMarkers
from omni.isaac.orbit.markers.config import CUBOID_MARKER_CFG
from omni.isaac.orbit.terrains import TerrainImporter
from omni.isaac.orbit.utils.math import quat_rotate_inverse, wrap_to_pi, yaw_quat

from .command_generator_base import CommandGeneratorBase

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv

    from .command_generator_cfg import TerrainBasedPositionCommandGeneratorCfg


class TerrainBasedPositionCommandGenerator(CommandGeneratorBase):
    """Command generator that generates position commands based on the terrain.

    The position commands are sampled from the terrain mesh and the heading commands are either set
    to point towards the target or are sampled uniformly.
    """

    cfg: TerrainBasedPositionCommandGeneratorCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: TerrainBasedPositionCommandGeneratorCfg, env: BaseEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        super().__init__(cfg, env)
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]
        # -- terrain
        self.terrain: TerrainImporter = env.scene.terrain
        # -- commands: (x, y, z, heading)
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)
        # -- metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)
        # -- debug vis
        self.box_goal_visualizer = None

    def __str__(self) -> str:
        msg = "TerrainBasedPositionCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base position in base frame. Shape is (num_envs, 3)."""
        return self.pos_command_b

    """
    Operations.
    """

    def set_debug_vis(self, debug_vis: bool):
        super().set_debug_vis(debug_vis)
        if self.box_goal_visualizer is not None:
            self.box_goal_visualizer.set_visibility(debug_vis)

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new position targets from the terrain
        # TODO: need to add that here directly
        self.pos_command_w[env_ids] = self.terrain.sample_new_targets(env_ids)
        # offset the position command by the current root position
        self.pos_command_w[env_ids, 2] += self.robot.data.default_root_state_w[env_ids, 2]

        if self.cfg.simple_heading:
            # set heading command to point towards target
            target_vec = self.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids]
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_heading = wrap_to_pi(target_direction + torch.pi)
            self.heading_command_w[env_ids] = torch.where(
                wrap_to_pi(target_direction - self.robot.data.heading_w[env_ids]).abs()
                < wrap_to_pi(flipped_heading - self.robot.data.heading_w[env_ids]).abs(),
                target_direction,
                flipped_heading,
            )
        else:
            # random heading command
            r = torch.empty(len(env_ids), device=self.device)
            self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)

    def _update_command(self):
        """Re-target the position command to the current root position and heading."""
        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        self.pos_command_b[:] = quat_rotate_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)
        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - self.robot.heading_w)

    def _update_metrics(self):
        # logs data
        self.metrics["error_pos"] = torch.norm(self.pos_command_w - self.robot.data.root_pos_w[:, :3], dim=1)
        self.metrics["error_heading"] = torch.abs(wrap_to_pi(self.heading_command_w - self.robot.heading_w))

    def _debug_vis_impl(self):
        # create the box marker if necessary
        if self.box_goal_visualizer is None:
            marker_cfg = CUBOID_MARKER_CFG
            marker_cfg.markers["cuboid"].color = (1.0, 0.0, 0.0)
            marker_cfg.markers["cuboid"].scale = (0.1, 0.1, 0.1)
            self.box_goal_visualizer = VisualizationMarkers("/Visuals/Command/position_goal", marker_cfg)
        # update the box marker
        self.box_goal_visualizer.visualize(self.pos_command_w)
