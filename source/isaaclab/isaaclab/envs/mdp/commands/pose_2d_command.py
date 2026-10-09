# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the 2D-pose for locomotion tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains import TerrainImporter
from isaaclab.utils import env_mask_from_ids
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, wrap_to_pi, yaw_quat

if TYPE_CHECKING:
    from ... import ManagerBasedEnv
    from .commands_cfg import TerrainBasedPose2dCommandCfg, UniformPose2dCommandCfg


class UniformPose2dCommand(CommandTerm):
    """Command generator that generates pose commands containing a 3-D position and heading.

    The command generator samples uniform 2D positions around the environment origin. It sets
    the height of the position command to the default root height of the robot. The heading
    command is either set to point towards the target or is sampled uniformly.
    This can be configured through the :attr:`Pose2dCommandCfg.simple_heading` parameter in
    the configuration.
    """

    cfg: UniformPose2dCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformPose2dCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot and terrain assets
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- commands: (x, y, z, heading)
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)
        # -- metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)
        # -- per-episode sticky success bit (only used when cfg.position_success_threshold is set)
        self._track_success = cfg.position_success_threshold is not None
        if self._track_success:
            self._succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # adds (optional) cmd kind and element names for leapp export
        # during export, semantic data about this command will be used to annotate the command input
        self.cfg.cmd_kind = self.cfg.cmd_kind or "command/body/pose"
        self.cfg.element_names = self.cfg.element_names or ["x", "y", "z", "heading"]

    def __str__(self) -> str:
        msg = "PositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired 2D-pose in base frame. Shape is (num_envs, 4)."""
        return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        self.metrics["error_pos"] = torch.linalg.norm(
            self.pos_command_w[:, :2] - self.robot.data.root_pos_w.torch[:, :2], dim=1
        )
        self.metrics["error_heading"] = torch.abs(wrap_to_pi(self.heading_command_w - self.robot.data.heading_w.torch))
        if self._track_success:
            self._succeeded |= self.metrics["error_pos"] < self.cfg.position_success_threshold

    def reset(
        self, env_ids: Sequence[int] | None = None, env_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if env_mask is None:
            env_mask = env_mask_from_ids(env_ids, self.num_envs, self.device)
        extras = super().reset(env_mask=env_mask)
        if self._track_success:
            # Write the unified ``Metrics/success_rate`` directly to env extras so it shares
            # a TensorBoard card with the same metric from other tasks.
            success_rate = (self._succeeded & env_mask).sum() / env_mask.sum().clamp_min(1)
            self._env.extras.setdefault("log", {})["Metrics/success_rate"] = success_rate
            self._succeeded.masked_fill_(env_mask, False)
        return extras

    def _resample_command(self, env_mask: torch.Tensor):
        # obtain env origins for all the environments
        pos_command_w = self._env.scene.env_origins.clone()
        # offset the position command by the current root position
        r = torch.empty(self.num_envs, device=self.device)
        pos_command_w[:, 0] += r.uniform_(*self.cfg.ranges.pos_x)
        pos_command_w[:, 1] += r.uniform_(*self.cfg.ranges.pos_y)
        pos_command_w[:, 2] += self.robot.data.default_root_pose.torch[:, 2]
        self._set_resampled_command(env_mask, pos_command_w, r)

    def _set_resampled_command(self, env_mask: torch.Tensor, pos_command_w: torch.Tensor, r: torch.Tensor):
        """Write the sampled position and its heading command for the selected environments.

        Args:
            env_mask: Boolean mask of the environments to update. Shape is (num_envs,).
            pos_command_w: Sampled position commands in the world frame for all the environments.
                Shape is (num_envs, 3).
            r: Scratch buffer for random sampling. Shape is (num_envs,).
        """
        torch.where(env_mask[:, None], pos_command_w, self.pos_command_w, out=self.pos_command_w)

        if self.cfg.simple_heading:
            # set heading command to point towards target
            target_vec = pos_command_w - self.robot.data.root_pos_w.torch
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

            # compute errors to find the closest direction to the current heading
            # this is done to avoid the discontinuity at the -pi/pi boundary
            curr_to_target = wrap_to_pi(target_direction - self.robot.data.heading_w.torch).abs()
            curr_to_flipped_target = wrap_to_pi(flipped_target_direction - self.robot.data.heading_w.torch).abs()

            # set the heading command to the closest direction
            heading_command_w = torch.where(
                curr_to_target < curr_to_flipped_target,
                target_direction,
                flipped_target_direction,
            )
        else:
            # random heading command
            heading_command_w = r.uniform_(*self.cfg.ranges.heading)
        torch.where(env_mask, heading_command_w, self.heading_command_w, out=self.heading_command_w)

    def _update_command(self):
        """Re-target the position command to the current root state."""
        target_vec = self.pos_command_w - self.robot.data.root_pos_w.torch[:, :3]
        self.pos_command_b[:] = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w.torch), target_vec)
        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - self.robot.data.heading_w.torch)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the box marker
        self.goal_pose_visualizer.visualize(
            translations=self.pos_command_w,
            orientations=quat_from_euler_xyz(
                torch.zeros_like(self.heading_command_w),
                torch.zeros_like(self.heading_command_w),
                self.heading_command_w,
            ),
            environment_ids=self._env.scene._ALL_INDICES,
        )


class TerrainBasedPose2dCommand(UniformPose2dCommand):
    """Command generator that generates pose commands based on the terrain.

    This command generator samples the position commands from the valid patches of the terrain.
    The heading commands are either set to point towards the target or are sampled uniformly.

    It expects the terrain to have a valid flat patches under the key 'target'.
    """

    cfg: TerrainBasedPose2dCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: TerrainBasedPose2dCommandCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the terrain asset
        self.terrain: TerrainImporter = env.scene["terrain"]

        # obtain the valid targets from the terrain
        if "target" not in self.terrain.flat_patches:
            raise RuntimeError(
                "The terrain-based command generator requires a valid flat patch under 'target' in the terrain."
                f" Found: {list(self.terrain.flat_patches.keys())}"
            )
        # valid targets: (terrain_level, terrain_type, num_patches, 3)
        self.valid_targets: torch.Tensor = self.terrain.flat_patches["target"]

    def _resample_command(self, env_mask: torch.Tensor):
        # sample new position targets from the terrain for all the environments
        ids = torch.randint(0, self.valid_targets.shape[2], size=(self.num_envs,), device=self.device)
        pos_command_w = self.valid_targets[self.terrain.terrain_levels, self.terrain.terrain_types, ids]
        # offset the position command by the current root height
        pos_command_w[:, 2] += self.robot.data.default_root_pose.torch[:, 2]
        self._set_resampled_command(env_mask, pos_command_w, torch.empty(self.num_envs, device=self.device))
