# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Selector-aware pose command term for multi-task environments.

Like the core :class:`UniformPoseCommand` but scoped to a subset of
envs via :class:`SceneEntityCfg` ``selector``.  Only writes to
``env_ids`` rows; other rows stay at identity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import CommandTerm
from isaaclab.scene.env_view_index import filter_to_group
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

from .commands_cfg import PoseCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PoseCommand(CommandTerm):
    """Selector-aware uniform pose command generator.

    Generates pose commands for a single robot asset scoped to specific
    selector rows.  The command buffer is ``(num_envs, 7)`` but only the
    selected rows are populated.
    """

    cfg: PoseCommandCfg

    def __init__(self, cfg: PoseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # Allocate command buffers unconditionally — the command tensor must
        # always exist so that callers (e.g. obs terms) can read from it safely.
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        self._ee_pose_w = torch.zeros(self.num_envs, 7, device=self.device)
        self._ee_pose_w[:, 3] = 1.0
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

        cfg.asset_cfg.resolve(env.scene)

        # If the selector is disabled (weight=0), SceneEntityCfg.resolve() returns an
        # empty env_ids tensor instead of raising.  Enter no-op mode: skip all asset
        # access so this term acts as an inert zero-command buffer.
        self._disabled = isinstance(cfg.asset_cfg.env_ids, torch.Tensor) and cfg.asset_cfg.env_ids.numel() == 0
        if self._disabled:
            return

        self.robot = env.scene[cfg.asset_cfg.name]
        self.body_idx = cfg.asset_cfg.body_ids[0]
        self._env_ids = cfg.asset_cfg.env_ids
        self._view_ids = cfg.asset_cfg.view_ids

        env_to_view_map = env.scene.selector.get(cfg.asset_cfg.selector, asset=cfg.asset_cfg.name)
        self._selector_layout = env_to_view_map.layout

    def __str__(self) -> str:
        msg = "PoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tSelector: {self.cfg.asset_cfg.selector}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command [m, -]. Shape is (num_envs, 7)."""
        return self.pose_command_b

    def _update_metrics(self):
        if self._disabled:
            return
        root_pos = wp.to_torch(self.robot.data.root_pos_w)[self._view_ids]
        root_quat = wp.to_torch(self.robot.data.root_quat_w)[self._view_ids]

        self.pose_command_w[self._env_ids, :3], self.pose_command_w[self._env_ids, 3:] = combine_frame_transforms(
            root_pos,
            root_quat,
            self.pose_command_b[self._env_ids, :3],
            self.pose_command_b[self._env_ids, 3:],
        )
        ee_pos = wp.to_torch(self.robot.data.body_pos_w)[self._view_ids, self.body_idx]
        ee_quat = wp.to_torch(self.robot.data.body_quat_w)[self._view_ids, self.body_idx]
        self._ee_pose_w[self._env_ids, :3] = ee_pos
        self._ee_pose_w[self._env_ids, 3:] = ee_quat
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[self._env_ids, :3],
            self.pose_command_w[self._env_ids, 3:],
            ee_pos,
            ee_quat,
        )
        self.metrics["position_error"][self._env_ids] = torch.linalg.norm(pos_error, dim=-1)
        self.metrics["orientation_error"][self._env_ids] = torch.linalg.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: torch.Tensor):
        if self._disabled:
            return
        _, matched = filter_to_group(self._selector_layout, env_ids)
        if matched.numel() == 0:
            return
        ranges = self.cfg.ranges
        num_matched = matched.numel()
        rand = torch.empty(num_matched, device=self.device)
        self.pose_command_b[matched, 0] = rand.uniform_(*ranges.pos_x)
        self.pose_command_b[matched, 1] = rand.uniform_(*ranges.pos_y)
        self.pose_command_b[matched, 2] = rand.uniform_(*ranges.pos_z)

        euler = torch.zeros(num_matched, 3, device=self.device)
        euler[:, 0].uniform_(*ranges.roll)
        euler[:, 1].uniform_(*ranges.pitch)
        euler[:, 2].uniform_(*ranges.yaw)
        quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        self.pose_command_b[matched, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                from isaaclab.markers import VisualizationMarkers
                from isaaclab.markers.config import FRAME_MARKER_CFG

                goal_cfg = FRAME_MARKER_CFG.copy()
                goal_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
                goal_cfg.prim_path = "/Visuals/PoseCommand/goal"
                self.goal_pose_visualizer = VisualizationMarkers(goal_cfg)

                ee_cfg = FRAME_MARKER_CFG.copy()
                ee_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
                ee_cfg.prim_path = "/Visuals/PoseCommand/ee"
                self.current_pose_visualizer = VisualizationMarkers(ee_cfg)
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if self._disabled:
            return
        if not self.robot.is_initialized:
            return
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        self.current_pose_visualizer.visualize(self._ee_pose_w[:, :3], self._ee_pose_w[:, 3:])
