# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Selection-aware pose commands for contributed heterogeneous task batches."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, quat_from_euler_xyz, quat_unique

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import SelectedUniformPoseCommandCfg


class SelectedUniformPoseCommand(CommandTerm):
    """Sample root-frame pose commands only where the configured scene entity exists."""

    cfg: SelectedUniformPoseCommandCfg

    def __init__(self, cfg: SelectedUniformPoseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.pose_command = torch.zeros(self.num_envs, 7, device=self.device)
        self._all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._reference_cfg = cfg.reference_cfg
        self._tracked_cfg = cfg.tracked_cfg
        self._reference_cfg.resolve(env.scene)
        self._tracked_cfg.resolve(env.scene)
        self._reference = env.scene[self._reference_cfg.name]
        self._tracked = env.scene[self._tracked_cfg.name]

    @property
    def command(self) -> torch.Tensor:
        """Pose command, shape ``(num_envs, 7)``."""
        return self.pose_command

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        global_env_ids = self._all_env_ids[env_ids]
        _, selected_env_ids = self._reference_cfg.select(global_env_ids)
        ranges = self.cfg.ranges
        sample = torch.empty(len(selected_env_ids), device=self.device)
        self.pose_command[selected_env_ids, 0] = sample.uniform_(*ranges.pos_x)
        self.pose_command[selected_env_ids, 1] = sample.uniform_(*ranges.pos_y)
        self.pose_command[selected_env_ids, 2] = sample.uniform_(*ranges.pos_z)
        euler = torch.empty(len(selected_env_ids), 3, device=self.device)
        euler[:, 0].uniform_(*ranges.roll)
        euler[:, 1].uniform_(*ranges.pitch)
        euler[:, 2].uniform_(*ranges.yaw)
        self.pose_command[selected_env_ids, 3:] = quat_unique(quat_from_euler_xyz(*euler.unbind(-1)))

    def _update_metrics(self) -> None:
        pass

    def _update_command(self) -> None:
        pass

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis and not hasattr(self, "goal_pose_visualizer"):
            self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
        if not hasattr(self, "goal_pose_visualizer"):
            return
        self.goal_pose_visualizer.set_visibility(debug_vis)
        self.current_pose_visualizer.set_visibility(debug_vis)

    def _debug_vis_callback(self, event) -> None:
        if not self._reference.is_initialized or not self._tracked.is_initialized:
            return

        reference_pose_w = self._reference.data.root_link_pose_w.torch
        command = self.pose_command[self._reference_cfg.env_ids]
        goal_pos_w, goal_quat_w = combine_frame_transforms(
            reference_pose_w[:, :3], reference_pose_w[:, 3:7], command[:, :3], command[:, 3:]
        )
        self.goal_pose_visualizer.visualize(
            goal_pos_w,
            goal_quat_w,
            environment_ids=self._reference_cfg.env_ids,
        )

        if self._tracked_cfg.body_names is None:
            tracked_pose_w = self._tracked.data.root_link_pose_w.torch
        else:
            tracked_pose_w = self._tracked.data.body_link_pose_w.torch[:, self._tracked_cfg.body_ids]
            if tracked_pose_w.shape[1] != 1:
                raise ValueError(
                    f"Expected '{self._tracked_cfg.name}' to select one tracked body, got {tracked_pose_w.shape[1]}."
                )
            tracked_pose_w = tracked_pose_w[:, 0]
        self.current_pose_visualizer.visualize(
            tracked_pose_w[:, :3],
            tracked_pose_w[:, 3:7],
            environment_ids=self._tracked_cfg.env_ids,
        )
