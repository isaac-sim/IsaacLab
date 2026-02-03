# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.sensors.frame_transformer import BaseFrameTransformerData


class FrameTransformerData(BaseFrameTransformerData):
    """Data container for the PhysX frame transformer sensor."""

    @property
    def target_frame_names(self) -> list[str]:
        """Target frame names (order matches data ordering)."""
        return self._target_frame_names

    @property
    def target_pose_source(self) -> torch.Tensor:
        """Pose of target frame(s) relative to source frame. Shape is (N, M, 7). Quaternion in wxyz order."""
        return torch.cat([self._target_pos_source, self._target_quat_source], dim=-1)

    @property
    def target_pos_source(self) -> torch.Tensor:
        """Position of target frame(s) relative to source frame. Shape is (N, M, 3)."""
        return self._target_pos_source

    @property
    def target_quat_source(self) -> torch.Tensor:
        """Orientation of target frame(s) relative to source frame (x, y, z, w). Shape is (N, M, 4)."""
        return self._target_quat_source

    @property
    def target_pose_w(self) -> torch.Tensor:
        """Pose of target frame(s) after offset in world frame. Shape is (N, M, 7). Quaternion in xyzw order."""
        return torch.cat([self._target_pos_w, self._target_quat_w], dim=-1)

    @property
    def target_pos_w(self) -> torch.Tensor:
        """Position of target frame(s) after offset in world frame. Shape is (N, M, 3)."""
        return self._target_pos_w

    @property
    def target_quat_w(self) -> torch.Tensor:
        """Orientation of target frame(s) after offset in world frame (x, y, z, w). Shape is (N, M, 4)."""
        return self._target_quat_w

    @property
    def source_pose_w(self) -> torch.Tensor:
        """Pose of source frame after offset in world frame. Shape is (N, 7). Quaternion in xyzw order."""
        return torch.cat([self._source_pos_w, self._source_quat_w], dim=-1)

    @property
    def source_pos_w(self) -> torch.Tensor:
        """Position of source frame after offset in world frame. Shape is (N, 3)."""
        return self._source_pos_w

    @property
    def source_quat_w(self) -> torch.Tensor:
        """Orientation of source frame after offset in world frame (x, y, z, w). Shape is (N, 4)."""
        return self._source_quat_w

    def create_buffers(
        self,
        num_envs: int,
        num_target_frames: int,
        target_frame_names: list[str],
        device: str,
    ) -> None:
        """Create internal buffers for sensor data.

        Args:
            num_envs: Number of environments.
            num_target_frames: Number of target frames.
            target_frame_names: Names of target frames.
            device: Device for tensor storage.
        """
        self._target_frame_names = target_frame_names
        self._source_pos_w = torch.zeros(num_envs, 3, device=device)
        self._source_quat_w = torch.zeros(num_envs, 4, device=device)
        self._target_pos_w = torch.zeros(num_envs, num_target_frames, 3, device=device)
        self._target_quat_w = torch.zeros(num_envs, num_target_frames, 4, device=device)
        self._target_pos_source = torch.zeros(num_envs, num_target_frames, 3, device=device)
        self._target_quat_source = torch.zeros(num_envs, num_target_frames, 4, device=device)
