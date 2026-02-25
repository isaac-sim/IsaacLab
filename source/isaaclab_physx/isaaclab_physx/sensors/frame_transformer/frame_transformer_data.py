# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp

from isaaclab.sensors.frame_transformer import BaseFrameTransformerData


class FrameTransformerData(BaseFrameTransformerData):
    """Data container for the PhysX frame transformer sensor."""

    @property
    def target_frame_names(self) -> list[str]:
        """Target frame names (order matches data ordering)."""
        return self._target_frame_names

    @property
    def target_pose_source(self) -> None:
        """Not available for warp backend (cannot concatenate vec3f and quatf). Use target_pos_source /
        target_quat_source."""
        return None

    @property
    def target_pos_source(self) -> wp.array:
        """Position of target frame(s) relative to source frame. Shape is (N, M) vec3f."""
        return self._target_pos_source

    @property
    def target_quat_source(self) -> wp.array:
        """Orientation of target frame(s) relative to source frame (x, y, z, w). Shape is (N, M) quatf."""
        return self._target_quat_source

    @property
    def target_pose_w(self) -> None:
        """Not available for warp backend (cannot concatenate vec3f and quatf). Use target_pos_w / target_quat_w."""
        return None

    @property
    def target_pos_w(self) -> wp.array:
        """Position of target frame(s) after offset in world frame. Shape is (N, M) vec3f."""
        return self._target_pos_w

    @property
    def target_quat_w(self) -> wp.array:
        """Orientation of target frame(s) after offset in world frame (x, y, z, w). Shape is (N, M) quatf."""
        return self._target_quat_w

    @property
    def source_pose_w(self) -> None:
        """Not available for warp backend (cannot concatenate vec3f and quatf). Use source_pos_w / source_quat_w."""
        return None

    @property
    def source_pos_w(self) -> wp.array:
        """Position of source frame after offset in world frame. Shape is (N,) vec3f."""
        return self._source_pos_w

    @property
    def source_quat_w(self) -> wp.array:
        """Orientation of source frame after offset in world frame (x, y, z, w). Shape is (N,) quatf."""
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
        self._source_pos_w = wp.zeros(num_envs, dtype=wp.vec3f, device=device)
        self._source_quat_w = wp.zeros(num_envs, dtype=wp.quatf, device=device)
        self._target_pos_w = wp.zeros((num_envs, num_target_frames), dtype=wp.vec3f, device=device)
        self._target_quat_w = wp.zeros((num_envs, num_target_frames), dtype=wp.quatf, device=device)
        self._target_pos_source = wp.zeros((num_envs, num_target_frames), dtype=wp.vec3f, device=device)
        self._target_quat_source = wp.zeros((num_envs, num_target_frames), dtype=wp.quatf, device=device)

        # Initialize quaternions to identity (w=1). wp.zeros gives (0,0,0,0) not (0,0,0,1).

        wp.to_torch(self._source_quat_w)[:, 3] = 1.0
        wp.to_torch(self._target_quat_w)[:, :, 3] = 1.0
        wp.to_torch(self._target_quat_source)[:, :, 3] = 1.0
