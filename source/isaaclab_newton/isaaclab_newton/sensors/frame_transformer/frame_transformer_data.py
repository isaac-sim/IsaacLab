# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp

from isaaclab.sensors.frame_transformer import BaseFrameTransformerData


class FrameTransformerData(BaseFrameTransformerData):
    """Data container for the Newton frame transformer sensor.

    The quaternion fields store components in (w, x, y, z) order as :class:`wp.vec4f` arrays.
    """

    @property
    def target_frame_names(self) -> list[str]:
        """Target frame names (order matches data ordering)."""
        return self._target_frame_names

    @property
    def target_pose_source(self) -> wp.array | None:
        """Pose of target frame(s) relative to source frame.

        Not computed for the Newton backend. Returns ``None``.
        """
        return None

    @property
    def target_pos_source(self) -> wp.array:
        """Position of target frame(s) relative to source frame.

        Shape is (num_instances, num_target_frames), dtype = :class:`wp.vec3f`.
        """
        return self._target_pos_source

    @property
    def target_quat_source(self) -> wp.array:
        """Orientation of target frame(s) relative to source frame.

        Shape is (num_instances, num_target_frames), dtype = :class:`wp.vec4f` in (w, x, y, z) order.
        """
        return self._target_quat_source

    @property
    def target_pose_w(self) -> wp.array | None:
        """Pose of target frame(s) after offset in world frame.

        Not computed for the Newton backend. Returns ``None``.
        """
        return None

    @property
    def target_pos_w(self) -> wp.array:
        """Position of target frame(s) after offset in world frame.

        Shape is (num_instances, num_target_frames), dtype = :class:`wp.vec3f`.
        """
        return self._target_pos_w

    @property
    def target_quat_w(self) -> wp.array:
        """Orientation of target frame(s) after offset in world frame.

        Shape is (num_instances, num_target_frames), dtype = :class:`wp.vec4f` in (w, x, y, z) order.
        """
        return self._target_quat_w

    @property
    def source_pose_w(self) -> wp.array | None:
        """Pose of source frame after offset in world frame.

        Not computed for the Newton backend. Returns ``None``.
        """
        return None

    @property
    def source_pos_w(self) -> wp.array:
        """Position of source frame after offset in world frame.

        Shape is (num_instances,), dtype = :class:`wp.vec3f`.
        """
        return self._source_pos_w

    @property
    def source_quat_w(self) -> wp.array:
        """Orientation of source frame after offset in world frame.

        Shape is (num_instances,), dtype = :class:`wp.vec4f` in (w, x, y, z) order.
        """
        return self._source_quat_w

    def create_buffers(
        self,
        num_envs: int,
        num_target_frames: int,
        target_frame_names: list[str],
        device: str,
    ) -> None:
        """Allocate internal warp buffers.

        Args:
            num_envs: Number of parallel environments.
            num_target_frames: Number of tracked target frames.
            target_frame_names: Ordered list of target frame names.
            device: Warp device string (e.g. ``"cuda:0"``).
        """
        self._num_envs = num_envs
        self._device = device
        self._num_target_frames = num_target_frames
        self._target_frame_names = target_frame_names
        self._source_pos_w = wp.zeros(num_envs, dtype=wp.vec3f, device=device)
        self._source_quat_w = wp.zeros(num_envs, dtype=wp.vec4f, device=device)
        self._target_pos_w = wp.zeros((num_envs, num_target_frames), dtype=wp.vec3f, device=device)
        self._target_quat_w = wp.zeros((num_envs, num_target_frames), dtype=wp.vec4f, device=device)
        self._target_pos_source = wp.zeros((num_envs, num_target_frames), dtype=wp.vec3f, device=device)
        self._target_quat_source = wp.zeros((num_envs, num_target_frames), dtype=wp.vec4f, device=device)
