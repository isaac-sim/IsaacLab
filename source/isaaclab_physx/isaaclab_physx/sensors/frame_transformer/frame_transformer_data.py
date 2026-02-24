# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp

from isaaclab.sensors.frame_transformer import BaseFrameTransformerData

from isaaclab_physx.sensors.kernels import concat_pos_and_quat_to_pose_1d_kernel, concat_pos_and_quat_to_pose_kernel


class FrameTransformerData(BaseFrameTransformerData):
    """Data container for the PhysX frame transformer sensor."""

    @property
    def target_frame_names(self) -> list[str]:
        """Target frame names (order matches data ordering)."""
        return self._target_frame_names

    @property
    def target_pose_source(self) -> wp.array:
        """Pose of target frame(s) relative to source frame.

        Shape is (num_instances, num_target_frames), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_target_frames, 7). The pose is provided in (x, y, z, qx, qy, qz, qw) format.
        """
        wp.launch(
            concat_pos_and_quat_to_pose_kernel,
            dim=(self._num_envs, self._num_target_frames),
            inputs=[self._target_pos_source, self._target_quat_source],
            outputs=[self._target_pose_source],
            device=self._device,
        )
        return self._target_pose_source

    @property
    def target_pos_source(self) -> wp.array:
        """Position of target frame(s) relative to source frame.

        Shape is (num_instances, num_target_frames), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_target_frames, 3).
        """
        return self._target_pos_source

    @property
    def target_quat_source(self) -> wp.array:
        """Orientation of target frame(s) relative to source frame.

        Shape is (num_instances, num_target_frames), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_target_frames, 4). The orientation is provided in (x, y, z, w) format.
        """
        return self._target_quat_source

    @property
    def target_pose_w(self) -> wp.array:
        """Pose of target frame(s) after offset in world frame.

        Shape is (num_instances, num_target_frames), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_target_frames, 7). The pose is provided in (x, y, z, qx, qy, qz, qw) format.
        """
        wp.launch(
            concat_pos_and_quat_to_pose_kernel,
            dim=(self._num_envs, self._num_target_frames),
            inputs=[self._target_pos_w, self._target_quat_w],
            outputs=[self._target_pose_w],
            device=self._device,
        )
        return self._target_pose_w

    @property
    def target_pos_w(self) -> wp.array:
        """Position of target frame(s) after offset in world frame.

        Shape is (num_instances, num_target_frames), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_target_frames, 3).
        """
        return self._target_pos_w

    @property
    def target_quat_w(self) -> wp.array:
        """Orientation of target frame(s) after offset in world frame.

        Shape is (num_instances, num_target_frames), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_target_frames, 4). The orientation is provided in (x, y, z, w) format.
        """
        return self._target_quat_w

    @property
    def source_pose_w(self) -> wp.array:
        """Pose of source frame after offset in world frame.

        Shape is (num_instances,), dtype = wp.transformf. In torch this resolves to (num_instances, 7).
        The pose is provided in (x, y, z, qx, qy, qz, qw) format.
        """
        wp.launch(
            concat_pos_and_quat_to_pose_1d_kernel,
            dim=self._num_envs,
            inputs=[self._source_pos_w, self._source_quat_w],
            outputs=[self._source_pose_w],
            device=self._device,
        )
        return self._source_pose_w

    @property
    def source_pos_w(self) -> wp.array:
        """Position of source frame after offset in world frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        return self._source_pos_w

    @property
    def source_quat_w(self) -> wp.array:
        """Orientation of source frame after offset in world frame.

        Shape is (num_instances,), dtype = wp.quatf. In torch this resolves to (num_instances, 4).
        The orientation is provided in (x, y, z, w) format.
        """
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
        self._num_envs = num_envs
        self._device = device
        self._num_target_frames = num_target_frames
        self._target_frame_names = target_frame_names
        self._source_pose_w = wp.zeros(num_envs, dtype=wp.transformf, device=device)
        self._source_pos_w = wp.zeros(num_envs, dtype=wp.vec3f, device=device)
        self._source_quat_w = wp.zeros(num_envs, dtype=wp.quatf, device=device)
        self._target_pose_w = wp.zeros((num_envs, num_target_frames), dtype=wp.transformf, device=device)
        self._target_pos_w = wp.zeros((num_envs, num_target_frames), dtype=wp.vec3f, device=device)
        self._target_quat_w = wp.zeros((num_envs, num_target_frames), dtype=wp.quatf, device=device)
        self._target_pose_source = wp.zeros((num_envs, num_target_frames), dtype=wp.transformf, device=device)
        self._target_pos_source = wp.zeros((num_envs, num_target_frames), dtype=wp.vec3f, device=device)
        self._target_quat_source = wp.zeros((num_envs, num_target_frames), dtype=wp.quatf, device=device)

        # Initialize quaternions to identity (w=1). wp.zeros gives (0,0,0,0) not (0,0,0,1).

        wp.to_torch(self._source_quat_w)[:, 3] = 1.0
        wp.to_torch(self._target_quat_w)[:, :, 3] = 1.0
        wp.to_torch(self._target_quat_source)[:, :, 3] = 1.0
