# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for frame transformer sensor data containers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import warp as wp


class BaseFrameTransformerData(ABC):
    """Data container for the frame transformer sensor.

    This base class defines the interface for frame transformer sensor data. Backend-specific
    implementations should inherit from this class and provide the actual data storage.
    """

    @property
    @abstractmethod
    def target_frame_names(self) -> list[str]:
        """Target frame names (order matches data ordering).

        Resolved from :attr:`FrameTransformerCfg.FrameCfg.name`.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def target_pose_source(self) -> wp.array | None:
        """Pose of the target frame(s) relative to source frame.

        Shape is (num_instances, num_target_frames), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_target_frames, 7). The pose is provided in (x, y, z, qx, qy, qz, qw) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def target_pos_source(self) -> wp.array:
        """Position of the target frame(s) relative to source frame.

        Shape is (num_instances, num_target_frames), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_target_frames, 3).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def target_quat_source(self) -> wp.array:
        """Orientation of the target frame(s) relative to source frame.

        Shape is (num_instances, num_target_frames), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_target_frames, 4). The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def target_pose_w(self) -> wp.array | None:
        """Pose of the target frame(s) after offset in world frame.

        Shape is (num_instances, num_target_frames), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_target_frames, 7). The pose is provided in (x, y, z, qx, qy, qz, qw) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def target_pos_w(self) -> wp.array:
        """Position of the target frame(s) after offset in world frame.

        Shape is (num_instances, num_target_frames), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_target_frames, 3).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def target_quat_w(self) -> wp.array:
        """Orientation of the target frame(s) after offset in world frame.

        Shape is (num_instances, num_target_frames), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_target_frames, 4). The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def source_pose_w(self) -> wp.array | None:
        """Pose of the source frame after offset in world frame.

        Shape is (num_instances,), dtype = wp.transformf. In torch this resolves to (num_instances, 7).
        The pose is provided in (x, y, z, qx, qy, qz, qw) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def source_pos_w(self) -> wp.array:
        """Position of the source frame after offset in world frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def source_quat_w(self) -> wp.array:
        """Orientation of the source frame after offset in world frame.

        Shape is (num_instances,), dtype = wp.quatf. In torch this resolves to (num_instances, 4).
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError
