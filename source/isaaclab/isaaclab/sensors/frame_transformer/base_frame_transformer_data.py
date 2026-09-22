# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for frame transformer sensor data containers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from isaaclab.utils.leapp import (
    POSE7_ELEMENT_NAMES,
    QUAT_XYZW_ELEMENT_NAMES,
    XYZ_ELEMENT_NAMES,
    InputKindEnum,
    leapp_tensor_semantics,
    target_frame_pose_resolver,
    target_frame_quat_resolver,
    target_frame_xyz_resolver,
)
from isaaclab.utils.warp import ProxyArray


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
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSE, element_names_resolver=target_frame_pose_resolver)
    def target_pose_source(self) -> ProxyArray | None:
        """Pose of the target frame(s) relative to source frame.

        Shape is (num_instances, num_target_frames), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_target_frames, 7). The pose is provided in (x, y, z, qx, qy, qz, qw) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSITION, element_names_resolver=target_frame_xyz_resolver)
    def target_pos_source(self) -> ProxyArray:
        """Position of the target frame(s) relative to source frame.

        Shape is (num_instances, num_target_frames), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_target_frames, 3).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ROTATION, element_names_resolver=target_frame_quat_resolver)
    def target_quat_source(self) -> ProxyArray:
        """Orientation of the target frame(s) relative to source frame.

        Shape is (num_instances, num_target_frames), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_target_frames, 4). The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSE, element_names_resolver=target_frame_pose_resolver)
    def target_pose_w(self) -> ProxyArray | None:
        """Pose of the target frame(s) after offset in world frame.

        Shape is (num_instances, num_target_frames), dtype = wp.transformf. In torch this resolves to
        (num_instances, num_target_frames, 7). The pose is provided in (x, y, z, qx, qy, qz, qw) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSITION, element_names_resolver=target_frame_xyz_resolver)
    def target_pos_w(self) -> ProxyArray:
        """Position of the target frame(s) after offset in world frame.

        Shape is (num_instances, num_target_frames), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_target_frames, 3).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ROTATION, element_names_resolver=target_frame_quat_resolver)
    def target_quat_w(self) -> ProxyArray:
        """Orientation of the target frame(s) after offset in world frame.

        Shape is (num_instances, num_target_frames), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_target_frames, 4). The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSE, element_names=POSE7_ELEMENT_NAMES)
    def source_pose_w(self) -> ProxyArray | None:
        """Pose of the source frame after offset in world frame.

        Shape is (num_instances,), dtype = wp.transformf. In torch this resolves to (num_instances, 7).
        The pose is provided in (x, y, z, qx, qy, qz, qw) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSITION, element_names=XYZ_ELEMENT_NAMES)
    def source_pos_w(self) -> ProxyArray:
        """Position of the source frame after offset in world frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ROTATION, element_names=QUAT_XYZW_ELEMENT_NAMES)
    def source_quat_w(self) -> ProxyArray:
        """Orientation of the source frame after offset in world frame.

        Shape is (num_instances,), dtype = wp.quatf. In torch this resolves to (num_instances, 4).
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError
