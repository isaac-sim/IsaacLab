# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for PVA sensor data containers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from isaaclab.utils.leapp import (
    POSE7_ELEMENT_NAMES,
    QUAT_XYZW_ELEMENT_NAMES,
    XYZ_ELEMENT_NAMES,
    InputKindEnum,
    leapp_tensor_semantics,
)
from isaaclab.utils.warp import ProxyArray


class BasePvaData(ABC):
    """Data container for the PVA sensor.

    This base class defines the interface for PVA sensor data. Backend-specific
    implementations should inherit from this class and provide the actual data storage.
    """

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSE, element_names=POSE7_ELEMENT_NAMES)
    def pose_w(self) -> ProxyArray | None:
        """Pose of the sensor origin in world frame [m, unitless].

        Shape is (num_instances,), dtype = wp.transformf. In torch this resolves to (num_instances, 7).
        The pose is provided in (x, y, z, qx, qy, qz, qw) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_POSITION, element_names=XYZ_ELEMENT_NAMES)
    def pos_w(self) -> ProxyArray:
        """Position of the sensor origin in world frame [m].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ROTATION, element_names=QUAT_XYZW_ELEMENT_NAMES)
    def quat_w(self) -> ProxyArray:
        """Orientation of the sensor origin in world frame.

        Shape is (num_instances,), dtype = wp.quatf. In torch this resolves to (num_instances, 4).
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.VECTOR3D, element_names=XYZ_ELEMENT_NAMES)
    def projected_gravity_b(self) -> ProxyArray:
        """Gravity direction unit vector projected on the PVA frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_LINEAR_VELOCITY, element_names=XYZ_ELEMENT_NAMES)
    def lin_vel_b(self) -> ProxyArray:
        """PVA frame linear velocity relative to the world expressed in PVA frame [m/s].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ANGULAR_VELOCITY, element_names=XYZ_ELEMENT_NAMES)
    def ang_vel_b(self) -> ProxyArray:
        """PVA frame angular velocity relative to the world expressed in PVA frame [rad/s].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_LINEAR_ACCELERATION, element_names=XYZ_ELEMENT_NAMES)
    def lin_acc_b(self) -> ProxyArray:
        """Linear acceleration (coordinate) in the PVA frame [m/s^2].

        Equal to -g in freefall, zero at rest.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ANGULAR_ACCELERATION, element_names=XYZ_ELEMENT_NAMES)
    def ang_acc_b(self) -> ProxyArray:
        """PVA frame angular acceleration relative to the world expressed in PVA frame [rad/s^2].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        raise NotImplementedError
