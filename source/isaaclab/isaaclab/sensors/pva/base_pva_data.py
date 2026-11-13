# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for PVA sensor data containers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ...utils.leapp import (
    ANGULAR_VELOCITY_COMPONENT_AXIS,
    LINEAR_VELOCITY_COMPONENT_AXIS,
    POSE7_COMPONENT_AXIS,
    QUAT_XYZW_COMPONENT_AXIS,
    XYZ_COMPONENT_AXIS,
    Axis,
    AxisKind,
    ExpressionFrame,
    Kind,
    leapp_tensor_semantics,
)
from ...utils.warp import ProxyArray


class BasePvaData(ABC):
    """Data container for the PVA sensor.

    This base class defines the interface for PVA sensor data. Backend-specific
    implementations should inherit from this class and provide the actual data storage.
    """

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=Kind.FRAME_POSE, axes=[Axis(AxisKind.SYSTEM), POSE7_COMPONENT_AXIS])
    def pose_w(self) -> ProxyArray | None:
        """Pose of the sensor origin in world frame [m, unitless].

        Shape is (num_instances,), dtype = wp.transformf. In torch this resolves to (num_instances, 7).
        The pose is provided in (x, y, z, qx, qy, qz, qw) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=Kind.FRAME_POSITION, axes=[Axis(AxisKind.SYSTEM), XYZ_COMPONENT_AXIS])
    def pos_w(self) -> ProxyArray:
        """Position of the sensor origin in world frame [m].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=Kind.FRAME_ORIENTATION, axes=[Axis(AxisKind.SYSTEM), QUAT_XYZW_COMPONENT_AXIS])
    def quat_w(self) -> ProxyArray:
        """Orientation of the sensor origin in world frame.

        Shape is (num_instances,), dtype = wp.quatf. In torch this resolves to (num_instances, 4).
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=Kind.VECTOR3D, axes=[Axis(AxisKind.SYSTEM), XYZ_COMPONENT_AXIS])
    def projected_gravity_b(self) -> ProxyArray:
        """Gravity direction unit vector projected on the PVA frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(
        kind=Kind.FRAME_LINEAR_VELOCITY,
        axes=[Axis(AxisKind.SYSTEM), LINEAR_VELOCITY_COMPONENT_AXIS],
        expressed_in=ExpressionFrame.SELF,
    )
    def lin_vel_b(self) -> ProxyArray:
        """PVA frame linear velocity relative to the world expressed in PVA frame [m/s].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(
        kind=Kind.FRAME_ANGULAR_VELOCITY,
        axes=[Axis(AxisKind.SYSTEM), ANGULAR_VELOCITY_COMPONENT_AXIS],
        expressed_in=ExpressionFrame.SELF,
    )
    def ang_vel_b(self) -> ProxyArray:
        """PVA frame angular velocity relative to the world expressed in PVA frame [rad/s].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(
        kind=Kind.FRAME_LINEAR_ACCELERATION,
        axes=[Axis(AxisKind.SYSTEM), XYZ_COMPONENT_AXIS],
        expressed_in=ExpressionFrame.SELF,
    )
    def lin_acc_b(self) -> ProxyArray:
        """Linear acceleration (coordinate) in the PVA frame [m/s^2].

        Equal to -g in freefall, zero at rest.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(
        kind=Kind.FRAME_ANGULAR_ACCELERATION,
        axes=[Axis(AxisKind.SYSTEM), XYZ_COMPONENT_AXIS],
        expressed_in=ExpressionFrame.SELF,
    )
    def ang_acc_b(self) -> ProxyArray:
        """PVA frame angular acceleration relative to the world expressed in PVA frame [rad/s^2].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        raise NotImplementedError
