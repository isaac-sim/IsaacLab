# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for IMU sensor data containers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from isaaclab.utils.leapp import (
    XYZ_ELEMENT_NAMES,
    InputKindEnum,
    leapp_tensor_semantics,
)
from isaaclab.utils.warp import ProxyArray


class BaseImuData(ABC):
    """Data container for the IMU sensor.

    This base class defines the interface for IMU sensor data. Backend-specific
    implementations should inherit from this class and provide the actual data storage.

    Unlike the PVA sensor, the IMU only provides the two physical quantities that a
    real inertial measurement unit measures: angular velocity and linear acceleration.
    """

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_ANGULAR_VELOCITY, element_names=XYZ_ELEMENT_NAMES)
    def ang_vel_b(self) -> ProxyArray:
        """IMU frame angular velocity relative to the world expressed in IMU frame [rad/s].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    @leapp_tensor_semantics(kind=InputKindEnum.BODY_LINEAR_ACCELERATION, element_names=XYZ_ELEMENT_NAMES)
    def lin_acc_b(self) -> ProxyArray:
        """Linear acceleration (proper) in the IMU frame [m/s^2].

        Zero in freefall, +g upward at rest.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        raise NotImplementedError
