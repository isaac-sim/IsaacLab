# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for IMU sensor data containers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import warp as wp


class BaseImuData(ABC):
    """Data container for the Imu sensor.

    This base class defines the interface for IMU sensor data. Backend-specific
    implementations should inherit from this class and provide the actual data storage.
    """

    @property
    @abstractmethod
    def pose_w(self) -> wp.array | None:
        """Pose of the sensor origin in world frame.

        Shape is (num_instances,), dtype = wp.transformf. In torch this resolves to (num_instances, 7).
        The pose is provided in (x, y, z, qx, qy, qz, qw) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def pos_w(self) -> wp.array:
        """Position of the sensor origin in world frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def quat_w(self) -> wp.array:
        """Orientation of the sensor origin in world frame.

        Shape is (num_instances,), dtype = wp.quatf. In torch this resolves to (num_instances, 4).
        The orientation is provided in (x, y, z, w) format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def projected_gravity_b(self) -> wp.array:
        """Gravity direction unit vector projected on the IMU frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def lin_vel_b(self) -> wp.array:
        """IMU frame linear velocity relative to the world expressed in IMU frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def ang_vel_b(self) -> wp.array:
        """IMU frame angular velocity relative to the world expressed in IMU frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def lin_acc_b(self) -> wp.array:
        """IMU frame linear acceleration relative to the world expressed in IMU frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def ang_acc_b(self) -> wp.array:
        """IMU frame angular acceleration relative to the world expressed in IMU frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        raise NotImplementedError
