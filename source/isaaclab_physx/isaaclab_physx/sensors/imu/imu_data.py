# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging

import torch

from isaaclab.sensors.imu import BaseImuData

logger = logging.getLogger(__name__)


class ImuData(BaseImuData):
    """Data container for the PhysX Imu sensor."""

    @property
    def pose_w(self) -> torch.Tensor:
        """Pose of the sensor origin in world frame. Shape is (N, 7). Quaternion in xyzw order."""
        logger.warning(
            "The `pose_w` property will be deprecated in a future release. Please use a dedicated sensor to measure"
            "sensor poses in world frame."
        )
        return torch.cat((self._pos_w, self._quat_w), dim=-1)

    @property
    def pos_w(self) -> torch.Tensor:
        """Position of the sensor origin in world frame. Shape is (N, 3)."""
        logger.warning(
            "The `pos_w` property will be deprecated in a future release. Please use a dedicated sensor to measure"
            "sensor positions in world frame."
        )
        return self._pos_w

    @property
    def quat_w(self) -> torch.Tensor:
        """Orientation of the sensor origin in quaternion (x, y, z, w) in world frame. Shape is (N, 4)."""
        logger.warning(
            "The `quat_w` property will be deprecated in a future release. Please use a dedicated sensor to measure"
            "sensor orientations in world frame."
        )
        return self._quat_w

    @property
    def projected_gravity_b(self) -> torch.Tensor:
        """Gravity direction unit vector projected on the imu frame. Shape is (N, 3)."""
        return self._projected_gravity_b

    @property
    def lin_vel_b(self) -> torch.Tensor:
        """IMU frame linear velocity relative to the world expressed in IMU frame. Shape is (N, 3)."""
        return self._lin_vel_b

    @property
    def ang_vel_b(self) -> torch.Tensor:
        """IMU frame angular velocity relative to the world expressed in IMU frame. Shape is (N, 3)."""
        return self._ang_vel_b

    @property
    def lin_acc_b(self) -> torch.Tensor:
        """IMU frame linear acceleration relative to the world expressed in IMU frame. Shape is (N, 3)."""
        return self._lin_acc_b

    @property
    def ang_acc_b(self) -> torch.Tensor:
        """IMU frame angular acceleration relative to the world expressed in IMU frame. Shape is (N, 3)."""
        return self._ang_acc_b

    def create_buffers(self, num_envs: int, device: str) -> None:
        """Create internal buffers for sensor data.

        Args:
            num_envs: Number of environments.
            device: Device for tensor storage.
        """
        self._pos_w = torch.zeros(num_envs, 3, device=device)
        self._quat_w = torch.zeros(num_envs, 4, device=device)
        self._quat_w[:, 0] = 1.0
        self._projected_gravity_b = torch.zeros(num_envs, 3, device=device)
        self._lin_vel_b = torch.zeros(num_envs, 3, device=device)
        self._ang_vel_b = torch.zeros(num_envs, 3, device=device)
        self._lin_acc_b = torch.zeros(num_envs, 3, device=device)
        self._ang_acc_b = torch.zeros(num_envs, 3, device=device)
