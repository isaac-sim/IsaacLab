# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp

from isaaclab.sensors.imu import BaseImuData
from isaaclab.utils.warp import TorchArray


class ImuData(BaseImuData):
    """Data container for the PhysX IMU sensor."""

    @property
    def ang_vel_b(self) -> TorchArray:
        """IMU frame angular velocity relative to the world expressed in IMU frame [rad/s].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        if self._ang_vel_b_ta is None:
            self._ang_vel_b_ta = TorchArray(self._ang_vel_b)
        return self._ang_vel_b_ta

    @property
    def lin_acc_b(self) -> TorchArray:
        """IMU frame linear acceleration relative to the world expressed in IMU frame [m/s^2].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        if self._lin_acc_b_ta is None:
            self._lin_acc_b_ta = TorchArray(self._lin_acc_b)
        return self._lin_acc_b_ta

    def create_buffers(self, num_envs: int, device: str) -> None:
        """Create internal buffers for sensor data.

        Args:
            num_envs: Number of environments.
            device: Device for tensor storage.
        """
        self._num_envs = num_envs
        self._device = device
        self._ang_vel_b = wp.zeros(num_envs, dtype=wp.vec3f, device=device)
        self._lin_acc_b = wp.zeros(num_envs, dtype=wp.vec3f, device=device)

        # -- Pinned TorchArray cache (one per read property, lazily created on first access)
        self._ang_vel_b_ta: TorchArray | None = None
        self._lin_acc_b_ta: TorchArray | None = None
