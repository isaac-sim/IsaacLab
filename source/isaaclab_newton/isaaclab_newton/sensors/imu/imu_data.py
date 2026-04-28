# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp

from isaaclab.sensors.imu import BaseImuData
from isaaclab.utils.warp import ProxyArray


class ImuData(BaseImuData):
    """Data container for the Newton IMU sensor."""

    def __init__(self):
        self._ang_vel_b: wp.array | None = None
        self._lin_acc_b: wp.array | None = None
        self._ang_vel_b_ta: ProxyArray | None = None
        self._lin_acc_b_ta: ProxyArray | None = None

    @property
    def ang_vel_b(self) -> ProxyArray | None:
        """IMU frame angular velocity relative to the world expressed in IMU frame [rad/s].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        ``None`` before the simulation is initialized.
        """
        if self._ang_vel_b is None:
            return None
        if self._ang_vel_b_ta is None:
            self._ang_vel_b_ta = ProxyArray(self._ang_vel_b)
        return self._ang_vel_b_ta

    @property
    def lin_acc_b(self) -> ProxyArray | None:
        """Linear acceleration (proper) in the IMU frame [m/s^2].

        Zero in freefall, +g upward at rest.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        ``None`` before the simulation is initialized.
        """
        if self._lin_acc_b is None:
            return None
        if self._lin_acc_b_ta is None:
            self._lin_acc_b_ta = ProxyArray(self._lin_acc_b)
        return self._lin_acc_b_ta

    def create_buffers(self, num_envs: int, device: str) -> None:
        """Create internal buffers for sensor data.

        Args:
            num_envs: Number of environments.
            device: Device for array storage.
        """
        self._ang_vel_b = wp.zeros(num_envs, dtype=wp.vec3f, device=device)
        self._lin_acc_b = wp.zeros(num_envs, dtype=wp.vec3f, device=device)
        self._ang_vel_b_ta = None
        self._lin_acc_b_ta = None
