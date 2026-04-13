# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock IMU sensor for testing without Isaac Sim."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import warp as wp

try:
    from isaaclab.sensors.imu.base_imu_data import BaseImuData
except (ImportError, ModuleNotFoundError):
    # Direct import bypassing isaaclab.sensors.__init__.py (which needs omni)
    import importlib.util
    from pathlib import Path

    _file = Path(__file__).resolve().parents[3] / "sensors" / "imu" / "base_imu_data.py"
    _spec = importlib.util.spec_from_file_location("_base_imu_data", str(_file))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    BaseImuData = _mod.BaseImuData


class MockImuData(BaseImuData):
    """Mock data container for IMU sensor.

    This class mimics the interface of BaseImuData for testing purposes.
    The IMU only provides angular velocity and linear acceleration.
    """

    def __init__(self, num_instances: int, device: str = "cpu"):
        """Initialize mock IMU data.

        Args:
            num_instances: Number of sensor instances.
            device: Device for tensor allocation.
        """
        self._num_instances = num_instances
        self.device = device

        self._ang_vel_b: wp.array | None = None
        self._lin_acc_b: wp.array | None = None

    # -- Properties --

    @property
    def ang_vel_b(self) -> wp.array:
        """Angular velocity in IMU body frame [rad/s]. Shape: (N, 3)."""
        if self._ang_vel_b is None:
            return wp.zeros(shape=(self._num_instances, 3), dtype=wp.float32, device=self.device)
        return self._ang_vel_b

    @property
    def lin_acc_b(self) -> wp.array:
        """Linear acceleration in IMU body frame [m/s^2]. Shape: (N, 3)."""
        if self._lin_acc_b is None:
            return wp.zeros(shape=(self._num_instances, 3), dtype=wp.float32, device=self.device)
        return self._lin_acc_b

    # -- Setters --

    def set_ang_vel_b(self, value: torch.Tensor) -> None:
        """Set angular velocity in body frame."""
        self._ang_vel_b = wp.from_torch(value.to(self.device).contiguous(), dtype=wp.float32)

    def set_lin_acc_b(self, value: torch.Tensor) -> None:
        """Set linear acceleration in body frame."""
        self._lin_acc_b = wp.from_torch(value.to(self.device).contiguous(), dtype=wp.float32)

    def set_mock_data(
        self,
        ang_vel_b: torch.Tensor | None = None,
        lin_acc_b: torch.Tensor | None = None,
    ) -> None:
        """Bulk setter for mock data.

        Args:
            ang_vel_b: Angular velocity in body frame [rad/s]. Shape: (N, 3).
            lin_acc_b: Linear acceleration in body frame [m/s^2]. Shape: (N, 3).
        """
        if ang_vel_b is not None:
            self.set_ang_vel_b(ang_vel_b)
        if lin_acc_b is not None:
            self.set_lin_acc_b(lin_acc_b)


class MockImu:
    """Mock IMU sensor for testing without Isaac Sim.

    This class mimics the interface of BaseImu for testing purposes.
    It provides the same properties and methods but without simulation dependencies.
    """

    def __init__(
        self,
        num_instances: int,
        device: str = "cpu",
    ):
        """Initialize mock IMU sensor.

        Args:
            num_instances: Number of sensor instances.
            device: Device for tensor allocation.
        """
        self._num_instances = num_instances
        self._device = device
        self._data = MockImuData(num_instances, device)

    # -- Properties --

    @property
    def data(self) -> MockImuData:
        """Data container for the sensor."""
        return self._data

    @property
    def num_instances(self) -> int:
        """Number of sensor instances."""
        return self._num_instances

    @property
    def device(self) -> str:
        """Device for tensor allocation."""
        return self._device

    # -- Methods --

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset sensor state for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, resets all.
        """
        pass

    def update(self, dt: float, force_recompute: bool = False) -> None:
        """Update sensor.

        Args:
            dt: Time step since last update.
            force_recompute: Force recomputation of buffers.
        """
        pass
