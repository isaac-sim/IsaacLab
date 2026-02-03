# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock IMU sensor for testing without Isaac Sim."""

from __future__ import annotations

from collections.abc import Sequence

import torch


class MockImuData:
    """Mock data container for IMU sensor.

    This class mimics the interface of BaseImuData for testing purposes.
    All tensor properties return zero tensors with correct shapes if not explicitly set.
    """

    def __init__(self, num_instances: int, device: str = "cpu"):
        """Initialize mock IMU data.

        Args:
            num_instances: Number of sensor instances.
            device: Device for tensor allocation.
        """
        self._num_instances = num_instances
        self._device = device

        # Internal storage for mock data
        self._pos_w: torch.Tensor | None = None
        self._quat_w: torch.Tensor | None = None
        self._projected_gravity_b: torch.Tensor | None = None
        self._lin_vel_b: torch.Tensor | None = None
        self._ang_vel_b: torch.Tensor | None = None
        self._lin_acc_b: torch.Tensor | None = None
        self._ang_acc_b: torch.Tensor | None = None

    # -- Properties --

    @property
    def pos_w(self) -> torch.Tensor:
        """Position of sensor origin in world frame. Shape: (N, 3)."""
        if self._pos_w is None:
            return torch.zeros(self._num_instances, 3, device=self._device)
        return self._pos_w

    @property
    def quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) in world frame. Shape: (N, 4)."""
        if self._quat_w is None:
            # Default to identity quaternion (w, x, y, z) = (1, 0, 0, 0)
            quat = torch.zeros(self._num_instances, 4, device=self._device)
            quat[:, 0] = 1.0
            return quat
        return self._quat_w

    @property
    def pose_w(self) -> torch.Tensor:
        """Pose in world frame (pos + quat). Shape: (N, 7)."""
        return torch.cat([self.pos_w, self.quat_w], dim=-1)

    @property
    def projected_gravity_b(self) -> torch.Tensor:
        """Gravity direction in IMU body frame. Shape: (N, 3)."""
        if self._projected_gravity_b is None:
            # Default gravity pointing down in body frame
            gravity = torch.zeros(self._num_instances, 3, device=self._device)
            gravity[:, 2] = -1.0
            return gravity
        return self._projected_gravity_b

    @property
    def lin_vel_b(self) -> torch.Tensor:
        """Linear velocity in IMU body frame. Shape: (N, 3)."""
        if self._lin_vel_b is None:
            return torch.zeros(self._num_instances, 3, device=self._device)
        return self._lin_vel_b

    @property
    def ang_vel_b(self) -> torch.Tensor:
        """Angular velocity in IMU body frame. Shape: (N, 3)."""
        if self._ang_vel_b is None:
            return torch.zeros(self._num_instances, 3, device=self._device)
        return self._ang_vel_b

    @property
    def lin_acc_b(self) -> torch.Tensor:
        """Linear acceleration in IMU body frame. Shape: (N, 3)."""
        if self._lin_acc_b is None:
            return torch.zeros(self._num_instances, 3, device=self._device)
        return self._lin_acc_b

    @property
    def ang_acc_b(self) -> torch.Tensor:
        """Angular acceleration in IMU body frame. Shape: (N, 3)."""
        if self._ang_acc_b is None:
            return torch.zeros(self._num_instances, 3, device=self._device)
        return self._ang_acc_b

    # -- Setters --

    def set_pos_w(self, value: torch.Tensor) -> None:
        """Set position in world frame."""
        self._pos_w = value.to(self._device)

    def set_quat_w(self, value: torch.Tensor) -> None:
        """Set orientation in world frame."""
        self._quat_w = value.to(self._device)

    def set_projected_gravity_b(self, value: torch.Tensor) -> None:
        """Set projected gravity in body frame."""
        self._projected_gravity_b = value.to(self._device)

    def set_lin_vel_b(self, value: torch.Tensor) -> None:
        """Set linear velocity in body frame."""
        self._lin_vel_b = value.to(self._device)

    def set_ang_vel_b(self, value: torch.Tensor) -> None:
        """Set angular velocity in body frame."""
        self._ang_vel_b = value.to(self._device)

    def set_lin_acc_b(self, value: torch.Tensor) -> None:
        """Set linear acceleration in body frame."""
        self._lin_acc_b = value.to(self._device)

    def set_ang_acc_b(self, value: torch.Tensor) -> None:
        """Set angular acceleration in body frame."""
        self._ang_acc_b = value.to(self._device)

    def set_mock_data(
        self,
        pos_w: torch.Tensor | None = None,
        quat_w: torch.Tensor | None = None,
        projected_gravity_b: torch.Tensor | None = None,
        lin_vel_b: torch.Tensor | None = None,
        ang_vel_b: torch.Tensor | None = None,
        lin_acc_b: torch.Tensor | None = None,
        ang_acc_b: torch.Tensor | None = None,
    ) -> None:
        """Bulk setter for mock data.

        Args:
            pos_w: Position in world frame. Shape: (N, 3).
            quat_w: Orientation (w, x, y, z) in world frame. Shape: (N, 4).
            projected_gravity_b: Gravity direction in body frame. Shape: (N, 3).
            lin_vel_b: Linear velocity in body frame. Shape: (N, 3).
            ang_vel_b: Angular velocity in body frame. Shape: (N, 3).
            lin_acc_b: Linear acceleration in body frame. Shape: (N, 3).
            ang_acc_b: Angular acceleration in body frame. Shape: (N, 3).
        """
        if pos_w is not None:
            self.set_pos_w(pos_w)
        if quat_w is not None:
            self.set_quat_w(quat_w)
        if projected_gravity_b is not None:
            self.set_projected_gravity_b(projected_gravity_b)
        if lin_vel_b is not None:
            self.set_lin_vel_b(lin_vel_b)
        if ang_vel_b is not None:
            self.set_ang_vel_b(ang_vel_b)
        if lin_acc_b is not None:
            self.set_lin_acc_b(lin_acc_b)
        if ang_acc_b is not None:
            self.set_ang_acc_b(ang_acc_b)


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
        # No-op for mock - data persists until explicitly changed
        pass

    def update(self, dt: float, force_recompute: bool = False) -> None:
        """Update sensor.

        Args:
            dt: Time step since last update.
            force_recompute: Force recomputation of buffers.
        """
        # No-op for mock - data is set explicitly
        pass
