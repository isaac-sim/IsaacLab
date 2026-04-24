# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock PVA sensor for testing without Isaac Sim."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray

try:
    from isaaclab.sensors.pva.base_pva_data import BasePvaData
except (ImportError, ModuleNotFoundError):
    # Direct import bypassing isaaclab.sensors.__init__.py (which needs omni)
    import importlib.util
    from pathlib import Path

    _file = Path(__file__).resolve().parents[3] / "sensors" / "pva" / "base_pva_data.py"
    _spec = importlib.util.spec_from_file_location("_base_pva_data", str(_file))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    BasePvaData = _mod.BasePvaData


class MockPvaData(BasePvaData):
    """Mock data container for PVA sensor.

    This class mimics the interface of BasePvaData for testing purposes.
    All tensor properties return zero warp arrays with correct shapes if not explicitly set.
    """

    def __init__(self, num_instances: int, device: str = "cpu"):
        """Initialize mock PVA data.

        Args:
            num_instances: Number of sensor instances.
            device: Device for tensor allocation.
        """
        self._num_instances = num_instances
        self.device = device

        # Internal storage for mock data
        self._pos_w: wp.array | None = None
        self._quat_w: wp.array | None = None
        self._projected_gravity_b: wp.array | None = None
        self._lin_vel_b: wp.array | None = None
        self._ang_vel_b: wp.array | None = None
        self._lin_acc_b: wp.array | None = None
        self._ang_acc_b: wp.array | None = None

        # ProxyArray caches
        self._pos_w_ta: ProxyArray | None = None
        self._quat_w_ta: ProxyArray | None = None
        self._projected_gravity_b_ta: ProxyArray | None = None
        self._lin_vel_b_ta: ProxyArray | None = None
        self._ang_vel_b_ta: ProxyArray | None = None
        self._lin_acc_b_ta: ProxyArray | None = None
        self._ang_acc_b_ta: ProxyArray | None = None

    # -- Properties --

    @property
    def pos_w(self) -> ProxyArray:
        """Position of sensor origin in world frame. Shape: (N, 3)."""
        if self._pos_w is None:
            self._pos_w = wp.zeros(shape=(self._num_instances, 3), dtype=wp.float32, device=self.device)
            self._pos_w_ta = None
        if self._pos_w_ta is None:
            self._pos_w_ta = ProxyArray(self._pos_w)
        return self._pos_w_ta

    @property
    def quat_w(self) -> ProxyArray:
        """Orientation (x, y, z, w) in world frame. Shape: (N, 4)."""
        if self._quat_w is None:
            # Default to identity quaternion (x, y, z, w) = (0, 0, 0, 1)
            quat_np = np.zeros((self._num_instances, 4), dtype=np.float32)
            quat_np[:, 3] = 1.0  # w component is at index 3 in XYZW format
            self._quat_w = wp.array(quat_np, dtype=wp.float32, device=self.device)
            self._quat_w_ta = None
        if self._quat_w_ta is None:
            self._quat_w_ta = ProxyArray(self._quat_w)
        return self._quat_w_ta

    @property
    def pose_w(self) -> ProxyArray:
        """Pose in world frame (pos + quat). Shape: (N, 7)."""
        pos_t = self.pos_w.torch
        quat_t = self.quat_w.torch
        pose_t = torch.cat([pos_t, quat_t], dim=-1)
        return ProxyArray(wp.from_torch(pose_t.contiguous(), dtype=wp.float32))

    @property
    def projected_gravity_b(self) -> ProxyArray:
        """Gravity direction in PVA body frame. Shape: (N, 3)."""
        if self._projected_gravity_b is None:
            # Default gravity pointing down in body frame
            gravity_np = np.zeros((self._num_instances, 3), dtype=np.float32)
            gravity_np[:, 2] = -1.0
            self._projected_gravity_b = wp.array(gravity_np, dtype=wp.float32, device=self.device)
            self._projected_gravity_b_ta = None
        if self._projected_gravity_b_ta is None:
            self._projected_gravity_b_ta = ProxyArray(self._projected_gravity_b)
        return self._projected_gravity_b_ta

    @property
    def lin_vel_b(self) -> ProxyArray:
        """Linear velocity in PVA body frame. Shape: (N, 3)."""
        if self._lin_vel_b is None:
            self._lin_vel_b = wp.zeros(shape=(self._num_instances, 3), dtype=wp.float32, device=self.device)
            self._lin_vel_b_ta = None
        if self._lin_vel_b_ta is None:
            self._lin_vel_b_ta = ProxyArray(self._lin_vel_b)
        return self._lin_vel_b_ta

    @property
    def ang_vel_b(self) -> ProxyArray:
        """Angular velocity in PVA body frame. Shape: (N, 3)."""
        if self._ang_vel_b is None:
            self._ang_vel_b = wp.zeros(shape=(self._num_instances, 3), dtype=wp.float32, device=self.device)
            self._ang_vel_b_ta = None
        if self._ang_vel_b_ta is None:
            self._ang_vel_b_ta = ProxyArray(self._ang_vel_b)
        return self._ang_vel_b_ta

    @property
    def lin_acc_b(self) -> ProxyArray:
        """Linear acceleration in PVA body frame. Shape: (N, 3)."""
        if self._lin_acc_b is None:
            self._lin_acc_b = wp.zeros(shape=(self._num_instances, 3), dtype=wp.float32, device=self.device)
            self._lin_acc_b_ta = None
        if self._lin_acc_b_ta is None:
            self._lin_acc_b_ta = ProxyArray(self._lin_acc_b)
        return self._lin_acc_b_ta

    @property
    def ang_acc_b(self) -> ProxyArray:
        """Angular acceleration in PVA body frame. Shape: (N, 3)."""
        if self._ang_acc_b is None:
            self._ang_acc_b = wp.zeros(shape=(self._num_instances, 3), dtype=wp.float32, device=self.device)
            self._ang_acc_b_ta = None
        if self._ang_acc_b_ta is None:
            self._ang_acc_b_ta = ProxyArray(self._ang_acc_b)
        return self._ang_acc_b_ta

    # -- Setters --

    def set_pos_w(self, value: torch.Tensor) -> None:
        """Set position in world frame."""
        self._pos_w = wp.from_torch(value.to(self.device).contiguous(), dtype=wp.float32)
        self._pos_w_ta = None

    def set_quat_w(self, value: torch.Tensor) -> None:
        """Set orientation in world frame."""
        self._quat_w = wp.from_torch(value.to(self.device).contiguous(), dtype=wp.float32)
        self._quat_w_ta = None

    def set_projected_gravity_b(self, value: torch.Tensor) -> None:
        """Set projected gravity in body frame."""
        self._projected_gravity_b = wp.from_torch(value.to(self.device).contiguous(), dtype=wp.float32)
        self._projected_gravity_b_ta = None

    def set_lin_vel_b(self, value: torch.Tensor) -> None:
        """Set linear velocity in body frame."""
        self._lin_vel_b = wp.from_torch(value.to(self.device).contiguous(), dtype=wp.float32)
        self._lin_vel_b_ta = None

    def set_ang_vel_b(self, value: torch.Tensor) -> None:
        """Set angular velocity in body frame."""
        self._ang_vel_b = wp.from_torch(value.to(self.device).contiguous(), dtype=wp.float32)
        self._ang_vel_b_ta = None

    def set_lin_acc_b(self, value: torch.Tensor) -> None:
        """Set linear acceleration in body frame."""
        self._lin_acc_b = wp.from_torch(value.to(self.device).contiguous(), dtype=wp.float32)
        self._lin_acc_b_ta = None

    def set_ang_acc_b(self, value: torch.Tensor) -> None:
        """Set angular acceleration in body frame."""
        self._ang_acc_b = wp.from_torch(value.to(self.device).contiguous(), dtype=wp.float32)
        self._ang_acc_b_ta = None

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
            quat_w: Orientation (x, y, z, w) in world frame. Shape: (N, 4).
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


class MockPva:
    """Mock PVA sensor for testing without Isaac Sim.

    This class mimics the interface of BasePva for testing purposes.
    It provides the same properties and methods but without simulation dependencies.
    """

    def __init__(
        self,
        num_instances: int,
        device: str = "cpu",
    ):
        """Initialize mock PVA sensor.

        Args:
            num_instances: Number of sensor instances.
            device: Device for tensor allocation.
        """
        self._num_instances = num_instances
        self._device = device
        self._data = MockPvaData(num_instances, device)

    # -- Properties --

    @property
    def data(self) -> MockPvaData:
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
