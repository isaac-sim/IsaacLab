# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp

from isaaclab.sensors.pva import BasePvaData
from isaaclab.utils.warp import TorchArray


class PvaData(BasePvaData):
    """Data container for the Newton PVA sensor."""

    def __init__(self):
        self._pose_w: wp.array | None = None
        self._pos_w: wp.array | None = None
        self._quat_w: wp.array | None = None
        self._projected_gravity_b: wp.array | None = None
        self._lin_vel_b: wp.array | None = None
        self._ang_vel_b: wp.array | None = None
        self._lin_acc_b: wp.array | None = None
        self._ang_acc_b: wp.array | None = None
        # TorchArray caches
        self._pose_w_ta: TorchArray | None = None
        self._pos_w_ta: TorchArray | None = None
        self._quat_w_ta: TorchArray | None = None
        self._projected_gravity_b_ta: TorchArray | None = None
        self._lin_vel_b_ta: TorchArray | None = None
        self._ang_vel_b_ta: TorchArray | None = None
        self._lin_acc_b_ta: TorchArray | None = None
        self._ang_acc_b_ta: TorchArray | None = None

    @property
    def pose_w(self) -> TorchArray | None:
        """Pose of the sensor origin in world frame [m, unitless].

        Shape is (num_instances,), dtype = wp.transformf. In torch this resolves to (num_instances, 7).
        The pose is provided in (x, y, z, qx, qy, qz, qw) format.

        ``None`` before the simulation is initialized.
        """
        if self._pose_w is None:
            return None
        if self._pose_w_ta is None:
            self._pose_w_ta = TorchArray(self._pose_w)
        return self._pose_w_ta

    @property
    def pos_w(self) -> TorchArray | None:
        """Position of the sensor origin in world frame [m].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        ``None`` before the simulation is initialized.
        """
        if self._pos_w is None:
            return None
        if self._pos_w_ta is None:
            self._pos_w_ta = TorchArray(self._pos_w)
        return self._pos_w_ta

    @property
    def quat_w(self) -> TorchArray | None:
        """Orientation of the sensor origin in world frame.

        Shape is (num_instances,), dtype = wp.quatf. In torch this resolves to (num_instances, 4).
        The orientation is provided in (x, y, z, w) format.

        ``None`` before the simulation is initialized.
        """
        if self._quat_w is None:
            return None
        if self._quat_w_ta is None:
            self._quat_w_ta = TorchArray(self._quat_w)
        return self._quat_w_ta

    @property
    def projected_gravity_b(self) -> TorchArray | None:
        """Gravity direction unit vector projected on the PVA frame [unitless].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        ``None`` before the simulation is initialized.
        """
        if self._projected_gravity_b is None:
            return None
        if self._projected_gravity_b_ta is None:
            self._projected_gravity_b_ta = TorchArray(self._projected_gravity_b)
        return self._projected_gravity_b_ta

    @property
    def lin_vel_b(self) -> TorchArray | None:
        """PVA frame linear velocity relative to the world expressed in PVA frame [m/s].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        ``None`` before the simulation is initialized.
        """
        if self._lin_vel_b is None:
            return None
        if self._lin_vel_b_ta is None:
            self._lin_vel_b_ta = TorchArray(self._lin_vel_b)
        return self._lin_vel_b_ta

    @property
    def ang_vel_b(self) -> TorchArray | None:
        """PVA frame angular velocity relative to the world expressed in PVA frame [rad/s].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        ``None`` before the simulation is initialized.
        """
        if self._ang_vel_b is None:
            return None
        if self._ang_vel_b_ta is None:
            self._ang_vel_b_ta = TorchArray(self._ang_vel_b)
        return self._ang_vel_b_ta

    @property
    def lin_acc_b(self) -> TorchArray | None:
        """Linear acceleration (coordinate) in the PVA frame [m/s^2].

        Equal to -g in freefall, zero at rest.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        ``None`` before the simulation is initialized.
        """
        if self._lin_acc_b is None:
            return None
        if self._lin_acc_b_ta is None:
            self._lin_acc_b_ta = TorchArray(self._lin_acc_b)
        return self._lin_acc_b_ta

    @property
    def ang_acc_b(self) -> TorchArray | None:
        """PVA frame angular acceleration relative to the world expressed in PVA frame [rad/s^2].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        ``None`` before the simulation is initialized.
        """
        if self._ang_acc_b is None:
            return None
        if self._ang_acc_b_ta is None:
            self._ang_acc_b_ta = TorchArray(self._ang_acc_b)
        return self._ang_acc_b_ta

    def create_buffers(self, num_envs: int, device: str) -> None:
        """Create internal buffers for sensor data.

        Args:
            num_envs: Number of environments.
            device: Device for array storage.
        """
        self._pose_w = wp.zeros(num_envs, dtype=wp.transformf, device=device)
        self._pos_w = wp.zeros(num_envs, dtype=wp.vec3f, device=device)
        self._quat_w = wp.zeros(num_envs, dtype=wp.quatf, device=device)
        self._projected_gravity_b = wp.zeros(num_envs, dtype=wp.vec3f, device=device)
        self._lin_vel_b = wp.zeros(num_envs, dtype=wp.vec3f, device=device)
        self._ang_vel_b = wp.zeros(num_envs, dtype=wp.vec3f, device=device)
        self._lin_acc_b = wp.zeros(num_envs, dtype=wp.vec3f, device=device)
        self._ang_acc_b = wp.zeros(num_envs, dtype=wp.vec3f, device=device)
        # Reset TorchArray caches
        self._pose_w_ta = None
        self._pos_w_ta = None
        self._quat_w_ta = None
        self._projected_gravity_b_ta = None
        self._lin_vel_b_ta = None
        self._ang_vel_b_ta = None
        self._lin_acc_b_ta = None
        self._ang_acc_b_ta = None
