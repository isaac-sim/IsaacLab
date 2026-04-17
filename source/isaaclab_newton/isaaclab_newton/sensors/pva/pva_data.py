# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp

from isaaclab.sensors.pva import BasePvaData


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

    @property
    def pose_w(self) -> wp.array | None:
        """Pose of the sensor origin in world frame [m, unitless].

        Shape is (num_instances,), dtype = wp.transformf. In torch this resolves to (num_instances, 7).
        The pose is provided in (x, y, z, qx, qy, qz, qw) format.

        ``None`` before the simulation is initialized.
        """
        return self._pose_w

    @property
    def pos_w(self) -> wp.array | None:
        """Position of the sensor origin in world frame [m].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        ``None`` before the simulation is initialized.
        """
        return self._pos_w

    @property
    def quat_w(self) -> wp.array | None:
        """Orientation of the sensor origin in world frame.

        Shape is (num_instances,), dtype = wp.quatf. In torch this resolves to (num_instances, 4).
        The orientation is provided in (x, y, z, w) format.

        ``None`` before the simulation is initialized.
        """
        return self._quat_w

    @property
    def projected_gravity_b(self) -> wp.array | None:
        """Gravity direction unit vector projected on the PVA frame [unitless].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        ``None`` before the simulation is initialized.
        """
        return self._projected_gravity_b

    @property
    def lin_vel_b(self) -> wp.array | None:
        """PVA frame linear velocity relative to the world expressed in PVA frame [m/s].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        ``None`` before the simulation is initialized.
        """
        return self._lin_vel_b

    @property
    def ang_vel_b(self) -> wp.array | None:
        """PVA frame angular velocity relative to the world expressed in PVA frame [rad/s].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        ``None`` before the simulation is initialized.
        """
        return self._ang_vel_b

    @property
    def lin_acc_b(self) -> wp.array | None:
        """Linear acceleration (coordinate) in the PVA frame [m/s^2].

        Equal to -g in freefall, zero at rest.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        ``None`` before the simulation is initialized.
        """
        return self._lin_acc_b

    @property
    def ang_acc_b(self) -> wp.array | None:
        """PVA frame angular acceleration relative to the world expressed in PVA frame [rad/s^2].

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).

        ``None`` before the simulation is initialized.
        """
        return self._ang_acc_b

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
