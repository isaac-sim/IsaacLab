# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging

import warp as wp

from isaaclab.sensors.imu import BaseImuData

from isaaclab_physx.sensors.kernels import concat_pos_and_quat_to_pose_1d_kernel

logger = logging.getLogger(__name__)


class ImuData(BaseImuData):
    """Data container for the PhysX Imu sensor."""

    @property
    def pose_w(self) -> wp.array:
        """Pose of the sensor origin in world frame.

        Shape is (num_instances,), dtype = wp.transformf. In torch this resolves to (num_instances, 7).
        The pose is provided in (x, y, z, qx, qy, qz, qw) format.
        """
        wp.launch(
            concat_pos_and_quat_to_pose_1d_kernel,
            dim=self._num_envs,
            inputs=[self._pos_w, self._quat_w],
            outputs=[self._pose_w],
            device=self._device,
        )
        return self._pose_w

    @property
    def pos_w(self) -> wp.array:
        """Position of the sensor origin in world frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        return self._pos_w

    @property
    def quat_w(self) -> wp.array:
        """Orientation of the sensor origin in world frame.

        Shape is (num_instances,), dtype = wp.quatf. In torch this resolves to (num_instances, 4).
        The orientation is provided in (x, y, z, w) format.
        """
        return self._quat_w

    @property
    def projected_gravity_b(self) -> wp.array:
        """Gravity direction unit vector projected on the IMU frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        return self._projected_gravity_b

    @property
    def lin_vel_b(self) -> wp.array:
        """IMU frame linear velocity relative to the world expressed in IMU frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        return self._lin_vel_b

    @property
    def ang_vel_b(self) -> wp.array:
        """IMU frame angular velocity relative to the world expressed in IMU frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        return self._ang_vel_b

    @property
    def lin_acc_b(self) -> wp.array:
        """IMU frame linear acceleration relative to the world expressed in IMU frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        return self._lin_acc_b

    @property
    def ang_acc_b(self) -> wp.array:
        """IMU frame angular acceleration relative to the world expressed in IMU frame.

        Shape is (num_instances,), dtype = wp.vec3f. In torch this resolves to (num_instances, 3).
        """
        return self._ang_acc_b

    def create_buffers(self, num_envs: int, device: str) -> None:
        """Create internal buffers for sensor data.

        Args:
            num_envs: Number of environments.
            device: Device for tensor storage.
        """
        self._num_envs = num_envs
        self._device = device
        self._pose_w = wp.zeros(num_envs, dtype=wp.transformf, device=device)
        self._pos_w = wp.zeros(num_envs, dtype=wp.vec3f, device=device)
        self._quat_w = wp.zeros(num_envs, dtype=wp.quatf, device=device)
        # Initialize quaternion to identity (w=1): warp quatf is (x,y,z,w)
        # Use torch interop to set the w component
        quat_torch = wp.to_torch(self._quat_w)
        quat_torch[:, 3] = 1.0
        self._projected_gravity_b = wp.zeros(num_envs, dtype=wp.vec3f, device=device)
        # Initialize projected gravity to (0, 0, -1)
        pg_torch = wp.to_torch(self._projected_gravity_b)
        pg_torch[:, 2] = -1.0
        self._lin_vel_b = wp.zeros(num_envs, dtype=wp.vec3f, device=device)
        self._ang_vel_b = wp.zeros(num_envs, dtype=wp.vec3f, device=device)
        self._lin_acc_b = wp.zeros(num_envs, dtype=wp.vec3f, device=device)
        self._ang_acc_b = wp.zeros(num_envs, dtype=wp.vec3f, device=device)
