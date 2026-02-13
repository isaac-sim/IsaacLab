# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: torch.Tensor | None
from __future__ import annotations

import logging

import warp as wp

from isaaclab.sensors.contact_sensor.base_contact_sensor_data import BaseContactSensorData

logger = logging.getLogger(__name__)


class ContactSensorData(BaseContactSensorData):
    """Data container for the contact reporting sensor."""

    _pos_w: wp.array | None
    _quat_w: wp.array | None

    _net_forces_w: wp.array | None
    _net_forces_w_history: wp.array | None
    _force_matrix_w: wp.array | None
    _force_matrix_w_history: wp.array | None
    _last_air_time: wp.array | None
    _current_air_time: wp.array | None
    _last_contact_time: wp.array | None
    _current_contact_time: wp.array | None
    _first_transition: wp.array | None

    @property
    def pos_w(self) -> wp.array | None:
        """Position of the sensor origin in world frame.

        `wp.vec3f` array whose shape is (N,) where N is the number of sensors. Note, that when casted to as a
        `torch.Tensor`, the shape will be (N, 3).

        Note:
            If the :attr:`ContactSensorCfg.track_pose` is False, then this quantity is None.
        """
        return self._pos_w

    @property
    def quat_w(self) -> wp.array | None:
        """Orientation of the sensor origin in quaternion (x, y, z, w) in world frame.

        `wp.quatf` whose shape is (N,) where N is the number of sensors. Note, that when casted to as a `torch.Tensor`,
        the shape will be (N, 4).

        Note:
            If the :attr:`ContactSensorCfg.track_pose` is False, then this quantity is None.
        """
        return self._quat_w

    @property
    def net_forces_w(self) -> wp.array2d | None:
        """The net (total) contact forces in world frame.

        `wp.vec3f` array whose shape is (N, S) where N is the number of environments and S is the number of sensors.
        Note, that when casted to as a `torch.Tensor`, the shape will be (N, S, 3).

        Note:
            This quantity is the sum of the contact forces acting on each sensor. It must not be confused
            with the total contact forces acting on the sensors (which also includes the tangential forces).
        """
        return self._net_forces_w

    @property
    def net_forces_w_history(self) -> wp.array3d | None:
        """The net (total) contact forces in world frame.

        `wp.vec3f` array whose shape is (N, T, S) where N is the number of environments, T is the configured history
        length and S is the number of sensors. Note, that when casted to as a `torch.Tensor`, the shape will be
        (N, T, S, 3).

        In the history dimension, the first index is the most recent and the last index is the oldest.

        Note:
            This quantity is the sum of the contact forces acting on each sensor. It must not be confused
            with the total contact forces acting on the sensors (which also includes the tangential forces).
        """
        return self._net_forces_w_history

    @property
    def force_matrix_w(self) -> wp.array3d | None:
        """The contact forces between sensors and filter objects in world frame.

        `wp.vec3f` array whose shape is (N, S, F) where N is the number of environments, S is number of sensors
        and F is the number of filter objects. Note, that when casted to as a `torch.Tensor`, the shape will be
        (N, S, F, 3).

        Note:
            If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is None.
        """
        return self._force_matrix_w

    @property
    def force_matrix_w_history(self) -> wp.array4d | None:
        """The contact forces between sensors and filter objects in world frame.

        `wp.vec3f` array whose shape is (N, T, S, F) where N is the number of environments, T is the configured history
        length, S is number of sensors and F is the number of filter objects. Note, that when casted to as a
        `torch.Tensor`, the shape will be (N, T, S, F, 3).

        In the history dimension, the first index is the most recent and the last index is the oldest.

        Note:
            If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is None.
        """
        return self._force_matrix_w_history

    @property
    def last_air_time(self) -> wp.array2d | None:
        """Time spent (in s) in the air before the last contact.

        `wp.float32` array whose shape is (N, S) where N is the number of environments and S is the number of sensors.
        Note, that when casted to as a `torch.Tensor`, the shape will be (N, S).

        Note:
            If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
        """
        return self._last_air_time

    @property
    def current_air_time(self) -> wp.array2d | None:
        """Time spent (in s) in the air since the last detach.

        `wp.float32` array whose shape is (N, S) where N is the number of environments and S is the number of sensors.
        Note, that when casted to as a `torch.Tensor`, the shape will be (N, S).

        Note:
            If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
        """
        return self._current_air_time

    @property
    def last_contact_time(self) -> wp.array2d | None:
        """Time spent (in s) in contact before the last detach.

        `wp.float32` array whose shape is (N, S) where N is the number of environments and S is the number of sensors.
        Note, that when casted to as a `torch.Tensor`, the shape will be (N, S).

        Note:
            If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
        """
        return self._last_contact_time

    @property
    def current_contact_time(self) -> wp.array2d | None:
        """Time spent (in s) in contact since the last contact.

        `wp.float32` array whose shape is (N, S) where N is the number of environments and S is the number of sensors.
        Note, that when casted to as a `torch.Tensor`, the shape will be (N, S).

        Note:
            If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
        """
        return self._current_contact_time

    def create_buffers(
        self,
        num_envs: int,
        num_sensors: int,
        num_filter_objects: int,
        history_length: int,
        generate_force_matrix: bool,
        track_air_time: bool,
        track_pose: bool,
        device: str,
    ) -> None:
        """Creates the buffers for the contact sensor data.

        Args:
            num_envs: The number of environments.
            num_sensors: The number of sensors.
            num_filter_objects: The number of filter objects (counterparts).
            history_length: The history length.
            generate_force_matrix: Whether to generate the force matrix.
            track_air_time: Whether to track the air time.
            track_pose: Whether to track the pose.
            device: The device to use.
        """
        logger.info(
            f"Creating buffers for contact sensor data with num_envs: {num_envs}, num_sensors: {num_sensors},"
            f" num_filter_objects: {num_filter_objects}, history_length: {history_length}, generate_force_matrix:"
            f" {generate_force_matrix}, track_air_time: {track_air_time}, track_pose: {track_pose}, device: {device}"
        )
        # Track pose if requested
        if track_pose:
            self._pose = wp.zeros((num_envs,), dtype=wp.transformf, device=device)
            pos_scalars = wp.array(self._pose, dtype=wp.float32, device=device, copy=False)
            self._pos_w = wp.array(pos_scalars[:, :3], dtype=wp.vec3f, device=device, copy=False)
            self._quat_w = wp.array(pos_scalars[:, 3:], dtype=wp.quatf, device=device, copy=False)
        else:
            self._pose = None
            self._pos_w = None
            self._quat_w = None

        # Create owned buffer for net (total) forces - shape: (num_envs, num_sensors)
        self._net_forces_w = wp.zeros((num_envs, num_sensors), dtype=wp.vec3f, device=device)
        # Track net forces history if requested
        if history_length > 0:
            self._net_forces_w_history = wp.zeros(
                (num_envs, history_length, num_sensors), dtype=wp.vec3f, device=device
            )
            self._force_matrix_w_history = None  # TODO: implement force matrix history if needed
        else:
            self._net_forces_w_history = None
            self._force_matrix_w_history = None

        # Create owned buffer for force matrix - shape: (num_envs, num_sensors, num_filter_objects)
        # None if no filter objects configured
        if num_filter_objects > 0:
            self._force_matrix_w = wp.zeros((num_envs, num_sensors, num_filter_objects), dtype=wp.vec3f, device=device)
        else:
            self._force_matrix_w = None

        # Track air time if requested
        if track_air_time:
            self._last_air_time = wp.zeros((num_envs, num_sensors), dtype=wp.float32, device=device)
            self._current_air_time = wp.zeros((num_envs, num_sensors), dtype=wp.float32, device=device)
            self._last_contact_time = wp.zeros((num_envs, num_sensors), dtype=wp.float32, device=device)
            self._current_contact_time = wp.zeros((num_envs, num_sensors), dtype=wp.float32, device=device)
            self._first_transition = wp.zeros((num_envs, num_sensors), dtype=wp.float32, device=device)
        else:
            self._last_air_time = None
            self._current_air_time = None
            self._last_contact_time = None
            self._current_contact_time = None
            self._first_transition = None
