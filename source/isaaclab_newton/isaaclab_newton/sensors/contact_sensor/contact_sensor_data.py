# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: torch.Tensor | None
from __future__ import annotations

import logging
import torch

from isaaclab.sensors.contact_sensor.base_contact_sensor_data import BaseContactSensorData

logger = logging.getLogger(__name__)


class ContactSensorData(BaseContactSensorData):
    """Data container for the contact reporting sensor."""

    @property
    def pos_w(self) -> torch.Tensor | None:
        """Position of the sensor origin in world frame.

        `wp.vec3f` array whose shape is (N,) where N is the number of sensors. Note, that when casted to as a
        `torch.Tensor`, the shape will be (N, 3).

        Note:
            If the :attr:`ContactSensorCfg.track_pose` is False, then this quantity is None.
        """
        return self._pos_w

    @property
    def quat_w(self) -> torch.Tensor | None:
        """Orientation of the sensor origin in quaternion (x, y, z, w) in world frame.

        `wp.quatf` whose shape is (N,) where N is the number of sensors. Note, that when casted to as a `torch.Tensor`,
        the shape will be (N, 4).

        Note:
            If the :attr:`ContactSensorCfg.track_pose` is False, then this quantity is None.
        """
        return self._quat_w

    @property
    def net_forces_w(self) -> torch.Tensor | None:
        """The net normal contact forces in world frame.

        `wp.vec3f` array whose shape is (N, B) where N is the number of sensors and B is the number of bodies in each
        sensor. Note, that when casted to as a `torch.Tensor`, the shape will be (N, B, 3).

        Note:
            This quantity is the sum of the normal contact forces acting on the sensor bodies. It must not be confused
            with the total contact forces acting on the sensor bodies (which also includes the tangential forces).
        """
        return self._net_forces_w

    @property
    def net_forces_w_history(self) -> torch.Tensor | None:
        """The net normal contact forces in world frame.

        `wp.vec3f` array whose shape is (N, T, B) where N is the number of sensors, T is the configured history length
        and B is the number of bodies in each sensor. Note, that when casted to as a `torch.Tensor`, the shape will be
        (N, T, B, 3).

        In the history dimension, the first index is the most recent and the last index is the oldest.

        Note:
            This quantity is the sum of the normal contact forces acting on the sensor bodies. It must not be confused
            with the total contact forces acting on the sensor bodies (which also includes the tangential forces).
        """
        return self._net_forces_w_history

    @property
    def force_matrix_w(self) -> torch.Tensor | None:
        """The normal contact forces filtered between the sensor bodies and filtered bodies in world frame.

        `wp.vec3f` array whose shape is (N, B, M) where N is the number of sensors, B is number of bodies in each sensor
        and M is the number of filtered bodies. Note, that when casted to as a `torch.Tensor`, the shape will be
        (N, B, M, 3).

        Note:
            If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is None.
        """
        return self._force_matrix_w

    @property
    def force_matrix_w_history(self) -> torch.Tensor | None:
        """The normal contact forces filtered between the sensor bodies and filtered bodies in world frame.

        `wp.vec3f` array whose shape is (N, T, B, M) where N is the number of sensors, T is the configured history
        length and B is number of bodies in each sensor and M is the number of filtered bodies. Note, that when casted
        to as a `torch.Tensor`, the shape will be (N, T, B, M, 3).

        In the history dimension, the first index is the most recent and the last index is the oldest.

        Note:
            If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is None.
        """
        return self._force_matrix_w_history

    @property
    def last_air_time(self) -> torch.Tensor | None:
        """Time spent (in s) in the air before the last contact.

        `wp.float32` array whose shape is (N, B) where N is the number of sensors and B is the number of bodies in each
        sensor. Note, that when casted to as a `torch.Tensor`, the shape will be (N, B).

        Note:
            If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
        """
        return self._last_air_time

    @property
    def current_air_time(self) -> torch.Tensor | None:
        """Time spent (in s) in the air since the last detach.

        `wp.float32` array whose shape is (N, B) where N is the number of sensors and B is the number of bodies in each
        sensor. Note, that when casted to as a `torch.Tensor`, the shape will be (N, B).

        Note:
            If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
        """
        return self._current_air_time

    @property
    def last_contact_time(self) -> torch.Tensor | None:
        """Time spent (in s) in contact before the last detach.

        `wp.float32` array whose shape is (N, B) where N is the number of sensors and B is the number of bodies in each
        sensor. Note, that when casted to as a `torch.Tensor`, the shape will be (N, B).

        Note:
            If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
        """
        return self._last_contact_time

    @property
    def current_contact_time(self) -> torch.Tensor | None:
        """Time spent (in s) in contact since the last contact.

        `wp.float32` array whose shape is (N, B) where N is the number of sensors and B is the number of bodies in each
        sensor. Note, that when casted to as a `torch.Tensor`, the shape will be (N, B).

        Note:
            If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
        """
        return self._current_contact_time

    def create_buffers(
        self,
        num_envs: int,
        num_bodies: int,
        num_filters: int,
        history_length: int,
        generate_force_matrix: bool,
        track_air_time: bool,
        track_pose: bool,
        device: str,
    ) -> None:
        """Creates the buffers for the contact sensor data.

        Args:
            num_envs: The number of environments.
            num_bodies: The number of bodies in each sensor.
            num_filters: The number of filtered bodies.
            history_length: The history length.
            generate_force_matrix: Whether to generate the force matrix.
            track_air_time: Whether to track the air time.
            track_pose: Whether to track the pose.
            device: The device to use.
        """
        logger.info(
            f"Creating buffers for contact sensor data with num_envs: {num_envs}, num_bodies: {num_bodies},"
            f" num_filters: {num_filters}, history_length: {history_length}, generate_force_matrix:"
            f" {generate_force_matrix}, track_air_time: {track_air_time}, track_pose: {track_pose}, device: {device}"
        )
        # Track pose if requested
        if track_pose:
            self._pos_w = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
            self._quat_w = torch.zeros((num_envs, 4), dtype=torch.float32, device=device)
        else:
            self._pos_w = None
            self._quat_w = None
        # Track net forces
        self._net_forces_w = torch.zeros((num_envs, num_bodies, 3), dtype=torch.float32, device=device)
        # Track net forces history if requested
        if history_length > 0:
            self._net_forces_w_history = torch.zeros(
                (num_envs, history_length, num_bodies, 3), dtype=torch.float32, device=device
            )
            # self._force_matrix_w_history = torch.zeros((num_envs, history_length, num_bodies, num_filter_bodies, 3), dtype=torch.float32, device=device)
            self._force_matrix_w_history = None
        else:
            self._net_forces_w_history = None
            self._force_matrix_w_history = None
        # Track force matrix if requested
        if generate_force_matrix:
            self._force_matrix_w = torch.zeros(
                (num_envs, num_bodies, num_filters, 3), dtype=torch.float32, device=device
            )
        else:
            self._force_matrix_w = None
        # Track air time if requested
        if track_air_time:
            self._last_air_time = torch.zeros((num_envs, num_bodies), dtype=torch.float32, device=device)
            self._current_air_time = torch.zeros((num_envs, num_bodies), dtype=torch.float32, device=device)
            self._last_contact_time = torch.zeros((num_envs, num_bodies), dtype=torch.float32, device=device)
            self._current_contact_time = torch.zeros((num_envs, num_bodies), dtype=torch.float32, device=device)
        else:
            self._last_air_time = None
            self._current_air_time = None
            self._last_contact_time = None
            self._current_contact_time = None
