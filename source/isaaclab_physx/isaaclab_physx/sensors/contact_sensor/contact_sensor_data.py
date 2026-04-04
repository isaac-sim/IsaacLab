# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import math

import warp as wp

from isaaclab.sensors.contact_sensor import BaseContactSensorData

from isaaclab_physx.sensors.kernels import concat_pos_and_quat_to_pose_kernel

logger = logging.getLogger(__name__)


class ContactSensorData(BaseContactSensorData):
    """Data container for the PhysX contact reporting sensor."""

    @property
    def pose_w(self) -> wp.array | None:
        """Pose of the sensor origin in world frame.

        None if :attr:`ContactSensorCfg.track_pose` is False.
        """
        wp.launch(
            concat_pos_and_quat_to_pose_kernel,
            dim=(self._num_envs, self._num_sensors),
            inputs=[self._pos_w, self._quat_w],
            outputs=[self._pose_w],
            device=self._device,
        )
        return self._pose_w

    @property
    def pos_w(self) -> wp.array | None:
        """Position of the sensor origin in world frame.

        Shape is (num_instances, num_sensors), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_sensors, 3).

        None if :attr:`ContactSensorCfg.track_pose` is False.
        """
        return self._pos_w

    @property
    def quat_w(self) -> wp.array | None:
        """Orientation of the sensor origin in world frame.

        Shape is (num_instances, num_sensors), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_sensors, 4). The orientation is provided in (x, y, z, w) format.

        None if :attr:`ContactSensorCfg.track_pose` is False.
        """
        return self._quat_w

    @property
    def net_forces_w(self) -> wp.array | None:
        """The net normal contact forces in world frame.

        Shape is (num_instances, num_sensors), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_sensors, 3).
        """
        return self._net_forces_w

    @property
    def net_forces_w_history(self) -> wp.array | None:
        """History of net normal contact forces.

        Shape is (num_instances, history_length, num_sensors), dtype = wp.vec3f. In torch this resolves to
        (num_instances, history_length, num_sensors, 3).
        """
        return self._net_forces_w_history

    @property
    def force_matrix_w(self) -> wp.array | None:
        """Normal contact forces filtered between sensor and filtered bodies.

        Shape is (num_instances, num_sensors, num_filter_shapes), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_sensors, num_filter_shapes, 3).

        None if :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty.
        """
        return self._force_matrix_w

    @property
    def force_matrix_w_history(self) -> wp.array | None:
        """History of filtered contact forces.

        Shape is (num_instances, history_length, num_sensors, num_filter_shapes), dtype = wp.vec3f.
        In torch this resolves to (num_instances, history_length, num_sensors, num_filter_shapes, 3).

        None if :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty.
        """
        return self._force_matrix_w_history

    @property
    def contact_pos_w(self) -> wp.array | None:
        """Average position of contact points.

        Shape is (num_instances, num_sensors, num_filter_shapes), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_sensors, num_filter_shapes, 3).

        None if :attr:`ContactSensorCfg.track_contact_points` is False.
        """
        return self._contact_pos_w

    @property
    def friction_forces_w(self) -> wp.array | None:
        """Sum of friction forces.

        Shape is (num_instances, num_sensors, num_filter_shapes), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_sensors, num_filter_shapes, 3).

        None if :attr:`ContactSensorCfg.track_friction_forces` is False.
        """
        return self._friction_forces_w

    @property
    def last_air_time(self) -> wp.array | None:
        """Time spent in air before last contact.

        Shape is (num_instances, num_sensors), dtype = wp.float32.

        None if :attr:`ContactSensorCfg.track_air_time` is False.
        """
        return self._last_air_time

    @property
    def current_air_time(self) -> wp.array | None:
        """Time spent in air since last detach.

        Shape is (num_instances, num_sensors), dtype = wp.float32.

        None if :attr:`ContactSensorCfg.track_air_time` is False.
        """
        return self._current_air_time

    @property
    def last_contact_time(self) -> wp.array | None:
        """Time spent in contact before last detach.

        Shape is (num_instances, num_sensors), dtype = wp.float32.

        None if :attr:`ContactSensorCfg.track_air_time` is False.
        """
        return self._last_contact_time

    @property
    def current_contact_time(self) -> wp.array | None:
        """Time spent in contact since last contact.

        Shape is (num_instances, num_sensors), dtype = wp.float32.

        None if :attr:`ContactSensorCfg.track_air_time` is False.
        """
        return self._current_contact_time

    def create_buffers(
        self,
        num_envs: int,
        num_sensors: int,
        num_filter_shapes: int,
        history_length: int,
        track_pose: bool,
        track_air_time: bool,
        track_contact_points: bool,
        track_friction_forces: bool,
        device: str,
    ) -> None:
        """Create internal buffers for sensor data.

        Args:
            num_envs: Number of environments.
            num_sensors: Number of sensors per environment.
            num_filter_shapes: Number of filtered shapes for force matrix.
            history_length: Length of force history buffer.
            track_pose: Whether to track sensor pose.
            track_air_time: Whether to track air/contact time.
            track_contact_points: Whether to track contact points.
            track_friction_forces: Whether to track friction forces.
            device: Device for tensor storage.
        """
        self._num_envs = num_envs
        self._num_sensors = num_sensors
        self._device = device
        # Ensure history_length >= 1 for consistent buffer shapes
        effective_history = max(history_length, 1)

        # Net forces (always tracked)
        self._net_forces_w = wp.zeros((num_envs, num_sensors), dtype=wp.vec3f, device=device)
        self._net_forces_w_history = wp.zeros((num_envs, effective_history, num_sensors), dtype=wp.vec3f, device=device)

        # Track force matrix if requested - only with filter
        if num_filter_shapes > 0:
            self._force_matrix_w = wp.zeros((num_envs, num_sensors, num_filter_shapes), dtype=wp.vec3f, device=device)
            self._force_matrix_w_history = wp.zeros(
                (num_envs, effective_history, num_sensors, num_filter_shapes), dtype=wp.vec3f, device=device
            )
        else:
            self._force_matrix_w = None
            self._force_matrix_w_history = None

        # Track pose if requested
        if track_pose:
            self._pos_w = wp.zeros((num_envs, num_sensors), dtype=wp.vec3f, device=device)
            self._quat_w = wp.zeros((num_envs, num_sensors), dtype=wp.quatf, device=device)
            self._pose_w = wp.zeros((num_envs, num_sensors), dtype=wp.transformf, device=device)
        else:
            self._pos_w = None
            self._quat_w = None
            self._pose_w = None

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

        # Track contact points if requested - filled with NaN
        if track_contact_points:
            self._contact_pos_w = wp.full(
                (num_envs, num_sensors, num_filter_shapes),
                dtype=wp.vec3f,
                device=device,
                value=wp.vec3f(math.nan, math.nan, math.nan),
            )
        else:
            self._contact_pos_w = None

        # Track friction forces if requested
        if track_friction_forces:
            self._friction_forces_w = wp.zeros(
                (num_envs, num_sensors, num_filter_shapes), dtype=wp.vec3f, device=device
            )
        else:
            self._friction_forces_w = None
