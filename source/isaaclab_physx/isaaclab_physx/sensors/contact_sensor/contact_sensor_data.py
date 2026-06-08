# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import math

import warp as wp

from isaaclab.sensors.contact_sensor import BaseContactSensorData
from isaaclab.utils.warp import ProxyArray

from isaaclab_physx.sensors.kernels import concat_pos_and_quat_to_pose_kernel

logger = logging.getLogger(__name__)


class ContactSensorData(BaseContactSensorData):
    """Data container for the PhysX contact reporting sensor."""

    @property
    def pose_w(self) -> ProxyArray | None:
        """Pose of the sensor origin in world frame.

        None if :attr:`ContactSensorCfg.track_pose` is False.
        """
        if self._pose_w is None:
            return None
        wp.launch(
            concat_pos_and_quat_to_pose_kernel,
            dim=(self._num_envs, self._num_sensors),
            inputs=[self._pos_w, self._quat_w],
            outputs=[self._pose_w],
            device=self._device,
        )
        if self._pose_w_ta is None:
            self._pose_w_ta = ProxyArray(self._pose_w)
        return self._pose_w_ta

    @property
    def pos_w(self) -> ProxyArray | None:
        """Position of the sensor origin in world frame.

        Shape is (num_instances, num_sensors), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_sensors, 3).

        None if :attr:`ContactSensorCfg.track_pose` is False.
        """
        if self._pos_w is None:
            return None
        if self._pos_w_ta is None:
            self._pos_w_ta = ProxyArray(self._pos_w)
        return self._pos_w_ta

    @property
    def quat_w(self) -> ProxyArray | None:
        """Orientation of the sensor origin in world frame.

        Shape is (num_instances, num_sensors), dtype = wp.quatf. In torch this resolves to
        (num_instances, num_sensors, 4). The orientation is provided in (x, y, z, w) format.

        None if :attr:`ContactSensorCfg.track_pose` is False.
        """
        if self._quat_w is None:
            return None
        if self._quat_w_ta is None:
            self._quat_w_ta = ProxyArray(self._quat_w)
        return self._quat_w_ta

    @property
    def net_forces_w(self) -> ProxyArray | None:
        """The net normal contact forces in world frame.

        Shape is (num_instances, num_sensors), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_sensors, 3).
        """
        if self._net_forces_w is None:
            return None
        if self._net_forces_w_ta is None:
            self._net_forces_w_ta = ProxyArray(self._net_forces_w)
        return self._net_forces_w_ta

    @property
    def net_forces_w_history(self) -> ProxyArray | None:
        """History of net normal contact forces.

        Shape is (num_instances, history_length, num_sensors), dtype = wp.vec3f. In torch this resolves to
        (num_instances, history_length, num_sensors, 3).
        """
        if self._net_forces_w_history is None:
            return None
        if self._net_forces_w_history_ta is None:
            self._net_forces_w_history_ta = ProxyArray(self._net_forces_w_history)
        return self._net_forces_w_history_ta

    @property
    def force_matrix_w(self) -> ProxyArray | None:
        """Normal contact forces filtered between sensor and filtered bodies.

        Shape is (num_instances, num_sensors, num_filter_shapes), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_sensors, num_filter_shapes, 3).

        None if :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty.
        """
        if self._force_matrix_w is None:
            return None
        if self._force_matrix_w_ta is None:
            self._force_matrix_w_ta = ProxyArray(self._force_matrix_w)
        return self._force_matrix_w_ta

    @property
    def force_matrix_w_history(self) -> ProxyArray | None:
        """History of filtered contact forces.

        Shape is (num_instances, history_length, num_sensors, num_filter_shapes), dtype = wp.vec3f.
        In torch this resolves to (num_instances, history_length, num_sensors, num_filter_shapes, 3).

        None if :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty.
        """
        if self._force_matrix_w_history is None:
            return None
        if self._force_matrix_w_history_ta is None:
            self._force_matrix_w_history_ta = ProxyArray(self._force_matrix_w_history)
        return self._force_matrix_w_history_ta

    @property
    def contact_pos_w(self) -> ProxyArray | None:
        """Average position of contact points.

        Shape is (num_instances, num_sensors, num_filter_shapes), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_sensors, num_filter_shapes, 3).

        None if :attr:`ContactSensorCfg.track_contact_points` is False.
        """
        if self._contact_pos_w is None:
            return None
        if self._contact_pos_w_ta is None:
            self._contact_pos_w_ta = ProxyArray(self._contact_pos_w)
        return self._contact_pos_w_ta

    @property
    def friction_forces_w(self) -> ProxyArray | None:
        """Sum of friction forces.

        Shape is (num_instances, num_sensors, num_filter_shapes), dtype = wp.vec3f. In torch this resolves to
        (num_instances, num_sensors, num_filter_shapes, 3).

        None if :attr:`ContactSensorCfg.track_friction_forces` is False.
        """
        if self._friction_forces_w is None:
            return None
        if self._friction_forces_w_ta is None:
            self._friction_forces_w_ta = ProxyArray(self._friction_forces_w)
        return self._friction_forces_w_ta

    @property
    def last_air_time(self) -> ProxyArray | None:
        """Time spent in air before last contact.

        Shape is (num_instances, num_sensors), dtype = wp.float32.

        None if :attr:`ContactSensorCfg.track_air_time` is False.
        """
        if self._last_air_time is None:
            return None
        if self._last_air_time_ta is None:
            self._last_air_time_ta = ProxyArray(self._last_air_time)
        return self._last_air_time_ta

    @property
    def current_air_time(self) -> ProxyArray | None:
        """Time spent in air since last detach.

        Shape is (num_instances, num_sensors), dtype = wp.float32.

        None if :attr:`ContactSensorCfg.track_air_time` is False.
        """
        if self._current_air_time is None:
            return None
        if self._current_air_time_ta is None:
            self._current_air_time_ta = ProxyArray(self._current_air_time)
        return self._current_air_time_ta

    @property
    def last_contact_time(self) -> ProxyArray | None:
        """Time spent in contact before last detach.

        Shape is (num_instances, num_sensors), dtype = wp.float32.

        None if :attr:`ContactSensorCfg.track_air_time` is False.
        """
        if self._last_contact_time is None:
            return None
        if self._last_contact_time_ta is None:
            self._last_contact_time_ta = ProxyArray(self._last_contact_time)
        return self._last_contact_time_ta

    @property
    def current_contact_time(self) -> ProxyArray | None:
        """Time spent in contact since last contact.

        Shape is (num_instances, num_sensors), dtype = wp.float32.

        None if :attr:`ContactSensorCfg.track_air_time` is False.
        """
        if self._current_contact_time is None:
            return None
        if self._current_contact_time_ta is None:
            self._current_contact_time_ta = ProxyArray(self._current_contact_time)
        return self._current_contact_time_ta

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
            self._first_transition_ta = ProxyArray(self._first_transition)
        else:
            self._last_air_time = None
            self._current_air_time = None
            self._last_contact_time = None
            self._current_contact_time = None
            self._first_transition = None
            self._first_transition_ta = None

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

        # -- Pinned ProxyArray cache (one per read property, lazily created on first access)
        self._pose_w_ta: ProxyArray | None = None
        self._pos_w_ta: ProxyArray | None = None
        self._quat_w_ta: ProxyArray | None = None
        self._net_forces_w_ta: ProxyArray | None = None
        self._net_forces_w_history_ta: ProxyArray | None = None
        self._force_matrix_w_ta: ProxyArray | None = None
        self._force_matrix_w_history_ta: ProxyArray | None = None
        self._contact_pos_w_ta: ProxyArray | None = None
        self._friction_forces_w_ta: ProxyArray | None = None
        self._last_air_time_ta: ProxyArray | None = None
        self._current_air_time_ta: ProxyArray | None = None
        self._last_contact_time_ta: ProxyArray | None = None
        self._current_contact_time_ta: ProxyArray | None = None
