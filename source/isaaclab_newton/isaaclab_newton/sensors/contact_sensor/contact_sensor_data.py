# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: torch.Tensor | None
from __future__ import annotations

import logging
import math

import warp as wp

from isaaclab.sensors.contact_sensor.base_contact_sensor_data import BaseContactSensorData
from isaaclab.utils.warp import ProxyArray

logger = logging.getLogger(__name__)


class ContactSensorData(BaseContactSensorData):
    """Data container for the contact reporting sensor."""

    _pos_w: wp.array | None
    _quat_w: wp.array | None

    _net_forces_w: wp.array | None
    _net_forces_w_history: wp.array | None
    _force_matrix_w: wp.array | None
    _force_matrix_w_history: wp.array | None
    _net_normal_forces_w: wp.array | None
    _net_normal_forces_w_history: wp.array | None
    _normal_force_matrix_w: wp.array | None
    _normal_force_matrix_w_history: wp.array | None
    _net_friction_forces_w: wp.array | None
    _net_friction_forces_w_history: wp.array | None
    _friction_force_matrix_w: wp.array | None
    _friction_force_matrix_w_history: wp.array | None
    _contact_pos_w: wp.array | None
    _last_air_time: wp.array | None
    _current_air_time: wp.array | None
    _last_contact_time: wp.array | None
    _current_contact_time: wp.array | None
    _first_transition: wp.array | None

    @property
    def pose_w(self) -> ProxyArray | None:
        """Not supported by Newton contact sensor."""
        raise NotImplementedError("pose_w is not supported by the Newton contact sensor.")

    @property
    def pos_w(self) -> ProxyArray | None:
        """Position of the sensor origin in world frame.

        `wp.vec3f` array whose shape is (N,) where N is the number of sensors. Note, that when casted to as a
        `torch.Tensor`, the shape will be (N, 3).

        Note:
            If the :attr:`ContactSensorCfg.track_pose` is False, then this quantity is None.
        """
        if self._pos_w is None:
            return None
        if self._pos_w_ta is None:
            self._pos_w_ta = ProxyArray(self._pos_w)
        return self._pos_w_ta

    @property
    def quat_w(self) -> ProxyArray | None:
        """Orientation of the sensor origin in quaternion (x, y, z, w) in world frame.

        `wp.quatf` whose shape is (N,) where N is the number of sensors. Note, that when casted to as a `torch.Tensor`,
        the shape will be (N, 4).

        Note:
            If the :attr:`ContactSensorCfg.track_pose` is False, then this quantity is None.
        """
        if self._quat_w is None:
            return None
        if self._quat_w_ta is None:
            self._quat_w_ta = ProxyArray(self._quat_w)
        return self._quat_w_ta

    @property
    def net_forces_w(self) -> ProxyArray | None:
        """The net total contact forces [N] in world frame.

        `wp.vec3f` array whose shape is (N, S) where N is the number of environments and S is the number of sensors.
        Note, that when casted to as a `torch.Tensor`, the shape will be (N, S, 3).

        This is the total contact force (normal + friction). Use :attr:`net_normal_forces_w` and
        :attr:`net_friction_forces_w` when the normal / friction split is needed.
        """
        if self._net_forces_w is None:
            return None
        if self._net_forces_w_ta is None:
            self._net_forces_w_ta = ProxyArray(self._net_forces_w)
        return self._net_forces_w_ta

    @property
    def net_forces_w_history(self) -> ProxyArray | None:
        """History of net total contact forces [N] in world frame.

        `wp.vec3f` array whose shape is (N, T, S). In the history dimension, the first index is the most
        recent and the last index is the oldest.
        """
        if self._net_forces_w_history is None:
            return None
        if self._net_forces_w_history_ta is None:
            self._net_forces_w_history_ta = ProxyArray(self._net_forces_w_history)
        return self._net_forces_w_history_ta

    @property
    def force_matrix_w(self) -> ProxyArray | None:
        """The total contact forces [N] between sensors and filter objects in world frame.

        `wp.vec3f` array whose shape is (N, S, F). Use :attr:`normal_force_matrix_w` and
        :attr:`friction_force_matrix_w` when the normal / friction split is needed.

        Note:
            If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is None.
        """
        if self._force_matrix_w is None:
            return None
        if self._force_matrix_w_ta is None:
            self._force_matrix_w_ta = ProxyArray(self._force_matrix_w)
        return self._force_matrix_w_ta

    @property
    def force_matrix_w_history(self) -> ProxyArray | None:
        """History of total filtered contact forces [N] in world frame.

        `wp.vec3f` array whose shape is (N, T, S, F). In the history dimension, the first index is the
        most recent and the last index is the oldest.

        Note:
            If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is None.
        """
        if self._force_matrix_w_history is None:
            return None
        if self._force_matrix_w_history_ta is None:
            self._force_matrix_w_history_ta = ProxyArray(self._force_matrix_w_history)
        return self._force_matrix_w_history_ta

    @property
    def net_normal_forces_w(self) -> ProxyArray | None:
        """The net normal contact forces [N] in world frame.

        `wp.vec3f` array whose shape is (N, S) where N is the number of environments and S is the number of sensors.
        Note, that when casted to as a `torch.Tensor`, the shape will be (N, S, 3).

        The net total contact force is the sum of this quantity and :attr:`net_friction_forces_w`.
        """
        if self._net_normal_forces_w is None:
            return None
        return self._net_normal_forces_w_ta

    @property
    def net_normal_forces_w_history(self) -> ProxyArray | None:
        """History of net normal contact forces [N] in world frame.

        `wp.vec3f` array whose shape is (N, T, S) where N is the number of environments, T is the configured history
        length or one when the configured length is zero, and S is the number of sensors. Note, that when casted to
        as a `torch.Tensor`, the shape will be (N, T, S, 3).

        In the history dimension, the first index is the most recent and the last index is the oldest.

        """
        if self._net_normal_forces_w_history is None:
            return None
        return self._net_normal_forces_w_history_ta

    @property
    def normal_force_matrix_w(self) -> ProxyArray | None:
        """The normal contact forces [N] between sensors and filter objects in world frame.

        `wp.vec3f` array whose shape is (N, S, F) where N is the number of environments, S is number of sensors
        and F is the number of filter objects. Note, that when casted to as a `torch.Tensor`, the shape will be
        (N, S, F, 3).

        Note:
            If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is None.
        """
        if self._normal_force_matrix_w is None:
            return None
        if self._normal_force_matrix_w_ta is None:
            self._normal_force_matrix_w_ta = ProxyArray(self._normal_force_matrix_w)
        return self._normal_force_matrix_w_ta

    @property
    def normal_force_matrix_w_history(self) -> ProxyArray | None:
        """The contact forces [N] between sensors and filter objects in world frame.

        `wp.vec3f` array whose shape is (N, T, S, F) where N is the number of environments, T is the configured history
        length or one when the configured length is zero, S is number of sensors and F is the number of filter objects.
        Note, that when casted to as a `torch.Tensor`, the shape will be (N, T, S, F, 3).

        In the history dimension, the first index is the most recent and the last index is the oldest.

        Note:
            If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is None.
        """
        if self._normal_force_matrix_w_history is None:
            return None
        if self._normal_force_matrix_w_history_ta is None:
            self._normal_force_matrix_w_history_ta = ProxyArray(self._normal_force_matrix_w_history)
        return self._normal_force_matrix_w_history_ta

    @property
    def net_friction_forces_w(self) -> ProxyArray | None:
        """The net friction contact forces [N] in world frame.

        `wp.vec3f` array whose shape is (N, S) where N is the number of environments and S is the number of sensors.
        When cast to a `torch.Tensor`, the shape is (N, S, 3).

        None if :attr:`ContactSensorCfg.track_friction_forces` is False.
        """
        if self._net_friction_forces_w is None:
            return None
        if self._net_friction_forces_w_ta is None:
            self._net_friction_forces_w_ta = ProxyArray(self._net_friction_forces_w)
        return self._net_friction_forces_w_ta

    @property
    def friction_forces_w(self) -> ProxyArray | None:
        """The net total friction contact forces [N] in world frame.

        `wp.vec3f` array whose shape is (N, S). This is the aggregate friction force, the same
        quantity as :attr:`net_friction_forces_w`.

        None if :attr:`ContactSensorCfg.track_friction_forces` is False.
        """
        return self.net_friction_forces_w

    @property
    def net_friction_forces_w_history(self) -> ProxyArray | None:
        """History of net friction contact forces [N] in world frame.

        `wp.vec3f` array whose shape is (N, T, S). In the history dimension, the first index is the most
        recent and the last index is the oldest.

        None if :attr:`ContactSensorCfg.track_friction_forces` is False.
        """
        if self._net_friction_forces_w_history is None:
            return None
        if self._net_friction_forces_w_history_ta is None:
            self._net_friction_forces_w_history_ta = ProxyArray(self._net_friction_forces_w_history)
        return self._net_friction_forces_w_history_ta

    @property
    def contact_pos_w(self) -> ProxyArray | None:
        """Average position of contact points [m] in world frame.

        `wp.vec3f` array whose shape is (N, S, F) where N is the number of environments, S is the number of
        sensors and F is the number of filter objects. Note that when cast to a `torch.Tensor`, the shape
        will be (N, S, F, 3).

        Each entry is the midpoint of all contacts between the sensor and the filter object, averaged with
        contact-force-magnitude weighting. Entries are NaN when the sensor and filter object are not in contact.

        Note:
            If :attr:`ContactSensorCfg.track_contact_points` is False or no filter objects are configured,
            this quantity is None.
        """
        if self._contact_pos_w is None:
            return None
        if self._contact_pos_w_ta is None:
            self._contact_pos_w_ta = ProxyArray(self._contact_pos_w)
        return self._contact_pos_w_ta

    @property
    def friction_force_matrix_w(self) -> ProxyArray | None:
        """Per-counterpart friction contact forces [N] in world frame.

        `wp.vec3f` array whose shape is (N, S, F), where F is the number of filter objects. When cast to a
        `torch.Tensor`, the shape is (N, S, F, 3).

        None if friction tracking is disabled or no filter objects are configured.
        """
        if self._friction_force_matrix_w is None:
            return None
        if self._friction_force_matrix_w_ta is None:
            self._friction_force_matrix_w_ta = ProxyArray(self._friction_force_matrix_w)
        return self._friction_force_matrix_w_ta

    @property
    def friction_force_matrix_w_history(self) -> ProxyArray | None:
        """History of per-counterpart friction contact forces [N] in world frame.

        `wp.vec3f` array whose shape is (N, T, S, F). In the history dimension, the first index is the most
        recent and the last index is the oldest.

        None if friction tracking is disabled or no filter objects are configured.
        """
        if self._friction_force_matrix_w_history is None:
            return None
        if self._friction_force_matrix_w_history_ta is None:
            self._friction_force_matrix_w_history_ta = ProxyArray(self._friction_force_matrix_w_history)
        return self._friction_force_matrix_w_history_ta

    @property
    def last_air_time(self) -> ProxyArray | None:
        """Time spent (in s) in the air before the last contact.

        `wp.float32` array whose shape is (N, S) where N is the number of environments and S is the number of sensors.
        Note, that when casted to as a `torch.Tensor`, the shape will be (N, S).

        Note:
            If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
        """
        if self._last_air_time is None:
            return None
        return self._last_air_time_ta

    @property
    def current_air_time(self) -> ProxyArray | None:
        """Time spent (in s) in the air since the last detach.

        `wp.float32` array whose shape is (N, S) where N is the number of environments and S is the number of sensors.
        Note, that when casted to as a `torch.Tensor`, the shape will be (N, S).

        Note:
            If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
        """
        if self._current_air_time is None:
            return None
        return self._current_air_time_ta

    @property
    def last_contact_time(self) -> ProxyArray | None:
        """Time spent (in s) in contact before the last detach.

        `wp.float32` array whose shape is (N, S) where N is the number of environments and S is the number of sensors.
        Note, that when casted to as a `torch.Tensor`, the shape will be (N, S).

        Note:
            If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
        """
        if self._last_contact_time is None:
            return None
        return self._last_contact_time_ta

    @property
    def current_contact_time(self) -> ProxyArray | None:
        """Time spent (in s) in contact since the last contact.

        `wp.float32` array whose shape is (N, S) where N is the number of environments and S is the number of sensors.
        Note, that when casted to as a `torch.Tensor`, the shape will be (N, S).

        Note:
            If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
        """
        if self._current_contact_time is None:
            return None
        return self._current_contact_time_ta

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
        *,
        track_contact_points: bool = False,
        track_friction_forces: bool = False,
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
            track_contact_points: Whether to track the contact point positions.
            track_friction_forces: Whether to track friction contact forces.
        """
        logger.info(
            f"Creating buffers for contact sensor data with num_envs: {num_envs}, num_sensors: {num_sensors},"
            f" num_filter_objects: {num_filter_objects}, history_length: {history_length}, generate_force_matrix:"
            f" {generate_force_matrix}, track_contact_points: {track_contact_points}, track_air_time:"
            f" {track_air_time}, track_pose: {track_pose}, device: {device}"
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

        # Ensure history_length >= 1 for consistent buffer shapes across backends.
        effective_history = max(history_length, 1)

        # Total forces from Newton (always tracked) - shape: (num_envs, num_sensors)
        self._net_forces_w = wp.zeros((num_envs, num_sensors), dtype=wp.vec3f, device=device)
        self._net_forces_w_history = wp.zeros((num_envs, effective_history, num_sensors), dtype=wp.vec3f, device=device)

        # Create owned buffer for net normal forces - shape: (num_envs, num_sensors)
        self._net_normal_forces_w = wp.zeros((num_envs, num_sensors), dtype=wp.vec3f, device=device)
        self._net_normal_forces_w_history = wp.zeros(
            (num_envs, effective_history, num_sensors), dtype=wp.vec3f, device=device
        )

        # Create owned buffer for force matrix - shape: (num_envs, num_sensors, num_filter_objects)
        # None if no filter objects configured
        if num_filter_objects > 0:
            self._force_matrix_w = wp.zeros((num_envs, num_sensors, num_filter_objects), dtype=wp.vec3f, device=device)
            self._force_matrix_w_history = wp.zeros(
                (num_envs, effective_history, num_sensors, num_filter_objects), dtype=wp.vec3f, device=device
            )
            self._normal_force_matrix_w = wp.zeros(
                (num_envs, num_sensors, num_filter_objects), dtype=wp.vec3f, device=device
            )
            self._normal_force_matrix_w_history = wp.zeros(
                (num_envs, effective_history, num_sensors, num_filter_objects), dtype=wp.vec3f, device=device
            )
        else:
            self._force_matrix_w = None
            self._force_matrix_w_history = None
            self._normal_force_matrix_w = None
            self._normal_force_matrix_w_history = None

        if track_friction_forces:
            self._net_friction_forces_w = wp.zeros((num_envs, num_sensors), dtype=wp.vec3f, device=device)
            self._net_friction_forces_w_history = wp.zeros(
                (num_envs, effective_history, num_sensors), dtype=wp.vec3f, device=device
            )
            if num_filter_objects > 0:
                self._friction_force_matrix_w = wp.zeros(
                    (num_envs, num_sensors, num_filter_objects), dtype=wp.vec3f, device=device
                )
                self._friction_force_matrix_w_history = wp.zeros(
                    (num_envs, effective_history, num_sensors, num_filter_objects), dtype=wp.vec3f, device=device
                )
            else:
                self._friction_force_matrix_w = None
                self._friction_force_matrix_w_history = None
        else:
            self._net_friction_forces_w = None
            self._net_friction_forces_w_history = None
            self._friction_force_matrix_w = None
            self._friction_force_matrix_w_history = None

        # Track contact point positions if requested - filled with NaN (no contact)
        if track_contact_points and num_filter_objects > 0:
            self._contact_pos_w = wp.full(
                (num_envs, num_sensors, num_filter_objects), dtype=wp.vec3f, device=device, value=math.nan
            )
        else:
            self._contact_pos_w = None

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

        # -- Pin ProxyArray instances for pre-allocated buffers
        self._net_forces_w_ta = ProxyArray(self._net_forces_w)
        self._net_forces_w_history_ta = ProxyArray(self._net_forces_w_history)
        self._net_normal_forces_w_ta = ProxyArray(self._net_normal_forces_w)
        self._net_normal_forces_w_history_ta = (
            ProxyArray(self._net_normal_forces_w_history) if self._net_normal_forces_w_history is not None else None
        )
        # -- Lazy ProxyArray instances for nullable buffers (pinned on first access)
        self._pos_w_ta: ProxyArray | None = None
        self._quat_w_ta: ProxyArray | None = None
        self._force_matrix_w_ta: ProxyArray | None = None
        self._force_matrix_w_history_ta: ProxyArray | None = None
        self._normal_force_matrix_w_ta: ProxyArray | None = None
        self._normal_force_matrix_w_history_ta: ProxyArray | None = None
        self._net_friction_forces_w_ta: ProxyArray | None = None
        self._net_friction_forces_w_history_ta: ProxyArray | None = None
        self._friction_force_matrix_w_ta: ProxyArray | None = None
        self._friction_force_matrix_w_history_ta: ProxyArray | None = None
        self._contact_pos_w_ta: ProxyArray | None = None
        # -- Pin ProxyArray instances for air/contact time buffers (eagerly when allocated)
        self._last_air_time_ta = ProxyArray(self._last_air_time) if self._last_air_time is not None else None
        self._current_air_time_ta = ProxyArray(self._current_air_time) if self._current_air_time is not None else None
        self._last_contact_time_ta = (
            ProxyArray(self._last_contact_time) if self._last_contact_time is not None else None
        )
        self._current_contact_time_ta = (
            ProxyArray(self._current_contact_time) if self._current_contact_time is not None else None
        )
