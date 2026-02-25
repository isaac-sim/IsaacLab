# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import math

import warp as wp

from isaaclab.sensors.contact_sensor import BaseContactSensorData

logger = logging.getLogger(__name__)


class ContactSensorData(BaseContactSensorData):
    """Data container for the PhysX contact reporting sensor."""

    @property
    def pose_w(self) -> wp.array | None:
        """Pose of the sensor origin in world frame. Shape is (N, 7). Quaternion in wxyz order."""
        logger.warning(
            "The `pose_w` property will be deprecated in a future release. Please use a dedicated sensor to measure"
            "sensor poses in world frame."
        )
        # Cannot simply cat vec3f + quatf in warp; return None to signal deprecation
        # Users should access pos_w and quat_w separately.
        return None

    @property
    def pos_w(self) -> wp.array | None:
        """Position of the sensor origin in world frame. Shape is (N, B) vec3f.

        None if :attr:`ContactSensorCfg.track_pose` is False.
        """
        logger.warning(
            "The `pos_w` property will be deprecated in a future release. Please use a dedicated sensor to measure"
            "sensor positions in world frame."
        )
        return self._pos_w

    @property
    def quat_w(self) -> wp.array | None:
        """Orientation of the sensor origin in quaternion (x, y, z, w) in world frame. Shape is (N, B) quatf.

        None if :attr:`ContactSensorCfg.track_pose` is False.
        """
        logger.warning(
            "The `quat_w` property will be deprecated in a future release. Please use a dedicated sensor to measure"
            "sensor orientations in world frame."
        )
        return self._quat_w

    @property
    def net_forces_w(self) -> wp.array | None:
        """The net normal contact forces in world frame. Shape is (N, B) vec3f."""
        return self._net_forces_w

    @property
    def net_forces_w_history(self) -> wp.array | None:
        """History of net normal contact forces. Shape is (N, T, B) vec3f."""
        return self._net_forces_w_history

    @property
    def force_matrix_w(self) -> wp.array | None:
        """Normal contact forces filtered between sensor and filtered bodies. Shape is (N, B, M) vec3f.

        None if :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty.
        """
        return self._force_matrix_w

    @property
    def force_matrix_w_history(self) -> wp.array | None:
        """History of filtered contact forces. Shape is (N, T, B, M) vec3f.

        None if :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty.
        """
        return self._force_matrix_w_history

    @property
    def contact_pos_w(self) -> wp.array | None:
        """Average position of contact points. Shape is (N, B, M) vec3f.

        None if :attr:`ContactSensorCfg.track_contact_points` is False.
        """
        return self._contact_pos_w

    @property
    def friction_forces_w(self) -> wp.array | None:
        """Sum of friction forces. Shape is (N, B, M) vec3f.

        None if :attr:`ContactSensorCfg.track_friction_forces` is False.
        """
        return self._friction_forces_w

    @property
    def last_air_time(self) -> wp.array | None:
        """Time spent in air before last contact. Shape is (N, B) float32.

        None if :attr:`ContactSensorCfg.track_air_time` is False.
        """
        return self._last_air_time

    @property
    def current_air_time(self) -> wp.array | None:
        """Time spent in air since last detach. Shape is (N, B) float32.

        None if :attr:`ContactSensorCfg.track_air_time` is False.
        """
        return self._current_air_time

    @property
    def last_contact_time(self) -> wp.array | None:
        """Time spent in contact before last detach. Shape is (N, B) float32.

        None if :attr:`ContactSensorCfg.track_air_time` is False.
        """
        return self._last_contact_time

    @property
    def current_contact_time(self) -> wp.array | None:
        """Time spent in contact since last contact. Shape is (N, B) float32.

        None if :attr:`ContactSensorCfg.track_air_time` is False.
        """
        return self._current_contact_time

    def create_buffers(
        self,
        num_envs: int,
        num_bodies: int,
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
            num_bodies: Number of bodies per environment.
            num_filter_shapes: Number of filtered shapes for force matrix.
            history_length: Length of force history buffer.
            track_pose: Whether to track sensor pose.
            track_air_time: Whether to track air/contact time.
            track_contact_points: Whether to track contact points.
            track_friction_forces: Whether to track friction forces.
            device: Device for tensor storage.
        """
        # Ensure history_length >= 1 for consistent buffer shapes
        effective_history = max(history_length, 1)

        # Net forces (always tracked)
        self._net_forces_w = wp.zeros((num_envs, num_bodies), dtype=wp.vec3f, device=device)
        self._net_forces_w_history = wp.zeros((num_envs, effective_history, num_bodies), dtype=wp.vec3f, device=device)

        # Force matrix (optional - only with filter)
        if num_filter_shapes > 0:
            self._force_matrix_w = wp.zeros((num_envs, num_bodies, num_filter_shapes), dtype=wp.vec3f, device=device)
            self._force_matrix_w_history = wp.zeros(
                (num_envs, effective_history, num_bodies, num_filter_shapes), dtype=wp.vec3f, device=device
            )
        else:
            self._force_matrix_w = None
            self._force_matrix_w_history = None

        # Pose tracking (optional)
        if track_pose:
            self._pos_w = wp.zeros((num_envs, num_bodies), dtype=wp.vec3f, device=device)
            self._quat_w = wp.zeros((num_envs, num_bodies), dtype=wp.quatf, device=device)
        else:
            self._pos_w = None
            self._quat_w = None

        # Air/contact time tracking (optional)
        if track_air_time:
            self._last_air_time = wp.zeros((num_envs, num_bodies), dtype=wp.float32, device=device)
            self._current_air_time = wp.zeros((num_envs, num_bodies), dtype=wp.float32, device=device)
            self._last_contact_time = wp.zeros((num_envs, num_bodies), dtype=wp.float32, device=device)
            self._current_contact_time = wp.zeros((num_envs, num_bodies), dtype=wp.float32, device=device)
        else:
            self._last_air_time = None
            self._current_air_time = None
            self._last_contact_time = None
            self._current_contact_time = None

        # Contact points (optional) - filled with NaN
        if track_contact_points:
            self._contact_pos_w = wp.full(
                (num_envs, num_bodies, num_filter_shapes),
                dtype=wp.vec3f,
                device=device,
                value=wp.vec3f(math.nan, math.nan, math.nan),
            )
        else:
            self._contact_pos_w = None

        # Friction forces (optional)
        if track_friction_forces:
            self._friction_forces_w = wp.zeros((num_envs, num_bodies, num_filter_shapes), dtype=wp.vec3f, device=device)
        else:
            self._friction_forces_w = None
