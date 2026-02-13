# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging

import torch

from isaaclab.sensors.contact_sensor import BaseContactSensorData

logger = logging.getLogger(__name__)


class ContactSensorData(BaseContactSensorData):
    """Data container for the PhysX contact reporting sensor."""

    @property
    def pose_w(self) -> torch.Tensor | None:
        """Pose of the sensor origin in world frame. Shape is (N, 7). Quaternion in wxyz order."""
        logger.warning(
            "The `pose_w` property will be deprecated in a future release. Please use a dedicated sensor to measure"
            "sensor poses in world frame."
        )
        return torch.cat([self._pos_w, self._quat_w], dim=-1)

    @property
    def pos_w(self) -> torch.Tensor | None:
        """Position of the sensor origin in world frame. Shape is (N, 3).

        None if :attr:`ContactSensorCfg.track_pose` is False.
        """
        logger.warning(
            "The `pos_w` property will be deprecated in a future release. Please use a dedicated sensor to measure"
            "sensor positions in world frame."
        )
        return self._pos_w

    @property
    def quat_w(self) -> torch.Tensor | None:
        """Orientation of the sensor origin in quaternion (x, y, z, w) in world frame. Shape is (N, 4).

        None if :attr:`ContactSensorCfg.track_pose` is False.
        """
        logger.warning(
            "The `quat_w` property will be deprecated in a future release. Please use a dedicated sensor to measure"
            "sensor orientations in world frame."
        )
        return self._quat_w

    @property
    def net_forces_w(self) -> torch.Tensor | None:
        """The net normal contact forces in world frame. Shape is (N, B, 3)."""
        return self._net_forces_w

    @property
    def net_forces_w_history(self) -> torch.Tensor | None:
        """History of net normal contact forces. Shape is (N, T, B, 3)."""
        return self._net_forces_w_history

    @property
    def force_matrix_w(self) -> torch.Tensor | None:
        """Normal contact forces filtered between sensor and filtered bodies. Shape is (N, B, M, 3).

        None if :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty.
        """
        return self._force_matrix_w

    @property
    def force_matrix_w_history(self) -> torch.Tensor | None:
        """History of filtered contact forces. Shape is (N, T, B, M, 3).

        None if :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty.
        """
        return self._force_matrix_w_history

    @property
    def contact_pos_w(self) -> torch.Tensor | None:
        """Average position of contact points. Shape is (N, B, M, 3).

        None if :attr:`ContactSensorCfg.track_contact_points` is False.
        """
        return self._contact_pos_w

    @property
    def friction_forces_w(self) -> torch.Tensor | None:
        """Sum of friction forces. Shape is (N, B, M, 3).

        None if :attr:`ContactSensorCfg.track_friction_forces` is False.
        """
        return self._friction_forces_w

    @property
    def last_air_time(self) -> torch.Tensor | None:
        """Time spent in air before last contact. Shape is (N, B).

        None if :attr:`ContactSensorCfg.track_air_time` is False.
        """
        return self._last_air_time

    @property
    def current_air_time(self) -> torch.Tensor | None:
        """Time spent in air since last detach. Shape is (N, B).

        None if :attr:`ContactSensorCfg.track_air_time` is False.
        """
        return self._current_air_time

    @property
    def last_contact_time(self) -> torch.Tensor | None:
        """Time spent in contact before last detach. Shape is (N, B).

        None if :attr:`ContactSensorCfg.track_air_time` is False.
        """
        return self._last_contact_time

    @property
    def current_contact_time(self) -> torch.Tensor | None:
        """Time spent in contact since last contact. Shape is (N, B).

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
        # Net forces (always tracked)
        self._net_forces_w = torch.zeros(num_envs, num_bodies, 3, device=device)
        if history_length > 0:
            self._net_forces_w_history = torch.zeros(num_envs, history_length, num_bodies, 3, device=device)
        else:
            self._net_forces_w_history = self._net_forces_w.unsqueeze(1)

        # Force matrix (optional - only with filter)
        if num_filter_shapes > 0:
            self._force_matrix_w = torch.zeros(num_envs, num_bodies, num_filter_shapes, 3, device=device)
            if history_length > 0:
                self._force_matrix_w_history = torch.zeros(
                    num_envs, history_length, num_bodies, num_filter_shapes, 3, device=device
                )
            else:
                self._force_matrix_w_history = self._force_matrix_w.unsqueeze(1)
        else:
            self._force_matrix_w = None
            self._force_matrix_w_history = None

        # Pose tracking (optional)
        if track_pose:
            self._pos_w = torch.zeros(num_envs, num_bodies, 3, device=device)
            self._quat_w = torch.zeros(num_envs, num_bodies, 4, device=device)
        else:
            self._pos_w = None
            self._quat_w = None

        # Air/contact time tracking (optional)
        if track_air_time:
            self._last_air_time = torch.zeros(num_envs, num_bodies, device=device)
            self._current_air_time = torch.zeros(num_envs, num_bodies, device=device)
            self._last_contact_time = torch.zeros(num_envs, num_bodies, device=device)
            self._current_contact_time = torch.zeros(num_envs, num_bodies, device=device)
        else:
            self._last_air_time = None
            self._current_air_time = None
            self._last_contact_time = None
            self._current_contact_time = None

        # Contact points (optional)
        if track_contact_points:
            self._contact_pos_w = torch.full((num_envs, num_bodies, num_filter_shapes, 3), torch.nan, device=device)
        else:
            self._contact_pos_w = None

        # Friction forces (optional)
        if track_friction_forces:
            self._friction_forces_w = torch.zeros(num_envs, num_bodies, num_filter_shapes, 3, device=device)
        else:
            self._friction_forces_w = None
