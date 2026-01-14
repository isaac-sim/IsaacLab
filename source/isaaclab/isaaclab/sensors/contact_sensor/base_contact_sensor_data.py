# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: torch.Tensor | None
from __future__ import annotations

from abc import ABC, abstractmethod

import warp as wp


class BaseContactSensorData(ABC):
    """Data container for the contact reporting sensor."""

    @property
    @abstractmethod
    def pos_w(self) -> wp.array | None:
        """Position of the sensor origin in world frame.

        `wp.vec3f` array whose shape is (N,) where N is the number of sensors. Note, that when casted to as a
        `torch.Tensor`, the shape will be (N, 3).

        Note:
            If the :attr:`ContactSensorCfg.track_pose` is False, then this quantity is None.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def quat_w(self) -> wp.array | None:
        """Orientation of the sensor origin in quaternion (x, y, z, w) in world frame.

        `wp.quatf` whose shape is (N,) where N is the number of sensors. Note, that when casted to as a `torch.Tensor`,
        the shape will be (N, 4).

        Note:
            If the :attr:`ContactSensorCfg.track_pose` is False, then this quantity is None.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def net_forces_w(self) -> wp.array | None:
        """The net normal contact forces in world frame.

        `wp.vec3f` array whose shape is (N, B) where N is the number of sensors and B is the number of bodies in each
        sensor. Note, that when casted to as a `torch.Tensor`, the shape will be (N, B, 3).

        Note:
            This quantity is the sum of the normal contact forces acting on the sensor bodies. It must not be confused
            with the total contact forces acting on the sensor bodies (which also includes the tangential forces).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def net_forces_w_history(self) -> wp.array | None:
        """The net normal contact forces in world frame.

        `wp.vec3f` array whose shape is (N, T, B) where N is the number of sensors, T is the configured history length
        and B is the number of bodies in each sensor. Note, that when casted to as a `torch.Tensor`, the shape will be
        (N, T, B, 3).

        In the history dimension, the first index is the most recent and the last index is the oldest.

        Note:
            This quantity is the sum of the normal contact forces acting on the sensor bodies. It must not be confused
            with the total contact forces acting on the sensor bodies (which also includes the tangential forces).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def force_matrix_w(self) -> wp.array | None:
        """The normal contact forces filtered between the sensor bodies and filtered bodies in world frame.

        `wp.vec3f` array whose shape is (N, B, M) where N is the number of sensors, B is number of bodies in each sensor
        and M is the number of filtered bodies. Note, that when casted to as a `torch.Tensor`, the shape will be
        (N, B, M, 3).

        Note:
            If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is None.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def force_matrix_w_history(self) -> wp.array | None:
        """The normal contact forces filtered between the sensor bodies and filtered bodies in world frame.

        `wp.vec3f` array whose shape is (N, T, B, M) where N is the number of sensors, T is the configured history
        length and B is number of bodies in each sensor and M is the number of filtered bodies. Note, that when casted
        to as a `torch.Tensor`, the shape will be (N, T, B, M, 3).

        In the history dimension, the first index is the most recent and the last index is the oldest.

        Note:
            If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is None.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def last_air_time(self) -> wp.array | None:
        """Time spent (in s) in the air before the last contact.

        `wp.float32` array whose shape is (N, B) where N is the number of sensors and B is the number of bodies in each
        sensor. Note, that when casted to as a `torch.Tensor`, the shape will be (N, B).

        Note:
            If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def current_air_time(self) -> wp.array | None:
        """Time spent (in s) in the air since the last detach.

        `wp.float32` array whose shape is (N, B) where N is the number of sensors and B is the number of bodies in each
        sensor. Note, that when casted to as a `torch.Tensor`, the shape will be (N, B).

        Note:
            If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def last_contact_time(self) -> wp.array | None:
        """Time spent (in s) in contact before the last detach.

        `wp.float32` array whose shape is (N, B) where N is the number of sensors and B is the number of bodies in each
        sensor. Note, that when casted to as a `torch.Tensor`, the shape will be (N, B).

        Note:
            If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def current_contact_time(self) -> wp.array | None:
        """Time spent (in s) in contact since the last contact.

        `wp.float32` array whose shape is (N, B) where N is the number of sensors and B is the number of bodies in each
        sensor. Note, that when casted to as a `torch.Tensor`, the shape will be (N, B).

        Note:
            If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
        """
        raise NotImplementedError

    @abstractmethod
    def create_buffers(self, num_envs: int, num_bodies: int, num_filter_bodies: int, history_length: int, device: str):
        """Creates the buffers for the contact sensor data.

        Args:
            num_envs: The number of environments.
            num_bodies: The number of bodies in each sensor.
            num_filter_bodies: The number of filtered bodies.
            history_length: The history length.
        """
        raise NotImplementedError
