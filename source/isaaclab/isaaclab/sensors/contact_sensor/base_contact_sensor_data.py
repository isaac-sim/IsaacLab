# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for contact sensor data containers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseContactSensorData(ABC):
    """Data container for the contact reporting sensor.

    This base class defines the interface for contact sensor data. Backend-specific
    implementations should inherit from this class and provide the actual data storage.
    """

    @property
    @abstractmethod
    def pose_w(self) -> torch.Tensor | None:
        """Pose of the sensor origin in world frame. Shape is (N, 7). Quaternion in xyzw order."""
        raise NotImplementedError

    @property
    @abstractmethod
    def pos_w(self) -> torch.Tensor | None:
        """Position of the sensor origin in world frame. Shape is (N, 3).

        None if :attr:`ContactSensorCfg.track_pose` is False.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def quat_w(self) -> torch.Tensor | None:
        """Orientation of the sensor origin in quaternion (x, y, z, w) in world frame.

        Shape is (N, 4). None if :attr:`ContactSensorCfg.track_pose` is False.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def net_forces_w(self) -> torch.Tensor | None:
        """The net normal contact forces in world frame. Shape is (N, B, 3)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def net_forces_w_history(self) -> torch.Tensor | None:
        """History of net normal contact forces. Shape is (N, T, B, 3)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def force_matrix_w(self) -> torch.Tensor | None:
        """Normal contact forces filtered between sensor and filtered bodies.

        Shape is (N, B, M, 3). None if :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def force_matrix_w_history(self) -> torch.Tensor | None:
        """History of filtered contact forces. Shape is (N, T, B, M, 3).

        None if :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty.
        """
        raise NotImplementedError

    # Make issues for this in Newton P1/P2s
    @property
    @abstractmethod
    def contact_pos_w(self) -> torch.Tensor | None:
        """Average position of contact points. Shape is (N, B, M, 3).

        None if :attr:`ContactSensorCfg.track_contact_points` is False.
        """
        raise NotImplementedError

    # Make issues for this in Newton P1/P2s
    @property
    @abstractmethod
    def friction_forces_w(self) -> torch.Tensor | None:
        """Sum of friction forces. Shape is (N, B, M, 3).

        None if :attr:`ContactSensorCfg.track_friction_forces` is False.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def last_air_time(self) -> torch.Tensor | None:
        """Time spent in air before last contact. Shape is (N, B).

        None if :attr:`ContactSensorCfg.track_air_time` is False.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def current_air_time(self) -> torch.Tensor | None:
        """Time spent in air since last detach. Shape is (N, B).

        None if :attr:`ContactSensorCfg.track_air_time` is False.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def last_contact_time(self) -> torch.Tensor | None:
        """Time spent in contact before last detach. Shape is (N, B).

        None if :attr:`ContactSensorCfg.track_air_time` is False.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def current_contact_time(self) -> torch.Tensor | None:
        """Time spent in contact since last contact. Shape is (N, B).

        None if :attr:`ContactSensorCfg.track_air_time` is False.
        """
        raise NotImplementedError
