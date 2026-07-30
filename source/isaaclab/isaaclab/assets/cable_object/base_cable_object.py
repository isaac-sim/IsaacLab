# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.assets.asset_base import AssetBase
from isaaclab.utils.warp import ProxyArray

if TYPE_CHECKING:
    from .base_cable_object_data import BaseCableObjectData
    from .cable_object_cfg import CableObjectCfg


class BaseCableObject(AssetBase):
    """Abstract base class for cable object assets.

    Cable objects expose the world-frame pose and velocity of each simulated segment.
    """

    cfg: CableObjectCfg
    """Configuration instance for the cable object."""

    __backend_name__: str = "base"
    """The name of the backend for the cable object."""

    def __init__(self, cfg: CableObjectCfg) -> None:
        """Initialize the cable object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)

    @property
    @abstractmethod
    def data(self) -> BaseCableObjectData:
        """Data container for the cable object."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_instances(self) -> int:
        """Number of cable instances."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_segments(self) -> int:
        """Number of rigid segments per cable."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def root_view(self):
        """Root articulation view for the cable object.

        .. note::
            Use this view with caution. It requires handling tensors in a backend-specific way.
        """
        raise NotImplementedError()

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the cable object.

        Args:
            env_ids: Environment indices. If None, all instances are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_data_to_sim(self) -> None:
        """Write buffered commands to the simulation."""
        raise NotImplementedError()

    @abstractmethod
    def update(self, dt: float) -> None:
        """Update the cable data.

        Args:
            dt: The time step for the update [s].
        """
        raise NotImplementedError()

    @abstractmethod
    def write_segment_pose_to_sim_index(
        self,
        *,
        segment_pose: torch.Tensor | wp.array(dtype=wp.transformf) | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array(dtype=wp.int32) | None = None,
    ) -> None:
        """Set segment poses for selected environments.

        Args:
            segment_pose: Segment actor-frame poses in simulation world frame. The Torch shape is
                (len(env_ids), num_segments, 7), with position ``(x, y, z)`` [m] followed by quaternion
                ``(x, y, z, w)``. The Warp shape is (len(env_ids), num_segments), dtype ``wp.transformf``.
            env_ids: Environment indices. If None, all instances are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_segment_pose_to_sim_mask(
        self,
        *,
        segment_pose: torch.Tensor | wp.array(dtype=wp.transformf) | ProxyArray,
        env_mask: wp.array(dtype=wp.bool) | None = None,
    ) -> None:
        """Set segment poses using an environment mask.

        Args:
            segment_pose: Segment actor-frame poses in simulation world frame. The Torch shape is
                (num_instances, num_segments, 7), with position (x, y, z) [m] followed by quaternion
                (x, y, z, w). The Warp shape is (num_instances, num_segments), dtype wp.transformf.
            env_mask: Environment mask. If None, all instances are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_segment_velocity_to_sim_index(
        self,
        *,
        segment_velocity: torch.Tensor | wp.array(dtype=wp.spatial_vectorf) | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array(dtype=wp.int32) | None = None,
    ) -> None:
        """Set segment velocities for selected environments.

        Args:
            segment_velocity: Segment center-of-mass velocities in simulation world frame. The Torch shape is
                (len(env_ids), num_segments, 6), with linear ``(x, y, z)`` [m/s] followed by angular
                ``(x, y, z)`` [rad/s] velocity. The Warp shape is (len(env_ids), num_segments), dtype
                ``wp.spatial_vectorf``.
            env_ids: Environment indices. If None, all instances are used.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_segment_velocity_to_sim_mask(
        self,
        *,
        segment_velocity: torch.Tensor | wp.array(dtype=wp.spatial_vectorf) | ProxyArray,
        env_mask: wp.array(dtype=wp.bool) | None = None,
    ) -> None:
        """Set segment velocities using an environment mask.

        Args:
            segment_velocity: Segment center-of-mass velocities in simulation world frame. The Torch shape is
                (num_instances, num_segments, 6), with linear (x, y, z) [m/s] followed by angular
                (x, y, z) [rad/s] velocity. The Warp shape is (num_instances, num_segments), dtype
                wp.spatial_vectorf.
            env_mask: Environment mask. If None, all instances are used.
        """
        raise NotImplementedError()
