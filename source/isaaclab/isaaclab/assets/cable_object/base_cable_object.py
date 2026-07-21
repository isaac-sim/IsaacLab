# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets.asset_base import AssetBase

if TYPE_CHECKING:
    from .base_cable_object_data import BaseCableObjectData
    from .cable_object_cfg import CableObjectCfg


class BaseCableObject(AssetBase):
    """Abstract base class for cable object assets.

    Cable objects expose the world-frame pose and velocity of each simulated segment. They are read-only
    because the physics backend owns the segment state.
    """

    cfg: CableObjectCfg
    """Configuration instance for the cable object."""

    __backend_name__: str = "base"
    """The name of the backend for the cable object."""

    def __init__(self, cfg: CableObjectCfg):
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
