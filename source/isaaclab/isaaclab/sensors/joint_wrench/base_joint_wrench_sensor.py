# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import warp as wp

import isaaclab.utils.string as string_utils

from ..sensor_base import SensorBase
from .base_joint_wrench_sensor_data import BaseJointWrenchSensorData

if TYPE_CHECKING:
    from .joint_wrench_sensor_cfg import JointWrenchSensorCfg


class BaseJointWrenchSensor(SensorBase):
    """The joint reaction wrench sensor.

    Reports incoming joint wrenches for the bodies selected by the backend as
    split force [N] / torque [N·m] pairs expressed in the
    ``INCOMING_JOINT_FRAME`` convention (child-side joint frame, child-side
    joint anchor reference point). Backends convert from their native
    representation to this convention internally. Use :attr:`body_names` or
    :meth:`find_bodies` to map entries to articulation bodies.
    """

    cfg: JointWrenchSensorCfg
    """The configuration parameters."""

    __backend_name__: str = "base"
    """The name of the backend for the joint wrench sensor."""

    def __init__(self, cfg: JointWrenchSensorCfg):
        """Initialize the joint wrench sensor.

        Args:
            cfg: The configuration parameters.
        """
        super().__init__(cfg)

    """
    Properties
    """

    @property
    @abstractmethod
    def data(self) -> BaseJointWrenchSensorData:
        """The sensor data container, populated after simulation initialization."""
        raise NotImplementedError

    @property
    @abstractmethod
    def body_names(self) -> list[str]:
        """Ordered names of the bodies whose incoming joint wrench is reported."""
        raise NotImplementedError

    @property
    def num_bodies(self) -> int:
        """Number of bodies whose incoming joint wrench is reported."""
        return len(self.body_names)

    """
    Operations
    """

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find reported bodies based on name keys.

        Args:
            name_keys: A regular expression or list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            The matching body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    """
    Implementation - Abstract methods to be implemented by backend-specific subclasses.
    """

    @abstractmethod
    def _initialize_impl(self) -> None:
        """Initialize the sensor handles and internal buffers.

        Subclasses should call ``super()._initialize_impl()`` first to
        initialize the common sensor infrastructure from
        :class:`~isaaclab.sensors.SensorBase`.
        """
        super()._initialize_impl()

    @abstractmethod
    def _update_buffers_impl(self, env_mask: wp.array) -> None:
        raise NotImplementedError
