# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import warp as wp

from ..sensor_base import SensorBase
from .base_imu_data import BaseImuData

if TYPE_CHECKING:
    from .imu_cfg import ImuCfg


class BaseImu(SensorBase):
    """The Inertia Measurement Unit (IMU) sensor.

    The sensor can be attached to any prim path with a rigid ancestor in its tree and produces body-frame
    linear acceleration and angular velocity, along with world-frame pose and body-frame linear and angular
    accelerations/velocities.

    If the provided path is not a rigid body, the closest rigid-body ancestor is used for simulation queries.
    The fixed transform from that ancestor to the target prim is computed once during initialization and
    composed with the configured sensor offset.

    .. note::

        We are computing the accelerations using numerical differentiation from the velocities. Consequently, the
        IMU sensor accuracy depends on the chosen phsyx timestep. For a sufficient accuracy, we recommend to keep the
        timestep at least as 200Hz.

    .. note::

        The user can configure the sensor offset in the configuration file. The offset is applied relative to the
        rigid source prim. If the target prim is not a rigid body, the offset is composed with the fixed transform
        from the rigid ancestor to the target prim. The offset is applied in the body frame of the rigid source prim.
        The offset is defined as a position vector and a quaternion rotation, which
        are applied in the order: position, then rotation. The position is applied as a translation
        in the body frame of the rigid source prim, and the rotation is applied as a rotation
        in the body frame of the rigid source prim.

    """

    cfg: ImuCfg
    """The configuration parameters."""

    __backend_name__: str = "base"
    """The name of the backend for the IMU sensor."""

    def __init__(self, cfg: ImuCfg):
        """Initializes the Imu sensor.

        Args:
            cfg: The configuration parameters.
        """
        # initialize base class
        super().__init__(cfg)

    """
    Properties
    """

    @property
    @abstractmethod
    def data(self) -> BaseImuData:
        raise NotImplementedError

    """
    Implementation - Abstract methods to be implemented by backend-specific subclasses.
    """

    @abstractmethod
    def _initialize_impl(self):
        """Initializes the sensor handles and internal buffers.

        Subclasses should call ``super()._initialize_impl()`` first to initialize
        the common sensor infrastructure from :class:`SensorBase`.
        """
        super()._initialize_impl()

    @abstractmethod
    def _update_buffers_impl(self, env_mask: wp.array):
        raise NotImplementedError
