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
    """The Inertial Measurement Unit (IMU) sensor.

    This sensor models a real IMU that measures angular velocity (gyroscope) and
    proper linear acceleration (accelerometer) in the sensor's body frame. Unlike the
    PVA sensor, it does not provide pose, linear velocity, angular acceleration,
    or projected gravity.

    The sensor can be attached to any prim path with a rigid ancestor in its tree.
    If the provided path is not a rigid body, the closest rigid-body ancestor is used
    for simulation queries. The fixed transform from that ancestor to the target prim
    is computed once during initialization and composed with the configured sensor offset.

    .. note::

        The accuracy of the acceleration readings depends on the physics backend and timestep.
        For sufficient accuracy, we recommend keeping the timestep at least 200 Hz.
    """

    cfg: ImuCfg
    """The configuration parameters."""

    __backend_name__: str = "base"
    """The name of the backend for the IMU sensor."""

    def __init__(self, cfg: ImuCfg):
        """Initializes the IMU sensor.

        Args:
            cfg: The configuration parameters.
        """
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
