# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory class for IMU data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_imu_data import BaseImuData

if TYPE_CHECKING:
    from isaaclab_physx.sensors.imu import ImuData as PhysXImuData


class ImuData(FactoryBase, BaseImuData):
    """Factory for creating IMU data instances."""

    def __new__(cls, *args, **kwargs) -> BaseImuData | PhysXImuData:
        """Create a new instance of IMU data based on the backend."""
        return super().__new__(cls, *args, **kwargs)
