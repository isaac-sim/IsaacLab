# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_imu import BaseImu
from .base_imu_data import BaseImuData

if TYPE_CHECKING:
    from isaaclab_physx.sensors.imu import Imu as PhysXImu
    from isaaclab_physx.sensors.imu import ImuData as PhysXImuData


class Imu(FactoryBase, BaseImu):
    """Factory for creating IMU sensor instances."""

    data: BaseImuData | PhysXImuData

    def __new__(cls, *args, **kwargs) -> BaseImu | PhysXImu:
        """Create a new instance of an IMU sensor based on the backend."""
        return super().__new__(cls, *args, **kwargs)
