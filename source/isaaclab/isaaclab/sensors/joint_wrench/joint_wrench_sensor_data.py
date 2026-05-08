# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory class for joint-wrench sensor data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_joint_wrench_sensor_data import BaseJointWrenchSensorData

if TYPE_CHECKING:
    from isaaclab_newton.sensors.joint_wrench import JointWrenchSensorData as NewtonJointWrenchSensorData
    from isaaclab_physx.sensors.joint_wrench import JointWrenchSensorData as PhysXJointWrenchSensorData


class JointWrenchSensorData(FactoryBase, BaseJointWrenchSensorData):
    """Factory for creating joint-wrench sensor data instances."""

    def __new__(
        cls, *args, **kwargs
    ) -> BaseJointWrenchSensorData | PhysXJointWrenchSensorData | NewtonJointWrenchSensorData:
        """Create a new instance of joint-wrench sensor data based on the backend."""
        return super().__new__(cls, *args, **kwargs)
