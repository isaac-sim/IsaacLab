# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_joint_wrench_sensor import BaseJointWrenchSensor
from .base_joint_wrench_sensor_data import BaseJointWrenchSensorData

if TYPE_CHECKING:
    from isaaclab_newton.sensors.joint_wrench import JointWrenchSensor as NewtonJointWrenchSensor
    from isaaclab_newton.sensors.joint_wrench import JointWrenchSensorData as NewtonJointWrenchSensorData
    from isaaclab_physx.sensors.joint_wrench import JointWrenchSensor as PhysXJointWrenchSensor
    from isaaclab_physx.sensors.joint_wrench import JointWrenchSensorData as PhysXJointWrenchSensorData


class JointWrenchSensor(FactoryBase, BaseJointWrenchSensor):
    """Factory for creating joint-wrench sensor instances."""

    data: BaseJointWrenchSensorData | PhysXJointWrenchSensorData | NewtonJointWrenchSensorData

    def __new__(cls, *args, **kwargs) -> BaseJointWrenchSensor | PhysXJointWrenchSensor | NewtonJointWrenchSensor:
        """Create a new instance of a joint-wrench sensor based on the backend."""
        return super().__new__(cls, *args, **kwargs)
