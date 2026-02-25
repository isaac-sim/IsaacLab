# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_contact_sensor import BaseContactSensor
from .base_contact_sensor_data import BaseContactSensorData

if TYPE_CHECKING:
    from isaaclab_physx.sensors.contact_sensor import ContactSensor as PhysXContactSensor
    from isaaclab_physx.sensors.contact_sensor import ContactSensorData as PhysXContactSensorData


class ContactSensor(FactoryBase, BaseContactSensor):
    """Factory for creating contact sensor instances."""

    data: BaseContactSensorData | PhysXContactSensorData

    def __new__(cls, *args, **kwargs) -> BaseContactSensor | PhysXContactSensor:
        """Create a new instance of a contact sensor based on the backend."""
        return super().__new__(cls, *args, **kwargs)
