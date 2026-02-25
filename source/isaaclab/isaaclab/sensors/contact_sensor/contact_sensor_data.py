# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Re-exports the base contact sensor data class for backwards compatibility."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_contact_sensor_data import BaseContactSensorData

if TYPE_CHECKING:
    from isaaclab_physx.sensors.contact_sensor import ContactSensorData as PhysXContactSensorData


class ContactSensorData(FactoryBase, BaseContactSensorData):
    """Factory for creating contact sensor data instances."""

    def __new__(cls, *args, **kwargs) -> BaseContactSensorData | PhysXContactSensorData:
        """Create a new instance of a contact sensor data based on the backend."""
        return super().__new__(cls, *args, **kwargs)
