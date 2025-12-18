# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Ignore optional memory usage warning globally
# pyright: reportOptionalSubscript=false

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_contact_sensor import BaseContactSensor
from .base_contact_sensor_data import BaseContactSensorData

if TYPE_CHECKING:
    from isaaclab_newton.sensors.contact_sensor import ContactSensor as NewtonContactSensor
    from isaaclab_newton.sensors.contact_sensor import ContactSensorData as NewtonContactSensorData


class ContactSensor(FactoryBase):
    """Factory for creating contact sensor instances."""

    data: BaseContactSensorData | NewtonContactSensorData

    def __new__(cls, *args, **kwargs) -> BaseContactSensor | NewtonContactSensor:
        """Create a new instance of a contact sensor based on the backend."""
        # The `FactoryBase` __new__ method will handle the logic and return
        # an instance of the correct backend-specific contact sensor class,
        # which is guaranteed to be a subclass of `BaseContactSensor` by convention.
        return super().__new__(cls, *args, **kwargs)
