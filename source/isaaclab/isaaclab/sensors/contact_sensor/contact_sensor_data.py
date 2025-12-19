# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: torch.Tensor | None
from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.backend_utils import FactoryBase

from .base_contact_sensor_data import BaseContactSensorData

if TYPE_CHECKING:
    from isaaclab_newton.sensors.contact_sensor.contact_sensor_data import ContactSensorData as NewtonContactSensorData


class ContactSensorData(FactoryBase):
    """Factory for creating contact sensor data instances."""

    def __new__(cls, *args, **kwargs) -> BaseContactSensorData | NewtonContactSensorData:
        """Create a new instance of a contact sensor data based on the backend."""
        return super().__new__(cls, *args, **kwargs)
