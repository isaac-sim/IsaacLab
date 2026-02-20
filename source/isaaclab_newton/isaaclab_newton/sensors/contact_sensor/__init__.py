# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for contact sensor based on :class:`newton.SensorContact`."""

from .contact_sensor import ContactSensor
from .contact_sensor_data import ContactSensorData

__all__ = [
    "ContactSensor",
    "ContactSensorData",
]
