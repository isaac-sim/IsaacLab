# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for PhysX rigid contact sensor."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .contact_sensor import ContactSensor
    from .contact_sensor_data import ContactSensorData

from isaaclab.utils.module import lazy_export

lazy_export(
    ("contact_sensor", "ContactSensor"),
    ("contact_sensor_data", "ContactSensorData"),
)
