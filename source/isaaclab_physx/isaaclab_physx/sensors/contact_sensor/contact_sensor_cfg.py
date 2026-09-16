# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg as _BaseContactSensorCfg
from isaaclab.utils import config_field

if TYPE_CHECKING:
    from .contact_sensor import ContactSensor


@dataclass
class ContactSensorCfg(_BaseContactSensorCfg):
    """PhysX contact sensor configuration."""

    class_type: type["ContactSensor"] | str = config_field("{DIR}.contact_sensor:ContactSensor")
