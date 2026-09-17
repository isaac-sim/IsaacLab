# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg as _BaseContactSensorCfg

if TYPE_CHECKING:
    from .contact_sensor import ContactSensor


@dataclass
class ContactSensorCfg(_BaseContactSensorCfg):
    """OVPhysX contact sensor configuration."""

    class_type: type["ContactSensor"] | str = "isaaclab_ov.sensors.contact_sensor.contact_sensor:ContactSensor"
