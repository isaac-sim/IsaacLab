# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import TYPE_CHECKING

from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg as _BaseContactSensorCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .contact_sensor import ContactSensor


@configclass
class ContactSensorCfg(_BaseContactSensorCfg):
    """PhysX contact sensor configuration."""

    class_type: type["ContactSensor"] | str = "{DIR}.contact_sensor:ContactSensor"
