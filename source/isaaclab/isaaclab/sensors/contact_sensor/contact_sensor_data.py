# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Re-exports the base contact sensor data class for backwards compatibility."""

from .base_contact_sensor_data import BaseContactSensorData

# Re-export for backwards compatibility
ContactSensorData = BaseContactSensorData

__all__ = ["BaseContactSensorData", "ContactSensorData"]
