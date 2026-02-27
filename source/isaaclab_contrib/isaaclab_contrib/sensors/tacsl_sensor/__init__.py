# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TacSL Tactile Sensor implementation for IsaacLab."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .visuotactile_sensor import VisuoTactileSensor
    from .visuotactile_sensor_cfg import GelSightRenderCfg, VisuoTactileSensorCfg
    from .visuotactile_sensor_data import VisuoTactileSensorData

from isaaclab.utils.module import lazy_export

lazy_export(
    ("visuotactile_sensor", "VisuoTactileSensor"),
    ("visuotactile_sensor_cfg", ["GelSightRenderCfg", "VisuoTactileSensorCfg"]),
    ("visuotactile_sensor_data", "VisuoTactileSensorData"),
)
