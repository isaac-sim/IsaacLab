# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TacSL Tactile Sensor implementation for IsaacLab."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "visuotactile_sensor": ["VisuoTactileSensor"],
        "visuotactile_sensor_cfg": ["GelSightRenderCfg", "VisuoTactileSensorCfg"],
        "visuotactile_sensor_data": ["VisuoTactileSensorData"],
    },
)
