# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for PhysX rigid contact sensor."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "contact_sensor": ["ContactSensor"],
        "contact_sensor_cfg": ["ContactSensorCfg"],
        "contact_sensor_data": ["ContactSensorData"],
    },
)
