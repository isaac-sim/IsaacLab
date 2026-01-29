# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for externally contributed sensors.

This package provides specialized sensor classes for simulating externally contributed
sensors in Isaac Lab. These sensors are not part of the core Isaac Lab framework yet,
but are planned to be added in the future. They are contributed by the community to
extend the capabilities of Isaac Lab.

Following the categorization in :mod:`isaaclab.sensors` sub-package, the prim paths passed
to the sensor's configuration class are interpreted differently based on the sensor type.
The following table summarizes the interpretation of the prim paths for different sensor types:

+---------------------+---------------------------+---------------------------------------------------------------+
| Sensor Type         | Example Prim Path         | Pre-check                                                     |
+=====================+===========================+===============================================================+
| Visuo-Tactile Sensor| /World/robot/base         | Leaf exists and is a physics body (Rigid Body)                |
+---------------------+---------------------------+---------------------------------------------------------------+

"""

from .tacsl_sensor import *
