# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for Warp-based ray-cast sensor.

The sub-module contains two implementations of the ray-cast sensor:

- :class:`isaaclab.sensors.ray_caster.RayCaster`: A basic ray-cast sensor that can be used to ray-cast against a single mesh.
- :class:`isaaclab.sensors.ray_caster.MultiMeshRayCaster`: A multi-mesh ray-cast sensor that can be used to ray-cast against
  multiple meshes. For these meshes, it tracks their transformations and updates the warp meshes accordingly.

Corresponding camera implementations are also provided for each of the sensor implementations. Internally, they perform
the same ray-casting operations as the sensor implementations, but return the results as images.
"""

from . import patterns

from isaaclab.utils.module import lazy_export

lazy_export()
