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

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["patterns"],
    submod_attrs={
        "multi_mesh_ray_caster": ["MultiMeshRayCaster"],
        "multi_mesh_ray_caster_camera": ["MultiMeshRayCasterCamera"],
        "multi_mesh_ray_caster_camera_cfg": ["MultiMeshRayCasterCameraCfg"],
        "multi_mesh_ray_caster_camera_data": ["MultiMeshRayCasterCameraData"],
        "multi_mesh_ray_caster_cfg": ["MultiMeshRayCasterCfg"],
        "multi_mesh_ray_caster_data": ["MultiMeshRayCasterData"],
        "ray_caster": ["RayCaster"],
        "ray_caster_camera": ["RayCasterCamera"],
        "ray_caster_camera_cfg": ["RayCasterCameraCfg"],
        "ray_caster_cfg": ["RayCasterCfg"],
        "ray_caster_data": ["RayCasterData"],
    },
)
