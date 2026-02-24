# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for spawning primitive shapes in the simulation.

NVIDIA Omniverse provides various primitive shapes that can be used to create USDGeom prims. Based
on the configuration, the spawned prim can be:

* a visual mesh (no physics)
* a static collider (no rigid body)
* a rigid body (with collision and rigid body properties).

"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "shapes": ["spawn_capsule", "spawn_cone", "spawn_cuboid", "spawn_cylinder", "spawn_sphere"],
        "shapes_cfg": ["CapsuleCfg", "ConeCfg", "CuboidCfg", "CylinderCfg", "ShapeCfg", "SphereCfg"],
    },
)
