# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
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

from .shapes import spawn_capsule, spawn_cone, spawn_cuboid, spawn_cylinder, spawn_sphere
from .shapes_cfg import CapsuleCfg, ConeCfg, CuboidCfg, CylinderCfg, ShapeCfg, SphereCfg
