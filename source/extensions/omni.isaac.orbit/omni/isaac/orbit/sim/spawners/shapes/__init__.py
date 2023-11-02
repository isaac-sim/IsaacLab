# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
NVIDIA Omniverse provides various primitive shapes that can be used to create USDGeom prims. Based
on the configuration, the spawned prim can be used a visual mesh (no physics), a static collider
(no rigid body), or a rigid body (with collision and rigid body properties).

Since this creates a prim manually, we follow the convention recommended by NVIDIA to prepare
`Sim-Ready assets <https://docs.omniverse.nvidia.com/simready/latest/simready-asset-creation.html>`_.
"""

from __future__ import annotations

from .shapes import spawn_capsule, spawn_cone, spawn_cuboid, spawn_cylinder, spawn_sphere
from .shapes_cfg import CapsuleCfg, ConeCfg, CuboidCfg, CylinderCfg, SphereCfg

__all__ = [
    # capsule
    "CapsuleCfg",
    "spawn_capsule",
    # cone
    "ConeCfg",
    "spawn_cone",
    # cuboid
    "CuboidCfg",
    "spawn_cuboid",
    # cylinder
    "CylinderCfg",
    "spawn_cylinder",
    # sphere
    "SphereCfg",
    "spawn_sphere",
]
