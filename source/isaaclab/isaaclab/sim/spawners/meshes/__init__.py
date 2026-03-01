# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for spawning meshes in the simulation.

NVIDIA Omniverse deals with meshes as `USDGeomMesh`_ prims. This sub-module provides various
configurations to spawn different types of meshes. Based on the configuration, the spawned prim can be:

* a visual mesh (no physics)
* a static collider (no rigid or deformable body)
* a deformable body (with deformable properties)

.. note::
    While rigid body properties can be set on a mesh, it is recommended to use the
    :mod:`isaaclab.sim.spawners.shapes` module to spawn rigid bodies. This is because USD shapes
    are more optimized for physics simulations.

.. _USDGeomMesh: https://openusd.org/release/api/class_usd_geom_mesh.html
"""

from isaaclab.utils.module import lazy_export

lazy_export()
