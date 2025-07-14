# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import Literal

from isaaclab.sim.spawners import materials
from isaaclab.sim.spawners.spawner_cfg import DeformableObjectSpawnerCfg, RigidObjectSpawnerCfg
from isaaclab.utils import configclass

from . import meshes


@configclass
class MeshCfg(RigidObjectSpawnerCfg, DeformableObjectSpawnerCfg):
    """Configuration parameters for a USD Geometry or Geom prim.

    This class is similar to :class:`ShapeCfg` but is specifically for meshes.

    Meshes support both rigid and deformable properties. However, their schemas are applied at
    different levels in the USD hierarchy based on the type of the object. These are described below:

    - Deformable body properties: Applied to the mesh prim: ``{prim_path}/geometry/mesh``.
    - Collision properties: Applied to the mesh prim: ``{prim_path}/geometry/mesh``.
    - Rigid body properties: Applied to the parent prim: ``{prim_path}``.

    where ``{prim_path}`` is the path to the prim in the USD stage and ``{prim_path}/geometry/mesh``
    is the path to the mesh prim.

    .. note::
        There are mututally exclusive parameters for rigid and deformable properties. If both are set,
        then an error will be raised. This also holds if collision and deformable properties are set together.

    """

    visual_material_path: str = "material"
    """Path to the visual material to use for the prim. Defaults to "material".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `visual_material` is not None.
    """

    visual_material: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """

    physics_material_path: str = "material"
    """Path to the physics material to use for the prim. Defaults to "material".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `physics_material` is not None.
    """

    physics_material: materials.PhysicsMaterialCfg | None = None
    """Physics material properties.

    Note:
        If None, then no physics material will be added.
    """


@configclass
class MeshSphereCfg(MeshCfg):
    """Configuration parameters for a sphere mesh prim with deformable properties.

    See :meth:`spawn_mesh_sphere` for more information.
    """

    func: Callable = meshes.spawn_mesh_sphere

    radius: float = MISSING
    """Radius of the sphere (in m)."""


@configclass
class MeshCuboidCfg(MeshCfg):
    """Configuration parameters for a cuboid mesh prim with deformable properties.

    See :meth:`spawn_mesh_cuboid` for more information.
    """

    func: Callable = meshes.spawn_mesh_cuboid

    size: tuple[float, float, float] = MISSING
    """Size of the cuboid (in m)."""


@configclass
class MeshCylinderCfg(MeshCfg):
    """Configuration parameters for a cylinder mesh prim with deformable properties.

    See :meth:`spawn_cylinder` for more information.
    """

    func: Callable = meshes.spawn_mesh_cylinder

    radius: float = MISSING
    """Radius of the cylinder (in m)."""
    height: float = MISSING
    """Height of the cylinder (in m)."""
    axis: Literal["X", "Y", "Z"] = "Z"
    """Axis of the cylinder. Defaults to "Z"."""


@configclass
class MeshCapsuleCfg(MeshCfg):
    """Configuration parameters for a capsule mesh prim.

    See :meth:`spawn_capsule` for more information.
    """

    func: Callable = meshes.spawn_mesh_capsule

    radius: float = MISSING
    """Radius of the capsule (in m)."""
    height: float = MISSING
    """Height of the capsule (in m)."""
    axis: Literal["X", "Y", "Z"] = "Z"
    """Axis of the capsule. Defaults to "Z"."""


@configclass
class MeshConeCfg(MeshCfg):
    """Configuration parameters for a cone mesh prim.

    See :meth:`spawn_cone` for more information.
    """

    func: Callable = meshes.spawn_mesh_cone

    radius: float = MISSING
    """Radius of the cone (in m)."""
    height: float = MISSING
    """Height of the v (in m)."""
    axis: Literal["X", "Y", "Z"] = "Z"
    """Axis of the cone. Defaults to "Z"."""
