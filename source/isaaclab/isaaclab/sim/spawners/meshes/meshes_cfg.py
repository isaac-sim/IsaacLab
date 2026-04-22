# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

    func: Callable | str = "{DIR}.meshes:spawn_mesh_sphere"

    radius: float = MISSING
    """Radius of the sphere (in m)."""


@configclass
class MeshCuboidCfg(MeshCfg):
    """Configuration parameters for a cuboid mesh prim with deformable properties.

    See :meth:`spawn_mesh_cuboid` for more information.
    """

    func: Callable | str = "{DIR}.meshes:spawn_mesh_cuboid"

    size: tuple[float, float, float] = MISSING
    """Size of the cuboid (in m)."""


@configclass
class MeshCylinderCfg(MeshCfg):
    """Configuration parameters for a cylinder mesh prim with deformable properties.

    See :meth:`spawn_cylinder` for more information.
    """

    func: Callable | str = "{DIR}.meshes:spawn_mesh_cylinder"

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

    func: Callable | str = "{DIR}.meshes:spawn_mesh_capsule"

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

    func: Callable | str = "{DIR}.meshes:spawn_mesh_cone"

    radius: float = MISSING
    """Radius of the cone (in m)."""
    height: float = MISSING
    """Height of the v (in m)."""
    axis: Literal["X", "Y", "Z"] = "Z"
    """Axis of the cone. Defaults to "Z"."""


@configclass
class MeshSquareCfg(MeshCfg):
    """Configuration parameters for a 2D square mesh prim.

    See :meth:`spawn_mesh_square` for more information.
    """

    func: Callable | str = "{DIR}.meshes:spawn_mesh_square"

    size: float = MISSING
    """Edge length of the square (in m)."""
    resolution: tuple[int, int] = (5, 5)
    """Resolution of the square (in elements/edges per side)."""


@configclass
class TetMeshCuboidCfg(MeshCfg):
    """Configuration parameters for a cuboid mesh prim with tetrahedral volumetric data.

    Generates a regular grid of vertices decomposed into tetrahedra (5 tets per hex cell).
    The surface triangles are written as ``UsdGeom.Mesh`` geometry for rendering, and the
    tet indices are stored as a custom ``int[]`` attribute ``newton:tetIndices`` on the mesh
    prim for the Newton backend to read.

    See :meth:`spawn_tet_mesh_cuboid` for more information.
    """

    func: Callable | str = "{DIR}.meshes:spawn_tet_mesh_cuboid"

    size: tuple[float, float, float] = MISSING
    """Size of the cuboid (in m) as (x, y, z)."""

    resolution: int = 4
    """Number of cells along each axis. Total tets = resolution^3 * 5."""


@configclass
class TetMeshFromFileCfg(MeshCfg):
    """Configuration parameters for spawning a tet mesh from a Gmsh ``.msh`` file.

    Loads tetrahedral mesh data via meshio and creates a ``UsdGeom.TetMesh`` prim
    with vertex positions, tet connectivity, and surface triangles -- the same
    structure as :class:`TetMeshCuboidCfg`.

    See :meth:`spawn_tet_mesh_from_file` for more information.
    """

    func: Callable | str = "{DIR}.meshes:spawn_tet_mesh_from_file"

    file_path: str = MISSING
    """Path to the tet mesh file (``.msh`` Gmsh format)."""

    scale: tuple[float, float, float] | float | None = None
    """Scale applied to vertex positions before writing to the stage.

    Use this to convert mesh units (e.g. ``0.01`` or ``(0.01, 0.01, 0.01)`` for cm to m).
    Unlike Xform scale, this modifies the actual vertex positions so the geometry in the
    stage is in the correct units. A scalar value is broadcast to all three axes.
    """


@configclass
class MeshFromFileCfg(MeshCfg):
    """Configuration parameters for spawning a mesh prim from a USD file.

    Loads mesh geometry (vertices + faces) from an external USD file and creates
    a ``UsdGeom.Mesh`` prim in the stage. This is useful for cloth or deformable meshes
    stored as bare geometry without physics APIs.

    See :meth:`spawn_mesh_from_file` for more information.
    """

    func: Callable | str = "{DIR}.meshes:spawn_mesh_from_file"

    usd_path: str = MISSING
    """Path to the USD file containing the mesh geometry."""

    usd_prim_path: str | None = None
    """Prim path within the USD file to read the mesh from (e.g. ``/root/shirt``).

    If None, the file's default prim is used. If the file has no default prim,
    the first ``UsdGeom.Mesh`` child of the pseudo-root is used.
    """

    scale: tuple[float, float, float] | float | None = None
    """Scale applied to mesh vertices before writing to the stage.

    Use this to convert mesh units (e.g. ``0.01`` or ``(0.01, 0.01, 0.01)`` for cm to m).
    Unlike Xform scale, this modifies the actual vertex positions so the geometry in the
    stage is in the correct units. A scalar value is broadcast to all three axes.
    """
