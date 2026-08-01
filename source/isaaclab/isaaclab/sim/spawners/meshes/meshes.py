# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import trimesh
import trimesh.transformations

from pxr import Usd, UsdPhysics

from isaaclab.sim import schemas
from isaaclab.sim.utils import bind_physics_material, bind_visual_material, clone, create_prim, get_current_stage
from isaaclab.utils.version import has_kit

from ..materials import (
    DeformableBodyMaterialBaseCfg,
    RigidBodyMaterialBaseCfg,
    RigidBodyMaterialFragment,
    SurfaceDeformableBodyMaterialBaseCfg,
)
from ..materials.physics_materials import spawn_physics_material

if TYPE_CHECKING:
    from . import meshes_cfg

# import logger
logger = logging.getLogger(__name__)


@clone
def spawn_mesh_sphere(
    prim_path: str,
    cfg: meshes_cfg.MeshSphereCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USD-Mesh sphere prim with the given attributes.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    # create a trimesh sphere
    sphere = trimesh.creation.uv_sphere(radius=cfg.radius)

    # obtain stage handle
    stage = get_current_stage()
    # spawn the sphere as a mesh
    _spawn_mesh_geom_from_mesh(prim_path, cfg, sphere, translation, orientation, stage=stage)
    # return the prim
    return stage.GetPrimAtPath(prim_path)


@clone
def spawn_mesh_cuboid(
    prim_path: str,
    cfg: meshes_cfg.MeshCuboidCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USD-Mesh cuboid prim with the given attributes.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    # create a trimesh box
    box = trimesh.creation.box(cfg.size)

    # obtain stage handle
    stage = get_current_stage()
    # spawn the cuboid as a mesh
    _spawn_mesh_geom_from_mesh(prim_path, cfg, box, translation, orientation, None, stage=stage)
    # return the prim
    return stage.GetPrimAtPath(prim_path)


@clone
def spawn_mesh_cylinder(
    prim_path: str,
    cfg: meshes_cfg.MeshCylinderCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USD-Mesh cylinder prim with the given attributes.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    # align axis from "Z" to input by rotating the cylinder
    axis = cfg.axis.upper()
    if axis == "X":
        transform = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    elif axis == "Y":
        transform = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    else:
        transform = None
    # create a trimesh cylinder
    cylinder = trimesh.creation.cylinder(radius=cfg.radius, height=cfg.height, transform=transform)

    # obtain stage handle
    stage = get_current_stage()
    # spawn the cylinder as a mesh
    _spawn_mesh_geom_from_mesh(prim_path, cfg, cylinder, translation, orientation, stage=stage)
    # return the prim
    return stage.GetPrimAtPath(prim_path)


@clone
def spawn_mesh_capsule(
    prim_path: str,
    cfg: meshes_cfg.MeshCapsuleCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USD-Mesh capsule prim with the given attributes.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    # align axis from "Z" to input by rotating the cylinder
    axis = cfg.axis.upper()
    if axis == "X":
        transform = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    elif axis == "Y":
        transform = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    else:
        transform = None
    # create a trimesh capsule
    capsule = trimesh.creation.capsule(radius=cfg.radius, height=cfg.height, transform=transform)

    # obtain stage handle
    stage = get_current_stage()
    # spawn capsule if it doesn't exist.
    _spawn_mesh_geom_from_mesh(prim_path, cfg, capsule, translation, orientation, stage=stage)
    # return the prim
    return stage.GetPrimAtPath(prim_path)


@clone
def spawn_mesh_cone(
    prim_path: str,
    cfg: meshes_cfg.MeshConeCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USD-Mesh cone prim with the given attributes.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    # align axis from "Z" to input by rotating the cylinder
    axis = cfg.axis.upper()
    if axis == "X":
        transform = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    elif axis == "Y":
        transform = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    else:
        transform = None
    # create a trimesh cone
    cone = trimesh.creation.cone(radius=cfg.radius, height=cfg.height, transform=transform)

    # obtain stage handle
    stage = get_current_stage()
    # spawn cone if it doesn't exist.
    _spawn_mesh_geom_from_mesh(prim_path, cfg, cone, translation, orientation, stage=stage)
    # return the prim
    return stage.GetPrimAtPath(prim_path)


@clone
def spawn_mesh_rectangle(
    prim_path: str,
    cfg: meshes_cfg.MeshRectangleCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USD-Mesh 2D rectangle prim with the given attributes.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    # create a 2D triangle mesh grid
    vertices, faces = _create_triangle_mesh_grid(cfg.resolution)
    vertices[:, 0] *= cfg.size[0]
    vertices[:, 1] *= cfg.size[1]
    grid = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # obtain stage handle
    stage = get_current_stage()
    # spawn the rectangle as a mesh
    _spawn_mesh_geom_from_mesh(prim_path, cfg, grid, translation, orientation, None, stage=stage)
    # return the prim
    return stage.GetPrimAtPath(prim_path)


def _create_triangle_mesh_grid(resolution: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Create a centered triangle grid for :class:`MeshRectangleCfg`."""
    if resolution[0] < 1 or resolution[1] < 1:
        raise ValueError(f"Rectangle mesh resolution must be positive, got {resolution}.")

    num_x, num_y = resolution
    xs = np.linspace(-0.5, 0.5, num_x + 1, dtype=np.float32)
    ys = np.linspace(-0.5, 0.5, num_y + 1, dtype=np.float32)
    vertices = np.array([(x, y, 0.0) for y in ys for x in xs], dtype=np.float32)

    faces = []
    row_stride = num_x + 1
    for iy in range(num_y):
        for ix in range(num_x):
            v0 = iy * row_stride + ix
            v1 = v0 + 1
            v2 = v0 + row_stride
            v3 = v2 + 1
            if (ix % 2 == 0) != (iy % 2 == 0):
                faces.append((v0, v1, v2))
                faces.append((v1, v3, v2))
            else:
                faces.append((v0, v1, v3))
                faces.append((v0, v3, v2))

    return vertices, np.asarray(faces, dtype=np.int64)


"""
Helper functions.
"""


def _spawn_mesh_geom_from_mesh(
    prim_path: str,
    cfg: meshes_cfg.MeshCfg,
    mesh: trimesh.Trimesh,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    scale: tuple[float, float, float] | None = None,
    stage: Usd.Stage | None = None,
    **kwargs,
):
    """Create a `USDGeomMesh`_ prim from the given mesh.

    This function is similar to :func:`shapes._spawn_geom_from_prim_type` but spawns the prim from a given mesh.
    In case of the mesh, it is spawned as a USDGeomMesh prim with the given vertices and faces.

    There is a difference in how the properties are applied to the prim based on the type of object:

    - Deformable body properties: The properties are applied to the parent prim: ``{prim_path}``.
    - Collision properties: The properties are applied to the simulation mesh ``{prim_path}/sim_mesh`` for
      deformable bodies, and to the mesh prim ``{prim_path}/geometry/mesh`` otherwise.
    - Rigid body properties: The properties are applied to the parent prim: ``{prim_path}``.

    Args:
        prim_path: The prim path to spawn the asset at.
        cfg: The config containing the properties to apply.
        mesh: The mesh to spawn the prim from.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        scale: The scale to apply to the prim. Defaults to None, in which case this is set to identity.
        stage: The stage to spawn the asset at. Defaults to None, in which case the current stage is used.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Raises:
        ValueError: If a prim already exists at the given path.
        ValueError: If both deformable and rigid properties are used.
        ValueError: If the physics material is not of the correct type. Deformable properties require a deformable
            physics material, and rigid properties require a rigid physics material.
        ValueError: If deformable properties are used with non-fragment collision properties.

    .. _USDGeomMesh: https://openusd.org/dev/api/class_usd_geom_mesh.html
    """
    # obtain stage handle
    stage = stage if stage is not None else get_current_stage()

    # spawn geometry if it doesn't exist.
    if not stage.GetPrimAtPath(prim_path).IsValid():
        create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation, stage=stage)
    else:
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")
    # check that invalid schema types are not used
    if cfg.deformable_props is not None and cfg.rigid_props is not None:
        raise ValueError("Cannot use both deformable and rigid properties at the same time.")
    if cfg.deformable_props is not None and cfg.collision_props is not None:
        # only fragments resolve onto the simulation mesh, legacy cfgs would target the inert body prim
        frags = cfg.collision_props if isinstance(cfg.collision_props, (list, tuple)) else [cfg.collision_props]
        if not all(isinstance(frag, schemas.SchemaFragment) for frag in frags):
            raise ValueError("Deformable bodies require 'collision_props' as collision fragments.")
    # check material types are correct
    if cfg.deformable_props is not None and cfg.physics_material is not None:
        if not isinstance(cfg.physics_material, DeformableBodyMaterialBaseCfg):
            raise ValueError("Deformable properties require a deformable physics material.")
    if cfg.rigid_props is not None and cfg.physics_material is not None:
        # accept anything spawn_physics_material accepts for the rigid case: a legacy rigid-body
        # material cfg, a single fragment, or a list/tuple of fragments
        physics_material_frags = (
            cfg.physics_material if isinstance(cfg.physics_material, (list, tuple)) else [cfg.physics_material]
        )
        is_rigid_material = isinstance(cfg.physics_material, RigidBodyMaterialBaseCfg) or all(
            isinstance(frag, RigidBodyMaterialFragment) for frag in physics_material_frags
        )
        if not is_rigid_material:
            raise ValueError("Rigid properties require a rigid physics material.")

    # create all the paths we need for clarity
    geom_prim_path = prim_path + "/geometry"
    mesh_prim_path = geom_prim_path + "/mesh"

    # create the mesh prim
    mesh_prim = create_prim(
        mesh_prim_path,
        prim_type="Mesh",
        scale=scale,
        attributes={
            "points": mesh.vertices,
            "faceVertexIndices": mesh.faces.flatten(),
            "faceVertexCounts": np.asarray([3] * len(mesh.faces)),
            "subdivisionScheme": "bilinear",
        },
        stage=stage,
    )

    if cfg.deformable_props is not None:
        # apply deformable body properties
        deformable_type = (
            "surface" if isinstance(cfg.physics_material, SurfaceDeformableBodyMaterialBaseCfg) else "volume"
        )
        schemas.define_deformable_body_properties(
            prim_path, cfg.deformable_props, stage=stage, deformable_type=deformable_type
        )
        # the collider is the simulation mesh, resolved from the body prim
        if cfg.collision_props is not None:
            frags = cfg.collision_props if isinstance(cfg.collision_props, (list, tuple)) else [cfg.collision_props]
            schemas.apply_collision_properties(prim_path, frags, stage=stage)
        if cfg.mass_props is not None:
            raise ValueError(
                """MassPropertiesCfg are not supported for deformable bodies
                and should be set through deformable_props with mass=<value>."""
            )
    elif cfg.collision_props is not None:
        # decide on type of collision approximation based on the mesh
        if cfg.__class__.__name__ == "MeshSphereCfg":
            collision_approximation = "boundingSphere"
        elif cfg.__class__.__name__ == "MeshCuboidCfg":
            collision_approximation = "boundingCube"
        else:
            # for: MeshCylinderCfg, MeshCapsuleCfg, MeshConeCfg
            collision_approximation = "convexHull"
        # apply collision approximation to mesh
        # note: for primitives, we use the convex hull approximation -- this should be sufficient for most cases.
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        mesh_collision_api.GetApproximationAttr().Set(collision_approximation)
        # apply collision properties
        # transition shim, remove later: new fragment list -> apply_*; legacy single cfg -> define_*
        coll_frags = cfg.collision_props if isinstance(cfg.collision_props, (list, tuple)) else [cfg.collision_props]
        if coll_frags and all(isinstance(f, schemas.SchemaFragment) for f in coll_frags):
            schemas.apply_collision_properties(mesh_prim_path, coll_frags, stage=stage)
        else:
            schemas.define_collision_properties(mesh_prim_path, cfg.collision_props, stage=stage)

    # apply visual material
    if cfg.visual_material is not None:
        if not has_kit():
            logger.warning("Skipping visual material application for '%s' in kitless mode.", mesh_prim_path)
        else:
            if not cfg.visual_material_path.startswith("/"):
                material_path = f"{geom_prim_path}/{cfg.visual_material_path}"
            else:
                material_path = cfg.visual_material_path
            # create material
            cfg.visual_material.func(material_path, cfg.visual_material)
            # apply material
            bind_visual_material(mesh_prim_path, material_path, stage=stage)

    # apply physics material
    if cfg.physics_material is not None:
        if not cfg.physics_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.physics_material_path}"
        else:
            material_path = cfg.physics_material_path
        # create material (accepts a legacy material cfg or rigid-body fragment(s))
        spawn_physics_material(material_path, cfg.physics_material, stage=stage)
        # apply material
        bind_physics_material(prim_path, material_path, stage=stage)

    # note: we apply the rigid properties to the parent prim in case of rigid objects.
    if cfg.rigid_props is not None:
        # apply mass properties (transition shim, remove later: fragment list -> apply_*; legacy cfg -> define_*)
        if cfg.mass_props is not None:
            # normalize a single fragment to a list so the convenience form routes like a list
            mass_frags = [cfg.mass_props] if isinstance(cfg.mass_props, schemas.SchemaFragment) else cfg.mass_props
            if isinstance(mass_frags, (list, tuple)) and all(isinstance(f, schemas.SchemaFragment) for f in mass_frags):
                schemas.apply_mass_properties(prim_path, mass_frags, stage=stage)
            else:
                schemas.define_mass_properties(prim_path, cfg.mass_props, stage=stage)
        # apply rigid properties (transition shim, remove later: fragment list -> apply_*; legacy cfg -> define_*)
        rigid_frags = cfg.rigid_props if isinstance(cfg.rigid_props, (list, tuple)) else [cfg.rigid_props]
        if rigid_frags and all(isinstance(f, schemas.SchemaFragment) for f in rigid_frags):
            schemas.apply_rigid_body_properties(prim_path, rigid_frags, stage=stage)
        else:
            schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)
