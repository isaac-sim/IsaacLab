# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import trimesh
import trimesh.transformations
from typing import TYPE_CHECKING

import isaacsim.core.utils.prims as prim_utils
from pxr import Usd, UsdPhysics

from isaaclab.sim import schemas
from isaaclab.sim.utils import bind_physics_material, bind_visual_material, clone

from ..materials import DeformableBodyMaterialCfg, RigidBodyMaterialCfg

if TYPE_CHECKING:
    from . import meshes_cfg


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
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    # create a trimesh sphere
    sphere = trimesh.creation.uv_sphere(radius=cfg.radius)
    # spawn the sphere as a mesh
    _spawn_mesh_geom_from_mesh(prim_path, cfg, sphere, translation, orientation)
    # return the prim
    return prim_utils.get_prim_at_path(prim_path)


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
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """  # create a trimesh box
    box = trimesh.creation.box(cfg.size)
    # spawn the cuboid as a mesh
    _spawn_mesh_geom_from_mesh(prim_path, cfg, box, translation, orientation, None)
    # return the prim
    return prim_utils.get_prim_at_path(prim_path)


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
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
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
    # spawn the cylinder as a mesh
    _spawn_mesh_geom_from_mesh(prim_path, cfg, cylinder, translation, orientation)
    # return the prim
    return prim_utils.get_prim_at_path(prim_path)


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
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
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
    # spawn capsule if it doesn't exist.
    _spawn_mesh_geom_from_mesh(prim_path, cfg, capsule, translation, orientation)
    # return the prim
    return prim_utils.get_prim_at_path(prim_path)


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
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
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
    # spawn cone if it doesn't exist.
    _spawn_mesh_geom_from_mesh(prim_path, cfg, cone, translation, orientation)
    # return the prim
    return prim_utils.get_prim_at_path(prim_path)


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
    **kwargs,
):
    """Create a `USDGeomMesh`_ prim from the given mesh.

    This function is similar to :func:`shapes._spawn_geom_from_prim_type` but spawns the prim from a given mesh.
    In case of the mesh, it is spawned as a USDGeomMesh prim with the given vertices and faces.

    There is a difference in how the properties are applied to the prim based on the type of object:

    - Deformable body properties: The properties are applied to the mesh prim: ``{prim_path}/geometry/mesh``.
    - Collision properties: The properties are applied to the mesh prim: ``{prim_path}/geometry/mesh``.
    - Rigid body properties: The properties are applied to the parent prim: ``{prim_path}``.

    Args:
        prim_path: The prim path to spawn the asset at.
        cfg: The config containing the properties to apply.
        mesh: The mesh to spawn the prim from.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        scale: The scale to apply to the prim. Defaults to None, in which case this is set to identity.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Raises:
        ValueError: If a prim already exists at the given path.
        ValueError: If both deformable and rigid properties are used.
        ValueError: If both deformable and collision properties are used.
        ValueError: If the physics material is not of the correct type. Deformable properties require a deformable
            physics material, and rigid properties require a rigid physics material.

    .. _USDGeomMesh: https://openusd.org/dev/api/class_usd_geom_mesh.html
    """
    # spawn geometry if it doesn't exist.
    if not prim_utils.is_prim_path_valid(prim_path):
        prim_utils.create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation)
    else:
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    # check that invalid schema types are not used
    if cfg.deformable_props is not None and cfg.rigid_props is not None:
        raise ValueError("Cannot use both deformable and rigid properties at the same time.")
    if cfg.deformable_props is not None and cfg.collision_props is not None:
        raise ValueError("Cannot use both deformable and collision properties at the same time.")
    # check material types are correct
    if cfg.deformable_props is not None and cfg.physics_material is not None:
        if not isinstance(cfg.physics_material, DeformableBodyMaterialCfg):
            raise ValueError("Deformable properties require a deformable physics material.")
    if cfg.rigid_props is not None and cfg.physics_material is not None:
        if not isinstance(cfg.physics_material, RigidBodyMaterialCfg):
            raise ValueError("Rigid properties require a rigid physics material.")

    # create all the paths we need for clarity
    geom_prim_path = prim_path + "/geometry"
    mesh_prim_path = geom_prim_path + "/mesh"

    # create the mesh prim
    mesh_prim = prim_utils.create_prim(
        mesh_prim_path,
        prim_type="Mesh",
        scale=scale,
        attributes={
            "points": mesh.vertices,
            "faceVertexIndices": mesh.faces.flatten(),
            "faceVertexCounts": np.asarray([3] * len(mesh.faces)),
            "subdivisionScheme": "bilinear",
        },
    )

    # note: in case of deformable objects, we need to apply the deformable properties to the mesh prim.
    #   this is different from rigid objects where we apply the properties to the parent prim.
    if cfg.deformable_props is not None:
        # apply mass properties
        if cfg.mass_props is not None:
            schemas.define_mass_properties(mesh_prim_path, cfg.mass_props)
        # apply deformable body properties
        schemas.define_deformable_body_properties(mesh_prim_path, cfg.deformable_props)
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
        schemas.define_collision_properties(mesh_prim_path, cfg.collision_props)

    # apply visual material
    if cfg.visual_material is not None:
        if not cfg.visual_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.visual_material_path}"
        else:
            material_path = cfg.visual_material_path
        # create material
        cfg.visual_material.func(material_path, cfg.visual_material)
        # apply material
        bind_visual_material(mesh_prim_path, material_path)

    # apply physics material
    if cfg.physics_material is not None:
        if not cfg.physics_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.physics_material_path}"
        else:
            material_path = cfg.physics_material_path
        # create material
        cfg.physics_material.func(material_path, cfg.physics_material)
        # apply material
        bind_physics_material(mesh_prim_path, material_path)

    # note: we apply the rigid properties to the parent prim in case of rigid objects.
    if cfg.rigid_props is not None:
        # apply mass properties
        if cfg.mass_props is not None:
            schemas.define_mass_properties(prim_path, cfg.mass_props)
        # apply rigid properties
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props)
