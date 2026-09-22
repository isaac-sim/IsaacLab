# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh
import trimesh.transformations

from pxr import Usd, UsdPhysics

from ... import schemas
from ...utils import bind_physics_material, bind_visual_material, clone, create_prim, get_current_stage
from .._utils import apply_schema_props, fragment_mapping, props_expr, resolve_material_path
from ..materials import (
    DeformableBodyMaterialBaseCfg,
    RigidBodyMaterialBaseCfg,
    RigidBodyMaterialFragment,
    SurfaceDeformableBodyMaterialBaseCfg,
)
from ..materials.physics_materials import spawn_physics_material

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
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    sphere = trimesh.creation.uv_sphere(radius=cfg.radius)
    return _spawn_mesh_geom_from_mesh(prim_path, cfg, sphere, translation, orientation)


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
    box = trimesh.creation.box(cfg.size)
    return _spawn_mesh_geom_from_mesh(prim_path, cfg, box, translation, orientation)


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
    cylinder = trimesh.creation.cylinder(radius=cfg.radius, height=cfg.height, transform=_axis_transform(cfg.axis))
    return _spawn_mesh_geom_from_mesh(prim_path, cfg, cylinder, translation, orientation)


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
    capsule = trimesh.creation.capsule(radius=cfg.radius, height=cfg.height, transform=_axis_transform(cfg.axis))
    return _spawn_mesh_geom_from_mesh(prim_path, cfg, capsule, translation, orientation)


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
    cone = trimesh.creation.cone(radius=cfg.radius, height=cfg.height, transform=_axis_transform(cfg.axis))
    return _spawn_mesh_geom_from_mesh(prim_path, cfg, cone, translation, orientation)


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
    # create a 2D triangle mesh
    half_x, half_y = cfg.size[0] / 2, cfg.size[1] / 2
    vertices = np.array(
        [(-half_x, -half_y, 0.0), (half_x, -half_y, 0.0), (half_x, half_y, 0.0), (-half_x, half_y, 0.0)],
        dtype=np.float32,
    )
    rectangle = trimesh.Trimesh(vertices=vertices, faces=((0, 1, 2), (0, 2, 3)), process=False)
    return _spawn_mesh_geom_from_mesh(prim_path, cfg, rectangle, translation, orientation)


"""
Helper functions.
"""


def _axis_transform(axis: str) -> np.ndarray | None:
    """Return the rotation that aligns a Z-axis-aligned trimesh primitive with the requested axis."""
    axis = axis.upper()
    if axis == "X":
        return trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    if axis == "Y":
        return trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    return None


def _refine_surface_mesh(mesh: trimesh.Trimesh, cfg: meshes_cfg.MeshCfg) -> trimesh.Trimesh:
    """Subdivide a deformable's surface mesh to the configured edge-length target.

    Args:
        mesh: The mesh to refine.
        cfg: The config carrying :attr:`~isaaclab.sim.MeshCfg.edge_refinement`.

    Returns:
        The refined mesh, or the input mesh when refinement does not apply.

    Raises:
        ValueError: If the edge refinement is less than ``1.0``.
    """
    if cfg.edge_refinement < 1.0:
        raise ValueError(f"Mesh edge refinement must be at least 1.0, got {cfg.edge_refinement}.")
    if cfg.deformable_props is None or cfg.edge_refinement == 1.0:
        return mesh

    max_edge = float(np.linalg.norm(mesh.bounding_box.extents)) / cfg.edge_refinement
    vertices, faces = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=max_edge)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _apply_deformable_collision_props(prim_path: str, collision_props, stage: Usd.Stage) -> None:
    """Apply collision fragments to the simulation mesh of a deformable body.

    The collider is the simulation mesh authored under the body prim, so mapping keys anchor there
    (e.g. ``{"/sim_mesh": [...]}``) while a bare fragment list sweeps the subtree to reach it.

    Args:
        prim_path: The prim path of the deformable body.
        collision_props: A mapping from target pattern to collision fragments, or a fragment or
            sequence of fragments.
        stage: The stage where the prims live.
    """
    if isinstance(collision_props, dict):
        for pattern, fragments in collision_props.items():
            schemas.apply_collision_properties(props_expr(prim_path, pattern), fragments, stage=stage)
        return
    fragments = collision_props if isinstance(collision_props, (list, tuple)) else [collision_props]
    schemas.apply_collision_properties(props_expr(prim_path, "/.*"), fragments, stage=stage)


def _spawn_mesh_geom_from_mesh(
    prim_path: str,
    cfg: meshes_cfg.MeshCfg,
    mesh: trimesh.Trimesh,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    scale: tuple[float, float, float] | None = None,
    stage: Usd.Stage | None = None,
    **kwargs,
) -> Usd.Prim:
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

    Returns:
        The created root prim.

    Raises:
        ValueError: If a prim already exists at the given path.
        ValueError: If edge refinement is less than ``1.0``.
        ValueError: If both deformable and rigid properties are used.
        ValueError: If the physics material is not of the correct type. Deformable properties require a deformable
            physics material, and rigid properties require a rigid physics material.
        ValueError: If deformable properties are used with non-fragment collision properties.

    .. _USDGeomMesh: https://openusd.org/dev/api/class_usd_geom_mesh.html
    """
    mesh = _refine_surface_mesh(mesh, cfg)
    stage = stage if stage is not None else get_current_stage()

    prim = create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation, stage=stage)
    # check that invalid schema types are not used
    if cfg.deformable_props is not None and cfg.rigid_props is not None:
        raise ValueError("Cannot use both deformable and rigid properties at the same time.")
    if cfg.deformable_props is not None and cfg.collision_props is not None:
        # only fragments resolve onto the simulation mesh, legacy cfgs would target the inert body prim
        collision_props_mapping = fragment_mapping(cfg.collision_props)
        if collision_props_mapping is not None:
            frags = [frag for fragments in collision_props_mapping.values() for frag in fragments]
        else:
            frags = [cfg.collision_props]
        if not frags or not all(isinstance(frag, schemas.SchemaFragment) for frag in frags):
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

    geom_prim_path = prim_path + "/geometry"
    mesh_prim_path = geom_prim_path + "/mesh"
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
        deformable_kwargs = {}
        if deformable_type == "volume":
            deformable_kwargs["tetrahedralization_edge_length_fac"] = 1.0 / cfg.edge_refinement
        schemas.define_deformable_body_properties(
            prim_path,
            cfg.deformable_props,
            stage=stage,
            deformable_type=deformable_type,
            **deformable_kwargs,
        )
        if cfg.collision_props is not None:
            _apply_deformable_collision_props(prim_path, cfg.collision_props, stage)
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
        UsdPhysics.MeshCollisionAPI.Apply(mesh_prim).GetApproximationAttr().Set(collision_approximation)
        # collision properties anchor at the geometry prim
        apply_schema_props(
            cfg.collision_props,
            mesh_prim_path,
            schemas.apply_collision_properties,
            schemas.define_collision_properties,
            stage,
        )

    if cfg.visual_material is not None:
        material_path = resolve_material_path(cfg.visual_material_path, geom_prim_path)
        cfg.visual_material.func(material_path, cfg.visual_material)
        bind_visual_material(mesh_prim_path, material_path, stage=stage)
    if cfg.physics_material is not None:
        material_path = resolve_material_path(cfg.physics_material_path, geom_prim_path)
        spawn_physics_material(material_path, cfg.physics_material, stage=stage)
        bind_physics_material(prim_path, material_path, stage=stage)

    # mass and rigid body properties anchor at the container prim
    if cfg.rigid_props is not None:
        if cfg.mass_props is not None:
            apply_schema_props(
                cfg.mass_props, prim_path, schemas.apply_mass_properties, schemas.define_mass_properties, stage
            )
        apply_schema_props(
            cfg.rigid_props, prim_path, schemas.apply_rigid_body_properties, schemas.define_rigid_body_properties, stage
        )
    return prim
