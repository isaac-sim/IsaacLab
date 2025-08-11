# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import isaacsim.core.utils.prims as prim_utils
from pxr import Usd

from isaaclab.sim import schemas
from isaaclab.sim.utils import bind_physics_material, bind_visual_material, clone

if TYPE_CHECKING:
    from . import shapes_cfg


@clone
def spawn_sphere(
    prim_path: str,
    cfg: shapes_cfg.SphereCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USDGeom-based sphere prim with the given attributes.

    For more information, see `USDGeomSphere <https://openusd.org/dev/api/class_usd_geom_sphere.html>`_.

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
    # spawn sphere if it doesn't exist.
    attributes = {"radius": cfg.radius}
    _spawn_geom_from_prim_type(prim_path, cfg, "Sphere", attributes, translation, orientation)
    # return the prim
    return prim_utils.get_prim_at_path(prim_path)


@clone
def spawn_cuboid(
    prim_path: str,
    cfg: shapes_cfg.CuboidCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USDGeom-based cuboid prim with the given attributes.

    For more information, see `USDGeomCube <https://openusd.org/dev/api/class_usd_geom_cube.html>`_.

    Note:
        Since USD only supports cubes, we set the size of the cube to the minimum of the given size and
        scale the cube accordingly.

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
        If a prim already exists at the given path.
    """
    # resolve the scale
    size = min(cfg.size)
    scale = [dim / size for dim in cfg.size]
    # spawn cuboid if it doesn't exist.
    attributes = {"size": size}
    _spawn_geom_from_prim_type(prim_path, cfg, "Cube", attributes, translation, orientation, scale)
    # return the prim
    return prim_utils.get_prim_at_path(prim_path)


@clone
def spawn_cylinder(
    prim_path: str,
    cfg: shapes_cfg.CylinderCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USDGeom-based cylinder prim with the given attributes.

    For more information, see `USDGeomCylinder <https://openusd.org/dev/api/class_usd_geom_cylinder.html>`_.

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
    # spawn cylinder if it doesn't exist.
    attributes = {"radius": cfg.radius, "height": cfg.height, "axis": cfg.axis.upper()}
    _spawn_geom_from_prim_type(prim_path, cfg, "Cylinder", attributes, translation, orientation)
    # return the prim
    return prim_utils.get_prim_at_path(prim_path)


@clone
def spawn_capsule(
    prim_path: str,
    cfg: shapes_cfg.CapsuleCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USDGeom-based capsule prim with the given attributes.

    For more information, see `USDGeomCapsule <https://openusd.org/dev/api/class_usd_geom_capsule.html>`_.

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
    # spawn capsule if it doesn't exist.
    attributes = {"radius": cfg.radius, "height": cfg.height, "axis": cfg.axis.upper()}
    _spawn_geom_from_prim_type(prim_path, cfg, "Capsule", attributes, translation, orientation)
    # return the prim
    return prim_utils.get_prim_at_path(prim_path)


@clone
def spawn_cone(
    prim_path: str,
    cfg: shapes_cfg.ConeCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USDGeom-based cone prim with the given attributes.

    For more information, see `USDGeomCone <https://openusd.org/dev/api/class_usd_geom_cone.html>`_.

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
    # spawn cone if it doesn't exist.
    attributes = {"radius": cfg.radius, "height": cfg.height, "axis": cfg.axis.upper()}
    _spawn_geom_from_prim_type(prim_path, cfg, "Cone", attributes, translation, orientation)
    # return the prim
    return prim_utils.get_prim_at_path(prim_path)


"""
Helper functions.
"""


def _spawn_geom_from_prim_type(
    prim_path: str,
    cfg: shapes_cfg.ShapeCfg,
    prim_type: str,
    attributes: dict,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    scale: tuple[float, float, float] | None = None,
):
    """Create a USDGeom-based prim with the given attributes.

    To make the asset instanceable, we must follow a certain structure dictated by how USD scene-graph
    instancing and physics work. The rigid body component must be added to each instance and not the
    referenced asset (i.e. the prototype prim itself). This is because the rigid body component defines
    properties that are specific to each instance and cannot be shared under the referenced asset. For
    more information, please check the `documentation <https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/rigid-bodies.html#instancing-rigid-bodies>`_.

    Due to the above, we follow the following structure:

    * ``{prim_path}`` - The root prim that is an Xform with the rigid body and mass APIs if configured.
    * ``{prim_path}/geometry`` - The prim that contains the mesh and optionally the materials if configured.
      If instancing is enabled, this prim will be an instanceable reference to the prototype prim.

    Args:
        prim_path: The prim path to spawn the asset at.
        cfg: The config containing the properties to apply.
        prim_type: The type of prim to create.
        attributes: The attributes to apply to the prim.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        scale: The scale to apply to the prim. Defaults to None, in which case this is set to identity.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    # spawn geometry if it doesn't exist.
    if not prim_utils.is_prim_path_valid(prim_path):
        prim_utils.create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation)
    else:
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    # create all the paths we need for clarity
    geom_prim_path = prim_path + "/geometry"
    mesh_prim_path = geom_prim_path + "/mesh"

    # create the geometry prim
    prim_utils.create_prim(mesh_prim_path, prim_type, scale=scale, attributes=attributes)
    # apply collision properties
    if cfg.collision_props is not None:
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

    # note: we apply rigid properties in the end to later make the instanceable prim
    # apply mass properties
    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props)
    # apply rigid body properties
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props)
