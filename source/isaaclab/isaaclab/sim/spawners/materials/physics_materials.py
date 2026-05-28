# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses

from pxr import Usd, UsdPhysics, UsdShade

from isaaclab.sim.schemas.schemas import _apply_namespaced_schemas
from isaaclab.sim.utils import clone
from isaaclab.sim.utils.stage import get_current_stage

from . import physics_materials_cfg


@clone
def spawn_rigid_body_material(prim_path: str, cfg: physics_materials_cfg.RigidBodyMaterialBaseCfg) -> Usd.Prim:
    """Create material with rigid-body physics properties.

    Rigid body materials are used to define the physical properties to meshes of a rigid body. These
    include the friction, restitution, and (PhysX-only) compliant-contact spring and combine-mode
    tokens. For more information on rigid body material, please refer to the `documentation on
    PxMaterial <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxBaseMaterial.html>`_.

    The writer is metadata-driven: it always applies the standard ``UsdPhysics.MaterialAPI`` and
    writes the friction/restitution fields, then reads ``_usd_applied_schema``, ``_usd_namespace``,
    and ``_usd_attr_name_map`` from the cfg to author solver-specific attributes. The applied
    schema (e.g. ``PhysxMaterialAPI``) is added only when at least one solver-specific field has a
    non-``None`` value at the instance level.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration for the physics material.

    Returns:
        The spawned rigid body material prim.

    Raises:
        ValueError: When a prim already exists at the specified prim path and is not a material.
        ValueError: When the cfg defines solver-specific fields but does not define ``_usd_namespace``.
    """
    # get stage handle
    stage = get_current_stage()

    # create material prim if no prim exists
    if not stage.GetPrimAtPath(prim_path).IsValid():
        _ = UsdShade.Material.Define(stage, prim_path)

    # obtain prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim is a material
    if not prim.IsA(UsdShade.Material):
        raise ValueError(f"A prim already exists at path: '{prim_path}' but is not a material.")

    # apply the standard UsdPhysics MaterialAPI (always)
    if not UsdPhysics.MaterialAPI(prim):
        UsdPhysics.MaterialAPI.Apply(prim)

    # build cfg dict, dropping underscore-prefixed metadata keys and the spawner ``func`` field
    cfg_dict = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg) if f.name != "func"}

    # All fields routed by the helper: base friction/restitution under ``physics:*``,
    # PhysX-subclass fields (compliant-contact, combine modes) under ``physxMaterial:*``.
    _apply_namespaced_schemas(prim, cfg, cfg_dict)

    # return the prim
    return prim


@clone
def spawn_deformable_body_material(
    prim_path: str, cfg: physics_materials_cfg.DeformableBodyMaterialBaseCfg
) -> Usd.Prim:
    """Create material with deformable-body physics properties.

    Deformable body materials are used to define the physical properties to meshes of a deformable body. These
    include the friction and deformable body properties. For more information on deformable body material,
    please refer to the documentation on `PxFEMSoftBodyMaterial`_.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration for the physics material.

    Returns:
        The spawned deformable body material prim.

    Raises:
        ValueError: When a prim already exists at the specified prim path and is not a material.

    .. _PxFEMSoftBodyMaterial: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/structPxFEMSoftBodyMaterialModel.html
    """
    # get stage handle
    stage = get_current_stage()

    # create material prim if no prim exists
    if not stage.GetPrimAtPath(prim_path).IsValid():
        _ = UsdShade.Material.Define(stage, prim_path)

    # obtain prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim is a material
    if not prim.IsA(UsdShade.Material):
        raise ValueError(f"A prim already exists at path: '{prim_path}' but is not a material.")
    cfg_dict = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg) if f.name != "func"}
    _apply_namespaced_schemas(prim, cfg, cfg_dict)
    # return the prim
    return prim
