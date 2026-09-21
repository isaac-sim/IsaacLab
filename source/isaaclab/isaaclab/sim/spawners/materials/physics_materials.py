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
from isaaclab.utils.string import string_to_callable

from . import physics_materials_cfg


def spawn_rigid_body_material_from_fragments(
    prim_path: str,
    fragments: physics_materials_cfg.RigidBodyMaterialFragment
    | list[physics_materials_cfg.RigidBodyMaterialFragment]
    | tuple[physics_materials_cfg.RigidBodyMaterialFragment, ...],
    stage: Usd.Stage | None = None,
) -> Usd.Prim:
    """Spawn a rigid-body physics material from one or more single-namespace fragments.

    Creates (or reuses) the ``UsdShade.Material`` prim at ``prim_path``, applies the standard
    ``UsdPhysics.MaterialAPI`` anchor, then dispatches each fragment via its
    :attr:`~isaaclab.sim.schemas.SchemaFragment.func` to author its namespace onto the material prim.
    Backend fragments carry backend-specific namespaces (e.g. PhysX ``physxMaterial:*``) without core
    importing a backend.

    .. note::
        Unlike the ``@clone``-decorated :func:`spawn_rigid_body_material`, this writer expects a
        concrete ``prim_path`` and does not resolve regex prim-path patterns. Physics materials are
        spawned at a single derived path by :func:`spawn_physics_material` and the spawner internals,
        so regex resolution does not apply here.

    Args:
        prim_path: The prim path to spawn the material at.
        fragments: A single :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialFragment`,
            or a list/tuple of them.
        stage: The stage to spawn on. Defaults to None, in which case the current stage is used.

    Returns:
        The spawned rigid body material prim.

    Raises:
        ValueError: When ``fragments`` is empty.
        ValueError: When a prim already exists at the path and is not a material.
        TypeError: When an item is not a rigid-body material fragment.
    """
    if not isinstance(fragments, (list, tuple)):
        fragments = [fragments]
    else:
        fragments = list(fragments)
    if not fragments:
        raise ValueError(f"Cannot spawn a physics material at '{prim_path}' from an empty fragment collection.")
    invalid_types = [
        type(fragment).__name__
        for fragment in fragments
        if not isinstance(fragment, physics_materials_cfg.RigidBodyMaterialFragment)
    ]
    if invalid_types:
        raise TypeError(
            "A physics-material fragment collection must contain only RigidBodyMaterialFragment instances; got"
            f" {invalid_types}."
        )
    if stage is None:
        stage = get_current_stage()

    # create the material prim if none exists yet
    if not stage.GetPrimAtPath(prim_path).IsValid():
        UsdShade.Material.Define(stage, prim_path)
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsA(UsdShade.Material):
        raise ValueError(f"A prim already exists at path: '{prim_path}' but is not a material.")

    # apply the standard UsdPhysics MaterialAPI anchor (the defining schema for a physics material)
    if not UsdPhysics.MaterialAPI(prim):
        UsdPhysics.MaterialAPI.Apply(prim)

    # dispatch each fragment's applier (writes its single namespace onto the material prim)
    for cfg in fragments:
        func = cfg.func if callable(cfg.func) else string_to_callable(cfg.func)
        func(cfg, prim_path, stage)
    return prim


def spawn_physics_material(
    prim_path: str,
    material: physics_materials_cfg.PhysicsMaterialCfg
    | physics_materials_cfg.RigidBodyMaterialFragment
    | list[physics_materials_cfg.RigidBodyMaterialFragment]
    | tuple[physics_materials_cfg.RigidBodyMaterialFragment, ...],
    stage: Usd.Stage | None = None,
) -> Usd.Prim:
    """Spawn a physics material from a spawner ``physics_material`` slot value.

    Dispatches the two accepted slot forms: a single rigid-body fragment or list/tuple is spawned via
    :func:`spawn_rigid_body_material_from_fragments`; otherwise the value is a legacy material cfg and is
    spawned via its own :attr:`func`. Lets spawners accept both the fragment and the legacy interface
    from one call site.

    Args:
        prim_path: The prim path to spawn the material at.
        material: A :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialFragment` (or list/tuple
            of them), or a legacy material cfg carrying a :attr:`func`.
        stage: The stage to spawn on. Defaults to None, in which case the current stage is used. A
            legacy material cfg only supports the current stage (or None); passing a different
            stage raises.
    Returns:
        The spawned material prim.

    Raises:
        ValueError: When ``material`` is an empty fragment collection.
        ValueError: When ``material`` is a legacy material cfg and ``stage`` is neither ``None``
            nor the current stage.
        TypeError: When ``material`` is a fragment collection containing anything other than
            :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialFragment` instances.
        TypeError: When ``material`` is neither a physics-material cfg nor a rigid-body fragment.
    """
    # The family writer owns fragment validation so direct and dispatched calls have one contract.
    if isinstance(material, (list, tuple, physics_materials_cfg.RigidBodyMaterialFragment)):
        return spawn_rigid_body_material_from_fragments(prim_path, material, stage)
    if not isinstance(material, physics_materials_cfg.PhysicsMaterialCfg):
        raise TypeError(
            "A physics material must be a PhysicsMaterialCfg or rigid-body material fragment; got"
            f" '{type(material).__name__}'."
        )
    # legacy single-cfg path (rigid or deformable material cfg with its own spawner ``func``).
    # Legacy material funcs take only ``(prim_path, cfg)`` and resolve the stage internally via
    # ``get_current_stage()`` (their path matching is also current-stage-bound), so an explicit
    # different stage cannot be honored on this path -- reject it loudly rather than authoring on
    # the wrong stage. The fragment path above supports explicit stages.
    if stage is not None and stage != get_current_stage():
        raise ValueError(
            f"Legacy material cfg '{type(material).__name__}' can only be spawned on the current"
            " stage. Pass the current stage (or None), or use the fragment-based API for"
            " explicit-stage authoring."
        )
    return material.func(prim_path, material)


@clone
def spawn_rigid_body_material(prim_path: str, cfg: physics_materials_cfg.RigidBodyMaterialBaseCfg) -> Usd.Prim:
    """Create material with rigid-body physics properties.

    Rigid body materials are used to define the physical properties to meshes of a rigid body. These
    include the friction, restitution, and (PhysX-only) compliant-contact spring and combine-mode
    tokens. For more information on rigid body material, please refer to the `documentation on
    PxMaterial <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxBaseMaterial.html>`_.

    The writer is metadata-driven: it always applies the standard ``UsdPhysics.MaterialAPI`` and
    writes the friction/restitution/density fields, then reads ``_usd_applied_schema``,
    ``_usd_namespace``, and ``_usd_attr_name_map`` from the cfg to author solver-specific attributes.
    The applied schema (e.g. ``PhysxMaterialAPI``) is added only when at least one solver-specific
    field has a non-``None`` value at the instance level.

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
    prim_path: str,
    cfg: physics_materials_cfg.CableMaterialCfg | physics_materials_cfg.DeformableBodyMaterialBaseCfg,
) -> Usd.Prim:
    """Create material with deformable physics properties.

    Deformable materials define the physical properties of curves, surfaces, and volume meshes.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration for the physics material.

    Returns:
        The spawned deformable material prim.

    Raises:
        ValueError: When a prim already exists at the specified prim path and is not a material.

    """
    if isinstance(cfg, physics_materials_cfg.CableMaterialCfg):
        cfg.validate()

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
