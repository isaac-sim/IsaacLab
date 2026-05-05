# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from pxr import Usd, UsdPhysics, UsdShade

from isaaclab.sim.utils import clone, safe_set_attribute_on_usd_prim, safe_set_attribute_on_usd_schema
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.string import to_camel_case

if TYPE_CHECKING:
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
    usd_physics_material_api = UsdPhysics.MaterialAPI(prim)
    if not usd_physics_material_api:
        usd_physics_material_api = UsdPhysics.MaterialAPI.Apply(prim)

    # read class metadata for the namespaced (solver-specific) write phase
    namespace = getattr(cfg, "_usd_namespace", None)
    applied_schema = getattr(cfg, "_usd_applied_schema", None)
    attr_name_map = getattr(cfg, "_usd_attr_name_map", {}) or {}

    # build cfg dict, dropping underscore-prefixed metadata keys and the spawner ``func`` field
    cfg_dict = {k: v for k, v in cfg.to_dict().items() if not k.startswith("_") and k != "func"}

    # write standard UsdPhysics.MaterialAPI fields (friction + restitution)
    for attr_name in ("static_friction", "dynamic_friction", "restitution"):
        value = cfg_dict.pop(attr_name, None)
        safe_set_attribute_on_usd_schema(usd_physics_material_api, attr_name, value, camel_case=True)

    # collect instance-level namespaced writes; exclude None values
    namespaced_writes: list[tuple[str, object]] = []
    # 2a. fields with explicit USD-attribute-name overrides
    for cfg_field in list(attr_name_map):
        value = cfg_dict.pop(cfg_field, None)
        if value is not None:
            namespaced_writes.append((attr_name_map[cfg_field], value))
    # 2b. remaining fields use snake -> camelCase auto-conversion
    for cfg_field, value in list(cfg_dict.items()):
        if value is not None:
            namespaced_writes.append((to_camel_case(cfg_field, "cC"), value))

    # gate schema application AND attribute authoring on the instance-level set
    if namespaced_writes:
        if namespace is None:
            raise ValueError(
                f"{type(cfg).__name__} has solver-specific fields"
                f" {[k for k, _ in namespaced_writes]} but does not define '_usd_namespace'."
            )
        if applied_schema and applied_schema not in prim.GetAppliedSchemas():
            prim.AddAppliedSchema(applied_schema)
        for usd_attr, value in namespaced_writes:
            safe_set_attribute_on_usd_prim(prim, f"{namespace}:{usd_attr}", value, camel_case=False)

    # return the prim
    return prim
