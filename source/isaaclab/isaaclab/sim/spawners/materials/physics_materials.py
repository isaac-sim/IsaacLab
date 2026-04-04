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
def spawn_rigid_body_material(prim_path: str, cfg: physics_materials_cfg.RigidBodyMaterialCfg) -> Usd.Prim:
    """Create material with rigid-body physics properties.

    Rigid body materials are used to define the physical properties to meshes of a rigid body. These
    include the friction, restitution, and their respective combination modes. For more information on
    rigid body material, please refer to the `documentation on PxMaterial <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxBaseMaterial.html>`_.

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
        ValueError:  When a prim already exists at the specified prim path and is not a material.
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
    # retrieve the USD rigid-body api
    usd_physics_material_api = UsdPhysics.MaterialAPI(prim)
    if not usd_physics_material_api:
        usd_physics_material_api = UsdPhysics.MaterialAPI.Apply(prim)
    # ensure PhysX material API is applied
    applied = prim.GetAppliedSchemas()
    if "PhysxMaterialAPI" not in applied:
        prim.AddAppliedSchema("PhysxMaterialAPI")

    # convert to dict
    cfg = cfg.to_dict()
    del cfg["func"]
    # set into USD API
    for attr_name in ["static_friction", "dynamic_friction", "restitution"]:
        value = cfg.pop(attr_name, None)
        safe_set_attribute_on_usd_schema(usd_physics_material_api, attr_name, value, camel_case=True)
    # set into PhysX API (prim attributes: physxMaterial:*)
    for attr_name, value in cfg.items():
        safe_set_attribute_on_usd_prim(prim, f"physxMaterial:{to_camel_case(attr_name, 'cC')}", value, camel_case=False)
    # return the prim
    return prim


@clone
def spawn_deformable_body_material(prim_path: str, cfg: physics_materials_cfg.DeformableBodyMaterialCfg) -> Usd.Prim:
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
        ValueError:  When a prim already exists at the specified prim path and is not a material.

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
    # ensure PhysX deformable body material API is applied
    applied = prim.GetAppliedSchemas()
    if "PhysxDeformableBodyMaterialAPI" not in applied:
        prim.AddAppliedSchema("PhysxDeformableBodyMaterialAPI")

    # convert to dict
    cfg = cfg.to_dict()
    del cfg["func"]
    # set into PhysX API (prim attributes: physxDeformableBodyMaterial:*)
    for attr_name, value in cfg.items():
        safe_set_attribute_on_usd_prim(
            prim, f"physxDeformableBodyMaterial:{to_camel_case(attr_name, 'cC')}", value, camel_case=False
        )
    # return the prim
    return prim
