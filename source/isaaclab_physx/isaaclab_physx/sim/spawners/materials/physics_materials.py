# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pxr import Usd, UsdShade

from isaaclab.sim.utils import clone, safe_set_attribute_on_usd_prim
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.string import to_camel_case

from . import physics_materials_cfg


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
    if "OmniPhysicsDeformableMaterialAPI" not in applied:
        prim.AddAppliedSchema("OmniPhysicsDeformableMaterialAPI")
    if "PhysxDeformableMaterialAPI" not in applied:
        prim.AddAppliedSchema("PhysxDeformableMaterialAPI")
    # surface deformable material API
    is_surface_deformable = isinstance(cfg, physics_materials_cfg.SurfaceDeformableBodyMaterialCfg)
    if is_surface_deformable:
        if "OmniPhysicsSurfaceDeformableMaterialAPI" not in applied:
            prim.AddAppliedSchema("OmniPhysicsSurfaceDeformableMaterialAPI")
        if "PhysxSurfaceDeformableMaterialAPI" not in applied:
            prim.AddAppliedSchema("PhysxSurfaceDeformableMaterialAPI")

    # convert to dict
    cfg = cfg.to_dict()
    del cfg["func"]
    # set into PhysX API, gather prefixes for each attribute
    property_prefixes = cfg["_property_prefix"]
    for prefix, attr_list in property_prefixes.items():
        for attr_name in attr_list:
            safe_set_attribute_on_usd_prim(
                prim, f"{prefix}:{to_camel_case(attr_name, 'cC')}", cfg[attr_name], camel_case=False
            )
    # return the prim
    return prim
