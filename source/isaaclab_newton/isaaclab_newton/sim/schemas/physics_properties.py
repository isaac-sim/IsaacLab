# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Native Newton contact declarations shared by export and terrain reconstruction."""

from newton.usd import PrimType, SchemaResolverNewton

from pxr import UsdPhysics, UsdShade

from isaaclab.assets.physics_properties import UsdAttribute
from isaaclab.sim.usd_export_properties import UsdMaterialWriter

# Reuse resolver names; this module adds only the model/builder correspondence.
COLLISION_FIELDS = {
    "shape_" + key: (
        key,
        UsdAttribute(SchemaResolverNewton.mapping[PrimType.SHAPE][key].name, "NewtonCollisionAPI", type_name="float"),
    )
    for key in ("margin", "gap")
}
MATERIAL_FIELDS = {
    "shape_material_mu": ("mu", UsdAttribute("physics:dynamicFriction", "PhysicsMaterialAPI")),
    "shape_material_restitution": ("restitution", UsdAttribute("physics:restitution", "PhysicsMaterialAPI")),
    **{
        "shape_material_" + key: (key, UsdAttribute(value.name, "NewtonMaterialAPI", type_name="float"))
        for key, value in SchemaResolverNewton.mapping[PrimType.MATERIAL].items()
    },
}
# No scene resolver maps these global soft-contact constants; they affect particle contacts.
SOFT_CONTACT_FIELDS = ("soft_contact_ke", "soft_contact_kd", "soft_contact_kf", "soft_contact_mu")


def author_contacts(writer, prim, index: int, values: dict) -> None:
    """Author native collision/material schemas, preserving equivalent source bindings."""
    for source, (_, target) in COLLISION_FIELDS.items():
        writer.write_attribute(str(prim.GetPath()), target, float(values[source][index]))
    original, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
    if original and all(
        original.GetPrim().GetAttribute(target.attribute).Get() == float(values[source][index])
        for source, (_, target) in MATERIAL_FIELDS.items()
    ):
        return
    material = UsdMaterialWriter(writer).for_override(prim)
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    mu = float(values["shape_material_mu"][index])
    if not original or physics.GetDynamicFrictionAttr().Get() != mu:
        physics.CreateStaticFrictionAttr().Set(mu)
    for source, (_, target) in MATERIAL_FIELDS.items():
        writer.write_attribute(str(material.GetPath()), target, float(values[source][index]))


def read_contacts(prim, shape_cfg) -> None:
    """Restore contacts on a terrain shape skipped by the native mesh importer."""
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
    for declarations, owner in ((COLLISION_FIELDS, prim), (MATERIAL_FIELDS, material.GetPrim() if material else None)):
        if not owner:
            continue
        for field, target in declarations.values():
            value = owner.GetAttribute(target.attribute).Get()
            if value is not None:
                setattr(shape_cfg, field, float(value))
