# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Shared PhysX tensor-view contact mapping for Isaac Sim and OVPhysX owners."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pxr import Usd, UsdPhysics, UsdShade

from isaaclab.assets.physics_properties import UsdAttribute
from isaaclab.sim.usd_export_properties import UsdMaterialWriter

if TYPE_CHECKING:
    from isaaclab.sim.usd_export import UsdWriter


def author_body_contacts(writer: UsdWriter, path: str, disabled: bool, materials, offsets, rest_offsets) -> None:
    """Author body gravity and PhysX contact buffers without guessing shape ordering.

    Pre-startup export preserves source contacts. Otherwise, native tensor views
    expose body identities but not collider paths. A single
    collider or equal per-body values are unambiguous; distinct per-shape values
    require a backend identity API and are rejected.
    """
    body = writer.stage.GetPrimAtPath(path)
    writer.write_attribute(
        path, UsdAttribute("physxRigidBody:disableGravity", "PhysxRigidBodyAPI", type_name="bool"), disabled
    )
    if writer.preserve_source_contacts:
        # Before buffer randomization, authored bindings and native defaults remain authoritative.
        return
    colliders = []
    descendants = iter(Usd.PrimRange(body))
    for prim in descendants:
        if prim != body and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            descendants.PruneChildren()
        elif prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders.append(prim)
    materials = np.asarray(materials).reshape(-1, 3)
    offsets, rest_offsets = np.asarray(offsets).reshape(-1), np.asarray(rest_offsets).reshape(-1)
    count = len(colliders)
    if count == 0:
        return
    # A cooked mesh can yield several native convex shapes. Uniform values
    # remain representable on its original geometry without reconstructing it.
    if not len(materials) or any(len(values) != len(materials) for values in (offsets, rest_offsets)):
        raise NotImplementedError(f"Unmatched collider identities/count at {path}.")
    if any(not np.all(values == values[0]) for values in (materials, offsets, rest_offsets)):
        raise NotImplementedError(f"Distinct per-shape contact values lack stable collider identities at {path}.")
    for prim in colliders:
        for name, value in (("contactOffset", offsets[0]), ("restOffset", rest_offsets[0])):
            writer.write_attribute(
                str(prim.GetPath()),
                UsdAttribute(f"physxCollision:{name}", "PhysxCollisionAPI", type_name="float"),
                float(value),
            )
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        original, _ = binding.ComputeBoundMaterial("physics")
        names = ("staticFriction", "dynamicFriction", "restitution")
        if original and all(
            original.GetPrim().GetAttribute(f"physics:{name}").Get() == float(value)
            for name, value in zip(names, materials[0])
        ):
            continue
        material = UsdMaterialWriter(writer).for_override(prim)
        destination = material.GetPath()
        for name, value in zip(names, materials[0]):
            writer.write_attribute(
                str(destination), UsdAttribute(f"physics:{name}", "PhysicsMaterialAPI"), float(value)
            )
