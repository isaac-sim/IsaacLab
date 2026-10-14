# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX USD export: every engine value the exported stage does not already state."""

from __future__ import annotations

import numpy as np

from pxr import Usd, UsdPhysics

from isaaclab.sim.usd_export.physx_usd_exporter import PhysxUsdExporter, Readings, articulation_scope, physics_prims

from isaaclab_ov import tensor_types as TT
from isaaclab_ov.physics import OvPhysxManager
from isaaclab_ov.sim.views.ovphysx_view import OvPhysxView

_DOF_TENSORS = {
    "stiffness": TT.DOF_STIFFNESS,
    "damping": TT.DOF_DAMPING,
    "max_force": TT.DOF_MAX_FORCE,
    "max_velocity": TT.DOF_MAX_VELOCITY,
    "armature": TT.DOF_ARMATURE,
}

ARTICULATION_TENSORS = {
    **_DOF_TENSORS,
    "limits": TT.DOF_LIMIT,
    "friction": TT.DOF_FRICTION_PROPERTIES,
    "mass": TT.BODY_MASS,
    "inertia": TT.BODY_INERTIA,
    "com_pose": TT.BODY_COM_POSE,
    "disable_gravity": TT.BODY_DISABLE_GRAVITY,
}
"""Raw reading name -> OVPhysX binding read from each articulation."""

RIGID_BODY_TENSORS = {
    "mass": TT.RIGID_BODY_MASS,
    "inertia": TT.RIGID_BODY_INERTIA,
    "com_pose": TT.RIGID_BODY_COM_POSE,
    "disable_gravity": TT.RIGID_BODY_DISABLE_GRAVITY,
    "contact_offset": TT.RIGID_BODY_CONTACT_OFFSET,
    "rest_offset": TT.RIGID_BODY_REST_OFFSET,
    "material": TT.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION,
}
"""Raw reading name -> OVPhysX binding read from each rigid body outside articulations."""


class UsdExporter(PhysxUsdExporter):
    """Export an OVPhysX scene environment: its USD plus every engine value the USD does not state."""

    def read_engine(self, stage: Usd.Stage) -> tuple[Readings, tuple[float, float, float]]:
        articulations, bodies = physics_prims(stage)
        raw = read(OvPhysxManager.get_physx_instance(), articulations, bodies, self.scene.sim.device)
        return readings(stage, raw), OvPhysxManager.get_gravity()


def read(physx, articulations: list[str], bodies: list[str], device: str) -> dict:
    """Copy the engine values of each object before its bindings close.

    Args:
        physx: OVPhysX instance.
        articulations: Articulation root paths.
        bodies: Rigid bodies outside articulations.
        device: Simulation device.

    Returns:
        Raw values keyed by object path, as consumed by :func:`readings`.
    """
    raw = {}
    for root in articulations:
        view = OvPhysxView(physx, prim_paths=[root], device=device)
        try:
            raw[root] = {name: _get(view, tensor) for name, tensor in ARTICULATION_TENSORS.items()}
            raw[root].update(dof_names=list(view.dof_names), body_names=list(view.body_names))
        finally:
            view.close()
        dofs = {name: raw[root].pop(name) for name in _DOF_TENSORS}
        limits, friction = raw[root].pop("limits"), raw[root].pop("friction")
        dofs.update(lower_limit=limits[..., 0], upper_limit=limits[..., 1])
        dofs.update(static_friction=friction[..., 0], dynamic_friction=friction[..., 1])
        raw[root]["dofs"] = {**dofs, "viscous_friction": friction[..., 2]}
    for body in bodies:
        view = OvPhysxView(physx, prim_paths=[body], device=device)
        try:
            raw[body] = {name: _get(view, tensor) for name, tensor in RIGID_BODY_TENSORS.items()}
        finally:
            view.close()
    return raw


def readings(stage: Usd.Stage, raw: dict) -> Readings:
    """Key raw engine readings by USD prim path, matching engine names to prims under each articulation."""
    result = {}
    for root, data in raw.items():
        if "dof_names" not in data:
            result[root, None] = {"mass": data["mass"], "com": data["com_pose"][:3], "inertia": data["inertia"]}
            for name in ("disable_gravity", "contact_offset", "rest_offset", "material"):
                result[root, None][name] = data[name]
            continue
        scope = articulation_scope(stage, root)
        joints = _named(scope, lambda prim: prim.IsA(UsdPhysics.Joint))
        links = _named(scope, lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI))
        for index, name in enumerate(data["dof_names"]):
            path = joints.get(name)
            prim = stage.GetPrimAtPath(path) if path else None
            if prim and prim.IsA(UsdPhysics.RevoluteJoint):
                key = (path, "angular")
            elif prim and prim.IsA(UsdPhysics.PrismaticJoint):
                key = (path, "linear")
            else:
                key = (f"{root}:{name}", None)
            result[key] = {prop: values[index] for prop, values in data["dofs"].items()}
        for index, name in enumerate(data["body_names"]):
            result[links[name], None] = {
                "mass": data["mass"][index],
                "com": data["com_pose"][index, :3],
                "inertia": data["inertia"][index],
                "disable_gravity": data["disable_gravity"][index],
            }
    return result


def _get(view: OvPhysxView, tensor) -> np.ndarray:
    return view.get_attribute(tensor).numpy()[0].copy()


def _named(scope: Usd.Prim, predicate) -> dict[str, str]:
    return {
        prim.GetName(): str(prim.GetPath())
        for prim in Usd.PrimRange(scope, Usd.TraverseInstanceProxies())
        if predicate(prim)
    }
