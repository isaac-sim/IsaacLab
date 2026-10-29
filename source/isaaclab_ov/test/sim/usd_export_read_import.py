# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read OVPhysX engine values by prim path, and load a USD file into a new OVPhysX instance to read them.

Run :func:`read_import` in a separate process: a process has one attached OVPhysX simulation.
"""

import numpy as np
from isaaclab_ov import tensor_types as TT
from isaaclab_ov.sim.usd_export import usd_exporter
from isaaclab_ov.sim.views.ovphysx_view import OvPhysxView

from pxr import Usd, UsdPhysics

from isaaclab.test.utils.usd_export import articulation_scope


def read_import(source: str, articulations: list[str], bodies: list[str]) -> dict:
    """Import a USD layer string into a new OVPhysX instance and read its physics objects.

    Args:
        source: USD layer contents.
        articulations: Articulation root paths.
        bodies: Rigid bodies outside articulations.

    Returns:
        Raw readings as returned by :func:`read`.
    """
    import ovstage
    from isaaclab_ov.physics import OvPhysxManager, ovphysx_compat
    from isaaclab_ov.physics.ovphysx_manager import OvPhysxBackend
    from isaaclab_ov.physics.ovphysx_manager_cfg import OvPhysxBackendCfg
    from isaaclab_ov.stage import create_ovstage

    OvPhysxManager._ensure_physx_schemas_registered()
    backend = OvPhysxBackend(OvPhysxBackendCfg(device="cpu"))
    try:
        backend.stage = create_ovstage("usd_export_test")
        ovstage.population.open_usd_from_string(backend.stage, source, ordinal=1, domains=ovstage.PopulationDomain.ALL)
        backend.stage.advance_write_floor(ordinal=1).wait()
        ovphysx_compat.finish_stage_operation(
            backend.physx, backend.physx.attach_ovstage(backend.stage, read_ordinal=1)
        )
        return read(backend.physx, articulations, bodies, "cpu")
    finally:
        backend.close()


_DOF_READINGS = ("stiffness", "damping", "max_force", "max_velocity", "armature")

# The exporter's bindings, and gravity, which the round trip also compares.
ARTICULATION_TENSORS = {**usd_exporter.ARTICULATION_TENSORS, "disable_gravity": TT.BODY_DISABLE_GRAVITY}
RIGID_BODY_TENSORS = {**usd_exporter.RIGID_BODY_TENSORS, "disable_gravity": TT.RIGID_BODY_DISABLE_GRAVITY}


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
        dofs = {name: raw[root].pop(name) for name in _DOF_READINGS}
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


def readings(stage, raw: dict) -> dict:
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
