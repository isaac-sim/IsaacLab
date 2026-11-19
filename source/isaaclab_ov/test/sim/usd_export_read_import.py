# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read OVPhysX engine values by prim path, and load a USD file into a new OVPhysX instance to read them.

Run :func:`read_import` and :func:`read_nodal_positions` in a separate process: a process has one attached OVPhysX
simulation.
"""

import contextlib

import numpy as np
from isaaclab_ov import tensor_types as TT
from isaaclab_ov.sim.views.ovphysx_view import OvPhysxView

from pxr import Usd, UsdPhysics

from isaaclab.test.utils.usd_export import articulation_scope


@contextlib.contextmanager
def _import(source: str, device: str):
    """A new OVPhysX instance on ``device`` with a USD layer string attached, as its physics interface."""
    from isaaclab_ov.physics import OvPhysxManager
    from isaaclab_ov.physics.ovphysx_manager import OvPhysxBackend
    from isaaclab_ov.physics.ovphysx_manager_cfg import OvPhysxBackendCfg

    OvPhysxManager._ensure_physx_schemas_registered()
    backend = OvPhysxBackend(OvPhysxBackendCfg(device=device))
    try:
        backend.load(source)
        yield backend.physx
    finally:
        backend.close()


def read_import(source: str, articulations: list[str], bodies: list[str]) -> dict:
    """Import a USD layer string into a new OVPhysX instance and read its physics objects.

    Args:
        source: USD layer contents.
        articulations: Articulation root paths.
        bodies: Rigid bodies outside articulations.

    Returns:
        Raw readings as returned by :func:`read`.
    """
    with _import(source, "cpu") as physx:
        return read(physx, articulations, bodies, "cpu")


def read_nodal_positions(source: str, paths: list[str]) -> dict[str, np.ndarray]:
    """Import a USD layer string into a new OVPhysX instance on the GPU and read its volume deformables' nodes.

    Args:
        source: USD layer contents.
        paths: Volume deformable prim paths.

    Returns:
        Per path, the simulation mesh's nodal positions [m] in world coordinates.
    """
    with _import(source, "cuda:0") as physx:
        views = {
            path: OvPhysxView(physx, prim_paths=[path], device="cuda:0", tensor_types=[NODAL_POSITION], eager=True)
            for path in paths
        }
        positions = {path: view.get_attribute(NODAL_POSITION).numpy().copy() for path, view in views.items()}
        for view in views.values():
            view.close()
        return positions


NODAL_POSITION = TT.DEFORMABLE_SIM_NODAL_POSITION
"""OVPhysX binding of a volume deformable's simulation mesh nodal positions."""

_DOF_READINGS = ("stiffness", "damping", "max_force", "max_velocity", "armature", "position", "velocity")

ARTICULATION_TENSORS = {
    "stiffness": TT.DOF_STIFFNESS,
    "damping": TT.DOF_DAMPING,
    "max_force": TT.DOF_MAX_FORCE,
    "limits": TT.DOF_LIMIT,
    "armature": TT.DOF_ARMATURE,
    "max_velocity": TT.DOF_MAX_VELOCITY,
    "friction": TT.DOF_FRICTION_PROPERTIES,
    "mass": TT.BODY_MASS,
    "com_pose": TT.BODY_COM_POSE,
    "inertia": TT.BODY_INERTIA,
    # gravity and the joint state, which the round trip also compares
    "disable_gravity": TT.BODY_DISABLE_GRAVITY,
    "position": TT.DOF_POSITION,
    "velocity": TT.DOF_VELOCITY,
}
"""Reading -> OVPhysX articulation binding, read natively as the reference the round trip compares against."""

RIGID_BODY_TENSORS = {
    "mass": TT.RIGID_BODY_MASS,
    "com_pose": TT.RIGID_BODY_COM_POSE,
    "inertia": TT.RIGID_BODY_INERTIA,
    "material": TT.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION,
    "contact_offset": TT.RIGID_BODY_CONTACT_OFFSET,
    "rest_offset": TT.RIGID_BODY_REST_OFFSET,
    "disable_gravity": TT.RIGID_BODY_DISABLE_GRAVITY,
}
"""Reading -> OVPhysX rigid body binding."""


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
        joints = _index_prims_by_name(scope, lambda prim: prim.IsA(UsdPhysics.Joint))
        links = _index_prims_by_name(scope, lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI))
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


def _index_prims_by_name(scope: Usd.Prim, predicate) -> dict[str, str]:
    return {
        prim.GetName(): str(prim.GetPath())
        for prim in Usd.PrimRange(scope, Usd.TraverseInstanceProxies())
        if predicate(prim)
    }
