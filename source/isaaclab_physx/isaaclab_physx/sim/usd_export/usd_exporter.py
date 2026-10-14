# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Sim PhysX USD export: every engine value the exported stage does not already state."""

from __future__ import annotations

from collections import Counter

import numpy as np

from pxr import Usd, UsdPhysics

from isaaclab.sim.usd_export.physx_usd_exporter import PhysxUsdExporter, Readings, physics_prims


class UsdExporter(PhysxUsdExporter):
    """Export an Isaac Sim PhysX scene environment: its USD plus every engine value the USD does not state."""

    def read_engine(self, stage: Usd.Stage) -> tuple[Readings, tuple[float, float, float]]:
        from isaaclab_physx.physics import PhysxManager

        articulations, bodies = physics_prims(stage)
        view = PhysxManager.get_physics_sim_view()
        return readings(stage, read(view, articulations, bodies)), view.get_gravity()


_DOF_GETTERS = {
    "stiffness": "get_dof_stiffnesses",
    "damping": "get_dof_dampings",
    "max_force": "get_dof_max_forces",
    "max_velocity": "get_dof_max_velocities",
    "armature": "get_dof_armatures",
}

ARTICULATION_GETTERS = {
    **_DOF_GETTERS,
    "limits": "get_dof_limits",
    "friction": "get_dof_friction_properties",
    "mass": "get_masses",
    "inertia": "get_inertias",
    "com_pose": "get_coms",
}
"""Raw reading name -> articulation view getter."""

RIGID_BODY_GETTERS = {
    "mass": "get_masses",
    "inertia": "get_inertias",
    "com_pose": "get_coms",
    "material": "get_material_properties",
    "contact_offset": "get_contact_offsets",
    "rest_offset": "get_rest_offsets",
}
"""Raw reading name -> rigid body view getter."""


def read(simulation_view, articulations: list[str], bodies: list[str]) -> dict:
    """Copy the engine values of each object through PhysX tensor views.

    Args:
        simulation_view: PhysX tensor simulation view.
        articulations: Articulation root paths.
        bodies: Rigid bodies outside articulations.

    Returns:
        Raw values keyed by object path, as consumed by :func:`readings`.
    """
    raw = {}
    for root in articulations:
        view = simulation_view.create_articulation_view(root)
        data = {name: _get(getattr(view, getter)()) for name, getter in ARTICULATION_GETTERS.items()}
        dofs = {name: data.pop(name) for name in _DOF_GETTERS}
        limits, friction = data.pop("limits"), data.pop("friction")
        dofs.update(lower_limit=limits[..., 0], upper_limit=limits[..., 1])
        dofs.update(static_friction=friction[..., 0], dynamic_friction=friction[..., 1])
        dofs.update(viscous_friction=friction[..., 2])
        paths = {"dof_paths": list(view.dof_paths[0]), "link_paths": list(view.link_paths[0])}
        raw[root] = {**data, **paths, "dofs": dofs}
    for body in bodies:
        view = simulation_view.create_rigid_body_view(body)
        raw[body] = {name: _get(getattr(view, getter)()) for name, getter in RIGID_BODY_GETTERS.items()}
    return raw


def readings(stage: Usd.Stage, raw: dict) -> Readings:
    """Key raw engine readings by the USD prim paths the tensor views report."""
    result = {}
    for root, data in raw.items():
        if "dof_paths" not in data:
            com, inertia = data["com_pose"][:3], data["inertia"]
            result[root, None] = {"mass": data["mass"].item(), "com": com, "inertia": inertia}
            for name in ("material", "contact_offset", "rest_offset"):
                result[root, None][name] = data[name]
            continue
        counts, seen = Counter(data["dof_paths"]), Counter()
        for index, path in enumerate(data["dof_paths"]):
            prim = stage.GetPrimAtPath(path)
            if counts[path] == 1 and prim.IsA(UsdPhysics.RevoluteJoint):
                key = (path, "angular")
            elif counts[path] == 1 and prim.IsA(UsdPhysics.PrismaticJoint):
                key = (path, "linear")
            else:
                key = (f"{path}:{seen[path]}", None)
                seen[path] += 1
            result[key] = {prop: values[index] for prop, values in data["dofs"].items()}
        for index, path in enumerate(data["link_paths"]):
            com, inertia = data["com_pose"][index, :3], data["inertia"][index]
            result[path, None] = {"mass": data["mass"][index], "com": com, "inertia": inertia}
    return result


def _get(values) -> np.ndarray:
    """First view entry as a NumPy array, from either tensor frontend."""
    return np.asarray(values.numpy() if hasattr(values, "numpy") else values)[0].copy()
