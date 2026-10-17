# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read PhysX engine values by prim path, and import a USD file into a headless Isaac Sim to read them.

The USD export test loads exported files with this script, in a separate process because a process
has one attached PhysX simulation::

    python usd_export_read_import.py <stage.usda> <output.pkl> '[[articulations], [bodies]]'
"""

import json
import pickle
import sys
from collections import Counter

import numpy as np


def main(source: str, output: str, articulations: list[str], bodies: list[str]):
    """Import ``source`` and write the readings of the given objects to ``output``."""
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    try:
        import omni.physics.tensors
        import omni.physx
        import omni.usd

        context = omni.usd.get_context()
        if not context.open_stage(source):
            raise RuntimeError(f"Could not open '{source}'.")
        physx = omni.physx.get_physx_interface()
        physx.force_load_physics_from_usd()
        physx.start_simulation()
        view = omni.physics.tensors.create_simulation_view("numpy", stage_id=context.get_stage_id())
        view.set_subspace_roots("/")
        with open(output, "wb") as file:
            pickle.dump(read(view, articulations, bodies), file)
    finally:
        app.close()


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


def readings(stage, raw: dict) -> dict:
    """Key raw engine readings by the USD prim paths the tensor views report."""
    from pxr import UsdPhysics

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


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *json.loads(sys.argv[3]))
