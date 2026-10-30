# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read PhysX engine values by prim path, and import a USD file into a headless Isaac Sim to read them.

The USD export test loads exported files with this script, in a separate process because a process
has one attached PhysX simulation::

    python usd_export_read_import.py <stage.usda> <output.pkl> '[[articulations], [bodies]]'

or, to step deformables from a given nodal state (:func:`simulate_deformables`), with a JSON object of its arguments.
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


def simulate_deformables(source: str, output: str, bodies: dict, dt: float, steps: int):
    """Import ``source`` on the GPU, set each deformable's nodal state, step, and write its nodal positions.

    A deformable given no positions keeps the import's, which are written as its ``initial`` positions.
    """
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    try:
        import torch

        import carb
        import omni.physics.tensors
        import omni.physx
        import omni.usd

        # Settings for GPU tensor access, as PhysxManager sets them; the stage enables GPU dynamics itself.
        settings = carb.settings.get_settings()
        settings.set_int("/physics/cudaDevice", 0)
        settings.set_bool("/physics/suppressReadback", True)
        context = omni.usd.get_context()
        if not context.open_stage(source):
            raise RuntimeError(f"Could not open '{source}'.")
        physx = omni.physx.get_physx_interface()
        simulation = omni.physx.get_physx_simulation_interface()
        physx.force_load_physics_from_usd()
        physx.start_simulation()
        # GPU tensor writes need one step; the state written afterwards replaces its result.
        simulation.simulate(dt, 0.0)
        simulation.fetch_results()
        sim_view = omni.physics.tensors.create_simulation_view("torch", stage_id=context.get_stage_id())
        sim_view.set_subspace_roots("/")
        views, initial = {}, {}
        for path, (kind, positions, velocities) in bodies.items():
            create = (
                sim_view.create_volume_deformable_body_view
                if kind == "volume"
                else sim_view.create_surface_deformable_body_view
            )
            view = create(path)
            indices = torch.arange(view.count, dtype=torch.int32, device="cuda:0")
            if positions is None:
                initial[path] = view.get_simulation_nodal_positions().cpu().numpy()
            else:
                view.set_simulation_nodal_positions(torch.tensor(positions, device="cuda:0"), indices)
            view.set_simulation_nodal_velocities(torch.tensor(velocities, device="cuda:0"), indices)
            views[path] = view
        for step in range(steps):
            simulation.simulate(dt, (step + 1) * dt)
            simulation.fetch_results()
        with open(output, "wb") as file:
            final = {path: view.get_simulation_nodal_positions().cpu().numpy() for path, view in views.items()}
            pickle.dump({"initial": initial, "final": final}, file)
    finally:
        app.close()


_DOF_READINGS = ("stiffness", "damping", "max_force", "max_velocity", "armature", "position", "velocity")


def read(simulation_view, articulations: list[str], bodies: list[str]) -> dict:
    """Copy the engine values of each object through PhysX tensor views.

    Args:
        simulation_view: PhysX tensor simulation view.
        articulations: Articulation root paths.
        bodies: Rigid bodies outside articulations.

    Returns:
        Raw values keyed by object path, as consumed by :func:`readings`.
    """
    # The exporter's getters; imported here because this file also runs before Isaac Sim starts.
    from isaaclab_physx.sim.usd_export.usd_exporter import ARTICULATION_GETTERS, RIGID_BODY_GETTERS

    # The joint state, which the round trip also compares.
    getters = {**ARTICULATION_GETTERS, "position": "get_dof_positions", "velocity": "get_dof_velocities"}
    raw = {}
    for root in articulations:
        view = simulation_view.create_articulation_view(root)
        data = {name: _get(getattr(view, getter)()) for name, getter in getters.items()}
        dofs = {name: data.pop(name) for name in _DOF_READINGS}
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
    request = json.loads(sys.argv[3])
    if isinstance(request, dict):
        simulate_deformables(sys.argv[1], sys.argv[2], **request)
    else:
        main(sys.argv[1], sys.argv[2], *request)
