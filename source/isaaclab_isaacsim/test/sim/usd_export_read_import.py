# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read PhysX engine values by prim path, and import a USD file into a headless Isaac Sim to read them.

The USD export test loads exported files with this script, in a separate process because a process
has one attached PhysX simulation::

    python usd_export_read_import.py <stage.usda> <output.pkl> '[[articulations], [bodies]]'

or, to read the nodal positions of volume deformables (:func:`read_import_nodal_positions`), with
``'{"paths": [...]}'``.
"""

import json
import os
import pickle
import subprocess
import sys
import tempfile
from collections import Counter

import numpy as np


def read_import(stage, articulations: list[str], bodies: list[str]) -> dict:
    """Import ``stage`` into a new headless Isaac Sim process and read the given objects there.

    The import runs on the CPU, with the broadphase that needs no GPU.

    Args:
        stage: Stage to import, flattened before writing it to a file.
        articulations: Articulation root paths.
        bodies: Rigid bodies outside articulations.

    Returns:
        Raw readings as returned by :func:`read`.
    """
    from pxr import Sdf, Usd, UsdPhysics

    flat = Usd.Stage.Open(stage.Flatten())
    for prim in flat.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            prim.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool).Set(False)
            prim.CreateAttribute("physxScene:broadphaseType", Sdf.ValueTypeNames.Token).Set("MBP")
    return _run_in_isaac_sim(flat, [articulations, bodies])


def read_import_nodal_positions(stage, paths: list[str]) -> dict[str, np.ndarray]:
    """Import ``stage`` into a new headless Isaac Sim process on the GPU and read its volume deformables' nodes.

    Args:
        stage: Stage to import, flattened before writing it to a file.
        paths: Volume deformable prim paths.

    Returns:
        Per path, the simulation mesh's nodal positions [m] in world coordinates.
    """
    from pxr import Usd

    return _run_in_isaac_sim(Usd.Stage.Open(stage.Flatten()), {"paths": paths})


def _run_in_isaac_sim(stage, request):
    """Run this script on ``stage``, written to a file, in a new headless Isaac Sim process, and return its result."""
    with tempfile.TemporaryDirectory() as directory:
        source, output = os.path.join(directory, "stage.usda"), os.path.join(directory, "result.pkl")
        stage.Export(source)
        subprocess.run([sys.executable, __file__, source, output, json.dumps(request)], check=True)
        with open(output, "rb") as file:
            return pickle.load(file)


def main(source: str, output: str, request):
    """Import ``source`` and write to ``output`` the readings of ``[articulations, bodies]``, or, on the GPU, the nodal
    positions of the volume deformables ``{"paths": [...]}``."""
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    try:
        import carb
        import omni.physics.tensors
        import omni.physx
        import omni.usd

        deformables = isinstance(request, dict)
        if deformables:
            # Settings for GPU tensor access, as PhysxManager sets them; the stage enables GPU dynamics itself.
            carb.settings.get_settings().set_int("/physics/cudaDevice", 0)
            carb.settings.get_settings().set_bool("/physics/suppressReadback", True)
        context = omni.usd.get_context()
        if not context.open_stage(source):
            raise RuntimeError(f"Could not open '{source}'.")
        physx = omni.physx.get_physx_interface()
        physx.force_load_physics_from_usd()
        physx.start_simulation()
        if deformables:
            # GPU tensor access needs the warm-up step, after which Isaac Lab also reads a deformable's reset state.
            omni.physx.get_physx_simulation_interface().simulate(1 / 120, 0.0)
            omni.physx.get_physx_simulation_interface().fetch_results()
        frontend = "torch" if deformables else "numpy"
        view = omni.physics.tensors.create_simulation_view(frontend, stage_id=context.get_stage_id())
        view.set_subspace_roots("/")
        if deformables:
            result = {
                path: view.create_volume_deformable_body_view(path).get_simulation_nodal_positions()[0].cpu().numpy()
                for path in request["paths"]
            }
        else:
            result = read(view, *request)
        with open(output, "wb") as file:
            pickle.dump(result, file)
    finally:
        app.close()


_DOF_READINGS = ("stiffness", "damping", "max_force", "max_velocity", "armature", "position", "velocity")

ARTICULATION_GETTERS = {
    "stiffness": "get_dof_stiffnesses",
    "damping": "get_dof_dampings",
    "max_force": "get_dof_max_forces",
    "limits": "get_dof_limits",
    "armature": "get_dof_armatures",
    "max_velocity": "get_dof_max_velocities",
    "friction": "get_dof_friction_properties",
    "mass": "get_masses",
    "com_pose": "get_coms",
    "inertia": "get_inertias",
    # the joint state, which the round trip also compares
    "position": "get_dof_positions",
    "velocity": "get_dof_velocities",
}
"""Reading -> PhysX articulation view getter, read natively as the reference the round trip compares against."""

RIGID_BODY_GETTERS = {
    "mass": "get_masses",
    "com_pose": "get_coms",
    "inertia": "get_inertias",
    "material": "get_material_properties",
    "contact_offset": "get_contact_offsets",
    "rest_offset": "get_rest_offsets",
}
"""Reading -> PhysX rigid body view getter."""


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
    main(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]))
