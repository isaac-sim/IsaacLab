# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Deployment construction from USD; solver semantics remain with their managers."""

from __future__ import annotations

from newton import Model
from newton.solvers import SolverBase

from pxr import Usd

from isaaclab.sim.utils.newton_model_utils import replace_newton_builder_shape_colors

from isaaclab_newton.sim.schemas.physics_properties import SOFT_CONTACT_FIELDS


def create_deployment_model(path: str, device: str = "cpu") -> tuple[Model, dict]:
    """Load a deployment USD using its exported physical schemas and import settings.

    Args:
        path: Path to a USD written by :meth:`InteractiveScene.export_to_usd`.
        device: Warp device on which to create the model.

    Returns:
        The initialized model and importer identity maps. ``particle_paths`` maps
        ``(mesh_path, local_node_index)`` to model particle indices for coupled solvers.
    """
    import newton

    builder = newton.ModelBuilder()
    stage = Usd.Stage.Open(str(path))
    options = stage.GetRootLayer().customLayerData.get("isaaclab:newtonImportOptions", {})
    from newton.usd import SchemaResolverMjc, SchemaResolverNewton, SchemaResolverPhysx

    driver = stage.GetRootLayer().customLayerData["isaaclab:newtonDriver"]["solver"]
    solver_type = {
        "xpbd": newton.solvers.SolverXPBD,
        "mujoco": newton.solvers.SolverMuJoCo,
        "kamino": newton.solvers.SolverKamino,
        "vbd": newton.solvers.SolverVBD,
        "coupled_proxy": newton.solvers.SolverMuJoCo,
    }[driver]
    solver_type.register_custom_attributes(builder)
    resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]
    if driver in {"mujoco", "coupled_proxy"}:
        resolvers.insert(0, SchemaResolverMjc())
    from isaaclab_newton.physics import NewtonManager

    ignored = NewtonManager._inject_terrain_heightfields(stage, builder, root_paths=("/",))
    from isaaclab_newton.assets.cable_object.cable_object import CableObject

    deformables = {}
    from isaaclab.scene_data.deformable_discovery import discover_deformables_on_stage

    if discover_deformables_on_stage(stage):
        from isaaclab_contrib.deformable.deformable_object import add_exported_deformables_to_builder

        deformables = add_exported_deformables_to_builder(stage, builder)
    ignored.extend(path for entry in deformables.values() for path in entry["ignore_paths"])
    stage_info = builder.add_usd(
        str(path), schema_resolvers=resolvers, ignore_paths=ignored, return_deformable_results=True, **options
    )
    from isaaclab_newton.cloner.newton_clone_utils import _name_root_joints_after_their_body

    _name_root_joints_after_their_body(builder)
    CableObject.restore_fixed_configuration(stage, builder, stage_info.get("path_cable_map", {}))
    replace_newton_builder_shape_colors(builder, stage)
    if driver in {"vbd", "coupled_proxy"}:
        builder.color()
    model = builder.finalize(device=device)
    if deformables:
        inverse_masses = model.particle_inv_mass.numpy()
        for entry in deformables.values():
            if entry["inverse_masses"] is not None:
                start, stop = entry["ranges"]["particle"]
                inverse_masses[start:stop] = entry["inverse_masses"]
        model.particle_inv_mass.assign(inverse_masses)
    for name, value in stage.GetRootLayer().customLayerData.get("isaaclab:newtonModel", {}).items():
        if name not in SOFT_CONTACT_FIELDS:
            raise ValueError(f"Unknown exported Newton model property {name!r}.")
        setattr(model, name, value)
    import json

    exported_driver = dict(stage.GetRootLayer().customLayerData["isaaclab:newtonDriver"])
    if driver in {"xpbd", "mujoco", "vbd"}:
        iterations = stage_info.get("max_solver_iterations")
        if iterations is None:
            raise ValueError("Deployment USD has no native scene solver iteration setting.")
        if driver == "xpbd":
            exported_driver["iterations"] = int(iterations)
        else:
            options = json.loads(exported_driver["options"])
            options["iterations"] = int(iterations)
            exported_driver["options"] = json.dumps(options)
    stage_info["driver"] = exported_driver
    stage_info["particle_paths"] = {
        (path, i - entry["ranges"]["particle"][0]): i
        for path, entry in deformables.items()
        for i in range(*entry["ranges"]["particle"])
    }
    return model, stage_info


def create_deployment_solver(
    model: Model, driver: dict, *, particle_paths: dict[tuple[str, int], int] | None = None
) -> SolverBase:
    """Construct the exported native solver without a task configuration.

    Args:
        model: Model returned by :meth:`create_deployment_model`.
        driver: The resolved ``driver`` entry returned with the deployment model mappings.
        particle_paths: Particle identity map from :meth:`create_deployment_model`.

    Returns:
        The solver initialized with the artifact's effective settings.
    """
    import json

    import newton

    driver = dict(driver)
    name = driver.pop("solver")
    if name == "xpbd":
        return newton.solvers.SolverXPBD(model, **driver)
    options = json.loads(driver["options"])
    if name == "vbd":
        from isaaclab_newton.physics.vbd_manager import NewtonVBDManager

        return NewtonVBDManager.load_exported_solver(model, options)
    if name == "coupled_proxy":
        from isaaclab_contrib.coupling.coupler import NewtonCouplerManager

        return NewtonCouplerManager.load_exported_solver(model, options, particle_paths or {})
    if name == "mujoco":
        return newton.solvers.SolverMuJoCo(model, **options)
    if name != "kamino":
        raise ValueError(f"Unknown deployment solver {name!r}.")
    from isaaclab_newton.physics.kamino_manager import NewtonKaminoManager

    return NewtonKaminoManager.load_exported_solver(model, options)
