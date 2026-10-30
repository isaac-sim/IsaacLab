# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX USD export: a PhysX import of the exported file reproduces the initialized engine."""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation()

import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pxr import Sdf, Usd, UsdPhysics

# The engine reader lives next to this test, which also runs it in a separate Isaac Sim process.
sys.path.insert(0, str(Path(__file__).parent))


def _read_import(stage, articulations, bodies) -> dict:
    """Read the objects of ``stage`` after a PhysX import in a new headless Isaac Sim process."""
    flat = Usd.Stage.Open(stage.Flatten())
    for prim in flat.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            prim.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool).Set(False)
            prim.CreateAttribute("physxScene:broadphaseType", Sdf.ValueTypeNames.Token).Set("MBP")
    with tempfile.TemporaryDirectory() as directory:
        source, output = os.path.join(directory, "stage.usda"), os.path.join(directory, "readings.pkl")
        flat.Export(source)
        script = Path(__file__).parent / "usd_export_read_import.py"
        subprocess.run([sys.executable, str(script), source, output, json.dumps([articulations, bodies])], check=True)
        with open(output, "rb") as file:
            return pickle.load(file)


def test_importing_the_export_reproduces_the_initialized_engine(tmp_path):
    """Every exported value of a PhysX import in a new process equals the live engine's."""
    from isaaclab_physx.physics import PhysxCfg
    from usd_export_read_import import read, readings

    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from isaaclab.test.utils.usd_export import make_export_scene_cfg, physics_prims

    cfg = make_export_scene_cfg(tmp_path)
    sim_cfg = SimulationCfg(device="cpu", dt=1 / 120, gravity=(0.2, -0.1, -4.0), physics=PhysxCfg())
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(cfg)
        sim.reset()
        output = scene.export_to_usd(str(tmp_path / "exported.usda"))
        # Isaac Lab's reset writes the initial joint state, which the engine holds only from then on.
        robot = scene["robot"]
        robot.write_joint_state_to_sim_index(
            position=robot.data.default_joint_pos.torch, velocity=robot.data.default_joint_vel.torch
        )
        stage = Usd.Stage.Open(output)
        articulations, bodies = physics_prims(stage)
        live = readings(stage, read(sim.backend_manager.get_physics_sim_view(), articulations, bodies))
        gravity = np.asarray(sim.backend_manager.get_gravity())
    loaded = readings(stage, _read_import(stage, articulations, bodies))

    hinge = live["/World/envs/env_0/Robot/Hinge", "angular"]
    assert hinge["position"] == pytest.approx(0.21)
    assert hinge["stiffness"] == pytest.approx(83.0)
    assert hinge["armature"] == pytest.approx(0.023)
    assert live.keys() == loaded.keys()
    for key, values in live.items():
        for name, value in values.items():
            expected = loaded[key][name]
            np.testing.assert_allclose(expected, value, rtol=1e-4, atol=1e-6, err_msg=f"{key} {name}")
    scene_prim = UsdPhysics.Scene(stage.GetPrimAtPath("/physicsScene"))
    exported = np.asarray(scene_prim.GetGravityDirectionAttr().Get()) * scene_prim.GetGravityMagnitudeAttr().Get()
    np.testing.assert_allclose(exported, gravity, rtol=1e-5)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="PhysX simulates deformables on the GPU")
def test_importing_the_export_reproduces_deformables(tmp_path):
    """Deformables of a PhysX import step like the live engine's from the same nodal state."""
    import warp as wp
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_physx.sim.schemas import PhysxDeformableBodyCfg
    from isaaclab_physx.sim.spawners.materials import (
        PhysxDeformableBodyMaterialCfg,
        PhysxSurfaceDeformableBodyMaterialCfg,
    )

    import isaaclab.sim as sim_utils
    from isaaclab.assets import DeformableObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, build_simulation_context

    body = [
        sim_utils.OmniPhysicsDeformableBodyCfg(),
        PhysxDeformableBodyCfg(solver_position_iteration_count=24, linear_damping=0.05),
    ]
    cfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.0)
    cfg.beam = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Beam",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, -0.3, 1.0)),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.3, 0.04, 0.04),
            volume_deformable_props=body,
            physics_material=PhysxDeformableBodyMaterialCfg(youngs_modulus=2e5, poissons_ratio=0.3, density=500.0),
        ),
    )
    cfg.cloth = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cloth",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.3, 1.0)),
        spawn=sim_utils.MeshRectangleCfg(
            size=(0.2, 0.2),
            surface_deformable_props=body,
            physics_material=PhysxSurfaceDeformableBodyMaterialCfg(youngs_modulus=1e5, surface_thickness=0.002),
        ),
    )
    kinds = {"beam": "volume", "cloth": "surface"}
    dt, steps = 1 / 120, 20
    with build_simulation_context(sim_cfg=SimulationCfg(device="cuda:0", dt=dt, physics=PhysxCfg())) as sim:
        scene = InteractiveScene(cfg)
        sim.reset()
        output = scene.export_to_usd(str(tmp_path / "exported.usda"))
        initial, live = {}, {}
        for name, asset in scene.deformable_objects.items():
            view = asset.root_view
            positions = view.get_simulation_nodal_positions().numpy()
            # Stretch and spin, so that the elastic forces act without any contact.
            offsets = positions - positions.mean(axis=1, keepdims=True)
            velocities = (2.0 * offsets + np.cross([0.0, 0.0, 3.0], offsets)).astype(np.float32)
            view.set_simulation_nodal_velocities(
                wp.array(velocities, device="cuda:0"), wp.array([0], dtype=wp.int32, device="cuda:0")
            )
            initial[view.prim_paths[0]] = (kinds[name], positions.tolist(), velocities.tolist())
        for _ in range(steps):
            sim.step(render=False)
        for name, asset in scene.deformable_objects.items():
            live[asset.root_view.prim_paths[0]] = asset.root_view.get_simulation_nodal_positions().numpy()

    with tempfile.TemporaryDirectory() as directory:
        result = os.path.join(directory, "positions.pkl")
        request = {"bodies": initial, "dt": dt, "steps": steps}
        script = Path(__file__).parent / "usd_export_read_import.py"
        subprocess.run([sys.executable, str(script), output, result, json.dumps(request)], check=True)
        with open(result, "rb") as file:
            loaded = pickle.load(file)

    assert loaded.keys() == live.keys()
    for path, positions in live.items():
        moved = np.abs(positions - np.asarray(initial[path][1])).max()
        assert moved > 0.01, f"{path} barely moved"
        # PhysX GPU runs differ by up to about 0.2 mm; a 10x stiffer material moves the beam by about 15 mm.
        np.testing.assert_allclose(loaded[path], positions, atol=1e-3, err_msg=path)


def test_read_getters_cover_physx_views(tmp_path):
    """Every getter of the PhysX articulation and rigid-body views is compared by the round trip or excluded."""
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_physx.sim.usd_export.usd_exporter import ARTICULATION_GETTERS, RIGID_BODY_GETTERS

    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from isaaclab.test.utils.usd_export import make_export_scene_cfg

    read = {*ARTICULATION_GETTERS.values(), *RIGID_BODY_GETTERS.values()}
    excluded = {
        # the initial joint state comes from the asset configuration, which the engine holds only after a reset
        "get_dof_positions",
        "get_dof_velocities",
        # body state, targets and forces are not part of the export
        "get_root_transforms",
        "get_root_velocities",
        "get_link_transforms",
        "get_link_velocities",
        "get_link_accelerations",
        "get_dof_position_targets",
        "get_dof_velocity_targets",
        "get_dof_actuation_forces",
        "get_transforms",
        "get_velocities",
        "get_accelerations",
        # derived from exported properties
        "get_inv_masses",
        "get_inv_inertias",
        "get_jacobians",
        "get_generalized_mass_matrices",
        "get_coriolis_and_centrifugal_compensation_forces",
        "get_gravity_compensation_forces",
        "get_link_incoming_joint_force",
        "get_dof_projected_joint_forces",
        "get_articulation_centroidal_momentum",
        "get_articulation_mass_center",
        # view metadata
        "get_metatype",
        # set only from USD, never written at runtime, so already in the exported file
        "get_disable_simulations",
        "get_disable_gravities",
        "get_dof_drive_model_properties",
        "get_dof_types",
        "get_drive_types",
        "get_dof_motions",
        "get_compliant_material_properties",
        # legacy view of the static friction in get_dof_friction_properties
        "get_dof_friction_coefficients",
        # tendons have no per-tendon USD identity to write
        "get_fixed_tendon_stiffnesses",
        "get_fixed_tendon_dampings",
        "get_fixed_tendon_limit_stiffnesses",
        "get_fixed_tendon_limits",
        "get_fixed_tendon_rest_lengths",
        "get_fixed_tendon_offsets",
        "get_spatial_tendon_stiffnesses",
        "get_spatial_tendon_dampings",
        "get_spatial_tendon_limit_stiffnesses",
        "get_spatial_tendon_offsets",
    }
    cfg = make_export_scene_cfg(tmp_path)
    with build_simulation_context(sim_cfg=SimulationCfg(device="cpu", physics=PhysxCfg())) as sim:
        InteractiveScene(cfg)
        sim.reset()
        view = sim.backend_manager.get_physics_sim_view()
        getters = {
            name
            for cls in (
                type(view.create_articulation_view("/World/envs/env_0/Robot")),
                type(view.create_rigid_body_view("/World/envs/env_0/Box")),
            )
            for name in dir(cls)
            if name.startswith("get_")
        }
    unclassified = sorted(getters - read - excluded)
    assert not unclassified, f"Unclassified getters: {unclassified}"
