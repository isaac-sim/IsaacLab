# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX USD export: an OVPhysX import of the exported file reproduces the initialized engine."""

import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from pxr import Sdf, Usd, UsdPhysics

# The file loader runs in a spawned process and must import by module name.
sys.path.insert(0, str(Path(__file__).parent))


# Without a box per environment, environment 1 exists in physics only.
def test_importing_the_export_reproduces_the_initialized_engine(tmp_path):
    """Every exported value of a fresh OVPhysX import equals the live engine's, including config overrides."""
    from isaaclab_ov.physics import OvPhysxCfg
    from usd_export_read_import import read, read_import, readings

    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from isaaclab.test.utils.usd_export import make_export_scene_cfg, physics_prims

    cfg = make_export_scene_cfg(tmp_path)
    sim_cfg = SimulationCfg(device="cpu", dt=1 / 120, gravity=(0.2, -0.1, -4.0), physics=OvPhysxCfg())
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(cfg)
        sim.reset()
        output = scene.export_to_usd(str(tmp_path / "exported.usda"))
        stage = Usd.Stage.Open(output)
        articulations, bodies = physics_prims(stage)
        live = readings(stage, read(sim.backend_manager.get_physx_instance(), articulations, bodies, "cpu"))
        gravity = np.asarray(sim.backend_manager.get_gravity())
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        imported = pool.submit(read_import, stage.Flatten().ExportToString(), articulations, bodies).result()
    loaded = readings(stage, imported)

    hinge = live["/World/envs/env_0/Robot/Hinge", "angular"]
    assert hinge["stiffness"] == pytest.approx(83.0)
    assert hinge["armature"] == pytest.approx(0.023)
    exported_hinge = stage.GetPrimAtPath("/World/envs/env_0/Robot/Hinge")
    assert exported_hinge.GetAttribute("physxJointAxis:angular:armature").HasAuthoredValue()
    # Zero joint friction equals the schema default, so the export leaves it unset.
    assert not exported_hinge.GetAttribute("physxJointAxis:angular:staticFrictionEffort").HasAuthoredValue()
    assert live.keys() == loaded.keys()
    for key, values in live.items():
        for name, value in values.items():
            expected = loaded[key][name]
            np.testing.assert_allclose(expected, value, rtol=1e-4, atol=1e-6, err_msg=f"{key} {name}")
    scene_prim = UsdPhysics.Scene(stage.GetPrimAtPath("/physicsScene"))
    exported = np.asarray(scene_prim.GetGravityDirectionAttr().Get()) * scene_prim.GetGravityMagnitudeAttr().Get()
    np.testing.assert_allclose(exported, gravity, rtol=1e-5)


def test_values_the_engine_derives_are_not_written(tmp_path):
    """A contact offset the USD leaves to PhysX (-inf), which it derives from the collider, is not restated."""
    from isaaclab_ov.physics import OvPhysxCfg
    from usd_export_read_import import read, read_import, readings

    from pxr import Sdf

    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from isaaclab.test.utils.usd_export import make_export_scene_cfg, physics_prims

    sim_cfg = SimulationCfg(device="cpu", dt=1 / 120, physics=OvPhysxCfg())
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(make_export_scene_cfg(tmp_path))
        box = "/World/envs/env_0/Box"
        collider = next(p for p in Usd.PrimRange(sim.stage.GetPrimAtPath(box)) if p.HasAPI(UsdPhysics.CollisionAPI))
        collider.AddAppliedSchema("PhysxCollisionAPI")
        offset = collider.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float)
        offset.Set(float("-inf"))
        sim.reset()
        output = scene.export_to_usd(str(tmp_path / "exported.usda"))
        stage = Usd.Stage.Open(output)
        articulations, bodies = physics_prims(stage)
        live = readings(stage, read(sim.backend_manager.get_physx_instance(), articulations, bodies, "cpu"))
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        imported = pool.submit(read_import, stage.Flatten().ExportToString(), articulations, bodies).result()

    assert np.all(np.isfinite(live[box, None]["contact_offset"]))  # derived by the engine
    assert stage.GetPrimAtPath(collider.GetPath()).GetAttribute("physxCollision:contactOffset").Get() == float("-inf")
    loaded = readings(stage, imported)[box, None]["contact_offset"]
    np.testing.assert_allclose(loaded, live[box, None]["contact_offset"], rtol=1e-5)


def test_legacy_joint_attributes_isaac_lab_migrates_are_restated(tmp_path):
    """OVPhysX 0.7 ignores legacy PhysX joint attributes, which Isaac Lab migrates before the engine loads the stage;
    the export states them as the attributes an import reads."""
    from isaaclab_ov.physics import OvPhysxCfg, ovphysx_compat
    from usd_export_read_import import read, read_import, readings

    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from isaaclab.test.utils.usd_export import make_export_scene_cfg, physics_prims

    if not ovphysx_compat.OVPHYSX_USES_TASKS:
        pytest.skip("OVPhysX before 0.7 reads the legacy attributes.")
    cfg = make_export_scene_cfg(tmp_path)
    robot = Usd.Stage.Open(cfg.robot.spawn.usd_path)
    hinge = robot.GetPrimAtPath("/Robot/Hinge")
    hinge.AddAppliedSchema("PhysxJointAPI")
    hinge.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(0.3)
    hinge.CreateAttribute("physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float).Set(120.0)
    robot.GetRootLayer().Save()
    cfg.robot.actuators["hinge"].armature = None
    cfg.robot.actuators["hinge"].joint_velocity_limit = None
    with build_simulation_context(sim_cfg=SimulationCfg(device="cpu", dt=1 / 120, physics=OvPhysxCfg())) as sim:
        scene = InteractiveScene(cfg)
        sim.reset()
        stage = Usd.Stage.Open(scene.export_to_usd(str(tmp_path / "exported.usda")))
        articulations, bodies = physics_prims(stage)
        live = readings(stage, read(sim.backend_manager.get_physx_instance(), articulations, bodies, "cpu"))
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        imported = pool.submit(read_import, stage.Flatten().ExportToString(), articulations, bodies).result()
    loaded = readings(stage, imported)

    key = "/World/envs/env_0/Robot/Hinge", "angular"
    assert live[key]["armature"] == pytest.approx(0.3)
    assert stage.GetPrimAtPath(key[0]).GetAttribute("physxJointAxis:angular:armature").Get() == pytest.approx(0.3)
    for name in ("armature", "max_velocity"):
        assert loaded[key][name] == pytest.approx(live[key][name], rel=1e-5), name


def test_a_stopped_simulation_reads_the_engine_again(tmp_path):
    """The engine read at startup is dropped when the simulation stops and taken again when it is ready."""
    from isaaclab_ov.physics import OvPhysxCfg

    from isaaclab.physics import PhysicsEvent
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from isaaclab.test.utils.usd_export import make_export_scene_cfg

    sim_cfg = SimulationCfg(device="cpu", dt=1 / 120, physics=OvPhysxCfg())
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(make_export_scene_cfg(tmp_path))
        sim.reset()
        first = scene._usd_export_base
        assert first is not None
        sim.backend_manager.dispatch_event(PhysicsEvent.STOP, payload={})
        assert scene._usd_export_base is None
        sim.backend_manager.dispatch_event(PhysicsEvent.PHYSICS_READY, payload={})
        assert scene._usd_export_base is not None and scene._usd_export_base is not first


def test_read_bindings_cover_ovphysx_properties():
    """Every OVPhysX articulation and rigid-body binding is compared by the round trip or deliberately excluded."""
    from isaaclab_ov import tensor_types as TT
    from usd_export_read_import import ARTICULATION_TENSORS, RIGID_BODY_TENSORS

    read = {*ARTICULATION_TENSORS.values(), *RIGID_BODY_TENSORS.values()}
    excluded = {
        # state, targets and forces are not part of the export
        TT.ROOT_POSE,
        TT.ROOT_VELOCITY,
        TT.LINK_POSE,
        TT.LINK_VELOCITY,
        TT.LINK_ACCELERATION,
        TT.DOF_POSITION,
        TT.DOF_VELOCITY,
        TT.DOF_POSITION_TARGET,
        TT.DOF_VELOCITY_TARGET,
        TT.DOF_ACTUATION_FORCE,
        TT.LINK_WRENCH,
        TT.RIGID_BODY_POSE,
        TT.RIGID_BODY_VELOCITY,
        TT.RIGID_BODY_ACCELERATION,
        TT.RIGID_BODY_WRENCH,
        # derived from exported properties
        TT.BODY_INV_MASS,
        TT.BODY_INV_INERTIA,
        TT.RIGID_BODY_INV_MASS,
        TT.RIGID_BODY_INV_INERTIA,
        TT.JACOBIAN,
        TT.MASS_MATRIX,
        TT.CORIOLIS,
        TT.GRAVITY_FORCE,
        TT.LINK_INCOMING_JOINT_FORCE,
        TT.DOF_PROJECTED_JOINT_FORCE,
        # drive type and model come from the USD drive schema and are not written at runtime
        TT.DOF_DRIVE_TYPE,
        TT.DOF_DRIVE_MODEL,
        # articulation shapes and tendons have no per-shape or per-tendon USD identity to write
        TT.CONTACT_OFFSET,
        TT.REST_OFFSET,
        TT.SHAPE_FRICTION_AND_RESTITUTION,
        TT.FIXED_TENDON_STIFFNESS,
        TT.FIXED_TENDON_DAMPING,
        TT.FIXED_TENDON_LIMIT_STIFFNESS,
        TT.FIXED_TENDON_LIMIT,
        TT.FIXED_TENDON_REST_LENGTH,
        TT.FIXED_TENDON_OFFSET,
        TT.SPATIAL_TENDON_STIFFNESS,
        TT.SPATIAL_TENDON_DAMPING,
        TT.SPATIAL_TENDON_LIMIT_STIFFNESS,
        TT.SPATIAL_TENDON_OFFSET,
    }
    bindings = {
        value
        for name, value in vars(TT).items()
        if name.isupper() and "DEFORMABLE" not in name and isinstance(value, type(TT.DOF_STIFFNESS))
    }
    unclassified = sorted(str(binding) for binding in bindings - read - excluded)
    assert not unclassified, f"Unclassified bindings: {unclassified}"
