# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher
from isaaclab.test.utils import DeviceScope, resolve_test_sim_device, test_devices

simulation_app = AppLauncher(headless=True, device=resolve_test_sim_device()).app

"""Rest everything follows."""

import math

import pytest

from pxr import Usd, UsdPhysics


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("native_actuator", [False, True])
def test_fixed_configuration_round_trip_in_isaac_sim(device, tmp_path, native_actuator):
    """Load the complete fixed scene into fresh Isaac Sim without reconstructing its task cfg."""
    import numpy as np
    from isaaclab_physx.physics import PhysxCfg

    from pxr import Gf, Sdf, UsdGeom

    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from isaaclab.test.utils.usd_export import (
        assert_physics_structure_equal,
        make_fixed_scene_cfg,
        read_physics_structure,
    )

    cfg = make_fixed_scene_cfg(tmp_path)
    # A physical link can carry mass and state without owning collision geometry.
    source = Usd.Stage.Open(cfg.robot.spawn.usd_path)
    source.RemovePrim("/Robot/Base/Collision")
    source.GetRootLayer().Save()
    if native_actuator:
        from isaaclab.actuators import IdealPDActuatorCfg

        cfg.robot.actuators["hinge"] = IdealPDActuatorCfg(
            joint_names_expr=["Hinge"],
            stiffness=83.0,
            damping=4.5,
            armature=0.023,
            joint_effort_limit=19.0,
            joint_velocity_limit=2.5,
            actuator_effort_limit=17.0,
            actuator_velocity_limit=2.0,
        )
    sim_cfg = SimulationCfg(
        device=device,
        dt=1 / 120,
        gravity=(0.2, -0.1, -4.0),
        physics=PhysxCfg(bounce_threshold_velocity=0.31, friction_offset_threshold=0.025),
    )
    output = tmp_path / "fixed_environment.usda"
    expected = {}
    expected_structure = {}
    authored = {}
    getter_names = (
        "get_masses",
        "get_inertias",
        "get_coms",
        "get_disable_gravities",
        "get_material_properties",
        "get_contact_offsets",
        "get_rest_offsets",
    )
    joint_getters = (
        "get_dof_stiffnesses",
        "get_dof_dampings",
        "get_dof_limits",
        "get_dof_max_forces",
        "get_dof_max_velocities",
        "get_dof_armatures",
        "get_dof_friction_properties",
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(cfg)
        sim.reset()
        scene.reset_to_default()
        sim.forward()
        scene.update(0.0)
        # Independent reference: the live backend and complete original USD, not exporter selection.
        expected_structure.update(read_physics_structure(scene.sim.stage))
        expected_structure["/World/envs/env_0/Robot/FixedRoot", "localPose0Position"] = np.asarray(
            cfg.robot.init_state.pos
        )
        for prim in scene.sim.stage.Traverse():
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI) and not prim.IsA(UsdPhysics.Joint):
                authored[str(prim.GetPath())] = {prop.GetName(): prop.Get() for prop in prim.GetAuthoredAttributes()}
        assets = {**scene.articulations, **scene.rigid_objects, **scene.rigid_object_collections}
        for name, asset in assets.items():
            view = asset.root_view
            articulation = name in scene.articulations
            getters = getter_names + joint_getters if articulation else getter_names
            for row, root in enumerate(view.prim_paths):
                expected[root] = (
                    articulation,
                    list(view.link_paths[row]) if articulation else [root],
                    list(view.dof_paths[row]) if articulation else [],
                    {getter: getattr(view, getter)().numpy()[row].copy() for getter in getters},
                )
        before = scene.sim.stage.GetRootLayer().ExportToString()
        scene.export_to_usd(str(output), preserve_source_contacts=True)
        assert scene.sim.stage.GetRootLayer().ExportToString() == before
    stage = Usd.Stage.Open(str(output))
    assert_physics_structure_equal(expected_structure, read_physics_structure(stage))
    bodies = {str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
    assert bodies == {
        "/World/envs/env_0/Robot/Base",
        "/World/envs/env_0/Robot/Link",
        "/World/envs/env_0/Box",
        "/World/envs/env_0/CollectedFirst",
        "/World/envs/env_0/CollectedSecond",
    }
    for path, attributes in authored.items():
        prim = stage.GetPrimAtPath(path)
        assert prim, path
        for name, value in attributes.items():
            assert prim.GetAttribute(name).Get() == value, (path, name)
    hinge = stage.GetPrimAtPath("/World/envs/env_0/Robot/Hinge")
    expected_gain = 0.0 if native_actuator else 83 * math.pi / 180
    assert UsdPhysics.DriveAPI(hinge, "angular").GetStiffnessAttr().Get() == pytest.approx(expected_gain)
    native_prims = [prim for prim in stage.Traverse() if prim.GetTypeName() == "NewtonActuator"]
    assert len(native_prims) == int(native_actuator)
    if native_actuator:
        native_prim = native_prims[0]
        assert native_prim.GetAttribute("newton:kp").Get() == pytest.approx(83)
        assert native_prim.GetAttribute("newton:kd").Get() == pytest.approx(4.5)
        assert native_prim.GetRelationship("newton:targets").GetTargets() == [hinge.GetPath()]
    assert not hinge.GetAttribute("state:angular:physics:position").HasAuthoredValueOpinion()
    for name, mass, x in (("Box", 2.5, 1), ("CollectedFirst", 1.5, 2), ("CollectedSecond", 3.5, 3)):
        prim = stage.GetPrimAtPath(f"/World/envs/env_0/{name}")
        assert UsdPhysics.MassAPI(prim).GetMassAttr().Get() == pytest.approx(mass, rel=1e-6, abs=1e-7)
        pose = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        np.testing.assert_allclose(pose.ExtractTranslation(), (x, 0, 1), atol=1e-6)
        np.testing.assert_allclose(UsdPhysics.RigidBodyAPI(prim).GetVelocityAttr().Get(), (0, 0, 0))
    physics = stage.GetPrimAtPath("/physicsScene")
    assert physics.GetAttribute("physxScene:bounceThreshold").Get() == pytest.approx(0.31)
    assert physics.GetAttribute("physxScene:frictionOffsetThreshold").Get() == pytest.approx(0.025)
    # The fresh driver gets its rate from the file, never from the original SimulationCfg.
    dt = 1 / physics.GetAttribute("physxScene:timeStepsPerSecond").Get()
    with build_simulation_context(device=device, dt=dt) as fresh:
        fresh.stage.GetRootLayer().TransferContent(Sdf.Layer.FindOrOpen(str(output)))
        fresh.reset()
        native = fresh.physics_sim_view
        np.testing.assert_allclose(native.get_gravity(), (0.2, -0.1, -4), rtol=1e-6)
        assert fresh.get_physics_dt() == pytest.approx(1 / 120)
        for path, (articulation, body_paths, joint_paths, properties) in expected.items():
            view = native.create_articulation_view(path) if articulation else native.create_rigid_body_view(path)
            assert view.count == 1, path
            for name, value in properties.items():
                actual = getattr(view, name)().numpy()[0]
                if articulation and name not in {"get_material_properties", "get_contact_offsets", "get_rest_offsets"}:
                    paths = list(view.dof_paths[0]) if name.startswith("get_dof_") else list(view.link_paths[0])
                    reference = joint_paths if name.startswith("get_dof_") else body_paths
                    assert set(paths) == set(reference)
                    actual = actual[[paths.index(p) for p in reference]]
                if name == "get_coms":
                    # Compare COM orientation as a rotation, not a quaternion sign convention.
                    for a, b in zip(actual.reshape(-1, 7), value.reshape(-1, 7)):
                        np.testing.assert_allclose(a[:3], b[:3], atol=1e-6)
                        qa = Gf.Quatd(float(a[6]), Gf.Vec3d(*map(float, a[3:6])))
                        qb = Gf.Quatd(float(b[6]), Gf.Vec3d(*map(float, b[3:6])))
                        np.testing.assert_allclose(Gf.Matrix3d(qa), Gf.Matrix3d(qb), atol=1e-5)
                else:
                    np.testing.assert_allclose(actual, value, rtol=3e-4, atol=1e-5, err_msg=f"{path}: {name}")
