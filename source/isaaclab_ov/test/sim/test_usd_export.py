# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export of a running OVPhysX articulation back to USD.

Mirrors ``isaaclab_physx/test/sim/test_usd_export.py``. Both backends share the authoring layer in
:mod:`isaaclab.sim.usd_export`; what differs, and what these tests cover, is how each recovers the
prim path of every body and joint. OVPhysX resolves them from the stage rather than from the view.

Runs kitless -- :class:`~isaaclab.sim.SimulationContext` is driven directly with
``physics=OvPhysxCfg()``, with no ``AppLauncher`` boot.
"""

from __future__ import annotations

import math

import pytest
import torch
from isaaclab_ov.assets import Articulation
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_ov.sim.usd_export import export_articulation_to_usd, resolve_articulation_prim_paths

from pxr import Usd, UsdPhysics

from isaaclab.sim import SimulationCfg, build_simulation_context

from isaaclab_assets import ANT_CFG, FRANKA_PANDA_CFG

# Values written into the simulation after the stage is parsed. They are deliberately unlike any
# plausible authored default, so a stage still carrying the spawn-time value is unmistakable.
OVERRIDE_STIFFNESS = 1234.5
OVERRIDE_DAMPING = 67.25
OVERRIDE_MASS = 9.75
# Static effort, dynamic effort and viscous coefficient of the friction model; distinct and non-zero
# so a component landing in the wrong attribute is caught.
OVERRIDE_FRICTION = (0.375, 0.25, 0.125)


@pytest.fixture
def sim(request):
    """Simulation context backed by OVPhysX rather than PhysX."""
    device = request.getfixturevalue("device")
    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), device=device, dt=1.0 / 60.0, gravity=(0.0, 0.0, 0.0))
    with build_simulation_context(device=device, sim_cfg=sim_cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        yield sim


def _spawn(sim) -> Articulation:
    """Spawn a single articulation whose stage authors drive gains and limits to be overridden."""
    articulation = Articulation(FRANKA_PANDA_CFG.replace(prim_path="/World/Robot"))
    sim.reset()
    assert articulation.is_initialized, "articulation failed to initialize; the test would be vacuous"
    return articulation


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_resolved_prim_paths_exist_and_cover_the_articulation(sim, device):
    """Every body and joint resolves to a real prim under the articulation root.

    OVPhysX does not record provenance on its view, so the paths are recovered from the stage. If
    that resolution drifts, the export silently describes a different prim -- or none.
    """
    articulation = _spawn(sim)
    paths = resolve_articulation_prim_paths(articulation)

    assert len(paths.bodies) == articulation.num_bodies
    assert len(paths.joints) == articulation.num_joints
    for path in paths.bodies + paths.joints:
        assert articulation.stage.GetPrimAtPath(path).IsValid(), f"resolved path {path} is not a prim"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_export_captures_overrides_the_stage_never_saw(sim, device, tmp_path):
    """Values written through the backend reach the exported stage.

    This is the property the exporter exists for. Drive gains and masses written after the stage is
    parsed live only in the backend's buffers, so an exporter that simply saved the stage would emit
    the spawn-time values and look correct while describing a simulation that is not running.
    """
    articulation = _spawn(sim)

    stiffness = torch.full_like(articulation.data.joint_stiffness.torch, OVERRIDE_STIFFNESS)
    damping = torch.full_like(articulation.data.joint_damping.torch, OVERRIDE_DAMPING)
    articulation.write_joint_stiffness_to_sim_index(stiffness=stiffness)
    articulation.write_joint_damping_to_sim_index(damping=damping)
    masses = articulation.data.body_mass.torch.clone()
    masses[:] = OVERRIDE_MASS
    articulation.set_masses_index(masses=masses)
    sim.step()
    articulation.update(sim.get_physics_dt())

    out = tmp_path / "exported.usda"
    export_articulation_to_usd(articulation, str(out))
    exported = Usd.Stage.Open(str(out))
    paths = resolve_articulation_prim_paths(articulation)

    checked = 0
    for path in paths.joints:
        prim = exported.GetPrimAtPath(path)
        assert prim.IsValid(), f"joint prim {path} missing from the exported stage"
        token = {"PhysicsPrismaticJoint": "linear", "PhysicsRevoluteJoint": "angular"}.get(prim.GetTypeName())
        if token is None:
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, token)
        assert drive, f"joint {path} carries no drive in the export"
        # Angular drive gains are per degree on the stage and per radian in the simulation.
        gain_scale = math.pi / 180.0 if token == "angular" else 1.0
        assert drive.GetStiffnessAttr().Get() == pytest.approx(OVERRIDE_STIFFNESS * gain_scale, rel=1e-4), (
            f"joint {path} exported the stage's spawn-time stiffness instead of the simulated one"
        )
        assert drive.GetDampingAttr().Get() == pytest.approx(OVERRIDE_DAMPING * gain_scale, rel=1e-4)
        checked += 1
    assert checked > 0, "fixture produced no drivable joints; the test would be vacuous"

    for path in paths.bodies:
        mass_api = UsdPhysics.MassAPI(exported.GetPrimAtPath(path))
        assert mass_api.GetMassAttr().Get() == pytest.approx(OVERRIDE_MASS, rel=1e-4), (
            f"body {path} exported the stage's spawn-time mass instead of the simulated one"
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_export_writes_joint_limits_in_stage_units(sim, device, tmp_path):
    """Joint limits are authored in the unit the joint schema declares, not the simulation's.

    The simulation reports revolute limits in radians while USD authors them in degrees, so an
    export that copies the number across unconverted narrows every limit by a factor of 57.
    """
    articulation = _spawn(sim)
    out = tmp_path / "exported.usda"
    export_articulation_to_usd(articulation, str(out))

    exported = Usd.Stage.Open(str(out))
    limits = articulation.data.joint_pos_limits.torch[0]
    joint_row = {name: index for index, name in enumerate(articulation.joint_names)}
    paths = resolve_articulation_prim_paths(articulation)

    checked = 0
    for backend_index, path in enumerate(paths.joints):
        prim = exported.GetPrimAtPath(path)
        if prim.GetTypeName() != "PhysicsRevoluteJoint":
            continue
        row = joint_row.get(articulation.backend_joint_names[backend_index])
        if row is None:
            continue
        lower = prim.GetAttribute("physics:lowerLimit").Get()
        if lower is None or not math.isfinite(float(limits[row][0])):
            continue
        assert lower == pytest.approx(math.degrees(float(limits[row][0])), abs=1e-3), (
            f"joint {path} exported its lower limit in radians rather than degrees"
        )
        checked += 1
    assert checked > 0, "fixture produced no revolute joints with finite limits; the test would be vacuous"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_export_rejects_an_environment_the_view_does_not_have(sim, device, tmp_path):
    """Selecting a missing environment fails instead of exporting another one's state."""
    articulation = _spawn(sim)
    with pytest.raises(ValueError, match="out of range"):
        export_articulation_to_usd(articulation, str(tmp_path / "out.usda"), env_index=99)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_exported_armature_is_readable_by_physx(sim, device, tmp_path):
    """Armature is authored where the solver reads it, since UsdPhysics has no home for it."""
    articulation = _spawn(sim)
    out = tmp_path / "exported.usda"
    export_articulation_to_usd(articulation, str(out))

    exported = Usd.Stage.Open(str(out))
    armature = articulation.data.joint_armature.torch[0]
    joint_row = {name: index for index, name in enumerate(articulation.joint_names)}
    paths = resolve_articulation_prim_paths(articulation)

    checked = 0
    for backend_index, path in enumerate(paths.joints):
        row = joint_row.get(articulation.backend_joint_names[backend_index])
        if row is None:
            continue
        prim = exported.GetPrimAtPath(path)
        attribute = prim.GetAttribute("physxJoint:armature")
        assert attribute, f"joint {path} carries no physxJoint:armature in the export"
        assert attribute.Get() == pytest.approx(float(armature[row]), abs=1e-5)
        # the per-axis schema, applied for friction, shadows the joint-level value once present
        token = {"PhysicsPrismaticJoint": "linear", "PhysicsRevoluteJoint": "angular"}.get(prim.GetTypeName())
        if token is not None:
            axis_attribute = prim.GetAttribute(f"physxJointAxis:{token}:armature")
            assert axis_attribute, f"joint {path} carries no physxJointAxis:{token}:armature in the export"
            assert axis_attribute.Get() == pytest.approx(float(armature[row]), abs=1e-5)
        checked += 1
    assert checked > 0, "fixture produced no joints; the test would be vacuous"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_exported_friction_is_authored_on_the_drive_axis(sim, device, tmp_path):
    """Friction lands on the drive axis' per-axis PhysX schema, the only place the runtimes read it from.

    The legacy ``physxJoint:jointFriction`` scalar parses into nothing on current runtimes, so a stage
    carrying only that reimports with zero friction.
    """
    articulation = _spawn(sim)
    static, dynamic, viscous = (
        torch.full_like(articulation.data.joint_friction_coeff.torch, value) for value in OVERRIDE_FRICTION
    )
    articulation.write_joint_friction_coefficient_to_sim_index(
        joint_friction_coeff=static, joint_dynamic_friction_coeff=dynamic, joint_viscous_friction_coeff=viscous
    )
    sim.step()
    articulation.update(sim.get_physics_dt())

    out = tmp_path / "exported.usda"
    export_articulation_to_usd(articulation, str(out))
    exported = Usd.Stage.Open(str(out))
    paths = resolve_articulation_prim_paths(articulation)

    names = ("staticFrictionEffort", "dynamicFrictionEffort", "viscousFrictionCoefficient")
    checked = 0
    for path in paths.joints:
        prim = exported.GetPrimAtPath(path)
        token = {"PhysicsPrismaticJoint": "linear", "PhysicsRevoluteJoint": "angular"}.get(prim.GetTypeName())
        if token is None:
            continue
        assert f"PhysxJointAxisAPI:{token}" in prim.GetAppliedSchemas(), f"joint {path} lacks PhysxJointAxisAPI:{token}"
        # the viscous coefficient is per unit angular rate, so it follows the drive gains onto degrees
        gain_scale = math.pi / 180.0 if token == "angular" else 1.0
        expected = (*OVERRIDE_FRICTION[:2], OVERRIDE_FRICTION[2] * gain_scale)
        for name, value in zip(names, expected):
            attribute = prim.GetAttribute(f"physxJointAxis:{token}:{name}")
            assert attribute, f"joint {path} carries no physxJointAxis:{token}:{name} in the export"
            assert attribute.Get() == pytest.approx(value, abs=1e-5)
        checked += 1
    assert checked > 0, "fixture produced no drivable joints; the test would be vacuous"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_export_leaves_the_live_stage_untouched(sim, device, tmp_path):
    """The export authors onto a flattened snapshot, never onto the stage the simulation reads.

    On PhysX, applying a schema to a prim that is an articulation root invalidates every articulation
    view on the stage for the rest of the session, so an in-place export would break the very
    simulation it describes. The snapshot carries the new schema; the live prim does not.
    """
    articulation = _spawn(sim)
    out = tmp_path / "exported.usda"
    export_articulation_to_usd(articulation, str(out))
    exported = Usd.Stage.Open(str(out))

    checked = 0
    for path in resolve_articulation_prim_paths(articulation).joints:
        live = articulation.stage.GetPrimAtPath(path)
        token = {"PhysicsPrismaticJoint": "linear", "PhysicsRevoluteJoint": "angular"}.get(live.GetTypeName())
        if token is None:
            continue
        schema = f"PhysxJointAxisAPI:{token}"
        assert schema in exported.GetPrimAtPath(path).GetAppliedSchemas(), f"snapshot of {path} lacks {schema}"
        assert schema not in live.GetAppliedSchemas(), f"live prim {path} was edited by the export"
        checked += 1
    assert checked > 0, "fixture produced no drivable joints; the test would be vacuous"
    assert articulation.root_view is not None and list(articulation.joint_names), "view unreadable after export"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_resolves_floating_base_with_colliding_body_and_joint_names(sim, device):
    """A floating-base asset whose bodies and joints share names resolves to the right prims.

    Ant carries ``ArticulationRootAPI`` on its torso link, so the other links are siblings of the
    matched root rather than descendants, and it names a joint after each leg body. Anchoring on the
    root prim finds nothing; a single name index sends joint writes onto the body. Both are silent
    on Franka, whose root is the top-level Xform and whose names are unique.
    """
    articulation = Articulation(ANT_CFG.replace(prim_path="/World/Robot"))
    sim.reset()
    assert articulation.is_initialized, "articulation failed to initialize; the test would be vacuous"
    body_names = set(articulation.backend_body_names)
    joint_names = set(articulation.backend_joint_names)
    assert body_names & joint_names, "fixture has no body/joint name collision; the test would be vacuous"

    paths = resolve_articulation_prim_paths(articulation)
    stage = articulation.stage
    assert len(paths.bodies) == articulation.num_bodies and len(paths.joints) == articulation.num_joints
    for path in paths.joints:
        assert stage.GetPrimAtPath(path).IsA(UsdPhysics.Joint), f"{path} resolved for a joint is not a joint prim"
    for path in paths.bodies:
        assert not stage.GetPrimAtPath(path).IsA(UsdPhysics.Joint), f"{path} resolved for a body is a joint prim"
    assert len(set(paths.bodies) | set(paths.joints)) == len(paths.bodies) + len(paths.joints), (
        "a prim was resolved twice"
    )


def test_complete_environment_round_trip_cpu(tmp_path):
    """Reload two articulations and two rigid objects without applying the task's overrides again."""
    import numpy as np
    import ovphysx
    import ovstage
    from isaaclab_ov import tensor_types as TT
    from isaaclab_ov.physics import OvPhysxManager
    from isaaclab_ov.sim.views import OvPhysxView
    from isaaclab_ov.stage import create_ovstage

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import export_environment_to_usd
    from isaaclab.sim.usd_export import create_environment_snapshot
    from isaaclab.test.utils.usd_export import assert_physics_structure_equal, capture_physics_structure

    cfg = InteractiveSceneCfg(num_envs=2, env_spacing=4.0)
    cfg.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.other_robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/OtherRobot")
    cfg.other_robot.init_state.pos = (1.0, 0.0, 0.0)
    for name, position in (("box", (0.0, 1.0, 1.0)), ("ball", (1.0, 1.0, 1.0))):
        spawn = sim_utils.CuboidCfg(size=(0.2, 0.3, 0.4)) if name == "box" else sim_utils.SphereCfg(radius=0.15)
        spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg()
        spawn.mass_props = sim_utils.MassPropertiesCfg(mass=0.5)
        spawn.collision_props = sim_utils.CollisionPropertiesCfg()
        setattr(
            cfg,
            name,
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name}",
                spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=position),
            ),
        )
    cfg.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(size=(1.0, 1.0, 0.1), collision_props=sim_utils.CollisionPropertiesCfg()),
    )
    cfg.ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    cfg.collection = RigidObjectCollectionCfg(
        rigid_objects={
            name: RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Collected{name}",
                spawn=sim_utils.SphereCfg(
                    radius=0.1,
                    rigid_props=sim_utils.RigidBodyBaseCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionBaseCfg(),
                ),
            )
            for name in ("First", "Second")
        }
    )
    output = tmp_path / "environment.usda"
    expected = {}
    identities = {}
    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), device="cpu", gravity=(0.0, 0.0, -9.81))
    with build_simulation_context(device="cpu", sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(cfg)
        sim.reset()
        scene.update(0.0)
        for number, (name, asset) in enumerate({**scene.articulations, **scene.rigid_objects}.items()):
            is_art = name in scene.articulations
            view = asset.root_view
            mass = asset.data.body_mass.torch.clone()
            mass[0] = 2.0 + number
            mass[1] = 7.0 + number
            asset.set_masses_index(masses=mass)
            inertia = asset.data.body_inertia.torch.clone()
            inertia[0] *= 1.2
            inertia[1] *= 2.3
            asset.set_inertias_index(inertias=inertia)
            com = asset.data.body_com_pose_b.torch.clone()
            com[1, :, :3] += torch.tensor([0.01, -0.02, 0.03])
            asset.set_coms_index(coms=com)
            if is_art:
                asset.write_joint_stiffness_to_sim_index(
                    stiffness=torch.full_like(asset.data.joint_stiffness.torch, 432.1)
                )
                asset.write_joint_damping_to_sim_index(damping=torch.full_like(asset.data.joint_damping.torch, 12.3))
            prefix = "ARTICULATION" if is_art else "RIGID_BODY"
            for suffix, value in (
                ("SHAPE_FRICTION_AND_RESTITUTION", 0.31),
                ("CONTACT_OFFSET", 0.025),
                ("REST_OFFSET", 0.003),
            ):
                token = getattr(TT.TensorType, f"{prefix}_{suffix}")
                buffer = view.get_attribute(token)
                values = buffer.numpy()
                values[0] = value * 0.5
                values[1] = value
                buffer.assign(values)
                view.set_attribute(token, buffer)
            token = TT.BODY_DISABLE_GRAVITY if is_art else TT.RIGID_BODY_DISABLE_GRAVITY
            buffer = view.get_attribute(token)
            values = buffer.numpy()
            values[1] = 1
            buffer.assign(values)
            view.set_attribute(token, buffer)
            properties = (
                [
                    TT.BODY_MASS,
                    TT.BODY_INERTIA,
                    TT.BODY_COM_POSE,
                    TT.BODY_DISABLE_GRAVITY,
                    TT.DOF_STIFFNESS,
                    TT.DOF_DAMPING,
                    TT.DOF_LIMIT,
                    TT.DOF_MAX_VELOCITY,
                    TT.DOF_MAX_FORCE,
                    TT.DOF_ARMATURE,
                    TT.DOF_FRICTION_PROPERTIES,
                    TT.SHAPE_FRICTION_AND_RESTITUTION,
                    TT.CONTACT_OFFSET,
                    TT.REST_OFFSET,
                ]
                if is_art
                else [
                    TT.RIGID_BODY_MASS,
                    TT.RIGID_BODY_INERTIA,
                    TT.RIGID_BODY_COM_POSE,
                    TT.RIGID_BODY_DISABLE_GRAVITY,
                    TT.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION,
                    TT.RIGID_BODY_CONTACT_OFFSET,
                    TT.RIGID_BODY_REST_OFFSET,
                ]
            )
            if is_art:
                identities[view.prim_paths[1]] = (view.body_names, view.dof_names)
            expected[view.prim_paths[1]] = (
                is_art,
                {token: view.get_attribute(token).numpy()[1].copy() for token in properties},
            )
        collection = scene.rigid_object_collections["collection"]
        import warp as wp

        collection.set_masses_index(masses=wp.array([[2.0, 3.0], [7.0, 11.0]], dtype=wp.float32, device="cpu"))
        view = collection.root_view
        properties = [
            TT.RIGID_BODY_MASS,
            TT.RIGID_BODY_INERTIA,
            TT.RIGID_BODY_COM_POSE,
            TT.RIGID_BODY_DISABLE_GRAVITY,
            TT.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION,
            TT.RIGID_BODY_CONTACT_OFFSET,
            TT.RIGID_BODY_REST_OFFSET,
        ]
        for row, path in enumerate(view.prim_paths):
            if "/env_1/" in path:
                expected[path] = (False, {token: view.get_attribute(token).numpy()[row].copy() for token in properties})
        OvPhysxManager.set_gravity((0.25, -0.5, -3.0))
        before = sim.stage.GetRootLayer().ExportToString()
        expected_structure = capture_physics_structure(create_environment_snapshot(scene, 1))
        export_environment_to_usd(scene, str(output), env_index=1)
        assert sim.stage.GetRootLayer().ExportToString() == before
    stage = Usd.Stage.Open(str(output))
    assert_physics_structure_equal(expected_structure, capture_physics_structure(stage))
    assert not stage.GetPrimAtPath("/World/envs/env_0")
    assert stage.GetPrimAtPath("/World/envs/env_1/Table")
    assert stage.GetPrimAtPath("/World/Ground")
    assert len([p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]) == 2
    # Check every relationship and shader connection, not just named scalar attributes.
    for prim in stage.Traverse():
        for prop in prim.GetProperties():
            targets = prop.GetTargets() if isinstance(prop, Usd.Relationship) else prop.GetConnections()
            assert all(stage.GetPrimAtPath(p.GetPrimPath()) for p in targets), prop.GetPath()
    gravity = UsdPhysics.Scene(next(p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)))
    np.testing.assert_allclose(
        np.array(gravity.GetGravityDirectionAttr().Get()) * gravity.GetGravityMagnitudeAttr().Get(),
        [0.25, -0.5, -3.0],
        rtol=1e-6,
    )
    # Direct runtime import: no SimulationCfg, asset spawn config, actuator config or event terms.
    fresh_stage = create_ovstage("export_round_trip")
    fresh = ovphysx.PhysX()
    try:
        ovstage.population.open_usd_from_string(
            fresh_stage, stage.ExportToString(), ordinal=1, domains=ovstage.PopulationDomain.ALL
        )
        fresh_stage.advance_write_floor(ordinal=1).wait()
        fresh.attach_ovstage(fresh_stage, read_ordinal=1)
        for path, (is_art, properties) in expected.items():
            view = OvPhysxView(fresh, prim_paths=[path], device="cpu", tensor_types=list(properties), eager=True)
            try:
                assert view.count == 1
                for token, value in properties.items():
                    actual = view.get_attribute(token).numpy()[0]
                    if is_art and token.name.startswith(("ARTICULATION_BODY_", "ARTICULATION_DOF_")):
                        is_body = token.name.startswith("ARTICULATION_BODY_")
                        names = view.body_names if is_body else view.dof_names
                        original = identities[path][0 if is_body else 1]
                        assert set(names) == set(original)
                        actual = actual[[names.index(name) for name in original]]
                    if token in (TT.BODY_COM_POSE, TT.RIGID_BODY_COM_POSE):
                        # Principal axes may be permuted or sign-flipped. The full inertia tensor
                        # above compares the physical orientation independently of that choice.
                        actual, value = actual[..., :3], value[..., :3]
                    if np.asarray(value).dtype.kind in "biu":
                        np.testing.assert_array_equal(actual, value, err_msg=f"{path}: {token.name}")
                    else:
                        np.testing.assert_allclose(actual, value, rtol=3e-4, atol=1e-5, err_msg=f"{path}: {token.name}")
            finally:
                view.close()
        for path in expected:
            with pytest.raises(OvPhysxView.AttributeUnavailable):
                OvPhysxView(
                    fresh,
                    prim_paths=[path.replace("env_1", "env_0")],
                    device="cpu",
                    tensor_types=[TT.BODY_MASS if expected[path][0] else TT.RIGID_BODY_MASS],
                    eager=True,
                )
    finally:
        fresh.release()
        fresh_stage.destroy()
