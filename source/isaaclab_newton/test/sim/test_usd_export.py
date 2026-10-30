# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton USD export: loading the exported file reproduces the initialized model."""

import newton
import numpy as np
import pytest
import warp as wp
from newton.usd import PrimType, SchemaResolverNewton

from pxr import Gf, Sdf, Usd, UsdPhysics, UsdShade


def _export(
    tmp_path,
    solver_cfg,
    device,
    hinge_actuator=None,
    num_substeps=1,
    edit_robot=None,
    edit_cfg=None,
    before_export=None,
    default_shape_cfg=None,
    world_hook=None,
):
    """Initialize the fixture scene, export it, and capture the live model and its rollout for comparison.

    ``edit_robot`` edits the robot's USD and ``edit_cfg`` the scene configuration before the scene is built;
    ``world_hook`` changes each world's Newton builder before it is finalized; ``before_export`` changes the
    initialized scene.
    """
    import contextlib

    from isaaclab_newton.cloner import newton_builder_world_hook
    from isaaclab_newton.physics import NewtonCfg

    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from isaaclab.test.utils.usd_export import make_export_scene_cfg

    cfg = make_export_scene_cfg(tmp_path)
    if edit_robot is not None:
        robot = Usd.Stage.Open(cfg.robot.spawn.usd_path)
        edit_robot(robot)
        robot.GetRootLayer().Save()
    if hinge_actuator is not None:
        cfg.robot.actuators = {"hinge": hinge_actuator}
    if edit_cfg is not None:
        edit_cfg(cfg)
    sim_cfg = SimulationCfg(
        device=device,
        dt=1 / 120,
        gravity=(0.2, -0.1, -4.0),
        physics=NewtonCfg(solver_cfg=solver_cfg, num_substeps=num_substeps),
    )
    if default_shape_cfg is not None:
        sim_cfg.physics.default_shape_cfg = default_shape_cfg
    output = str(tmp_path / "exported.usda")
    hook = newton_builder_world_hook(world_hook) if world_hook is not None else contextlib.nullcontext()
    with build_simulation_context(sim_cfg=sim_cfg) as sim, hook:
        scene = InteractiveScene(cfg)
        sim.reset()
        if before_export is not None:
            before_export(scene)
        scene.export_to_usd(output)
        # Isaac Lab's reset writes the initial joint state, which the engine holds only from then on.
        robot = scene["robot"]
        robot.write_joint_state_to_sim_index(
            position=robot.data.default_joint_pos.torch, velocity=robot.data.default_joint_vel.torch
        )
        manager = sim.backend_manager
        model, state = manager.backend.model, manager.backend.state_0
        live = _capture(model, state.body_q, state.joint_q, state.joint_qd)
        solver_type = type(manager.backend.solver)
        # The engine's own construction of the solver, stepped from the reset state.
        solver = solver_cfg.class_type.build(solver_cfg, model, wp.DeterministicMode.NOT_GUARANTEED)
        motion = _rollout(model, solver, state.joint_q, state.joint_qd)
    return output, live, solver_type, motion


def _rollout(model, solver, joint_q=None, joint_qd=None, steps=60, dt=1 / 120):
    """Body positions after stepping from a joint state (default: the model's) without control input, by body label."""
    state_0, state_1, control = model.state(), model.state(), model.control()
    if joint_q is not None:
        state_0.joint_q.assign(joint_q)
        state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
    return dict(zip(model.body_label, state_0.body_q.numpy()[:, :3]))


def _newton_import(output, solver, device):
    """Newton's own USD import of the exported file, as a consumer without Isaac Lab does it."""
    from newton.usd import SchemaResolverMjc, SchemaResolverPhysx

    builder = newton.ModelBuilder()
    resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]
    if solver == "mujoco":
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        resolvers.append(SchemaResolverMjc())
    builder.add_usd(Usd.Stage.Open(output), schema_resolvers=resolvers)
    return builder.finalize(device=device)


def _capture(model, body_q, joint_q=None, joint_qd=None):
    """Every float model array, and the joint target modes, of world 0's bodies, shapes, joints, DOFs and coordinates.

    Keyed by attribute name and entity label; a DOF or coordinate is labelled ``<joint label>:<index in the
    joint>``. Root body poses come from ``body_q`` and the joint state from ``joint_q`` and ``joint_qd`` (default:
    the model's); other poses follow the joint state. The world's gravity is keyed ``("gravity", "")``. Model
    values the export does not carry show up as differences.
    """
    frequency = newton.Model.AttributeFrequency
    labels = {
        frequency.BODY: list(model.body_label),
        frequency.SHAPE: list(model.shape_label),
        frequency.JOINT: list(model.joint_label),
        frequency.JOINT_DOF: [None] * model.joint_dof_count,
        frequency.JOINT_COORD: [None] * model.joint_coord_count,
    }
    worlds = {
        frequency.BODY: model.body_world.numpy(),
        frequency.SHAPE: model.shape_world.numpy(),
        frequency.JOINT: model.joint_world.numpy(),
        frequency.JOINT_DOF: np.empty(model.joint_dof_count, dtype=int),
        frequency.JOINT_COORD: np.empty(model.joint_coord_count, dtype=int),
    }
    types = model.joint_type.numpy()
    # Free joints of floating bodies are generated, under generated names; their bodies carry the pose.
    worlds[frequency.JOINT][types == int(newton.JointType.FREE)] = -2
    for entity, starts in ((frequency.JOINT_DOF, model.joint_qd_start), (frequency.JOINT_COORD, model.joint_q_start)):
        starts = starts.numpy()
        for joint, label in enumerate(model.joint_label):
            for axis, index in enumerate(range(starts[joint], starts[joint + 1])):
                labels[entity][index] = f"{label}:{axis}"
                worlds[entity][index] = worlds[frequency.JOINT][joint]
    state = {"body_q": body_q, "joint_q": joint_q, "joint_qd": joint_qd}
    parents, children = model.joint_parent.numpy(), model.joint_child.numpy()
    roots = {model.body_label[child] for parent, child in zip(parents, children) if parent == -1}
    values = {}
    for name, entity in model.attribute_frequency.items():
        namespace, _, attribute = name.rpartition(":")
        array = getattr(getattr(model, namespace) if namespace else model, attribute, None)
        if entity not in labels or not isinstance(array, wp.array):
            continue
        array = (array if state.get(name) is None else state[name]).numpy()
        if array.dtype.kind != "f" and name != "joint_target_mode":
            continue
        for index in np.flatnonzero(np.isin(worlds[entity], (-1, 0))):
            if name != "body_q" or labels[entity][index] in roots:
                values[name, labels[entity][index]] = array[index]
    gravity = model.gravity.numpy()
    values["gravity", ""] = gravity[0] if gravity.ndim == 2 else gravity
    return values


def _assert_pose_equal(actual, expected):
    np.testing.assert_allclose(actual[:3], expected[:3], atol=1e-5)
    sign = 1.0 if np.dot(actual[3:], expected[3:]) >= 0 else -1.0
    np.testing.assert_allclose(actual[3:], sign * expected[3:], atol=1e-5)


@pytest.mark.parametrize("solver,device", [("xpbd", "cpu"), ("mujoco", "cuda:0")])
def test_importing_the_export_reproduces_the_initialized_model(tmp_path, caplog, solver, device):
    """Newton's own import of the file reproduces every live model value, including config overrides."""
    from isaaclab_newton.physics import MJWarpSolverCfg, XPBDSolverCfg

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("MuJoCo Warp requires CUDA.")
    solver_cfg = (
        XPBDSolverCfg(iterations=7) if solver == "xpbd" else MJWarpSolverCfg(iterations=17, enable_multiccd=True)
    )
    output, live, _, motion = _export(tmp_path, solver_cfg, device)

    model = _newton_import(output, solver, device)
    loaded = _capture(model, model.body_q)
    if solver == "mujoco":
        # MuJoCo reads its options from the physics scene, so the default solver runs the configured one.
        imported_solver = newton.solvers.SolverMuJoCo(model)
        assert imported_solver.mjw_model.opt.iterations == 17
        # Configuration flags have their own USD names; disable flags are stated inverted.
        assert bool(model.mujoco.flag_multiccd.numpy()[0]) and bool(model.mujoco.flag_contact.numpy()[0])
        # Options equal to Newton's import defaults are left to the importer.
        stage = Usd.Stage.Open(output)
        scene = next(prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene))
        assert scene.GetAttribute("mjc:option:iterations").HasAuthoredValue()
        assert not scene.GetAttribute("mjc:option:cone").HasAuthoredValue()
        assert not scene.GetAttribute("mjc:flag:contact").HasAuthoredValue()
        moved = _rollout(model, imported_solver)
        for label, position in motion.items():
            np.testing.assert_allclose(moved[label], position, atol=1e-4, err_msg=label)
    # The actuator configuration differs from the USD drive, and the configured initial state from the USD's (zero),
    # so the export must carry them.
    hinge = "/World/envs/env_0/Robot/Hinge:0"
    assert live["joint_q", hinge] == pytest.approx(0.21)
    assert live["joint_target_ke", hinge] == pytest.approx(83.0)
    assert live["joint_armature", hinge] == pytest.approx(0.023)
    assert set(live) == set(loaded), (sorted(set(live) - set(loaded))[:8], sorted(set(loaded) - set(live))[:8])
    assert live["joint_target_mode", hinge] == int(newton.JointTargetMode.POSITION_VELOCITY)
    if solver != "mujoco":
        # The import infers position mode, which XPBD does not apply, so the export does not warn about it.
        assert "target mode" not in caplog.text
        assert "USD does not state the settings of XPBDSolverCfg" in caplog.text
        del live["joint_target_mode", hinge]
    for (name, label), value in live.items():
        if name in ("body_q", "joint_X_p", "joint_X_c"):
            _assert_pose_equal(loaded[name, label], value)
        else:
            np.testing.assert_allclose(loaded[name, label], value, rtol=1e-4, atol=1e-7, err_msg=f"{label} {name}")


def test_importing_an_exported_cable_reproduces_it_but_its_segment_gaps(tmp_path, caplog):
    """Newton's import builds a cable from its USD curve; the segments it generates take its own contact gap."""
    from isaaclab_newton.physics import NewtonCfg, VBDSolverCfg
    from newton.usd import SchemaResolverPhysx

    from isaaclab.assets import CableObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, UsdPhysicsCollisionCfg, build_simulation_context
    from isaaclab.sim.spawners.materials import CableMaterialCfg
    from isaaclab.sim.spawners.shapes import CableCfg

    cfg = InteractiveSceneCfg(num_envs=1, env_spacing=3.0)
    cfg.cable = CableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cable",
        spawn=CableCfg(
            positions=[(0.1 * index, 0.0, 0.5) for index in range(4)],
            physics_material=CableMaterialCfg(thickness=0.01, density=100.0, stretch_stiffness=1e6, bend_stiffness=1e3),
            collision_props=UsdPhysicsCollisionCfg(collision_enabled=True),
        ),
    )
    sim_cfg = SimulationCfg(device="cpu", dt=1 / 120, physics=NewtonCfg(solver_cfg=VBDSolverCfg()))
    output = str(tmp_path / "exported.usda")
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(cfg)
        sim.reset()
        scene.export_to_usd(output)
        live = _capture(sim.backend_manager.backend.model, sim.backend_manager.backend.state_0.body_q)
    builder = newton.ModelBuilder()
    newton.solvers.SolverVBD.register_custom_attributes(builder)
    builder.add_usd(Usd.Stage.Open(output), schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()])
    model = builder.finalize(device="cpu")
    loaded = _capture(model, model.body_q)

    assert set(live) == set(loaded)
    differ = set()
    for key, value in live.items():
        if key[0] in ("body_q", "joint_X_p", "joint_X_c"):
            _assert_pose_equal(loaded[key], value)
        elif not np.allclose(loaded[key], value, rtol=1e-4, atol=1e-7):
            differ.add(key)
    segments = {label for name, label in live if name == "shape_gap"}
    assert len(segments) == 3
    assert differ == {("shape_gap", label) for label in segments}
    assert "contact values" in caplog.text


@pytest.mark.parametrize("mass", [None, 4.0], ids=["unchanged", "changed mass"])
def test_inertia_newton_derives_on_import_is_not_written(tmp_path, mass):
    """Inertia Newton's import corrects (here, moments breaking the triangle inequality) is left to the import, also
    when Isaac Lab changes the body's mass."""
    from isaaclab_newton.physics import XPBDSolverCfg

    link = "/World/envs/env_0/Robot/Link"

    def unbalanced_link(stage):
        UsdPhysics.MassAPI(stage.GetPrimAtPath("/Robot/Link")).GetDiagonalInertiaAttr().Set(Gf.Vec3f(0.01, 0.01, 0.5))

    def heavier_link(scene):
        model = scene.sim.backend_manager.backend.model
        masses = model.body_mass.numpy()
        masses[list(model.body_label).index(link)] = mass
        model.body_mass.assign(masses)

    output, live, _, _ = _export(
        tmp_path, XPBDSolverCfg(), "cpu", edit_robot=unbalanced_link, before_export=heavier_link if mass else None
    )

    assert live["body_inertia", link][0, 0] > 0.01  # the import corrected it
    exported = Sdf.Layer.FindOrOpen(output)
    for name in ("physics:diagonalInertia", "physics:principalAxes", "newton:inertia"):
        assert exported.GetPropertyAtPath(f"{link}.{name}") is None, name
    if mass:
        assert exported.GetAttributeAtPath(f"{link}.physics:mass").default == pytest.approx(mass)
    model = _newton_import(output, "xpbd", "cpu")
    loaded = _capture(model, model.body_q)
    np.testing.assert_allclose(loaded["body_inertia", link], live["body_inertia", link], rtol=1e-5)


def test_a_changed_joint_of_a_type_not_exported_is_refused(tmp_path):
    """A joint other than revolute, prismatic and rotational D6 keeps its USD values, so changing it is refused."""
    from isaaclab_newton.physics import XPBDSolverCfg

    from isaaclab.actuators import ImplicitActuatorCfg

    def ball_hinge(stage):
        stage.GetPrimAtPath("/Robot/Hinge").SetTypeName("PhysicsSphericalJoint")

    def ball_joint_state(cfg):
        cfg.robot.init_state.joint_pos = {".*": 0.0}

    actuator = ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=83.0, damping=4.5)
    with pytest.raises(NotImplementedError, match="Joint '/World/envs/env_0/Robot/Hinge' changed"):
        _export(
            tmp_path, XPBDSolverCfg(), "cpu", hinge_actuator=actuator, edit_robot=ball_hinge, edit_cfg=ball_joint_state
        )


@pytest.mark.parametrize("solver", ["mujoco", "xpbd"])
def test_builder_hook_and_shape_default_changes_are_written(tmp_path, solver):
    """Changes Isaac Lab makes while building the model, which Newton's import alone does not make, are written.

    A world builder hook sets a link's contact stiffness before finalization, and Isaac Lab's shape defaults reach
    colliders with and without a physics material; a collider without one gets a new material that states them.
    """
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonShapeCfg, XPBDSolverCfg
    from isaaclab_newton.sim.usd_export.usd_exporter import MATERIAL_ATTRIBUTES

    link_collider = "/World/envs/env_0/Robot/Link/Collision"

    def stiffer_link(builder, world, position, orientation):
        for shape, label in enumerate(builder.shape_label):
            if label.endswith("/Robot/Link/Collision"):
                builder.shape_material_ke[shape] = 7.0e3

    solver_cfg = MJWarpSolverCfg() if solver == "mujoco" else XPBDSolverCfg()
    shape_cfg = NewtonShapeCfg(gap=0.02, ke=5.0e3, kd=200.0, mu=0.8)
    output, live, _, _ = _export(tmp_path, solver_cfg, "cpu", default_shape_cfg=shape_cfg, world_hook=stiffer_link)

    assert live["shape_material_ke", link_collider] == pytest.approx(7.0e3)
    table, ground = "/World/envs/env_0/Table/geometry/mesh", "/World/Ground/geometry/mesh"
    stage = Usd.Stage.Open(output)
    bound = [
        UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(c)).ComputeBoundMaterial("physics")[0] for c in (table, ground)
    ]
    assert bound[0] and bound[0].GetPath() == bound[1].GetPath()  # equal values share one new material
    # Only what an unset attribute would not give: the changed friction and gains.
    assert {a.GetName() for a in bound[0].GetPrim().GetAuthoredAttributes()} == {
        "physics:dynamicFriction",
        "newton:contactStiffness",
        "newton:contactDamping",
    }
    model = _newton_import(output, solver, "cpu")
    loaded = _capture(model, model.body_q)
    for collider in (table, link_collider):
        for name in (*MATERIAL_ATTRIBUTES, "shape_gap"):
            np.testing.assert_allclose(
                loaded[name, collider], live[name, collider], rtol=1e-5, err_msg=f"{collider} {name}"
            )


def test_a_mass_change_keeps_the_inertia_derived_from_geometry(tmp_path):
    """The box's USD leaves its inertia to the import, which derives it from the mass; a changed mass keeps it."""
    from isaaclab_newton.physics import XPBDSolverCfg

    box = "/World/envs/env_0/Box"

    def heavier_box(scene):
        model = scene.sim.backend_manager.backend.model
        mass = model.body_mass.numpy()
        mass[list(model.body_label).index(box)] *= 2.0
        model.body_mass.assign(mass)

    output, live, _, _ = _export(tmp_path, XPBDSolverCfg(), "cpu", before_export=heavier_box)

    model = _newton_import(output, "xpbd", "cpu")
    loaded = _capture(model, model.body_q)
    for name in ("body_mass", "body_inertia"):
        np.testing.assert_allclose(loaded[name, box], live[name, box], rtol=1e-5, err_msg=name)


def test_a_joint_actuated_without_a_usd_drive_gets_one(tmp_path):
    """Newton's import actuates only axes with a USD drive; an axis Isaac Lab actuates gets one, also at zero gains."""
    from isaaclab_newton.physics import XPBDSolverCfg

    from isaaclab.actuators import ImplicitActuatorCfg

    def undriven(stage):
        stage.GetPrimAtPath("/Robot/Hinge").RemoveAPI(UsdPhysics.DriveAPI, "angular")

    actuator = ImplicitActuatorCfg(joint_names_expr=["Hinge"], stiffness=0.0, damping=0.0)
    output, live, _, _ = _export(tmp_path, XPBDSolverCfg(), "cpu", hinge_actuator=actuator, edit_robot=undriven)

    model = _newton_import(output, "xpbd", "cpu")
    key = "joint_target_mode", "/World/envs/env_0/Robot/Hinge:0"
    assert live[key] == int(newton.JointTargetMode.EFFORT)
    assert _capture(model, model.body_q)[key] == live[key]


def test_the_solver_step_rate_is_stated(tmp_path):
    """With substeps, the file states the solver step rate, which Newton's import returns as its step."""
    from isaaclab_newton.physics import XPBDSolverCfg

    output, *_ = _export(tmp_path, XPBDSolverCfg(), "cpu", num_substeps=2)

    result = newton.ModelBuilder().add_usd(Usd.Stage.Open(output), schema_resolvers=[SchemaResolverNewton()])
    assert result["physics_dt"] == pytest.approx(1 / 240)


def test_export_references_assets(tmp_path):
    """The exported layer references the robot file instead of copying it."""
    from isaaclab_newton.physics import XPBDSolverCfg

    output, *_ = _export(tmp_path, XPBDSolverCfg(), "cpu")

    layer = Sdf.Layer.FindOrOpen(output)
    assert str(tmp_path / "robot.usda") in layer.externalReferences
    assert layer.GetPrimAtPath("/World/envs/env_0/Robot").referenceList.GetAddedOrExplicitItems()
    # The robot's own prims come from the reference; only edits are authored over them.
    link = layer.GetPrimAtPath("/World/envs/env_0/Robot/Link")
    assert not link or link.specifier == Sdf.SpecifierOver


def test_explicit_pd_actuator_is_exported_as_a_joint_drive(tmp_path):
    """An explicit PD actuator, whose gains the engine does not hold, becomes the joint's USD drive."""
    from isaaclab_newton.physics import XPBDSolverCfg

    from isaaclab.actuators import IdealPDActuatorCfg

    actuator = IdealPDActuatorCfg(joint_names_expr=["Hinge"], stiffness=83.0, damping=4.5, actuator_effort_limit=12.0)
    output, *_ = _export(tmp_path, XPBDSolverCfg(), "cpu", hinge_actuator=actuator)

    stage = Usd.Stage.Open(output)
    hinge = UsdPhysics.DriveAPI(stage.GetPrimAtPath("/World/envs/env_0/Robot/Hinge"), "angular")
    assert hinge.GetStiffnessAttr().Get() == pytest.approx(np.deg2rad(83.0))
    assert hinge.GetDampingAttr().Get() == pytest.approx(np.deg2rad(4.5))
    assert hinge.GetMaxForceAttr().Get() == pytest.approx(12.0)
    # the drive replaces the actuator prim, so importers reading both apply the PD once
    assert not [prim for prim in stage.Traverse() if prim.GetTypeName() == "NewtonActuator"]
    model = _newton_import(output, "xpbd", "cpu")
    hinge_dof = model.joint_qd_start.numpy()[model.joint_label.index("/World/envs/env_0/Robot/Hinge")]
    assert model.joint_target_ke.numpy()[hinge_dof] == pytest.approx(83.0, rel=1e-5)
    assert model.joint_target_kd.numpy()[hinge_dof] == pytest.approx(4.5, rel=1e-5)


def test_model_without_joint_dofs_has_no_changed_values():
    """A model without joint DOFs, e.g. one with only fixed joints, compares to empty builder lists."""
    from types import SimpleNamespace

    from isaaclab_newton.sim.usd_export import UsdExporter

    changed = UsdExporter._changed({"joint_armature": np.zeros(0)}, SimpleNamespace(joint_armature=[]))
    assert changed["joint_armature"].shape == (0,)


def test_merged_joint_dofs_map_to_their_usd_joints():
    """Newton merges single-axis joints between two bodies into one D6 joint; each DOF maps to its own USD joint."""
    from types import SimpleNamespace

    from isaaclab_newton.sim.usd_export import UsdExporter

    stage = Usd.Stage.CreateInMemory()
    for path in ("/Slide", "/HingeX", "/HingeY"):
        (UsdPhysics.PrismaticJoint if path == "/Slide" else UsdPhysics.RevoluteJoint).Define(stage, path)

    def array(values):
        return SimpleNamespace(numpy=lambda: np.asarray(values))

    model = SimpleNamespace(
        joint_qd_start=array([0, 3]),
        joint_type=array([int(newton.JointType.D6)]),
        joint_dof_dim=array([[1, 2]]),
        mujoco=SimpleNamespace(joint_dof_label=["/Slide", "/HingeX", "/HingeY"]),
    )
    units = UsdExporter._joint_prims(stage, model, 0, stage.GetPrimAtPath("/HingeX"))
    assert [(str(prim.GetPath()), axes) for prim, axes, _ in units] == [
        ("/Slide", [(0, "linear")]),
        ("/HingeX", [(1, "angular")]),
        ("/HingeY", [(2, "angular")]),
    ]
    del model.mujoco
    with pytest.raises(NotImplementedError, match="merges USD joints"):
        UsdExporter._joint_prims(stage, model, 0, stage.GetPrimAtPath("/HingeX"))


def test_raised_effort_limit_widens_the_mujoco_force_range():
    """Newton keeps the tighter of the drive force limit and MuJoCo's joint force range, so the range is widened."""
    from isaaclab_newton.sim.usd_export import UsdExporter

    from isaaclab.sim.usd_export import UsdEdit

    stage = Usd.Stage.CreateInMemory()
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Hinge").GetPrim()
    joint.CreateAttribute("mjc:actuatorfrcrange:min", Sdf.ValueTypeNames.Double).Set(-4.0)
    joint.CreateAttribute("mjc:actuatorfrcrange:max", Sdf.ValueTypeNames.Double).Set(4.0)

    edits = UsdExporter._force_range_edits(stage, [UsdEdit("/Hinge", "drive:angular:physics:maxForce", 12.0)])
    assert {(edit.attribute, edit.value) for edit in edits} == {
        ("mjc:actuatorfrcrange:min", -12.0),
        ("mjc:actuatorfrcrange:max", 12.0),
    }
    assert not UsdExporter._force_range_edits(stage, [UsdEdit("/Hinge", "drive:angular:physics:maxForce", 3.0)])


def test_d6_limit_gains_go_to_a_stated_joint_wide_attribute():
    """Newton reads a joint-wide limit gain ahead of per-axis ones, so a joint stating it is written there."""
    from isaaclab_newton.sim.usd_export import UsdExporter
    from isaaclab_newton.sim.usd_export.usd_exporter import D6_DOF_ATTRIBUTES

    stage = Usd.Stage.CreateInMemory()
    joint = UsdPhysics.Joint.Define(stage, "/Ball").GetPrim()
    per_axis = D6_DOF_ATTRIBUTES["joint_limit_ke"][0]
    assert UsdExporter._limit_gain_attribute(joint, "joint_limit_ke", per_axis) == per_axis
    joint.CreateAttribute("newton:limitStiffness", Sdf.ValueTypeNames.Float).Set(100.0)
    assert UsdExporter._limit_gain_attribute(joint, "joint_limit_ke", per_axis) == "newton:limitStiffness"


def test_changed_limits_are_written_with_the_mujoco_reference(tmp_path):
    """Newton's import subtracts a joint's MuJoCo reference from its USD limits, so a changed limit imports as live."""
    import torch
    from isaaclab_newton.physics import MJWarpSolverCfg

    def add_reference(stage):
        stage.GetPrimAtPath("/Robot/Hinge").CreateAttribute("mjc:ref", Sdf.ValueTypeNames.Double).Set(10.0)

    def change_limits(scene):
        scene["robot"].write_joint_position_limit_to_sim_index(limits=torch.tensor([[[-0.6, 0.9]]], device="cuda:0"))

    output, _, _, _ = _export(
        tmp_path, MJWarpSolverCfg(), "cuda:0", edit_robot=add_reference, before_export=change_limits
    )
    model = _newton_import(output, "mujoco", "cpu")
    dof = model.joint_qd_start.numpy()[model.joint_label.index("/World/envs/env_0/Robot/Hinge")]
    assert model.joint_limit_lower.numpy()[dof] == pytest.approx(-0.6, abs=1e-5)
    assert model.joint_limit_upper.numpy()[dof] == pytest.approx(0.9, abs=1e-5)


def test_kamino_constraint_gains_are_stated_on_the_scene():
    """Kamino reads its constraint stabilization gains from the physics scene, so the export states those that differ
    from Newton's defaults."""
    from types import SimpleNamespace

    from isaaclab_newton.physics import KaminoConstraintsCfg, KaminoPADMMSolverCfg
    from isaaclab_newton.sim.usd_export import UsdExporter

    exporter = UsdExporter.__new__(UsdExporter)
    physics = SimpleNamespace(num_substeps=1)
    exporter.scene = SimpleNamespace(
        physics_scene_path="/physicsScene",
        sim=SimpleNamespace(cfg=SimpleNamespace(physics=physics), get_physics_dt=lambda: 0.01),
    )
    cfg = KaminoPADMMSolverCfg(constraints=KaminoConstraintsCfg(alpha=0.1, beta=0.2, gamma=0.01))
    edits = {edit.attribute: edit.value for edit in exporter._scene_edits(cfg)}
    assert edits["newton:kamino:constraints:beta"] == pytest.approx(0.2)
    assert "newton:kamino:constraints:gamma" not in edits  # Newton's default


def test_attribute_tables_cover_newton_importer():
    """Every attribute Newton's importer maps is exported or deliberately excluded."""
    from isaaclab_newton.sim.usd_export.usd_exporter import (
        DOF_ATTRIBUTES,
        MATERIAL_ATTRIBUTES,
        SHAPE_ATTRIBUTES,
    )

    exported = {attribute for attribute, _, _ in DOF_ATTRIBUTES.values()}
    exported |= set(SHAPE_ATTRIBUTES.values()) | set(MATERIAL_ATTRIBUTES.values())
    excluded = {
        # joint state, written as JOINT_STATE_ATTRIBUTES unless the USD states these
        *(
            f"newton:{axis}:{kind}"
            for axis in ("angular", "linear", "rotX", "rotY", "rotZ")
            for kind in ("position", "velocity")
        ),
        # per-axis limit gains of multi-axis joints, which the export rejects when changed
        *(
            f"newton:{axis}:limit{gain}"
            for axis in ("linear", "angular", "rotX", "rotY", "rotZ")
            for gain in ("Stiffness", "Damping")
        ),
        # deprecated collider contact gains; the export states them on the collider's physics material
        *(SchemaResolverNewton.mapping[PrimType.SHAPE][key].name for key in ("ke", "kd", "kf", "ka")),
        # collision geometry generation, fixed at import
        "newton:maxHullVertices",
        "newton:sdfMaxResolution",
        "newton:sdfNarrowBandInner",
        "newton:sdfNarrowBandOuter",
        "newton:sdfTargetVoxelSize",
        "newton:sdfTextureFormat",
        "newton:sdfPadding",
        "newton:hydroelasticEnabled",
        "newton:hydroelasticStiffness",
        "newton:massModel",
        "newton:shellThickness",
    }
    mapped = {
        spec.name
        for prim_type in (PrimType.JOINT, PrimType.SHAPE, PrimType.MATERIAL)
        for spec in SchemaResolverNewton.mapping[prim_type].values()
    }
    assert mapped - exported - excluded == set()
