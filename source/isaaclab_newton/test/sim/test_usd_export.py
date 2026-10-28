# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton USD export: loading the exported file reproduces the initialized model."""

import json

import newton
import numpy as np
import pytest
import warp as wp
from newton.usd import PrimType, SchemaResolverNewton

from pxr import Sdf, Usd, UsdPhysics


def _export(tmp_path, solver_cfg, device, num_envs=1, env_id=0, hinge_actuator=None, num_substeps=1):
    """Initialize the fixture scene, export it, and capture the live model and its rollout for comparison."""
    from isaaclab_newton.physics import NewtonCfg

    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from isaaclab.test.utils.usd_export import make_export_scene_cfg

    cfg = make_export_scene_cfg(tmp_path, num_envs=num_envs)
    if hinge_actuator is not None:
        cfg.robot.actuators = {"hinge": hinge_actuator}
    sim_cfg = SimulationCfg(
        device=device,
        dt=1 / 120,
        gravity=(0.2, -0.1, -4.0),
        physics=NewtonCfg(solver_cfg=solver_cfg, num_substeps=num_substeps),
    )
    output = str(tmp_path / "exported.usda")
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(cfg)
        sim.reset()
        scene.export_to_usd(output, env_id=env_id)
        manager = sim.backend_manager
        live = _capture(manager.backend.model, manager.backend.state_0.body_q, env_id)
        solver_type = type(manager.backend.solver)
        # The engine's own construction of the solver, stepped from the model's initial state.
        motion = _rollout(manager.backend.model, type(manager)._create_solver(manager.backend.model, solver_cfg))
    return output, live, solver_type, motion


def _rollout(model, solver, steps=60, dt=1 / 120):
    """Body positions after stepping the model's initial state without control input, keyed by body label."""
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
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


def _capture(model, body_q, env_id):
    """Every float model array, and the joint target modes, of one world's bodies, shapes, joints and DOFs.

    Keyed by attribute name and entity label; a DOF is labelled ``<joint label>:<axis index>``. Root body
    poses come from ``body_q``; other poses follow the joint state, which the export leaves out. The world's
    gravity is keyed ``("gravity", "")``. Model values the export does not
    carry show up as differences.
    """
    frequency = newton.Model.AttributeFrequency
    labels = {
        frequency.BODY: list(model.body_label),
        frequency.SHAPE: list(model.shape_label),
        frequency.JOINT: list(model.joint_label),
        frequency.JOINT_DOF: [None] * model.joint_dof_count,
    }
    worlds = {
        frequency.BODY: model.body_world.numpy(),
        frequency.SHAPE: model.shape_world.numpy(),
        frequency.JOINT: model.joint_world.numpy(),
        frequency.JOINT_DOF: np.empty(model.joint_dof_count, dtype=int),
    }
    starts, types = model.joint_qd_start.numpy(), model.joint_type.numpy()
    # Free joints of floating bodies are generated, under generated names; their bodies carry the pose.
    worlds[frequency.JOINT][types == int(newton.JointType.FREE)] = -2
    for joint, label in enumerate(model.joint_label):
        for axis, dof in enumerate(range(starts[joint], starts[joint + 1])):
            labels[frequency.JOINT_DOF][dof] = f"{label}:{axis}"
            worlds[frequency.JOINT_DOF][dof] = worlds[frequency.JOINT][joint]
    parents, children = model.joint_parent.numpy(), model.joint_child.numpy()
    roots = {model.body_label[child] for parent, child in zip(parents, children) if parent == -1}
    values = {}
    for name, entity in model.attribute_frequency.items():
        namespace, _, attribute = name.rpartition(":")
        array = getattr(getattr(model, namespace) if namespace else model, attribute, None)
        if entity not in labels or not isinstance(array, wp.array):
            continue
        array = body_q.numpy() if name == "body_q" else array.numpy()
        if array.dtype.kind != "f" and name != "joint_target_mode":
            continue
        for index in np.flatnonzero(np.isin(worlds[entity], (-1, env_id))):
            if name != "body_q" or labels[entity][index] in roots:
                values[name, labels[entity][index]] = array[index]
    gravity = model.gravity.numpy()
    values["gravity", ""] = gravity[env_id] if gravity.ndim == 2 else gravity
    return values


def _assert_pose_equal(actual, expected):
    np.testing.assert_allclose(actual[:3], expected[:3], atol=1e-5)
    sign = 1.0 if np.dot(actual[3:], expected[3:]) >= 0 else -1.0
    np.testing.assert_allclose(actual[3:], sign * expected[3:], atol=1e-5)


@pytest.mark.parametrize(
    "solver,device,num_envs,env_id",
    [("xpbd", "cpu", 1, 0), ("xpbd", "cpu", 2, 1), ("mujoco", "cuda:0", 1, 0)],
)
def test_importing_the_export_reproduces_the_initialized_model(tmp_path, caplog, solver, device, num_envs, env_id):
    """Newton's own import of the file reproduces every live model value, including config overrides."""
    from isaaclab_newton.physics import MJWarpSolverCfg, XPBDSolverCfg
    from isaaclab_newton.sim.usd_export import UsdExporter

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("MuJoCo Warp requires CUDA.")
    debug_output = tmp_path / "training.xml"
    solver_cfg = (
        XPBDSolverCfg(iterations=7)
        if solver == "xpbd"
        else MJWarpSolverCfg(iterations=17, enable_multiccd=True, save_to_mjcf=str(debug_output))
    )
    output, live, solver_type, motion = _export(tmp_path, solver_cfg, device, num_envs, env_id)

    debug_output.unlink(missing_ok=True)
    _, loaded_solver, dt = UsdExporter.load_usd(output, device=device)
    assert not debug_output.exists()  # loading does not write the training run's debug output
    assert type(loaded_solver) is solver_type
    assert dt == pytest.approx(1 / 120)
    model = _newton_import(output, solver, device)
    loaded = _capture(model, model.body_q, 0)
    if solver == "mujoco":
        # MuJoCo reads its options from the physics scene, so the default solver runs the configured one.
        imported_solver = newton.solvers.SolverMuJoCo(model)
        assert imported_solver.mjw_model.opt.iterations == 17
        # Configuration flags have their own USD names; disable flags are stated inverted.
        assert bool(model.mujoco.flag_multiccd.numpy()[0]) and bool(model.mujoco.flag_contact.numpy()[0])
        moved = _rollout(model, imported_solver)
        for label, position in motion.items():
            np.testing.assert_allclose(moved[label], position, atol=1e-4, err_msg=label)
    # The actuator configuration differs from the USD drive, so the export must carry it.
    hinge = f"/World/envs/env_{env_id}/Robot/Hinge:0"
    assert live["joint_target_ke", hinge] == pytest.approx(83.0)
    assert live["joint_armature", hinge] == pytest.approx(0.023)
    assert set(live) == set(loaded), (sorted(set(live) - set(loaded))[:8], sorted(set(loaded) - set(live))[:8])
    assert live["joint_target_mode", hinge] == int(newton.JointTargetMode.POSITION_VELOCITY)
    if solver != "mujoco":
        # USD cannot state a target mode; without MuJoCo's velocity actuator the import infers position mode.
        assert "target mode" in caplog.text
        del live["joint_target_mode", hinge]
    for (name, label), value in live.items():
        if name in ("body_q", "joint_X_p", "joint_X_c"):
            _assert_pose_equal(loaded[name, label], value)
        else:
            np.testing.assert_allclose(loaded[name, label], value, rtol=1e-4, atol=1e-7, err_msg=f"{label} {name}")


def test_loading_returns_the_solver_step(tmp_path):
    """With substeps, the file states the solver step rate, and loading returns the solver step size."""
    from isaaclab_newton.physics import XPBDSolverCfg
    from isaaclab_newton.sim.usd_export import UsdExporter

    output, *_ = _export(tmp_path, XPBDSolverCfg(), "cpu", num_substeps=2)

    assert Usd.Stage.Open(output).GetPrimAtPath("/physicsScene").GetAttribute("newton:timeStepsPerSecond").Get() == 240
    _, _, dt = UsdExporter.load_usd(output)
    assert dt == pytest.approx(1 / 240)


def test_export_references_assets_and_keeps_one_environment(tmp_path):
    """The exported layer references the robot file instead of copying it, and drops other environments."""
    from isaaclab_newton.physics import XPBDSolverCfg

    output, *_ = _export(tmp_path, XPBDSolverCfg(), "cpu", num_envs=2, env_id=0)

    layer = Sdf.Layer.FindOrOpen(output)
    assert str(tmp_path / "robot.usda") in layer.externalReferences
    assert layer.GetPrimAtPath("/World/envs/env_0/Robot").referenceList.GetAddedOrExplicitItems()
    assert not layer.GetPrimAtPath("/World/envs/env_1")
    # The robot's own prims come from the reference; only edits are authored over them.
    link = layer.GetPrimAtPath("/World/envs/env_0/Robot/Link")
    assert not link or link.specifier == Sdf.SpecifierOver


def test_explicit_pd_actuator_is_exported_as_a_joint_drive(tmp_path):
    """An explicit PD actuator, whose gains the engine does not hold, becomes the joint's USD drive."""
    from isaaclab_newton.physics import XPBDSolverCfg
    from isaaclab_newton.sim.usd_export import UsdExporter

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
    model, _, _ = UsdExporter.load_usd(output)
    hinge_dof = model.joint_qd_start.numpy()[model.joint_label.index("/World/envs/env_0/Robot/Hinge")]
    assert model.joint_target_ke.numpy()[hinge_dof] == pytest.approx(83.0, rel=1e-5)
    assert model.joint_target_kd.numpy()[hinge_dof] == pytest.approx(4.5, rel=1e-5)


@pytest.mark.parametrize(
    "settings",
    [
        {"solver_type": "builtins:dict", "solver": {}},
        {
            "solver_type": "isaaclab_newton.physics.xpbd_manager_cfg:XPBDSolverCfg",
            "solver": {"class_type": "builtins:dict"},
        },
        {"solver_type": "isaaclab_newton.physics.xpbd_manager_cfg:XPBDSolverCfg", "solver": {"__dict__": {}}},
    ],
    ids=["solver type", "manager type", "undeclared field"],
)
def test_loading_refuses_classes_other_than_newton_solvers(tmp_path, settings):
    """A file names the solver classes that loading instantiates; any other class is refused, not called."""
    from isaaclab_newton.sim.usd_export import UsdExporter
    from isaaclab_newton.sim.usd_export.usd_exporter import SETTINGS_KEY

    layer = Sdf.Layer.CreateNew(str(tmp_path / "export.usda"))
    layer.customLayerData = {SETTINGS_KEY: json.dumps(settings)}
    layer.Save()

    with pytest.raises(ValueError, match="is not a"):
        UsdExporter.load_usd(layer.realPath)


def test_loading_restores_nested_solver_configurations():
    """An optional nested configuration the file states is created from its declared type."""
    from isaaclab_newton.physics import KaminoDynamicsCfg, KaminoPADMMSolverCfg
    from isaaclab_newton.sim.usd_export import UsdExporter

    exported = KaminoPADMMSolverCfg(dynamics=KaminoDynamicsCfg(preconditioning=False)).to_dict()
    loaded = KaminoPADMMSolverCfg()
    UsdExporter._load_fields(loaded, json.loads(json.dumps(exported, default=str)), "export.usda")
    assert loaded.dynamics is not None and loaded.dynamics.preconditioning is False


def test_loading_does_not_import_modules_the_file_names(tmp_path):
    """A file naming an arbitrary module is refused before that module is imported."""
    import sys

    from isaaclab_newton.sim.usd_export import UsdExporter
    from isaaclab_newton.sim.usd_export.usd_exporter import SETTINGS_KEY

    xpbd = "isaaclab_newton.physics.xpbd_manager_cfg:XPBDSolverCfg"
    cases = (
        {"solver_type": "this:Cfg", "solver": {}},
        {"solver_type": xpbd, "solver": {"__dict__": {"class_type": "this:Cfg"}}},
        {"solver_type": xpbd, "solver": {"class_type": "this:Cfg"}},
    )
    for index, settings in enumerate(cases):
        sys.modules.pop("this", None)
        layer = Sdf.Layer.CreateNew(str(tmp_path / f"export_{index}.usda"))
        layer.customLayerData = {SETTINGS_KEY: json.dumps(settings)}
        layer.Save()
        with pytest.raises(ValueError, match="is not a"):
            UsdExporter.load_usd(layer.realPath)
        assert "this" not in sys.modules


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


def test_attribute_tables_cover_newton_importer():
    """Every attribute Newton's importer maps is exported or deliberately excluded."""
    from isaaclab_newton.sim.usd_export.usd_exporter import (
        DOF_ATTRIBUTES,
        MATERIAL_ATTRIBUTES,
        SHAPE_ATTRIBUTES,
        SHAPE_CONTACT_ATTRIBUTES,
    )

    exported = {attribute for attribute, _, _ in DOF_ATTRIBUTES.values()}
    exported |= (
        set(SHAPE_ATTRIBUTES.values()) | set(MATERIAL_ATTRIBUTES.values()) | set(SHAPE_CONTACT_ATTRIBUTES.values())
    )
    excluded = {
        # joint state is not part of the export
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
