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


def _export(tmp_path, solver_cfg, device, num_envs=1, env_id=0, hinge_actuator=None):
    """Initialize the fixture scene, export it, and capture the live model for comparison."""
    from isaaclab_newton.physics import NewtonCfg

    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from isaaclab.test.utils.usd_export import make_export_scene_cfg

    cfg = make_export_scene_cfg(tmp_path, num_envs=num_envs)
    if hinge_actuator is not None:
        cfg.robot.actuators = {"hinge": hinge_actuator}
    sim_cfg = SimulationCfg(
        device=device, dt=1 / 120, gravity=(0.2, -0.1, -4.0), physics=NewtonCfg(solver_cfg=solver_cfg)
    )
    output = str(tmp_path / "exported.usda")
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(cfg)
        sim.reset()
        scene.export_to_usd(output, env_id=env_id)
        manager = sim.backend_manager
        live = _capture(manager.get_model(), manager.get_state_0().body_q, env_id)
        solver_type = type(manager._solver)
    return output, live, solver_type


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
def test_loading_the_export_reproduces_the_initialized_model(tmp_path, solver, device, num_envs, env_id):
    """Every exported quantity of the loaded model equals the live model's, including config overrides."""
    from isaaclab_newton.physics import MJWarpSolverCfg, XPBDSolverCfg
    from isaaclab_newton.sim.usd_export import UsdExporter

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("MuJoCo Warp requires CUDA.")
    solver_cfg = XPBDSolverCfg(iterations=7) if solver == "xpbd" else MJWarpSolverCfg(iterations=17)
    output, live, solver_type = _export(tmp_path, solver_cfg, device, num_envs, env_id)

    model, loaded_solver, dt = UsdExporter.load_usd(output, device=device)
    loaded = _capture(model, model.body_q, 0)

    assert type(loaded_solver) is solver_type
    assert dt == pytest.approx(1 / 120)
    if solver == "mujoco":
        # MuJoCo options are stated on the physics scene, where Newton's importer reads them.
        stage = Usd.Stage.Open(output)
        scene = stage.GetPrimAtPath("/physicsScene")
        assert scene.GetAttribute("mjc:option:iterations").Get() == 17
        assert loaded_solver.mjw_model.opt.iterations == 17
    # The actuator configuration differs from the USD drive, so the export must carry it.
    hinge = f"/World/envs/env_{env_id}/Robot/Hinge:0"
    assert live["joint_target_ke", hinge] == pytest.approx(83.0)
    assert live["joint_armature", hinge] == pytest.approx(0.023)
    assert set(live) == set(loaded), (sorted(set(live) - set(loaded))[:8], sorted(set(loaded) - set(live))[:8])
    for (name, label), value in live.items():
        if name in ("body_q", "joint_X_p", "joint_X_c"):
            _assert_pose_equal(loaded[name, label], value)
        else:
            np.testing.assert_allclose(loaded[name, label], value, rtol=1e-4, atol=1e-7, err_msg=f"{label} {name}")


def test_export_references_assets_and_keeps_one_environment(tmp_path):
    """The exported layer references the robot file instead of copying it, and drops other environments."""
    from isaaclab_newton.physics import XPBDSolverCfg

    output, _, _ = _export(tmp_path, XPBDSolverCfg(), "cpu", num_envs=2, env_id=0)

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
    output, _, _ = _export(tmp_path, XPBDSolverCfg(), "cpu", hinge_actuator=actuator)

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
        {"solver_type": "isaaclab_newton.physics:XPBDSolverCfg", "solver": {"class_type": "builtins:dict"}},
    ],
    ids=["solver type", "manager type"],
)
def test_loading_refuses_classes_other_than_newton_solvers(tmp_path, settings):
    """A file names the solver classes that loading instantiates; any other class is refused, not called."""
    from isaaclab_newton.sim.usd_export import UsdExporter
    from isaaclab_newton.sim.usd_export.usd_exporter import SETTINGS_KEY

    layer = Sdf.Layer.CreateNew(str(tmp_path / "export.usda"))
    layer.customLayerData = {SETTINGS_KEY: json.dumps(settings)}
    layer.Save()

    with pytest.raises(ValueError, match="is not a Newton"):
        UsdExporter.load_usd(layer.realPath)


def test_model_without_joint_dofs_has_no_changed_values():
    """A model without joint DOFs, e.g. one with only fixed joints, compares to empty builder lists."""
    from types import SimpleNamespace

    from isaaclab_newton.sim.usd_export import UsdExporter

    changed = UsdExporter._changed({"joint_armature": np.zeros(0)}, SimpleNamespace(joint_armature=[]))
    assert changed["joint_armature"].shape == (0,)


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
