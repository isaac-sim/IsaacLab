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
    """Exported quantities of one world, keyed by USD path."""
    from isaaclab_newton.sim.usd_export.usd_exporter import DOF_ATTRIBUTES, MATERIAL_ATTRIBUTES, SHAPE_ATTRIBUTES

    def selected(worlds):
        return np.flatnonzero(np.isin(worlds.numpy(), (-1, env_id)))

    starts, types = model.joint_qd_start.numpy(), model.joint_type.numpy()
    parents, children = model.joint_parent.numpy(), model.joint_child.numpy()
    dof_arrays = {name: getattr(model, name).numpy() for name in (*DOF_ATTRIBUTES, "joint_target_mode")}
    joints, anchors, roots = {}, {}, set()
    for joint in selected(model.joint_world):
        if parents[joint] == -1:
            roots.add(model.body_label[children[joint]])
            if types[joint] != int(newton.JointType.FREE):
                anchors[model.joint_label[joint]] = model.joint_X_p.numpy()[joint]
        if types[joint] != int(newton.JointType.FREE):
            dofs = slice(starts[joint], starts[joint + 1])
            joints[model.joint_label[joint]] = {name: values[dofs] for name, values in dof_arrays.items()}
    body_arrays = {name: getattr(model, name).numpy() for name in ("body_mass", "body_com", "body_inertia")}
    poses = body_q.numpy()
    bodies = {}
    for body in selected(model.body_world):
        label = model.body_label[body]
        bodies[label] = {name: values[body] for name, values in body_arrays.items()}
        if label in roots:
            bodies[label]["pose"] = poses[body]
    shape_arrays = {name: getattr(model, name).numpy() for name in (*SHAPE_ATTRIBUTES, *MATERIAL_ATTRIBUTES)}
    shapes = {
        model.shape_label[shape]: {name: values[shape] for name, values in shape_arrays.items()}
        for shape in selected(model.shape_world)
    }
    gravity = model.gravity.numpy()
    return {
        "joints": joints,
        "anchors": anchors,
        "bodies": bodies,
        "shapes": shapes,
        "gravity": gravity[env_id] if gravity.ndim == 2 else gravity,
    }


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
    # The actuator configuration differs from the USD drive, so the export must carry it.
    hinge = live["joints"][f"/World/envs/env_{env_id}/Robot/Hinge"]
    assert hinge["joint_target_ke"][0] == pytest.approx(83.0)
    assert hinge["joint_armature"][0] == pytest.approx(0.023)
    assert live["joints"].keys() == loaded["joints"].keys()
    for label, values in live["joints"].items():
        for name, value in values.items():
            np.testing.assert_allclose(loaded["joints"][label][name], value, rtol=1e-4, err_msg=f"{label} {name}")
    assert live["anchors"].keys() == loaded["anchors"].keys()
    for label, anchor in live["anchors"].items():
        _assert_pose_equal(loaded["anchors"][label], anchor)
    assert live["bodies"].keys() == loaded["bodies"].keys()
    for label, values in live["bodies"].items():
        for name, value in values.items():
            if name == "pose":
                _assert_pose_equal(loaded["bodies"][label]["pose"], value)
            else:
                np.testing.assert_allclose(
                    loaded["bodies"][label][name], value, rtol=1e-4, atol=1e-7, err_msg=f"{label} {name}"
                )
    assert live["shapes"].keys() == loaded["shapes"].keys()
    for label, values in live["shapes"].items():
        for name, value in values.items():
            np.testing.assert_allclose(loaded["shapes"][label][name], value, rtol=1e-4, err_msg=f"{label} {name}")
    np.testing.assert_allclose(loaded["gravity"], live["gravity"], rtol=1e-5)


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
