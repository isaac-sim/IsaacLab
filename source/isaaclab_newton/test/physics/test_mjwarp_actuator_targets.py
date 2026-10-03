# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression coverage for imported actuator limits after composing and replicating assets."""

import mujoco
import numpy as np
from isaaclab_newton.cloner.newton_clone_utils import build_source_builders, replicate_builder_mapping
from isaaclab_newton.physics import NewtonMJWarpManager
from newton import ModelBuilder
from newton.solvers import SolverMuJoCo

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import make_clone_plan
from isaaclab.sim import SpawnerCfg


def test_imported_actuator_limits_follow_their_joints_after_composition():
    """A preceding free body must not route a finger's limits to an arm joint."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    table = UsdGeom.Cube.Define(stage, "/Sources/Table").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(table)
    UsdPhysics.CollisionAPI.Apply(table)
    robot = UsdGeom.Xform.Define(stage, "/Sources/Robot").GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(robot)
    base = UsdGeom.Cube.Define(stage, "/Sources/Robot/Base").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(base)
    UsdPhysics.CollisionAPI.Apply(base)
    fixed = UsdPhysics.FixedJoint.Define(stage, "/Sources/Robot/Fixed")
    fixed.CreateBody1Rel().SetTargets([base.GetPath()])
    for name, joint_type, axis, bounds, effort in (
        ("Arm", UsdPhysics.RevoluteJoint, "Z", (-1.0, 1.0), 87.0),
        ("Finger", UsdPhysics.PrismaticJoint, "Y", (0.0, 0.04), 200.0),
    ):
        body = UsdGeom.Cube.Define(stage, f"/Sources/Robot/{name}Body").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        UsdPhysics.CollisionAPI.Apply(body)
        joint = joint_type.Define(stage, f"/Sources/Robot/{name}")
        joint.CreateAxisAttr().Set(axis)
        joint.CreateBody0Rel().SetTargets([base.GetPath()])
        joint.CreateBody1Rel().SetTargets([body.GetPath()])
        actuator = stage.DefinePrim(f"/Sources/Robot/{name}Actuator", "MjcActuator")
        actuator.CreateRelationship("mjc:target").SetTargets([joint.GetPath()])
        actuator.CreateAttribute("mjc:biasType", Sdf.ValueTypeNames.Token).Set("affine")
        actuator.CreateAttribute("mjc:gainPrm", Sdf.ValueTypeNames.DoubleArray).Set([100.0] + [0.0] * 9)
        actuator.CreateAttribute("mjc:biasPrm", Sdf.ValueTypeNames.DoubleArray).Set([0.0, -100.0, -2.0] + [0.0] * 7)
        for attribute, value in (
            ("ctrlRange:min", bounds[0]),
            ("ctrlRange:max", bounds[1]),
            ("forceRange:min", -effort),
            ("forceRange:max", effort),
        ):
            actuator.CreateAttribute("mjc:" + attribute, Sdf.ValueTypeNames.Double).Set(value)

    def create_builder():
        builder = ModelBuilder()
        NewtonMJWarpManager._register_builder_attributes(builder)
        return builder

    sources = ("/Sources/Table", "/Sources/Robot")
    builders = build_source_builders(
        stage, sources, create_builder, NewtonMJWarpManager._get_usd_import_schema_resolvers(), load_visual_shapes=False
    )
    assets = tuple(
        AssetBaseCfg(
            prim_path="/World/envs/env_[^/]+/" + source.rsplit("/", 1)[-1], spawn=SpawnerCfg(spawn_path=source)
        )
        for source in sources
    )
    plan = make_clone_plan(assets, ((0, 1),), 2)
    builder = create_builder()
    replicate_builder_mapping(
        builder, plan, np.zeros((2, 3)), np.tile([0, 0, 0, 1], (2, 1)), builders, env_ids=np.arange(2)
    )
    NewtonMJWarpManager._prepare_builder_for_finalize(builder)
    solver = SolverMuJoCo(builder.finalize(device="cpu"))
    np.testing.assert_array_equal(solver.model.mujoco.actuator_trnid.numpy()[:, 0], [6, 7, 14, 15])
    model = solver.mj_model
    for name, bounds, effort in (("Arm", (-1.0, 1.0), 87.0), ("Finger", (0.0, 0.04), 200.0)):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"_World_envs_env_0_Robot_{name}")
        assert joint >= 0
        columns = np.flatnonzero((model.actuator_trnid[:, 0] == joint) & model.actuator_ctrllimited)
        assert columns.size == 1
        np.testing.assert_allclose(model.actuator_ctrlrange[columns[0]], bounds, atol=1e-7)
        np.testing.assert_allclose(model.actuator_forcerange[columns[0]], (-effort, effort))
