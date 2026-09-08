# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OvPhysX cloning."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from isaaclab_ov._clone import CloneRecipe
from isaaclab_ov.cloner import OvPhysxReplicateContext, ovphysx_replicate
from isaaclab_ov.physics.ovphysx_manager import OvPhysxManager

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab.cloner import ClonePlan
from isaaclab.physics import CollisionFilterCfg, CollisionGroupCfg, PhysicsManager


def _pose_matrix(position: tuple[float, float, float], quaternion: tuple[float, float, float, float]) -> Gf.Matrix4d:
    """Build a USD pose matrix from an xyzw quaternion."""
    matrix = Gf.Matrix4d(1.0)
    matrix.SetTranslateOnly(Gf.Vec3d(*position))
    matrix.SetRotateOnly(Gf.Quatd(quaternion[3], Gf.Vec3d(*quaternion[:3])))
    return matrix


def test_nested_clone_uses_final_target_pose(monkeypatch):
    """Nested clone rows keep their source-local pose under the target environment."""
    monkeypatch.setattr(OvPhysxManager, "_active_clone_recipes", [])
    monkeypatch.setattr(OvPhysxManager, "_pending_clones", [])
    half_sqrt_two = math.sqrt(0.5)
    source_half_angle_sin = 0.5
    source_half_angle_cos = math.sqrt(0.75)
    target_half_angle_sin = math.sin(math.pi / 8.0)
    target_half_angle_cos = math.cos(math.pi / 8.0)
    stage = Usd.Stage.CreateInMemory()

    source_env = UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    source_env.AddTransformOp().Set(_pose_matrix((4.0, 5.0, 6.0), (0.0, half_sqrt_two, 0.0, half_sqrt_two)))
    source_row = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot")
    source_row.AddTransformOp().Set(
        _pose_matrix((0.0, 1.0, 2.0), (source_half_angle_sin, 0.0, 0.0, source_half_angle_cos))
    )

    monkeypatch.setattr(PhysicsManager, "_sim", SimpleNamespace(physics_manager=OvPhysxManager))
    ovphysx_replicate(
        stage,
        sources=["/World/envs/env_0/Robot", "/World/envs/env_9/Inactive"],
        destinations=["/World/envs/env_{}/Robot", "/World/envs/env_{}/Inactive"],
        env_ids=np.array([0, 1], dtype=np.int64),
        mapping=np.array([[True, True], [False, False]], dtype=np.bool_),
        positions=np.array([[4.0, 5.0, 6.0], [10.0, 20.0, 30.0]], dtype=np.float32),
        quaternions=np.array(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, target_half_angle_sin, target_half_angle_cos]],
            dtype=np.float32,
        ),
    )

    expected_orientation = np.array(
        [
            target_half_angle_cos * source_half_angle_sin,
            target_half_angle_sin * source_half_angle_sin,
            target_half_angle_sin * source_half_angle_cos,
            target_half_angle_cos * source_half_angle_cos,
        ],
        dtype=np.float32,
    )
    expected_transform = (10.0 - half_sqrt_two, 20.0 + half_sqrt_two, 32.0, *expected_orientation.tolist())

    assert len(OvPhysxManager._pending_clones) == 1
    pending = OvPhysxManager._pending_clones[0]
    assert pending.source == "/World/envs/env_0/Robot"
    assert pending.targets == ("/World/envs/env_1/Robot",)
    assert len(pending.transforms) == 1
    assert pending.transforms[0][:3] == pytest.approx(expected_transform[:3])
    orientation = np.asarray(pending.transforms[0][3:], dtype=np.float32)
    if np.dot(orientation, expected_orientation) < 0.0:
        orientation = -orientation
    assert orientation.tolist() == pytest.approx(expected_orientation.tolist())


def test_ovphysx_context_consumes_plan():
    """The registered context publishes the rows routed to it by one clone plan."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/envs/env_10")
    UsdGeom.Xform.Define(stage, "/World/envs/env_10/Robot")
    recipes = []
    manager = SimpleNamespace(_register_clone_transforms=lambda *recipe: recipes.append(recipe))
    simulation = SimpleNamespace(stage=stage, physics_manager=manager)
    plan = ClonePlan(
        sources=("/World/envs/env_10/Robot",),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_ids=np.array([10, 20], dtype=np.int64),
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32),
        context_rows={OvPhysxReplicateContext: (0,)},
    )

    OvPhysxReplicateContext(simulation).replicate(plan)

    assert recipes[0][0:2] == ("/World/envs/env_10/Robot", ["/World/envs/env_20/Robot"])
    assert recipes[0][2][0] == pytest.approx((1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0))


def test_register_clone_preserves_translation_only_compatibility(monkeypatch):
    """World positions become target-root poses with identity rotations."""
    monkeypatch.setattr(OvPhysxManager, "_active_clone_recipes", [])
    monkeypatch.setattr(OvPhysxManager, "_pending_clones", [])

    OvPhysxManager.register_clone("/World/env_0", ["/World/env_1"], [(1.0, 2.0, 3.0)])

    expected_recipes = [CloneRecipe("/World/env_0", ("/World/env_1",), ((1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),))]
    assert OvPhysxManager._active_clone_recipes == expected_recipes
    assert OvPhysxManager._pending_clones == expected_recipes


def test_isolated_heterogeneous_recipes_reuse_plan_environment_ids(monkeypatch):
    """Every clone call uses the same plan ID for targets in the same world."""
    monkeypatch.setattr(OvPhysxManager, "_active_clone_recipes", [])
    monkeypatch.setattr(OvPhysxManager, "_pending_clones", [])
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    for name in ("Robot", "Object"):
        UsdGeom.Xform.Define(stage, f"/World/envs/env_0/{name}")
    simulation = SimpleNamespace(stage=stage, physics_manager=OvPhysxManager)
    plan = ClonePlan(
        sources=("/World/envs/env_0/Robot", "/World/envs/env_0/Object"),
        destinations=("/World/envs/env_{}/Robot", "/World/envs/env_{}/Object"),
        clone_mask=np.asarray([[True, True, False], [True, True, True]]),
        env_ids=np.asarray([0, 7, 9]),
        positions=np.zeros((3, 3), dtype=np.float32),
        context_rows={OvPhysxReplicateContext: (0, 1)},
    )
    context = OvPhysxReplicateContext(simulation)
    context.configure_environment_isolation(True)
    context.replicate(plan)

    assert [recipe.env_ids for recipe in OvPhysxManager._pending_clones] == [(7,), (7, 9)]

    class FakePhysX:
        def __init__(self):
            self.calls = []

        def clone(self, source, targets, transforms, *, env_ids):
            self.calls.append((source, targets, env_ids))
            return len(self.calls)

        def wait_op(self, operation):
            pass

    fake = FakePhysX()
    OvPhysxManager._replay_pending_clones(fake, requires_full_stage=False)
    assert fake.calls == [
        ("/World/envs/env_0/Robot", ["/World/envs/env_7/Robot"], [7]),
        ("/World/envs/env_0/Object", ["/World/envs/env_7/Object", "/World/envs/env_9/Object"], [7, 9]),
    ]


def test_manager_hook_configures_native_clone_isolation(monkeypatch):
    """The manager passes native isolation state to its simulation-scoped clone context."""
    calls = []
    context = SimpleNamespace(configure_environment_isolation=lambda enabled: calls.append(enabled))
    simulation = SimpleNamespace(get_or_create_backend=lambda *args: context)
    monkeypatch.setattr(PhysicsManager, "_sim", simulation)
    monkeypatch.setattr(OvPhysxManager, "_clone_environment_isolation", False)
    plan = ClonePlan(sources=(), destinations=(), clone_mask=np.zeros((0, 0), dtype=np.bool_))

    OvPhysxManager._apply_collision_filter_impl(plan, None, isolate_environments=True, replicate_physics=True)

    assert calls == [True]
    assert OvPhysxManager._clone_environment_isolation is True


def test_manager_hook_rejects_declarative_groups_explicitly(monkeypatch):
    """OVPhysX rejects manager groups instead of silently weakening their semantics."""
    monkeypatch.setattr(PhysicsManager, "_sim", SimpleNamespace())
    plan = ClonePlan(sources=(), destinations=(), clone_mask=np.zeros((0, 0), dtype=np.bool_))
    cfg = CollisionFilterCfg(groups={"all": CollisionGroupCfg(prim_path_exprs=(r"/World/.*",))})

    with pytest.raises(NotImplementedError, match="does not yet support PhysicsCfg.collision_filter"):
        OvPhysxManager._apply_collision_filter_impl(plan, cfg, isolate_environments=True, replicate_physics=True)


def test_manager_hook_retains_usd_isolation_without_native_replication(monkeypatch):
    """USD-only replication retains collision-group based environment isolation."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    UsdGeom.Xform.Define(stage, "/World/envs/env_1")
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    context = SimpleNamespace(configure_environment_isolation=lambda enabled: None)
    simulation = SimpleNamespace(
        stage=stage,
        cfg=SimpleNamespace(physics_prim_path="/physicsScene"),
        get_or_create_backend=lambda *args: context,
    )
    monkeypatch.setattr(PhysicsManager, "_sim", simulation)
    plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_ids=np.asarray([0, 1]),
    )

    OvPhysxManager._apply_collision_filter_impl(plan, None, isolate_environments=True, replicate_physics=False)

    assert stage.GetPrimAtPath("/physicsScene").GetAttribute("physxScene:invertCollisionGroupFilter").Get()
    assert stage.GetPrimAtPath("/World/collisions/group0").IsA(UsdPhysics.CollisionGroup)
    assert stage.GetPrimAtPath("/World/collisions/group1").IsA(UsdPhysics.CollisionGroup)


def test_raw_replicate_rejects_invalid_source_prim():
    """Active clone rows require a valid source prim."""
    stage = Usd.Stage.CreateInMemory()
    with pytest.raises(ValueError, match="/World/envs/env_0/Robot"):
        ovphysx_replicate(
            stage,
            sources=["/World/envs/env_0/Robot"],
            destinations=["/World/envs/env_{}/Robot"],
            env_ids=np.array([0, 1], dtype=np.int64),
            mapping=np.array([[True, True]], dtype=np.bool_),
        )


def test_raw_replicate_rejects_invalid_source_anchor():
    """Active nested clone rows require a valid source-environment anchor."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot")

    class StageWithoutAnchor:
        def GetPrimAtPath(self, path):
            if str(path) == "/World/envs/env_0":
                return Usd.Prim()
            return stage.GetPrimAtPath(path)

    with pytest.raises(ValueError, match="/World/envs/env_0"):
        ovphysx_replicate(
            StageWithoutAnchor(),
            sources=["/World/envs/env_0/Robot"],
            destinations=["/World/envs/env_{}/Robot"],
            env_ids=np.array([0, 1], dtype=np.int64),
            mapping=np.array([[True, True]], dtype=np.bool_),
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [("positions", np.zeros((1, 3), dtype=np.float32)), ("quaternions", np.zeros((1, 4), dtype=np.float32))],
)
def test_raw_replicate_rejects_pose_array_missing_selected_environment(name, value):
    """Provided pose arrays include every selected environment."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot")
    with pytest.raises(ValueError, match=name):
        ovphysx_replicate(
            stage,
            sources=["/World/envs/env_0/Robot"],
            destinations=["/World/envs/env_{}/Robot"],
            env_ids=np.array([0, 1], dtype=np.int64),
            mapping=np.array([[True, True]], dtype=np.bool_),
            **{name: value},
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [("positions", np.zeros((2, 2), dtype=np.float32)), ("quaternions", np.zeros((2, 3), dtype=np.float32))],
)
def test_raw_replicate_rejects_malformed_pose_array(name, value):
    """Provided pose arrays use the documented component counts."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot")
    with pytest.raises(ValueError, match=rf"{name} must have shape"):
        ovphysx_replicate(
            stage,
            sources=["/World/envs/env_0/Robot"],
            destinations=["/World/envs/env_{}/Robot"],
            env_ids=np.array([0, 1], dtype=np.int64),
            mapping=np.array([[True, True]], dtype=np.bool_),
            **{name: value},
        )
