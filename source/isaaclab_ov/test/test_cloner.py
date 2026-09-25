# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OvPhysX cloning."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from isaaclab_ov.cloner import OvPhysxReplicateContext, ovphysx_replicate
from isaaclab_ov.physics.ovphysx_manager import OvPhysxManager

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab.cloner import ClonePlan
from isaaclab.physics import PhysicsManager


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
    pending_source, pending_targets, pending_transforms, pending_env_ids = OvPhysxManager._pending_clones[0]
    assert pending_source == "/World/envs/env_0/Robot"
    assert pending_targets == ["/World/envs/env_1/Robot"]
    assert pending_env_ids == [1]
    assert len(pending_transforms) == 1
    assert pending_transforms[0][:3] == pytest.approx(expected_transform[:3])
    orientation = np.asarray(pending_transforms[0][3:], dtype=np.float32)
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
    assert recipes[0][3] == [20]


def test_raw_replicate_preserves_rigid_body_variants_and_env_ids(monkeypatch):
    """Each rigid-body geometry variant becomes one clone call with its selected env ids."""
    stage = Usd.Stage.CreateInMemory()
    for env_id, geometry_type in ((0, "Cube"), (1, "Sphere")):
        UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_id}")
        source = UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_id}/Object").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(source)
        stage.DefinePrim(f"/World/envs/env_{env_id}/Object/Geometry", geometry_type)

    recipes = []
    manager = SimpleNamespace(_register_clone_transforms=lambda *recipe: recipes.append(recipe))
    monkeypatch.setattr(PhysicsManager, "_sim", SimpleNamespace(physics_manager=manager))
    ovphysx_replicate(
        stage,
        sources=["/World/envs/env_0/Object", "/World/envs/env_1/Object"],
        destinations=["/World/envs/env_{}/Object", "/World/envs/env_{}/Object"],
        env_ids=np.arange(6, dtype=np.int64),
        mapping=np.array(
            [[True, False, True, False, True, False], [False, True, False, True, False, True]], dtype=np.bool_
        ),
    )

    assert [(source, targets, env_ids) for source, targets, _, env_ids in recipes] == [
        ("/World/envs/env_0/Object", ["/World/envs/env_2/Object", "/World/envs/env_4/Object"], [2, 4]),
        ("/World/envs/env_1/Object", ["/World/envs/env_3/Object", "/World/envs/env_5/Object"], [3, 5]),
    ]
    assert set(recipes[0][3]).isdisjoint(recipes[1][3])


def test_raw_replicate_preserves_source_only_geometry_variants(monkeypatch):
    """A nonzero source environment is retained even when its variant has no clone targets."""
    stage = Usd.Stage.CreateInMemory()
    for env_id in range(2):
        UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_id}")
        source = UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_id}/Object").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(source)

    recipes = []
    manager = SimpleNamespace(_register_clone_transforms=lambda *recipe: recipes.append(recipe))
    monkeypatch.setattr(PhysicsManager, "_sim", SimpleNamespace(physics_manager=manager))
    ovphysx_replicate(
        stage,
        sources=["/World/envs/env_0/Object", "/World/envs/env_1/Object"],
        destinations=["/World/envs/env_{}/Object", "/World/envs/env_{}/Object"],
        env_ids=np.arange(2, dtype=np.int64),
        mapping=np.eye(2, dtype=np.bool_),
    )

    assert recipes == [("/World/envs/env_1/Object", [], [], [])]


def test_raw_replicate_preserves_articulation_geometry_variants_and_env_ids(monkeypatch):
    """Equivalent articulations with distinct link geometry retain separate clone batches."""
    stage = Usd.Stage.CreateInMemory()
    for env_id, geometry_type in ((0, "Cube"), (1, "Sphere")):
        UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_id}")
        robot = UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_id}/Robot").GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(robot)
        link = UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_id}/Robot/Link").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(link)
        UsdPhysics.RevoluteJoint.Define(stage, f"/World/envs/env_{env_id}/Robot/Joint")
        stage.DefinePrim(f"/World/envs/env_{env_id}/Robot/Link/Geometry", geometry_type)

    recipes = []
    manager = SimpleNamespace(_register_clone_transforms=lambda *recipe: recipes.append(recipe))
    monkeypatch.setattr(PhysicsManager, "_sim", SimpleNamespace(physics_manager=manager))
    ovphysx_replicate(
        stage,
        sources=["/World/envs/env_0/Robot", "/World/envs/env_1/Robot"],
        destinations=["/World/envs/env_{}/Robot", "/World/envs/env_{}/Robot"],
        env_ids=np.arange(4, dtype=np.int64),
        mapping=np.array([[True, False, True, False], [False, True, False, True]], dtype=np.bool_),
    )

    assert [(source, targets, env_ids) for source, targets, _, env_ids in recipes] == [
        ("/World/envs/env_0/Robot", ["/World/envs/env_2/Robot"], [2]),
        ("/World/envs/env_1/Robot", ["/World/envs/env_3/Robot"], [3]),
    ]


def test_raw_replicate_rejects_incompatible_articulation_dof_structure():
    """Articulation variants with different joint counts fail before clone registration."""
    stage = Usd.Stage.CreateInMemory()
    for env_id, joint_count in ((0, 1), (1, 2)):
        UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_id}")
        robot = UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_id}/Robot").GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(robot)
        for joint_id in range(joint_count):
            UsdPhysics.RevoluteJoint.Define(stage, f"/World/envs/env_{env_id}/Robot/Joint_{joint_id}")

    with pytest.raises(ValueError, match="incompatible rigid-body or joint topology"):
        ovphysx_replicate(
            stage,
            sources=["/World/envs/env_0/Robot", "/World/envs/env_1/Robot"],
            destinations=["/World/envs/env_{}/Robot", "/World/envs/env_{}/Robot"],
            env_ids=np.arange(4, dtype=np.int64),
            mapping=np.array([[True, False, True, False], [False, True, False, True]], dtype=np.bool_),
        )


@pytest.mark.parametrize("changed_axis", ["rotX", "rotY", "transX"])
def test_raw_replicate_rejects_d6_axis_layout_mismatch(changed_axis):
    """Equal joint counts must not hide different enabled D6 axes."""
    stage = Usd.Stage.CreateInMemory()
    sources = [f"/World/envs/env_{i}/Robot" for i in range(2)]
    for source, unlocked in zip(sources, ["rotZ", changed_axis]):
        robot = UsdGeom.Xform.Define(stage, source).GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(robot)
        joint = UsdPhysics.Joint.Define(stage, source + "/Joint").GetPrim()
        for axis in ("rotX", "rotY", "rotZ", "transX", "transY", "transZ"):
            if axis != unlocked:
                limit = UsdPhysics.LimitAPI.Apply(joint, axis)
                limit.CreateLowAttr(1.0)
                limit.CreateHighAttr(-1.0)
    with pytest.raises(ValueError, match="incompatible rigid-body or joint topology"):
        ovphysx_replicate(
            stage,
            sources=sources,
            destinations=["/World/envs/env_{}/Robot"] * 2,
            env_ids=np.arange(4),
            mapping=np.array([[True, False, True, False], [False, True, False, True]]),
        )


def test_register_clone_preserves_translation_only_compatibility(monkeypatch):
    """World positions become target-root poses with identity rotations."""
    monkeypatch.setattr(OvPhysxManager, "_active_clone_recipes", [])
    monkeypatch.setattr(OvPhysxManager, "_pending_clones", [])

    OvPhysxManager.register_clone("/World/env_0", ["/World/env_1"], [(1.0, 2.0, 3.0)])

    expected_recipes = [("/World/env_0", ["/World/env_1"], [(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)], None)]
    assert OvPhysxManager._active_clone_recipes == expected_recipes
    assert OvPhysxManager._pending_clones == expected_recipes


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
    [
        ("positions", np.zeros((2, 2), dtype=np.float32)),
        ("quaternions", np.zeros((2, 3), dtype=np.float32)),
        # Missing the second selected environment.
        ("positions", np.zeros((1, 3), dtype=np.float32)),
        ("quaternions", np.zeros((1, 4), dtype=np.float32)),
    ],
)
def test_raw_replicate_rejects_malformed_pose_array(name, value):
    """Provided pose arrays use the documented component counts and include every selected environment."""
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
