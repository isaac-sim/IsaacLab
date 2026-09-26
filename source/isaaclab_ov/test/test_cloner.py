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

from pxr import Gf, Usd, UsdGeom

from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import make_clone_plan
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
    UsdGeom.Xform.Define(stage, "/World/envs/env_0").AddTranslateOp().Set((2, 0, 0))
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot").AddTranslateOp().Set((0.25, 0, 0))
    recipes = []
    manager = SimpleNamespace(_register_clone_transforms=lambda *recipe: recipes.append(recipe))
    plan = make_clone_plan(
        (AssetBaseCfg(prim_path="/World/envs/env_[^/]+/Robot"),),
        ((0, 0),),
        2,
        positions=np.array([[2, 0, 0], [5, 2, 3]], dtype=np.float32),
    )
    simulation = SimpleNamespace(stage=stage, physics_manager=manager)

    OvPhysxReplicateContext(simulation).replicate(plan, (0,))

    assert recipes[0][0:2] == ("/World/envs/env_0/Robot", ["/World/envs/env_1/Robot"])
    assert recipes[0][2][0] == pytest.approx((5.25, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0))
    assert recipes[1][1] == ["/World/envs/env_0/Robot_1", "/World/envs/env_1/Robot_1"]
    np.testing.assert_allclose(recipes[1][2], [(2.25, 0, 0, 0, 0, 0, 1), (5.25, 2, 3, 0, 0, 0, 1)])
    assert recipes[0][3] == [1]
    assert recipes[1][3] == [0, 1]


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
