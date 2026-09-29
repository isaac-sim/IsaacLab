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
from isaaclab_ov.cloner.replicate import _serialize_stage
from isaaclab_ov.physics.ovphysx_manager import OvPhysxManager

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

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
    monkeypatch.setattr(OvPhysxManager, "_clone_recipes", [])
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
    source_row.AddScaleOp().Set((2, 2, 2))
    body = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot/Body")
    body.AddTranslateOp().Set((1, 0, 0))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())

    monkeypatch.setattr(PhysicsManager, "_sim", SimpleNamespace(physics_manager=OvPhysxManager))
    ovphysx_replicate(
        stage,
        sources=["/World/envs/env_0/Robot", "/World/envs/env_9/Inactive", "/World/envs/env_0/Robot"],
        destinations=["/World/envs/env_{}/Robot", "/World/envs/env_{}/Inactive", "/World/envs/env_{}/Robot"],
        env_ids=np.array([0, 1], dtype=np.int64),
        mapping=np.array([[True, True], [False, False], [True, False]], dtype=np.bool_),
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

    assert len(OvPhysxManager._clone_recipes) == 1
    source, targets, poses, env_ids, source_env_id = OvPhysxManager._clone_recipes[0]
    assert source == "/World/envs/env_0/Robot"
    assert targets == ["/World/envs/env_0/Robot", "/World/envs/env_1/Robot"]
    assert env_ids == [0, 1]
    assert source_env_id == 0
    assert len(poses) == 2
    assert poses[1][:3] == pytest.approx(expected_transform[:3])
    orientation = np.asarray(poses[1][3:], dtype=np.float32)
    if np.dot(orientation, expected_orientation) < 0.0:
        orientation = -orientation
    assert orientation.tolist() == pytest.approx(expected_orientation.tolist())
    layer = Sdf.Layer.CreateAnonymous("export.usda")
    serialized, native = _serialize_stage(stage, OvPhysxManager._clone_recipes, full_stage=True)
    assert not native
    layer.ImportFromString(serialized)
    exported = Usd.Stage.Open(layer)
    target = exported.GetPrimAtPath(targets[1] + "/Body")
    actual = UsdGeom.XformCache().GetLocalToWorldTransform(target).ExtractTranslation()
    expected = _pose_matrix(expected_transform[:3], expected_transform[3:]).Transform(Gf.Vec3d(2, 0, 0))
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_ovphysx_context_consumes_plan():
    """The registered context publishes the rows routed to it by one clone plan."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/envs/env_0").AddTranslateOp().Set((2, 0, 0))
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot").AddTranslateOp().Set((0.25, 0, 0))
    recipes = []
    manager = SimpleNamespace(_clone_recipes=recipes)
    assets = (AssetBaseCfg(prim_path="/World/envs/env_[^/]+/Robot"),)
    positions = np.array([[2, 0, 0], [5, 2, 3], [8, 0, 0]], dtype=np.float32)
    plan = make_clone_plan(assets, ((0, 0), (0,)), 3, weights=(2, 1), positions=positions)
    OvPhysxReplicateContext(SimpleNamespace(stage=stage, physics_manager=manager)).replicate(plan, (0,))

    assert len(recipes) == 2
    assert recipes[0][0:2] == ("/World/envs/env_0/Robot", [f"/World/envs/env_{i}/Robot" for i in range(3)])
    assert recipes[0][2][1] == pytest.approx((5.25, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0))
    assert recipes[1][1] == ["/World/envs/env_0/Robot_1", "/World/envs/env_1/Robot_1"]
    np.testing.assert_allclose(recipes[1][2], [(2.25, 0, 0, 0, 0, 0, 1), (5.25, 2, 3, 0, 0, 0, 1)])
    assert recipes[0][3] == [0, 1, 2]
    assert recipes[1][3] == [0, 1]


def test_ovphysx_context_preserves_heterogeneous_world_prototypes():
    """Each world prototype keeps its authored geometry and disjoint destination IDs."""
    stage = Usd.Stage.CreateInMemory()
    for env_id, geometry_type in ((0, "Cube"), (1, "Sphere")):
        UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_id}")
        source = UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_id}/Object").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(source)
        stage.DefinePrim(f"/World/envs/env_{env_id}/Object/Geometry", geometry_type)

    recipes = []
    manager = SimpleNamespace(_clone_recipes=recipes)
    assets = (AssetBaseCfg(prim_path="/World/envs/env_[^/]+/Object"),) * 2
    plan = make_clone_plan(assets, ((0,), (1,)), 4, clone_strategy=lambda weights, count: np.arange(count) % 2)
    OvPhysxReplicateContext(SimpleNamespace(stage=stage, physics_manager=manager)).replicate(plan, (0, 1))

    assert [(source, targets, env_ids) for source, targets, _, env_ids, _ in recipes] == [
        ("/World/envs/env_0/Object", ["/World/envs/env_0/Object", "/World/envs/env_2/Object"], [0, 2]),
        ("/World/envs/env_1/Object", ["/World/envs/env_1/Object", "/World/envs/env_3/Object"], [1, 3]),
    ]


def test_raw_replicate_preserves_rigid_body_variants_and_env_ids(monkeypatch):
    """Each rigid-body geometry variant becomes one clone call with its selected env ids."""
    stage = Usd.Stage.CreateInMemory()
    for env_id, geometry_type in ((0, "Cube"), (1, "Sphere")):
        UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_id}")
        source = UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_id}/Object").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(source)
        stage.DefinePrim(f"/World/envs/env_{env_id}/Object/Geometry", geometry_type)

    recipes = []
    manager = SimpleNamespace(_clone_recipes=recipes)
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

    assert [(source, targets, env_ids) for source, targets, _, env_ids, _ in recipes] == [
        ("/World/envs/env_0/Object", [f"/World/envs/env_{i}/Object" for i in (0, 2, 4)], [0, 2, 4]),
        ("/World/envs/env_1/Object", [f"/World/envs/env_{i}/Object" for i in (1, 3, 5)], [1, 3, 5]),
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
    manager = SimpleNamespace(_clone_recipes=recipes)
    monkeypatch.setattr(PhysicsManager, "_sim", SimpleNamespace(physics_manager=manager))
    ovphysx_replicate(
        stage,
        sources=["/World/envs/env_0/Object", "/World/envs/env_1/Object"],
        destinations=["/World/envs/env_{}/Object", "/World/envs/env_{}/Object"],
        env_ids=np.arange(2, dtype=np.int64),
        mapping=np.eye(2, dtype=np.bool_),
    )

    assert [(source, targets, env_ids) for source, targets, _, env_ids, _ in recipes] == [
        (f"/World/envs/env_{i}/Object", [f"/World/envs/env_{i}/Object"], [i]) for i in range(2)
    ]
    assert not _serialize_stage(stage, recipes, full_stage=False)[1]


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
    manager = SimpleNamespace(_clone_recipes=recipes)
    monkeypatch.setattr(PhysicsManager, "_sim", SimpleNamespace(physics_manager=manager))
    ovphysx_replicate(
        stage,
        sources=["/World/envs/env_0/Robot", "/World/envs/env_1/Robot"],
        destinations=["/World/envs/env_{}/Robot", "/World/envs/env_{}/Robot"],
        env_ids=np.arange(4, dtype=np.int64),
        mapping=np.array([[True, False, True, False], [False, True, False, True]], dtype=np.bool_),
    )

    assert [(source, targets, env_ids) for source, targets, _, env_ids, _ in recipes] == [
        ("/World/envs/env_0/Robot", ["/World/envs/env_0/Robot", "/World/envs/env_2/Robot"], [0, 2]),
        ("/World/envs/env_1/Robot", ["/World/envs/env_1/Robot", "/World/envs/env_3/Robot"], [1, 3]),
    ]


def test_register_clone_preserves_translation_only_compatibility(monkeypatch):
    """World positions become target-root poses with identity rotations."""
    monkeypatch.setattr(OvPhysxManager, "_clone_recipes", [])

    OvPhysxManager.register_clone("/World/env_0", ["/World/env_1"], [(1.0, 2.0, 3.0)])

    expected_recipes = [("/World/env_0", ["/World/env_1"], [(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)], None, 0)]
    assert OvPhysxManager._clone_recipes == expected_recipes


def test_raw_replicate_validates_sources_and_pose_arrays():
    """Cloning requires source/anchor prims and a complete, correctly shaped pose array."""
    stage = Usd.Stage.CreateInMemory()
    sources, destinations = ["/World/envs/env_0/Robot"], ["/World/envs/env_{}/Robot"]
    options = dict(env_ids=np.array([0, 1], dtype=np.int64), mapping=np.array([[True, True]], dtype=np.bool_))
    with pytest.raises(ValueError, match="/World/envs/env_0/Robot"):
        ovphysx_replicate(stage, sources, destinations, **options)

    UsdGeom.Xform.Define(stage, sources[0])
    for name, shape in (
        ("positions", (2, 2)),
        ("quaternions", (2, 3)),  # Wrong component count.
        ("positions", (1, 3)),
        ("quaternions", (1, 4)),  # Missing a selected environment.
    ):
        with pytest.raises(ValueError, match=rf"{name} must have shape"):
            ovphysx_replicate(stage, sources, destinations, **options, **{name: np.zeros(shape, dtype=np.float32)})

    class StageWithoutAnchor:
        def GetPrimAtPath(self, path):
            if str(path) == "/World/envs/env_0":
                return Usd.Prim()
            return stage.GetPrimAtPath(path)

    with pytest.raises(ValueError, match="/World/envs/env_0"):
        ovphysx_replicate(StageWithoutAnchor(), sources, destinations, **options)


@pytest.mark.parametrize("full_stage", [False, True])
@pytest.mark.parametrize("env_template", ["/World/envs/env_{}", "/Scenes/World_{}"])
@pytest.mark.parametrize("independent_physics", [False, True])
def test_clone_export_preserves_sources_and_independent_assets(full_stage, env_template, independent_physics):
    """Export exact targets, preserve unrelated assets, and replay without consuming the recipes."""
    stage = Usd.Stage.CreateInMemory()
    for env_id, variant in enumerate(("cube", "sphere", "target", "target")):
        root = env_template.format(env_id)
        prim = stage.DefinePrim(root + "/Object", "Xform")
        prim.CreateAttribute("test:variant", Sdf.ValueTypeNames.String).Set(variant)
    stage.DefinePrim("/Shared/Ground", "Xform")
    for world in (1, 3):
        independent = stage.DefinePrim(env_template.format(world) + "/Independent", "Xform")
        if independent_physics:
            UsdPhysics.RigidBodyAPI.Apply(independent)
    assets = tuple(
        AssetBaseCfg(prim_path=env_template.format("[^/]+") + name)
        for name in ("/Object", "/Object", "/Support", "/Independent")
    )
    worlds = ((0, 2), (1, 2, 3))
    plan = make_clone_plan(assets, worlds, 4, env_template=env_template, clone_strategy=lambda _, n: np.arange(n) % 2)
    object_paths = [env_template.format(i) + "/Object" for i in range(4)]
    recipes = [(object_paths[i], [object_paths[i], object_paths[i + 2]], [], [i, i + 2], i) for i in range(2)]
    # A target can also be inside a world which retains a different prototype.
    support = stage.DefinePrim(env_template.format(0) + "/Support", "Xform")
    support.CreateAttribute("test:physics", Sdf.ValueTypeNames.Bool).Set(True)
    targets = [env_template.format(i) + "/Support" for i in range(4)]
    recipes.append((str(support.GetPath()), targets, [], list(range(4)), 0))
    stage.DefinePrim(targets[1], "Xform")
    before = stage.GetRootLayer().ExportToString()

    for _ in range(2):
        layer = Sdf.Layer.CreateAnonymous("export.usda")
        serialized, native = _serialize_stage(stage, recipes, full_stage, plan)
        assert layer.ImportFromString(serialized)
        exported = Usd.Stage.Open(layer)
        for env_id, variant in enumerate(("cube", "sphere")):
            prim = exported.GetPrimAtPath(object_paths[env_id])
            assert prim.GetAttribute("test:variant").Get() == variant
        assert exported.GetPrimAtPath("/Shared/Ground")
        assert exported.GetPrimAtPath(env_template.format(3) + "/Independent")
        assert bool(exported.GetPrimAtPath(env_template.format(2))) is full_stage
        for world, target in enumerate(targets):
            prim = exported.GetPrimAtPath(target)
            assert bool(prim) is (full_stage or world < 2 or (independent_physics and world == 3))
            if prim:
                assert prim.GetAttribute("test:physics").Get()
        if full_stage:
            assert not native
        else:
            assert {
                (source, target, world) for source, paths, _, ids, _ in native for target, world in zip(paths, ids)
            } == {
                (env_template.format(i) + name, env_template.format(i + 2) + name, i + 2)
                for i in range(2)
                for name in ("/Object", "/Support")
                if not independent_physics or i == 0
            }
        assert stage.GetRootLayer().ExportToString() == before
        assert len(recipes) == 3


def test_full_stage_export_preserves_nested_authored_opinions():
    """Copy parents before children and keep independently authored descendants."""
    stage = Usd.Stage.CreateInMemory()
    source = stage.DefinePrim("/Source/Robot", "Xform")
    source.CreateAttribute("test:physics", Sdf.ValueTypeNames.Bool).Set(True)
    child = stage.DefinePrim("/Source/Robot/Link", "Xform")
    child.CreateAttribute("test:mass", Sdf.ValueTypeNames.Float).Set(3.0)
    UsdGeom.Mesh.Define(stage, "/Target/Robot")
    stage.DefinePrim("/Target/Robot/Camera", "Camera")
    recipes = [
        ("/Source/Robot/Link", ["/Target/Robot/Link", "/Other/Nested/Robot/Link"], [], [1, 2], 0),
        ("/Source/Robot", ["/Target/Robot"], [], [1], 0),
    ]
    layer = Sdf.Layer.CreateAnonymous("export.usda")
    serialized, native = _serialize_stage(stage, recipes, full_stage=True)
    assert not native
    assert layer.ImportFromString(serialized)
    exported = Usd.Stage.Open(layer)
    assert exported.GetPrimAtPath("/Target/Robot").GetAttribute("test:physics").Get()
    assert exported.GetPrimAtPath("/Target/Robot").IsA(UsdGeom.Mesh)
    assert exported.GetPrimAtPath("/Target/Robot/Camera").IsA(UsdGeom.Camera)
    for root in ("/Target", "/Other/Nested"):
        prim = exported.GetPrimAtPath(root + "/Robot/Link")
        assert prim.IsDefined()
        assert prim.GetAttribute("test:mass").Get() == 3.0
    assert not stage.GetPrimAtPath("/Target/Robot/Link")


def test_native_clone_export_rejects_source_overlap():
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/Source", "Xform")
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match="overlaps a clone source"):
        _serialize_stage(stage, [("/World/Source", ["/World"], [], [0], 0)], full_stage=False)
    assert stage.GetRootLayer().ExportToString() == before
