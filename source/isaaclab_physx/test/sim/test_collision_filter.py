# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import pytest
from isaaclab_physx.physics.collision_filter import GENERATED_COLLISION_ROOT, apply_collision_filter

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.cloner import ClonePlan
from isaaclab.physics import CollisionFilterCfg, CollisionGroupCfg


def _collider(stage: Usd.Stage, path: str) -> None:
    UsdGeom.Cube.Define(stage, path)
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(path))


def _group_for(stage: Usd.Stage, collider_path: str) -> UsdPhysics.CollisionGroup:
    matches = []
    root = stage.GetPrimAtPath(GENERATED_COLLISION_ROOT)
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdPhysics.CollisionGroup):
            continue
        group = UsdPhysics.CollisionGroup(prim)
        if group.GetCollidersCollectionAPI().ComputeMembershipQuery().IsPathIncluded(collider_path):
            matches.append(group)
    assert len(matches) == 1
    return matches[0]


def _collides(stage: Usd.Stage, first: str, second: str) -> bool:
    first_group = _group_for(stage, first)
    second_path = _group_for(stage, second).GetPath()
    return second_path in first_group.GetFilteredGroupsRel().GetTargets()


def _two_env_stage() -> tuple[Usd.Stage, ClonePlan]:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    for env_id in (0, 3):
        for asset in ("Robot", "Object", "Support"):
            _collider(stage, f"/World/cells/cell_{env_id}/{asset}/shape")
    _collider(stage, "/World/Ground/shape")
    plan = ClonePlan(
        sources=("/World/cells/cell_0",),
        destinations=("/World/cells/cell_{}",),
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_ids=np.asarray([0, 3]),
        global_paths=("/World/Ground",),
        env_template="/World/cells/cell_{}",
    )
    return stage, plan


def test_compiles_authored_and_manager_groups_with_environment_isolation() -> None:
    stage, plan = _two_env_stage()
    objects = UsdPhysics.CollisionGroup.Define(stage, "/World/AuthoredObjects")
    objects.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets(
        ["/World/cells/cell_0/Object/shape", "/World/cells/cell_3/Object/shape"]
    )
    ground = UsdPhysics.CollisionGroup.Define(stage, "/World/AuthoredGround")
    ground.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets(["/World/Ground/shape"])
    objects.CreateFilteredGroupsRel().SetTargets([ground.GetPath()])
    filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath("/World/cells/cell_0/Robot/shape"))
    filtered_pairs.CreateFilteredPairsRel().SetTargets(["/World/cells/cell_0/Object/shape"])

    cfg = CollisionFilterCfg(
        groups={
            "robot": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Robot/shape",), filtered_groups=("support",)),
            "object": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Object/shape",)),
            "support": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Support/shape",)),
        }
    )
    apply_collision_filter(stage, "/physicsScene", plan, cfg, isolate_environments=True, replicate_physics=True)

    robot_0 = "/World/cells/cell_0/Robot/shape"
    object_0 = "/World/cells/cell_0/Object/shape"
    support_0 = "/World/cells/cell_0/Support/shape"
    assert _collides(stage, robot_0, object_0)
    assert _collides(stage, object_0, support_0)
    assert not _collides(stage, robot_0, support_0)
    assert not _collides(stage, object_0, "/World/Ground/shape")
    assert _collides(stage, robot_0, "/World/Ground/shape")
    assert not _collides(stage, robot_0, "/World/cells/cell_3/Robot/shape")

    assert not objects.GetCollidersCollectionAPI().ComputeMembershipQuery().IsPathIncluded(object_0)
    assert not ground.GetCollidersCollectionAPI().ComputeMembershipQuery().IsPathIncluded("/World/Ground/shape")
    assert filtered_pairs.GetFilteredPairsRel().GetTargets() == [Sdf.Path(object_0)]
    assert stage.GetPrimAtPath("/physicsScene").GetAttribute("physxScene:invertCollisionGroupFilter").Get()


def test_isolation_false_keeps_cross_environment_policy_edges() -> None:
    stage, plan = _two_env_stage()
    cfg = CollisionFilterCfg(groups={"robot": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Robot/shape",))})
    apply_collision_filter(stage, "/physicsScene", plan, cfg, isolate_environments=False, replicate_physics=False)

    assert _collides(stage, "/World/cells/cell_0/Robot/shape", "/World/cells/cell_3/Robot/shape")


def test_no_policy_and_no_isolation_is_a_noop() -> None:
    stage, plan = _two_env_stage()
    root_before = stage.GetRootLayer().ExportToString()

    apply_collision_filter(stage, "/physicsScene", plan, None, isolate_environments=False, replicate_physics=True)

    assert stage.GetRootLayer().ExportToString() == root_before


def test_preserves_preexisting_scene_inverted_allow_edges() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    scene.GetPrim().CreateAttribute("physxScene:invertCollisionGroupFilter", Sdf.ValueTypeNames.Bool, custom=True).Set(
        True
    )
    for path in ("/World/A1", "/World/A2", "/World/B", "/World/C", "/World/Ungrouped"):
        _collider(stage, path)
    groups = {}
    for name, members in {
        "A": ("/World/A1", "/World/A2"),
        "B": ("/World/B",),
        "C": ("/World/C",),
    }.items():
        groups[name] = UsdPhysics.CollisionGroup.Define(stage, f"/World/{name}Group")
        groups[name].GetCollidersCollectionAPI().CreateIncludesRel().SetTargets(members)
    groups["A"].CreateFilteredGroupsRel().SetTargets([groups["B"].GetPath()])
    plan = ClonePlan(sources=(), destinations=(), clone_mask=np.zeros((0, 0), dtype=np.bool_))
    cfg = CollisionFilterCfg(groups={"all": CollisionGroupCfg(prim_path_exprs=(r"/World/(A1|A2|B|C|Ungrouped)",))})

    apply_collision_filter(stage, "/physicsScene", plan, cfg, isolate_environments=False, replicate_physics=False)

    assert _collides(stage, "/World/A1", "/World/B")
    assert _collides(stage, "/World/B", "/World/A1")
    assert not _collides(stage, "/World/A1", "/World/C")
    assert not _collides(stage, "/World/A1", "/World/A2")
    assert _collides(stage, "/World/A1", "/World/Ungrouped")


def test_missing_usd_destination_collider_fails_before_native_replication() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    _collider(stage, "/World/envs/env_0/Robot/shape")
    plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_ids=np.asarray([0, 1]),
    )

    with pytest.raises(RuntimeError, match="USD destination collider topology"):
        apply_collision_filter(stage, "/physicsScene", plan, None, isolate_environments=True, replicate_physics=True)


def test_deinstances_only_relevant_group_roots_in_the_stage_root_layer() -> None:
    asset_stage = Usd.Stage.CreateInMemory()
    asset = UsdGeom.Xform.Define(asset_stage, "/Asset")
    asset_stage.SetDefaultPrim(asset.GetPrim())
    _collider(asset_stage, "/Asset/shape")
    group = UsdPhysics.CollisionGroup.Define(asset_stage, "/Asset/group")
    group.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets(["/Asset/shape"])
    asset_layer = asset_stage.GetRootLayer()
    asset_before = asset_layer.ExportToString()

    plain_stage = Usd.Stage.CreateInMemory()
    plain = UsdGeom.Xform.Define(plain_stage, "/Asset")
    plain_stage.SetDefaultPrim(plain.GetPrim())
    _collider(plain_stage, "/Asset/shape")

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    relevant = stage.DefinePrim("/World/Relevant", "Xform")
    relevant.GetReferences().AddReference(asset_layer.identifier, "/Asset")
    relevant.SetInstanceable(True)
    unrelated = stage.DefinePrim("/World/Unrelated", "Xform")
    unrelated.GetReferences().AddReference(plain_stage.GetRootLayer().identifier, "/Asset")
    unrelated.SetInstanceable(True)
    assert stage.GetPrimAtPath("/World/Relevant/group").IsInstanceProxy()

    plan = ClonePlan(sources=(), destinations=(), clone_mask=np.zeros((0, 0), dtype=np.bool_))
    cfg = CollisionFilterCfg(groups={"selected": CollisionGroupCfg(prim_path_exprs=(r"/World/Relevant/shape",))})
    apply_collision_filter(stage, "/physicsScene", plan, cfg, isolate_environments=False, replicate_physics=False)

    assert asset_layer.ExportToString() == asset_before
    assert not stage.GetPrimAtPath("/World/Relevant").IsInstance()
    assert stage.GetPrimAtPath("/World/Unrelated").IsInstance()
    composed_group = UsdPhysics.CollisionGroup.Get(stage, "/World/Relevant/group")
    assert (
        not composed_group.GetCollidersCollectionAPI().ComputeMembershipQuery().IsPathIncluded("/World/Relevant/shape")
    )
