# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import replace
from unittest import mock

import isaaclab_physx.physics.collision_filter as collision_filter_module
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
    for prim in Usd.PrimRange(stage.GetPrimAtPath(GENERATED_COLLISION_ROOT)):
        if prim.IsA(UsdPhysics.CollisionGroup):
            group = UsdPhysics.CollisionGroup(prim)
            if group.GetCollidersCollectionAPI().ComputeMembershipQuery().IsPathIncluded(collider_path):
                matches.append(group)
    assert len(matches) == 1
    return matches[0]


def _collides(stage: Usd.Stage, first: str, second: str) -> bool:
    return _group_for(stage, second).GetPath() in _group_for(stage, first).GetFilteredGroupsRel().GetTargets()


def _two_env_stage() -> tuple[Usd.Stage, ClonePlan]:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    for env_id in (0, 3):
        for asset in ("Robot", "Object", "Support"):
            _collider(stage, f"/World/cells/cell_{env_id}/{asset}/shape")
    _collider(stage, "/World/Ground/shape")
    return stage, ClonePlan(
        sources=("/World/cells/cell_0",),
        destinations=("/World/cells/cell_{}",),
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_ids=np.asarray([0, 3]),
        global_paths=("/World/Ground",),
        env_template="/World/cells/cell_{}",
    )


def test_isolation_only_reuses_compact_environment_groups() -> None:
    stage, plan = _two_env_stage()
    with mock.patch.object(collision_filter_module, "_discover_topology") as discover:
        apply_collision_filter(stage, "/physicsScene", plan, None)

    discover.assert_not_called()
    assert not stage.GetPrimAtPath(GENERATED_COLLISION_ROOT).IsValid()
    assert stage.GetPrimAtPath("/physicsScene").GetAttribute("physxScene:invertCollisionGroupFilter").Get()
    for index, env_id in enumerate((0, 3)):
        group = UsdPhysics.CollisionGroup.Get(stage, f"/World/collisions/group{index}")
        assert group.GetCollidersCollectionAPI().GetExpansionRuleAttr().Get() == "expandPrims"
        assert group.GetCollidersCollectionAPI().GetIncludesRel().GetTargets() == [
            Sdf.Path(f"/World/cells/cell_{env_id}")
        ]


def test_isolation_only_compiles_existing_authored_group_semantics() -> None:
    """World isolation must not reinterpret an asset's normal-mode filtered groups."""
    stage, plan = _two_env_stage()
    objects = UsdPhysics.CollisionGroup.Define(stage, "/World/AuthoredObjects")
    objects.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets(
        ["/World/cells/cell_0/Object/shape", "/World/cells/cell_3/Object/shape"]
    )
    ground = UsdPhysics.CollisionGroup.Define(stage, "/World/AuthoredGround")
    ground.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets(["/World/Ground/shape"])
    objects.CreateFilteredGroupsRel().SetTargets([ground.GetPath()])

    apply_collision_filter(stage, "/physicsScene", plan, None)

    assert stage.GetPrimAtPath(GENERATED_COLLISION_ROOT).IsValid()
    assert not _collides(stage, "/World/cells/cell_0/Object/shape", "/World/Ground/shape")
    assert _collides(stage, "/World/cells/cell_0/Robot/shape", "/World/Ground/shape")
    assert not _collides(stage, "/World/cells/cell_0/Robot/shape", "/World/cells/cell_3/Robot/shape")


def test_compiles_manager_and_authored_groups_with_environment_isolation() -> None:
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

    apply_collision_filter(stage, "/physicsScene", plan, cfg)

    robot = "/World/cells/cell_0/Robot/shape"
    obj = "/World/cells/cell_0/Object/shape"
    support = "/World/cells/cell_0/Support/shape"
    assert _collides(stage, robot, obj)
    assert _collides(stage, obj, support)
    assert not _collides(stage, robot, support)
    assert not _collides(stage, obj, "/World/Ground/shape")
    assert not _collides(stage, robot, "/World/cells/cell_3/Robot/shape")
    assert filtered_pairs.GetFilteredPairsRel().GetTargets() == [Sdf.Path(obj)]


def test_inverted_groups_select_the_sdf_path_without_disabling_convex_contacts() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    paths = {
        name: f"/World/envs/env_0/{asset}/mesh/colliders/{kind}"
        for name, asset, kind in (
            ("nut_sdf", "Nut", "sdf"),
            ("nut_convex", "Nut", "convex"),
            ("bolt_sdf", "Bolt", "sdf"),
            ("bolt_convex", "Bolt", "convex"),
        )
    }
    paths["other"] = "/World/envs/env_0/Other/collider"
    for path in paths.values():
        _collider(stage, path)
    cfg = CollisionFilterCfg(
        groups={
            "nut_sdf": CollisionGroupCfg(
                prim_path_exprs=(paths["nut_sdf"],),
                filtered_groups=("bolt_sdf",),
                invert_filtered_groups=True,
            ),
            "bolt_sdf": CollisionGroupCfg(
                prim_path_exprs=(paths["bolt_sdf"],),
                filtered_groups=("nut_sdf",),
                invert_filtered_groups=True,
            ),
            "nut_convex": CollisionGroupCfg(prim_path_exprs=(paths["nut_convex"],), filtered_groups=("bolt_convex",)),
            "bolt_convex": CollisionGroupCfg(prim_path_exprs=(paths["bolt_convex"],)),
        }
    )
    plan = ClonePlan(
        sources=(), destinations=(), clone_mask=np.zeros((0, 0), dtype=np.bool_), isolate_environments=False
    )

    apply_collision_filter(stage, "/physicsScene", plan, cfg)

    assert _collides(stage, paths["nut_sdf"], paths["bolt_sdf"])
    assert not _collides(stage, paths["nut_sdf"], paths["other"])
    assert not _collides(stage, paths["nut_convex"], paths["bolt_convex"])
    assert _collides(stage, paths["nut_convex"], paths["other"])
    assert _collides(stage, paths["bolt_convex"], paths["other"])


def test_isolation_false_keeps_cross_environment_policy_edges() -> None:
    stage, plan = _two_env_stage()
    plan = replace(plan, isolate_environments=False)
    cfg = CollisionFilterCfg(groups={"robot": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Robot/shape",))})

    apply_collision_filter(stage, "/physicsScene", plan, cfg)

    assert _collides(stage, "/World/cells/cell_0/Robot/shape", "/World/cells/cell_3/Robot/shape")


def test_missing_usd_destination_collider_fails_before_semantic_compilation() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/envs/env_1")
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    _collider(stage, "/World/envs/env_0/Robot/shape")
    plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_ids=np.asarray([0, 1]),
    )
    cfg = CollisionFilterCfg(groups={"robot": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Robot/shape",))})

    with pytest.raises(RuntimeError, match="USD destination collider topology"):
        apply_collision_filter(stage, "/physicsScene", plan, cfg)


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

    plan = ClonePlan(
        sources=(), destinations=(), clone_mask=np.zeros((0, 0), dtype=np.bool_), isolate_environments=False
    )
    cfg = CollisionFilterCfg(groups={"selected": CollisionGroupCfg(prim_path_exprs=(r"/World/Relevant/shape",))})
    apply_collision_filter(stage, "/physicsScene", plan, cfg)

    assert asset_layer.ExportToString() == asset_before
    assert not stage.GetPrimAtPath("/World/Relevant").IsInstance()
    assert stage.GetPrimAtPath("/World/Unrelated").IsInstance()
    composed_group = UsdPhysics.CollisionGroup.Get(stage, "/World/Relevant/group")
    assert (
        not composed_group.GetCollidersCollectionAPI().ComputeMembershipQuery().IsPathIncluded("/World/Relevant/shape")
    )
