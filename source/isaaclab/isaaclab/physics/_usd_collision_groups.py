# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Internal USD collision-group lowering shared by physics managers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Usd


def _author_environment_isolation_groups(
    stage: Usd.Stage,
    physics_scene_path: str,
    collision_root_path: str,
    prim_paths: Sequence[str],
    global_paths: Sequence[str] = (),
) -> None:
    """Author compact inverted collision groups for replicated environments."""
    # Deferred: importing pxr from the kit-less usd-core wheel before Kit boots corrupts Kit's
    # own USD runtime. Keep it in the manager-invoked function body.
    from pxr import Sdf, Usd, UsdGeom  # noqa: PLC0415

    scene_prim = stage.GetPrimAtPath(physics_scene_path)
    # We invert the collision group filters for more efficient collision filtering across environments
    invert_attr = scene_prim.CreateAttribute("physxScene:invertCollisionGroupFilter", Sdf.ValueTypeNames.Bool)
    invert_attr.Set(True)

    # Make sure we create the collision_scope in the RootLayer since the edit target
    # may be a live layer in the case of Live Sync.
    with Usd.EditContext(stage, Usd.EditTarget(stage.GetRootLayer())):
        UsdGeom.Scope.Define(stage, collision_root_path)

    with Sdf.ChangeBlock():
        if len(global_paths) > 0:
            global_collision_group_path = collision_root_path + "/global_group"
            # add collision group prim
            global_collision_group = Sdf.PrimSpec(
                stage.GetRootLayer().GetPrimAtPath(collision_root_path),
                "global_group",
                Sdf.SpecifierDef,
                "PhysicsCollisionGroup",
            )
            # prepend collision API schema
            global_collision_group.SetInfo(Usd.Tokens.apiSchemas, Sdf.TokenListOp.Create({"CollectionAPI:colliders"}))

            # expansion rule
            expansion_rule = Sdf.AttributeSpec(
                global_collision_group,
                "collection:colliders:expansionRule",
                Sdf.ValueTypeNames.Token,
                Sdf.VariabilityUniform,
            )
            expansion_rule.default = "expandPrims"

            # includes rel
            global_includes_rel = Sdf.RelationshipSpec(global_collision_group, "collection:colliders:includes", False)
            for global_path in global_paths:
                global_includes_rel.targetPathList.Append(global_path)

            # filteredGroups rel
            global_filtered_groups = Sdf.RelationshipSpec(global_collision_group, "physics:filteredGroups", False)
            # We are using inverted collision group filtering, which means objects by default don't collide across
            # groups. We need to add this group as a filtered group, so that objects within this group collide with
            # each other.
            global_filtered_groups.targetPathList.Append(global_collision_group_path)

        # set collision groups and filters
        for i, prim_path in enumerate(prim_paths):
            collision_group_path = collision_root_path + f"/group{i}"
            # add collision group prim
            collision_group = Sdf.PrimSpec(
                stage.GetRootLayer().GetPrimAtPath(collision_root_path),
                f"group{i}",
                Sdf.SpecifierDef,
                "PhysicsCollisionGroup",
            )
            # prepend collision API schema
            collision_group.SetInfo(Usd.Tokens.apiSchemas, Sdf.TokenListOp.Create({"CollectionAPI:colliders"}))

            # expansion rule
            expansion_rule = Sdf.AttributeSpec(
                collision_group,
                "collection:colliders:expansionRule",
                Sdf.ValueTypeNames.Token,
                Sdf.VariabilityUniform,
            )
            expansion_rule.default = "expandPrims"

            # includes rel
            includes_rel = Sdf.RelationshipSpec(collision_group, "collection:colliders:includes", False)
            includes_rel.targetPathList.Append(prim_path)

            # filteredGroups rel
            filtered_groups = Sdf.RelationshipSpec(collision_group, "physics:filteredGroups", False)
            # We are using inverted collision group filtering, which means objects by default don't collide across
            # groups. We need to add this group as a filtered group, so that objects within this group collide with
            # each other.
            filtered_groups.targetPathList.Append(collision_group_path)
            if len(global_paths) > 0:
                filtered_groups.targetPathList.Append(global_collision_group_path)
                global_filtered_groups.targetPathList.Append(collision_group_path)


def _matches_environment_isolation_groups(
    stage: Usd.Stage,
    physics_scene_path: str,
    collision_root_path: str,
    prim_paths: Sequence[str],
    global_paths: Sequence[str] = (),
) -> bool:
    """Return whether the stage contains exactly the compact environment groups."""
    from pxr import Usd, UsdPhysics  # noqa: PLC0415

    scene_prim = stage.GetPrimAtPath(physics_scene_path)
    inversion = scene_prim.GetAttribute("physxScene:invertCollisionGroupFilter")
    if not scene_prim.IsValid() or not inversion or inversion.Get() is not True:
        return False

    root = collision_root_path.rstrip("/")
    env_groups = [f"{root}/group{i}" for i in range(len(prim_paths))]
    global_group = f"{root}/global_group"
    expected_groups = {*env_groups, *([global_group] if global_paths else [])}
    authored_groups = {
        str(prim.GetPath()): UsdPhysics.CollisionGroup(prim)
        for prim in stage.Traverse(Usd.TraverseInstanceProxies())
        if prim.IsA(UsdPhysics.CollisionGroup)
    }
    if set(authored_groups) != expected_groups:
        return False

    def matches(group_path: str, includes: set[str], filtered_groups: set[str]) -> bool:
        group = authored_groups[group_path]
        prim = group.GetPrim()
        collection = group.GetCollidersCollectionAPI()
        return (
            not prim.IsInstanceProxy()
            and collection.GetExpansionRuleAttr().Get() == "expandPrims"
            and set(map(str, collection.GetIncludesRel().GetTargets())) == includes
            and not collection.GetExcludesRel().GetTargets()
            and not collection.GetIncludeRootAttr().Get()
            and not collection.GetMembershipExpressionAttr().Get()
            and set(map(str, group.GetFilteredGroupsRel().GetTargets())) == filtered_groups
            and not group.GetMergeGroupNameAttr().Get()
        )

    for group_path, prim_path in zip(env_groups, prim_paths):
        related = {group_path, *([global_group] if global_paths else [])}
        if not matches(group_path, {prim_path}, related):
            return False
    return not global_paths or matches(global_group, set(global_paths), {global_group, *env_groups})
