# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX collision-group authoring for clones."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Usd


def filter_collisions(
    stage: Usd.Stage,
    physicsscene_path: str,
    collision_root_path: str,
    prim_paths: list[str],
    global_paths: list[str] = [],
) -> None:
    """Create inverted collision groups for clones (PhysX only).

    Sets PhysX scene attributes and collision groups on the prim at ``physicsscene_path``
    (no PhysxSchema import). Call only when the physics backend is PhysX; Newton uses
    its own collision/world handling and does not use USD PhysX collision groups.

    Creates one PhysicsCollisionGroup per prim under ``collision_root_path``, enabling
    inverted filtering so clones don't collide across groups. Optionally adds a global
    group that collides with all.

    Args:
        stage: USD stage.
        physicsscene_path: Path to PhysicsScene prim.
        collision_root_path: Root scope for collision groups.
        prim_paths: Per-clone prim paths.
        global_paths: Optional global-collider paths.

    """
    # Deferred: importing pxr from the kit-less usd-core wheel before Kit boots corrupts Kit's
    # own USD runtime. Keeping it in the body means resolving the ``cloner.filter_collisions``
    # attribute stays pxr-free — only calling it, on a live PhysX stage, pulls pxr in.
    from pxr import Sdf, Usd, UsdGeom  # noqa: PLC0415

    scene_prim = stage.GetPrimAtPath(physicsscene_path)
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
