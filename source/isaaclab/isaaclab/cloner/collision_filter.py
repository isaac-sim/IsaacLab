# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX collision-group authoring for clones."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Usd


def filter_collisions(
    stage: Usd.Stage,
    physicsscene_path: str,
    collision_root_path: str,
    prim_paths: list[str],
    global_paths: Sequence[str] = (),
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
    root_spec = stage.GetRootLayer().GetPrimAtPath(collision_root_path)

    def define_group(name: str, includes: Sequence[str]):
        """Author one PhysicsCollisionGroup that collides with itself and return its ``filteredGroups`` rel."""
        group = Sdf.PrimSpec(root_spec, name, Sdf.SpecifierDef, "PhysicsCollisionGroup")
        group.SetInfo(Usd.Tokens.apiSchemas, Sdf.TokenListOp.Create({"CollectionAPI:colliders"}))
        expansion_rule = Sdf.AttributeSpec(
            group, "collection:colliders:expansionRule", Sdf.ValueTypeNames.Token, Sdf.VariabilityUniform
        )
        expansion_rule.default = "expandPrims"
        includes_rel = Sdf.RelationshipSpec(group, "collection:colliders:includes", False)
        for path in includes:
            includes_rel.targetPathList.Append(path)
        # With inverted filtering objects do not collide across groups by default, so a group must list
        # itself as a filtered group for its own members to collide with each other.
        filtered_groups = Sdf.RelationshipSpec(group, "physics:filteredGroups", False)
        filtered_groups.targetPathList.Append(f"{collision_root_path}/{name}")
        return filtered_groups

    with Sdf.ChangeBlock():
        global_filtered_groups = define_group("global_group", global_paths) if global_paths else None
        for i, prim_path in enumerate(prim_paths):
            filtered_groups = define_group(f"group{i}", [prim_path])
            if global_filtered_groups is not None:
                filtered_groups.targetPathList.Append(f"{collision_root_path}/global_group")
                global_filtered_groups.targetPathList.Append(f"{collision_root_path}/group{i}")
