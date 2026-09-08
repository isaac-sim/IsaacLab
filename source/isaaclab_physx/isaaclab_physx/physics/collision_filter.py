# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compile backend-neutral collision policy into PhysX collision groups.

PhysX currently assigns a collider to the first USD collision group that contains it and implements
only the scene-wide ``physxScene:invertCollisionGroupFilter`` switch. It does not implement the
standard per-group ``physics:invertFilteredGroups`` attribute. Consequently, independently authored
groups cannot be layered with environment isolation or manager groups without changing their
meaning. This module snapshots those inputs, partitions colliders into exclusive effective profiles,
and lowers the resulting allow graph to one generated PhysX group per profile.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Usd, UsdPhysics

    from isaaclab.cloner import ClonePlan
    from isaaclab.physics import CollisionFilterCfg


GENERATED_COLLISION_ROOT = "/World/IsaacLabCollisionFilter"
"""Root-layer namespace owned by the PhysX collision-policy compiler."""


@dataclass(frozen=True)
class _Collider:
    path: str
    world: int | None


@dataclass(frozen=True)
class _Profile:
    world: int | None
    manager_groups: tuple[str, ...]
    authored_groups: tuple[str, ...]


@dataclass(frozen=True)
class _AuthoredPolicy:
    memberships: dict[str, tuple[str, ...]]
    enabled_pairs: dict[tuple[str, str], bool]
    relevant_groups: tuple[str, ...]

    def allows(self, first: tuple[str, ...], second: tuple[str, ...]) -> bool:
        if not first or not second:
            return True
        return all(self.enabled_pairs[_ordered_pair(a, b)] for a in first for b in second)


def apply_collision_filter(
    stage: Usd.Stage,
    physics_scene_path: str,
    plan: ClonePlan,
    cfg: CollisionFilterCfg | None,
    *,
    isolate_environments: bool,
    replicate_physics: bool,
) -> None:
    """Realize manager collision policy and environment isolation on a PhysX USD stage.

    Destination colliders must already be present in USD. Production clone plans satisfy this by
    running :class:`isaaclab.cloner.UsdReplicateContext` before the manager barrier. Native PhysX
    replication alone cannot assign a different generated group to an unauthored destination.

    Existing collision groups are read before any override is authored. Groups that contain a
    managed collider are then emptied in the stage root layer, preserving referenced asset layers
    and :class:`UsdPhysics.PhysicsFilteredPairsAPI` relationships. If such a group is inside an
    instance, only its enclosing instance root is expanded.

    Args:
        stage: Stage containing assembled collider prims.
        physics_scene_path: Path of the active PhysX physics scene.
        plan: Completed clone layout.
        cfg: Optional backend-neutral manager policy.
        isolate_environments: Whether different planned worlds may collide.
        replicate_physics: Whether native physics replication will consume the assembled topology.

    Raises:
        RuntimeError: If required scene/destination topology is missing or the generated namespace
            is already occupied.
    """
    if not isolate_environments and (cfg is None or not cfg.groups):
        return

    # Deferred imports avoid binding usd-core before Kit has initialized its own USD runtime.
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

    scene_prim = stage.GetPrimAtPath(physics_scene_path)
    if not scene_prim.IsValid():
        raise RuntimeError(f"Cannot apply PhysX collision filtering: no physics scene at {physics_scene_path!r}.")
    if stage.GetPrimAtPath(GENERATED_COLLISION_ROOT).IsValid():
        raise RuntimeError(f"PhysX collision-filter namespace already exists at {GENERATED_COLLISION_ROOT!r}.")

    colliders = _discover_topology(stage, plan, replicate_physics=replicate_physics)
    if not colliders:
        return
    manager_memberships = _manager_memberships(colliders, plan, cfg)
    scene_inversion_attr = scene_prim.GetAttribute("physxScene:invertCollisionGroupFilter")
    scene_inverted = bool(scene_inversion_attr.Get()) if scene_inversion_attr else False
    authored_policy = _snapshot_authored_policy(stage, tuple(c.path for c in colliders), scene_inverted=scene_inverted)

    profiles: dict[_Profile, list[str]] = {}
    for collider in colliders:
        profile = _Profile(
            collider.world if isolate_environments else None,
            manager_memberships[collider.path],
            authored_policy.memberships[collider.path],
        )
        profiles.setdefault(profile, []).append(collider.path)

    profile_list = tuple(profiles)
    enabled = {
        (i, j): _profiles_allow(profile_list[i], profile_list[j], cfg, authored_policy)
        for i in range(len(profile_list))
        for j in range(i, len(profile_list))
    }

    root_layer = stage.GetRootLayer()
    with Usd.EditContext(stage, Usd.EditTarget(root_layer)):
        for group_path in authored_policy.relevant_groups:
            group = UsdPhysics.CollisionGroup.Get(stage, group_path)
            if not group or group.GetPrim().IsInstanceProxy():
                raise RuntimeError(
                    f"Cannot override authored collision group at {group_path!r} in the stage root layer."
                )
            group.GetCollidersCollectionAPI().BlockCollection()

        UsdGeom.Scope.Define(stage, GENERATED_COLLISION_ROOT)
        generated_scope = root_layer.GetPrimAtPath(GENERATED_COLLISION_ROOT)
        generated_paths = [Sdf.Path(f"{GENERATED_COLLISION_ROOT}/group_{i:04d}") for i in range(len(profile_list))]
        with Sdf.ChangeBlock():
            for i, profile in enumerate(profile_list):
                group_spec = Sdf.PrimSpec(generated_scope, f"group_{i:04d}", Sdf.SpecifierDef, "PhysicsCollisionGroup")
                group_spec.SetInfo(Usd.Tokens.apiSchemas, Sdf.TokenListOp.Create(["CollectionAPI:colliders"]))
                expansion_rule = Sdf.AttributeSpec(
                    group_spec,
                    "collection:colliders:expansionRule",
                    Sdf.ValueTypeNames.Token,
                    Sdf.VariabilityUniform,
                )
                expansion_rule.default = Usd.Tokens.explicitOnly
                includes = Sdf.RelationshipSpec(group_spec, "collection:colliders:includes", False)
                for path in profiles[profile]:
                    includes.targetPathList.Append(Sdf.Path(path))
                allowed = [generated_paths[j] for j in range(len(profile_list)) if enabled[_ordered_index_pair(i, j)]]
                filtered_groups = Sdf.RelationshipSpec(group_spec, "physics:filteredGroups", False)
                for path in allowed:
                    filtered_groups.targetPathList.Append(path)

        scene_prim.CreateAttribute("physxScene:invertCollisionGroupFilter", Sdf.ValueTypeNames.Bool, custom=True).Set(
            True
        )


def _discover_topology(stage: Usd.Stage, plan: ClonePlan, *, replicate_physics: bool) -> tuple[_Collider, ...]:
    from pxr import Usd, UsdPhysics  # noqa: PLC0415

    paths = {
        str(prim.GetPath())
        for prim in stage.Traverse(Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    }
    if not paths:
        return ()

    row_count = len(plan.sources)
    if len(plan.destinations) != row_count:
        raise ValueError("ClonePlan.sources and ClonePlan.destinations must have equal length.")
    if plan.env_ids is None:
        if row_count:
            raise ValueError("ClonePlan.env_ids is required to compile replicated collision topology.")
        env_ids: tuple[int, ...] = ()
    else:
        env_ids = tuple(map(int, plan.env_ids))
    expected_shape = (row_count, len(env_ids))
    if tuple(plan.clone_mask.shape) != expected_shape:
        raise ValueError(f"ClonePlan.clone_mask must have shape {expected_shape}, got {plan.clone_mask.shape}.")

    worlds: dict[str, int | None] = {path: None for path in paths}
    missing: list[str] = []
    for row, (source, destination) in enumerate(zip(plan.sources, plan.destinations)):
        source_colliders = [path for path in paths if _is_at_or_below(path, source)]
        for column, selected in enumerate(plan.clone_mask[row]):
            if not selected:
                continue
            world = env_ids[column]
            destination_root = destination.format(world)
            for source_path in source_colliders:
                target_path = destination_root + source_path[len(source) :]
                if target_path not in paths:
                    missing.append(target_path)
                    continue
                previous = worlds[target_path]
                if previous is not None and previous != world:
                    raise ValueError(f"Collider {target_path!r} is assigned to worlds {previous} and {world}.")
                worlds[target_path] = world

    if missing:
        mode = " before native PhysX replication" if replicate_physics else " at the manager barrier"
        preview = ", ".join(repr(path) for path in sorted(set(missing))[:5])
        extra = " ..." if len(set(missing)) > 5 else ""
        raise RuntimeError(
            "PhysX collision filtering requires USD destination collider topology"
            f"{mode}; missing {preview}{extra}. Ensure UsdReplicateContext runs before collision filtering."
        )

    global_roots = tuple(plan.global_paths)
    return tuple(
        _Collider(path, None if any(_is_at_or_below(path, root) for root in global_roots) else worlds[path])
        for path in sorted(paths)
    )


def _manager_memberships(
    colliders: tuple[_Collider, ...], plan: ClonePlan, cfg: CollisionFilterCfg | None
) -> dict[str, tuple[str, ...]]:
    if cfg is None:
        return {collider.path: () for collider in colliders}
    try:
        env_regex = plan.env_template.format("[^/]+")
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError(f"Invalid ClonePlan.env_template {plan.env_template!r}.") from exc
    patterns = {
        name: tuple(re.compile(selector.replace("{ENV_REGEX_NS}", env_regex)) for selector in group.prim_path_exprs)
        for name, group in cfg.groups.items()
    }
    return {
        collider.path: tuple(
            name for name, selectors in patterns.items() if any(regex.fullmatch(collider.path) for regex in selectors)
        )
        for collider in colliders
    }


def _snapshot_authored_policy(
    stage: Usd.Stage, collider_paths: tuple[str, ...], *, scene_inverted: bool
) -> _AuthoredPolicy:
    from pxr import Sdf, UsdPhysics  # noqa: PLC0415

    groups = _authored_groups(stage)
    relevant = _relevant_groups(groups, collider_paths)
    _deinstance_groups(stage, relevant)
    groups = _authored_groups(stage)
    relevant = _relevant_groups(groups, collider_paths)

    memberships: dict[str, list[str]] = {path: [] for path in collider_paths}
    for group_path in relevant:
        query = groups[group_path].GetCollidersCollectionAPI().ComputeMembershipQuery()
        for collider_path in collider_paths:
            if query.IsPathIncluded(Sdf.Path(collider_path)):
                memberships[collider_path].append(group_path)

    enabled_pairs: dict[tuple[str, str], bool] = {}
    if scene_inverted:
        allowed_targets = {path: set(map(str, groups[path].GetFilteredGroupsRel().GetTargets())) for path in relevant}
        for i, first in enumerate(relevant):
            for second in relevant[i:]:
                enabled_pairs[_ordered_pair(first, second)] = (
                    second in allowed_targets[first] or first in allowed_targets[second]
                )
    else:
        table = UsdPhysics.CollisionGroup.ComputeCollisionGroupTable(stage)
        for i, first in enumerate(relevant):
            for second in relevant[i:]:
                enabled_pairs[_ordered_pair(first, second)] = table.IsCollisionEnabled(
                    Sdf.Path(first), Sdf.Path(second)
                )

    return _AuthoredPolicy(
        {path: tuple(group_paths) for path, group_paths in memberships.items()}, enabled_pairs, relevant
    )


def _authored_groups(stage: Usd.Stage) -> dict[str, UsdPhysics.CollisionGroup]:
    from pxr import Usd, UsdPhysics  # noqa: PLC0415

    return {
        str(prim.GetPath()): UsdPhysics.CollisionGroup(prim)
        for prim in stage.Traverse(Usd.TraverseInstanceProxies())
        if prim.IsA(UsdPhysics.CollisionGroup) and not _is_at_or_below(str(prim.GetPath()), GENERATED_COLLISION_ROOT)
    }


def _relevant_groups(groups: dict[str, UsdPhysics.CollisionGroup], collider_paths: tuple[str, ...]) -> tuple[str, ...]:
    from pxr import Sdf  # noqa: PLC0415

    collider_sdf_paths = tuple(map(Sdf.Path, collider_paths))
    return tuple(
        path
        for path, group in groups.items()
        if any(group.GetCollidersCollectionAPI().ComputeMembershipQuery().IsPathIncluded(p) for p in collider_sdf_paths)
    )


def _deinstance_groups(stage: Usd.Stage, group_paths: tuple[str, ...]) -> None:
    from pxr import Usd  # noqa: PLC0415

    root_layer = stage.GetRootLayer()
    while True:
        proxy = next(
            (stage.GetPrimAtPath(path) for path in group_paths if stage.GetPrimAtPath(path).IsInstanceProxy()), None
        )
        if proxy is None:
            return
        instance_root = proxy.GetParent()
        while instance_root and (not instance_root.IsInstance() or instance_root.IsInstanceProxy()):
            instance_root = instance_root.GetParent()
        if not instance_root:
            raise RuntimeError(f"Cannot deinstance the enclosing root for collision group {proxy.GetPath()}.")
        with Usd.EditContext(stage, Usd.EditTarget(root_layer)):
            if not instance_root.SetInstanceable(False):
                raise RuntimeError(f"Cannot deinstance {instance_root.GetPath()} in the stage root layer.")


def _profiles_allow(
    first: _Profile, second: _Profile, cfg: CollisionFilterCfg | None, authored: _AuthoredPolicy
) -> bool:
    if first.world is not None and second.world is not None and first.world != second.world:
        return False
    if cfg is not None and (
        _manager_side_filters(first.manager_groups, second.manager_groups, cfg)
        or _manager_side_filters(second.manager_groups, first.manager_groups, cfg)
    ):
        return False
    return authored.allows(first.authored_groups, second.authored_groups)


def _manager_side_filters(first: tuple[str, ...], second: tuple[str, ...], cfg: CollisionFilterCfg) -> bool:
    for group_name in first:
        group = cfg.groups[group_name]
        if not second and group.invert_filtered_groups:
            return True
        for other_name in second:
            if (other_name in group.filtered_groups) != group.invert_filtered_groups:
                return True
    return False


def _is_at_or_below(path: str, root: str) -> bool:
    root = root.rstrip("/")
    return path == root or path.startswith(root + "/")


def _ordered_pair(first: str, second: str) -> tuple[str, str]:
    return (first, second) if first <= second else (second, first)


def _ordered_index_pair(first: int, second: int) -> tuple[int, int]:
    return (first, second) if first <= second else (second, first)


__all__ = ["GENERATED_COLLISION_ROOT", "apply_collision_filter"]
