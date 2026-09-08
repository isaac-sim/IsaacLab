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

from isaaclab.physics._collision_filter import CompiledCollisionFilter

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
    authored_groups: tuple[int, ...]


@dataclass(frozen=True)
class _AuthoredPolicy:
    memberships: dict[str, tuple[int, ...]]
    related_groups: dict[int, frozenset[int]]
    scene_inverted: bool
    relevant_groups: tuple[str, ...]

    def allows(self, first: tuple[int, ...], second: tuple[int, ...]) -> bool:
        if not first or not second:
            return True
        relations = (
            second_group in self.related_groups[first_group] or first_group in self.related_groups[second_group]
            for first_group in first
            for second_group in second
        )
        return all(relations) if self.scene_inverted else not any(relations)


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

    colliders = _discover_topology(
        stage,
        plan,
        isolate_environments=isolate_environments,
        replicate_physics=replicate_physics,
    )
    if not colliders:
        return
    manager_policy = None if cfg is None else CompiledCollisionFilter(cfg, plan.env_template)
    manager_memberships = _manager_memberships(colliders, manager_policy)
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
    profile_indices_by_world: dict[int | None, list[int]] = {}
    for index, profile in enumerate(profile_list):
        profile_indices_by_world.setdefault(profile.world, []).append(index)
    global_indices = profile_indices_by_world.get(None, [])

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
                candidates = (
                    range(len(profile_list))
                    if profile.world is None
                    else (*global_indices, *profile_indices_by_world[profile.world])
                )
                allowed = [
                    generated_paths[j]
                    for j in candidates
                    if _profiles_allow(profile, profile_list[j], manager_policy, authored_policy)
                ]
                filtered_groups = Sdf.RelationshipSpec(group_spec, "physics:filteredGroups", False)
                for path in allowed:
                    filtered_groups.targetPathList.Append(path)

        scene_prim.CreateAttribute("physxScene:invertCollisionGroupFilter", Sdf.ValueTypeNames.Bool, custom=True).Set(
            True
        )


def _discover_topology(
    stage: Usd.Stage,
    plan: ClonePlan,
    *,
    isolate_environments: bool,
    replicate_physics: bool,
) -> tuple[_Collider, ...]:
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

    global_roots = tuple(plan.global_paths)
    env_id_set = set(env_ids)
    environment_matcher = _environment_path_matcher(plan.env_template)
    worlds: dict[str, int | None] = {}
    for path in paths:
        match = environment_matcher.fullmatch(path)
        matched_world = None if match is None else int(match.group("env"))
        worlds[path] = matched_world if matched_world in env_id_set else None
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

    if isolate_environments and len(env_ids) > 1:
        unresolved = sorted(
            path
            for path, world in worlds.items()
            if world is None and not any(_is_at_or_below(path, root) for root in global_roots)
        )
        if unresolved:
            preview = ", ".join(repr(path) for path in unresolved[:5])
            extra = " ..." if len(unresolved) > 5 else ""
            raise RuntimeError(
                "PhysX environment isolation found colliders outside every planned environment and "
                f"ClonePlan.global_paths: {preview}{extra}. Declare shared collider roots as global paths."
            )

    return tuple(
        _Collider(path, None if any(_is_at_or_below(path, root) for root in global_roots) else worlds[path])
        for path in sorted(paths)
    )


def _environment_path_matcher(env_template: str) -> re.Pattern[str]:
    marker = "__ISAACLAB_ENV_ID__"
    try:
        concrete = env_template.format(marker)
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError(f"Invalid ClonePlan.env_template {env_template!r}.") from exc
    if concrete.count(marker) != 1:
        raise ValueError("ClonePlan.env_template must contain exactly one environment-id replacement field.")
    prefix, suffix = concrete.split(marker)
    return re.compile(rf"{re.escape(prefix)}(?P<env>-?[0-9]+){re.escape(suffix)}(?:/.*)?")


def _manager_memberships(
    colliders: tuple[_Collider, ...], policy: CompiledCollisionFilter | None
) -> dict[str, tuple[str, ...]]:
    if policy is None:
        return {collider.path: () for collider in colliders}
    return {collider.path: policy.memberships(collider.path) for collider in colliders}


def _snapshot_authored_policy(
    stage: Usd.Stage, collider_paths: tuple[str, ...], *, scene_inverted: bool
) -> _AuthoredPolicy:
    groups = _authored_groups(stage)
    group_memberships = _group_collider_memberships(stage, groups, collider_paths)
    _deinstance_groups(stage, tuple(group_memberships))
    groups = _authored_groups(stage)
    group_memberships = _group_collider_memberships(stage, groups, collider_paths)
    relevant = tuple(group_memberships)

    effective_keys = {}
    for path, group in groups.items():
        merge_name = group.GetMergeGroupNameAttr().Get()
        effective_keys[path] = ("merge", merge_name) if merge_name else ("path", path)
    key_ids = {key: index for index, key in enumerate(sorted(set(effective_keys.values())))}
    group_ids = {path: key_ids[key] for path, key in effective_keys.items()}

    memberships: dict[str, set[int]] = {path: set() for path in collider_paths}
    for group_path, member_paths in group_memberships.items():
        group_id = group_ids[group_path]
        for collider_path in member_paths:
            memberships[collider_path].add(group_id)

    related_groups: dict[int, set[int]] = {group_id: set() for group_id in key_ids.values()}
    for group_path, group in groups.items():
        source_id = group_ids[group_path]
        for target_path in map(str, group.GetFilteredGroupsRel().GetTargets()):
            if target_path in group_ids:
                related_groups[source_id].add(group_ids[target_path])

    return _AuthoredPolicy(
        {path: tuple(sorted(member_ids)) for path, member_ids in memberships.items()},
        {group_id: frozenset(targets) for group_id, targets in related_groups.items()},
        scene_inverted,
        relevant,
    )


def _authored_groups(stage: Usd.Stage) -> dict[str, UsdPhysics.CollisionGroup]:
    from pxr import Usd, UsdPhysics  # noqa: PLC0415

    return {
        str(prim.GetPath()): UsdPhysics.CollisionGroup(prim)
        for prim in stage.Traverse(Usd.TraverseInstanceProxies())
        if prim.IsA(UsdPhysics.CollisionGroup) and not _is_at_or_below(str(prim.GetPath()), GENERATED_COLLISION_ROOT)
    }


def _group_collider_memberships(
    stage: Usd.Stage,
    groups: dict[str, UsdPhysics.CollisionGroup],
    collider_paths: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    from pxr import Usd  # noqa: PLC0415

    collider_path_set = set(collider_paths)
    result = {}
    for group_path, group in groups.items():
        query = group.GetCollidersCollectionAPI().ComputeMembershipQuery()
        included = Usd.CollectionAPI.ComputeIncludedPaths(query, stage, Usd.TraverseInstanceProxies())
        members = tuple(sorted(collider_path_set.intersection(map(str, included))))
        if members:
            result[group_path] = members
    return result


def _deinstance_groups(stage: Usd.Stage, group_paths: tuple[str, ...]) -> None:
    from pxr import Usd  # noqa: PLC0415

    root_layer = stage.GetRootLayer()
    while True:
        instance_roots = {}
        for group_path in group_paths:
            proxy = stage.GetPrimAtPath(group_path)
            if not proxy.IsInstanceProxy():
                continue
            instance_root = proxy.GetParent()
            while instance_root and (not instance_root.IsInstance() or instance_root.IsInstanceProxy()):
                instance_root = instance_root.GetParent()
            if not instance_root:
                raise RuntimeError(f"Cannot deinstance the enclosing root for collision group {proxy.GetPath()}.")
            instance_roots[str(instance_root.GetPath())] = group_path
        if not instance_roots:
            return
        with Usd.EditContext(stage, Usd.EditTarget(root_layer)):
            for instance_root_path, group_path in instance_roots.items():
                instance_root = stage.GetPrimAtPath(instance_root_path)
                if not instance_root.SetInstanceable(False):
                    raise RuntimeError(
                        f"Cannot deinstance {instance_root_path} for collision group {group_path!r} "
                        "in the stage root layer."
                    )


def _profiles_allow(
    first: _Profile,
    second: _Profile,
    policy: CompiledCollisionFilter | None,
    authored: _AuthoredPolicy,
) -> bool:
    if first.world is not None and second.world is not None and first.world != second.world:
        return False
    if policy is not None and policy.filters(first.manager_groups, second.manager_groups):
        return False
    return authored.allows(first.authored_groups, second.authored_groups)


def _is_at_or_below(path: str, root: str) -> bool:
    root = root.rstrip("/")
    return path == root or path.startswith(root + "/")


__all__ = ["GENERATED_COLLISION_ROOT", "apply_collision_filter"]
