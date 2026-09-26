# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Query world topology and plain native path mappings without depending on clone contexts."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence

import numpy as np

from . import path as pth
from .clone_plan import ClonePlan


def get_asset_prototypes(plan: ClonePlan, path_expr: str | None = None) -> np.ndarray:
    """Return asset-prototype IDs selected by their declared cfg paths, without expanding instances.

    Args:
        plan: Asset prototypes and world compositions.
        path_expr: Exact cfg ``prim_path`` or a regular expression matching the complete declared
            path string. None selects all definitions, including unused prototypes.

    Returns:
        Ascending asset-prototype IDs, shape [num_matches], dtype int32, each included once.
        Generated native paths are not matched.
    """
    if path_expr is None:
        return np.arange(len(plan.asset_prototypes), dtype=np.int32)
    pattern = re.compile(path_expr)
    paths = (cfg.prim_path for cfg in plan.asset_prototypes)
    return np.fromiter(
        (index for index, path in enumerate(paths) if path == path_expr or pattern.fullmatch(path)), dtype=np.int32
    )


def get_world_prototypes(plan: ClonePlan, path_expr: str | None = None) -> np.ndarray:
    """Return world-prototype IDs containing assets selected by their declared cfg paths.

    Args:
        plan: Asset prototypes and world compositions.
        path_expr: Asset-path filter interpreted by :func:`get_asset_prototypes`. None selects
            all world definitions, including empty and unused prototypes and shared world -1.

    Returns:
        Ascending world-prototype IDs, shape [num_matches], dtype int32, not destination world IDs.
        Filtering selects complete compositions; repeated asset memberships remain in the plan.
    """
    prototype_ids = np.arange(-1, len(plan.world_prototype_starts) - 2, dtype=np.int32)
    if path_expr is None:
        return prototype_ids
    matched_assets = np.isin(plan.world_prototypes, get_asset_prototypes(plan, path_expr))
    match_counts = np.r_[0, np.cumsum(matched_assets)]
    return prototype_ids[np.diff(match_counts[plan.world_prototype_starts]) > 0]


def get_asset_prototype_world_index(plan: ClonePlan, asset_prototype: int | str) -> tuple[np.ndarray, np.ndarray]:
    """Return destination world indices for selected asset-prototype instances.

    Args:
        plan: Asset prototypes and world compositions.
        asset_prototype: Prototype index or a declared-path expression accepted by :func:`get_asset_prototypes`.

    Returns:
        (world_indices, world_starts), with one world index per selected asset instance.
        All arrays are 1-D with dtype int32. Shared instances use world -1 and occupy the first slice.
        World w's start/end offsets are world_starts[w + 1 : w + 3]. Starts have length num_worlds + 2, including
        empty worlds and the final end offset. These offsets address selected plan instances, not native buffers.
    """
    asset_ids = get_asset_prototypes(plan, asset_prototype) if isinstance(asset_prototype, str) else asset_prototype
    match_counts = np.r_[0, np.cumsum(np.isin(plan.world_prototypes, asset_ids))]
    counts = np.diff(match_counts[plan.world_prototype_starts])
    counts = np.r_[counts[0], counts[plan.world_prototype_layout + 1]]
    world_ids = np.arange(-1, len(plan.world_prototype_layout), dtype=np.int32)
    return np.repeat(world_ids, counts), np.r_[0, counts].cumsum(dtype=np.int32)


def get_asset_prototype_unique_world_index(plan: ClonePlan, asset_prototype: int | str) -> np.ndarray:
    """Return each destination world containing selected asset prototypes once.

    Args:
        plan: Asset prototypes and world compositions.
        asset_prototype: Prototype index or a declared-path expression accepted by :func:`get_asset_prototypes`.

    Returns:
        Ascending world indices, shape [num_matches], dtype int32. Shared instances use -1;
        assets with no instances return an empty array.
    """
    asset_ids = get_asset_prototypes(plan, asset_prototype) if isinstance(asset_prototype, str) else asset_prototype
    match_counts = np.r_[0, np.cumsum(np.isin(plan.world_prototypes, asset_ids))]
    counts = np.diff(match_counts[plan.world_prototype_starts])
    return np.flatnonzero(np.r_[counts[0], counts[plan.world_prototype_layout + 1]]).astype(np.int32) - 1


def get_world_prototype_world_index(plan: ClonePlan, world_prototype: int | str) -> np.ndarray:
    """Return the destination worlds using selected world prototypes.

    Args:
        plan: Asset prototypes and world compositions.
        world_prototype: World-prototype index or an asset-path expression accepted by :func:`get_world_prototypes`.
            Index -1 selects the shared world, even when empty.

    Returns:
        Ascending world indices, shape [num_matches], dtype int32, each included once.
    """
    prototype_ids = get_world_prototypes(plan, world_prototype) if isinstance(world_prototype, str) else world_prototype
    return np.flatnonzero(np.isin(np.r_[-1, plan.world_prototype_layout], prototype_ids)).astype(np.int32) - 1


def iter_clones(
    instances: Iterable[tuple[int, str | None, str, np.ndarray]],
) -> Iterator[tuple[int, str, str, np.ndarray]]:
    """Yield parent-first subtree copies, omitting children already copied by the same ancestor.

    Args:
        instances: Asset-prototype ID, source path, destination template, and world IDs per instance group.

    Yields:
        Asset-prototype ID, source path, destination template, and world IDs requiring a copy.
        Independently sourced children remain explicit overrides of their cloned parents.
    """
    instances = sorted((instance for instance in instances if len(instance[3])), key=lambda item: item[2].count("/"))
    for index, (asset_prototype_id, source, destination, world_ids) in enumerate(instances):
        covered = np.zeros(len(world_ids), dtype=np.bool_)
        redundant = covered.copy()
        # The nearest declared ancestor determines which source reaches each world.
        for _, parent_source, parent_destination, parent_world_ids in reversed(instances[:index]):
            if destination == parent_destination or not pth.under(destination, parent_destination):
                continue
            inherited = np.isin(world_ids, parent_world_ids) & ~covered
            if pth.rebase(source, parent_source, parent_destination) == destination:
                redundant |= inherited
            covered |= inherited
        if not redundant.all():
            yield asset_prototype_id, source, destination, world_ids[~redundant]


def path_env_ids(instances: Sequence[tuple[int, str | None, str, np.ndarray]], path: str) -> tuple[int, ...]:
    """Return the destination world IDs reached by a native prototype path.

    Args:
        instances: Asset-prototype ID, source path, destination template, and world IDs per instance group.
        path: Source path or one of its descendants.

    Returns:
        Ascending destination world IDs, empty when no prototype owns the path.
    """
    prototypes = [
        (source, world_ids)
        for _, source, _, world_ids in instances
        if len(world_ids) and world_ids[0] != -1 and pth.under(path, source)
    ]
    nearest = max((len(source.rstrip("/")) for source, _ in prototypes), default=0)
    return tuple(
        sorted(
            {
                int(world_id)
                for source, world_ids in prototypes
                if len(source.rstrip("/")) == nearest
                for world_id in world_ids
            }
        )
    )


def path_to_clone(instances: Sequence[tuple[int, str | None, str, np.ndarray]], path: str, env_id: int) -> str | None:
    """Resolve a prototype descendant to its single instance in one world.

    Args:
        instances: Asset-prototype ID, source path, destination template, and world IDs per instance group.
        path: Source path or one of its descendants.
        env_id: Destination world ID.

    Returns:
        The destination path, or None when the prototype does not populate that world.

    Raises:
        ValueError: If the world contains multiple instances of the prototype.
    """
    prototypes = [
        (source, template, world_ids)
        for _, source, template, world_ids in instances
        if len(world_ids) and world_ids[0] != -1 and pth.under(path, source)
    ]
    nearest = max((len(source.rstrip("/")) for source, _, _ in prototypes), default=0)
    paths = [
        pth.rebase(path, source, template.format(env_id))
        for source, template, world_ids in prototypes
        if len(source.rstrip("/")) == nearest and env_id in world_ids
    ]
    if len(paths) > 1:
        raise ValueError("The asset prototype has multiple instances in this world.")
    return paths[0] if paths else None


def path_to_source(
    instances: Sequence[tuple[int, str | None, str, np.ndarray]], path_expr: str, env_id: int | None = None
) -> tuple[str, str, str] | None:
    """Resolve a destination-side expression to its native prototype path.

    Args:
        instances: Asset-prototype ID, source path, destination template, and world IDs per instance group.
        path_expr: Concrete destination path or destination-side path expression.
        env_id: Destination world ID. A concrete expression selects its own world;
            otherwise the first populated prototype is selected.

    Returns:
        Source root, destination expression, and asset suffix, or None for an absent instance.
    """
    for source, template, world_ids, matched in _clone_sources(instances, path_expr, populated_only=False):
        selected_env = env_id
        if selected_env is None and matched.instance.isdigit():
            selected_env = int(matched.instance)
        if len(world_ids) and (selected_env is None or selected_env in world_ids):
            return source, template.format("[^/]+"), matched.suffix
    return None


def get_matched_sources(
    instances: Sequence[tuple[int, str | None, str, np.ndarray]], path_expr: str
) -> list[tuple[str, str, str, tuple[int, ...]]]:
    """Return the native prototypes and worlds behind the nearest destination declaration.

    Args:
        instances: Asset-prototype ID, source path, destination template, and world IDs per instance group.
        path_expr: Destination path or path expression.

    Returns:
        List of (source root, destination template, prototype descendant path, destination world IDs) groups.
    """
    return [
        (
            source,
            template,
            pth.rebase(path_expr, template.format(matched.instance), source),
            tuple(map(int, world_ids)),
        )
        for source, template, world_ids, matched in _clone_sources(instances, path_expr, populated_only=True)
    ]


def _clone_sources(instances, path_expr, *, populated_only):
    candidates = [
        (source, template, world_ids, matched)
        for _, source, template, world_ids in instances
        if not len(world_ids) or world_ids[0] != -1
        if not populated_only or len(world_ids)
        if (matched := pth.match(path_expr, template)) is not None
    ]
    nearest = min((len(matched.suffix) for _, _, _, matched in candidates), default=0)
    return [candidate for candidate in candidates if len(candidate[3].suffix) == nearest]
