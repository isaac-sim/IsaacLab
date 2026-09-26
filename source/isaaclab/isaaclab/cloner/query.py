# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Query world topology and plain native path mappings without depending on clone contexts."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

import numpy as np

from . import path as pth
from .clone_plan import ClonePlan


def get_world_prototypes(plan: ClonePlan) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Return each world prototype's members and destination worlds, including shared world -1.

    Args:
        plan: Asset prototypes, world compositions, and destination selections.

    Returns:
        List of (world-prototype ID, asset-prototype IDs, destination world IDs) groups.
        Repeated asset-prototype IDs remain repeated.
        The shared world has ID -1 and destination [-1]. Unselected prototypes have no destinations.
    """
    world_ids = np.argsort(plan.destinations, kind="stable")
    offsets = np.cumsum(np.bincount(plan.destinations, minlength=len(plan.world_prototype_starts) - 2))
    prototypes = [(-1, plan.world_prototypes[: plan.world_prototype_starts[1]], np.asarray([-1], dtype=np.int64))]
    start = 0
    for world_prototype_id, end in enumerate(offsets):
        begin, stop = plan.world_prototype_starts[world_prototype_id + 1 : world_prototype_id + 3]
        prototypes.append((world_prototype_id, plan.world_prototypes[begin:stop], world_ids[start:end]))
        start = end
    return prototypes


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
