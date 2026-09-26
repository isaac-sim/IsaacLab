# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Query asset prototypes, world compositions, and their destination world indices."""

from __future__ import annotations

import re

import numpy as np

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
