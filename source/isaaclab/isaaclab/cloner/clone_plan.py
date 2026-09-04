# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The :class:`ClonePlan` value type and the constructors that build one.

A plan is the whole description of a replication layout: which prototypes exist, where each
one is cloned to, and which envs each one populates. It is built once, queried through
:mod:`~isaaclab.cloner.query`, and executed by :func:`~isaaclab.cloner.replicate`.

Three constructors cover the ways a layout is specified:

* :func:`clone_plan_from_env_0` — every env is a copy of one prototype env.
* :func:`make_clone_plan` — the layout is derived from the scene's asset cfgs, expanding
  multi-asset spawners into per-variant prototypes.
* :func:`make_valid_clone_combinations` — restricts which variant combinations
  :func:`make_clone_plan` may draw from, weighted per combination.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.utils.string import string_to_callable
from isaaclab.utils.version import has_kit

from .cloner_cfg import DEFAULT_ENV_TEMPLATE, InclusionSet
from .cloner_strategies import sequential
from .path import match
from .usd import UsdReplicateContext


@dataclass(frozen=True, eq=False)
class ClonePlan:
    """Description of a single replication layout, consumed by :func:`~isaaclab.cloner.replicate`."""

    sources: tuple[str, ...]
    """Source prim paths, one per replication row."""

    destinations: tuple[str, ...]
    """Destination path templates with ``"{}"`` for the env id, one per row."""

    clone_mask: np.ndarray
    """Boolean array ``[len(sources), num_clones]``; ``True`` if env ``j`` comes from row ``i``."""

    env_ids: np.ndarray | None = None
    """Integer array ``[num_clones]`` of target env ids.

    Optional for plans used only with :func:`~isaaclab.cloner.query.iter_sources` or
    :func:`~isaaclab.cloner.query.path_to_source`; required by :func:`~isaaclab.cloner.replicate`.
    """

    positions: np.ndarray | None = None
    """Per-env world positions [m], shape ``[num_clones, 3]``, or ``None``."""

    cfg_rows: dict[int, tuple[int, ...]] = field(default_factory=dict)
    """``id(cfg)`` to the row indices the cfg owns."""

    context_rows: dict[type[object], tuple[int, ...]] = field(default_factory=dict)
    """Clone-context types to the rows they consume."""

    global_paths: tuple[str, ...] = ()
    """Unique prim paths for scene assets shared by every environment."""


def grid_transforms(N: int, spacing: float = 1.0, up_axis: str = "z") -> tuple[np.ndarray, np.ndarray]:
    """Create centered grid transforms as host arrays.

    Args:
        N: Number of instances.
        spacing: Distance between neighboring grid positions [m].
        up_axis: Up axis for positions (``"z"``, ``"y"``, or ``"x"``).

    Returns:
        Positions [m], shape ``[N, 3]``, and identity xyzw orientations, shape ``[N, 4]``.
    """
    num_rows = int(math.ceil(N / math.sqrt(N)))
    num_cols = int(math.ceil(N / num_rows))
    ii, jj = np.meshgrid(np.arange(num_rows, dtype=np.float32), np.arange(num_cols, dtype=np.float32), indexing="ij")
    ii = ii.reshape(-1)[:N]
    jj = jj.reshape(-1)[:N]
    x = -(ii - (num_rows - 1) / 2) * spacing
    y = (jj - (num_cols - 1) / 2) * spacing
    zero = np.zeros(N, dtype=np.float32)
    if up_axis.lower() == "z":
        positions = np.stack((x, y, zero), axis=1)
    elif up_axis.lower() == "y":
        positions = np.stack((x, zero, y), axis=1)
    else:
        positions = np.stack((zero, x, y), axis=1)
    orientations = np.zeros((N, 4), dtype=np.float32)
    orientations[:, 3] = 1.0
    return positions.astype(np.float32, copy=False), orientations


def num_spawn_variants(spawn_cfg: Any) -> int:
    """Return the number of spawn variants declared by one spawner configuration.

    :class:`~isaaclab.sim.MultiAssetSpawnerCfg` declares one variant per asset
    configuration and :class:`~isaaclab.sim.MultiUsdFileCfg` one per USD path;
    every other spawner declares a single variant.

    Args:
        spawn_cfg: Spawner configuration to inspect.

    Returns:
        The number of spawn variants the configuration expands into.
    """
    if isinstance(spawn_cfg, sim_utils.MultiAssetSpawnerCfg):
        return len(spawn_cfg.assets_cfg)
    if isinstance(spawn_cfg, sim_utils.MultiUsdFileCfg):
        return 1 if isinstance(spawn_cfg.usd_path, str) else len(spawn_cfg.usd_path)
    return 1


def make_valid_clone_combinations(
    asset_names: Sequence[str],
    variant_counts: Sequence[int],
    clone_combinations: Sequence[InclusionSet] | None = None,
    *,
    all_asset_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Build the valid clone-combination variant array.

    Each combination contributes rows in proportion to its weight, split evenly
    across its spawn variants and interleaved round-robin, so any prefix of the
    array samples every combination.

    Args:
        asset_names: Clone-planned scene asset names, one per array column.
        variant_counts: Number of spawn variants per clone-planned asset.
        clone_combinations: Legal clone combinations; assets not mentioned by
            any combination are active in every row. ``None`` uses the full
            cartesian product of variants.
        all_asset_names: Optional full scene asset-name list; combination
            entries may reference assets that are not clone-planned.

    Returns:
        A ``[num_valid_combinations, num_assets]`` array of source variant
        indices, ``-1`` where an asset is absent.

    Raises:
        ValueError: If the inputs are inconsistent or no valid rows result.
    """
    if len(asset_names) != len(variant_counts):
        raise ValueError(f"Expected one variant count per asset, got {len(variant_counts)} and {len(asset_names)}.")
    if not asset_names:
        raise ValueError("Expected at least one asset name.")
    if any(count <= 0 for count in variant_counts):
        raise ValueError("Variant counts must be positive.")

    if not clone_combinations:
        rows = itertools.product(*[range(count) for count in variant_counts])
        return np.asarray(list(rows), dtype=np.int64)

    clone_asset_names = set(asset_names)
    known_assets = set(all_asset_names) if all_asset_names is not None else clone_asset_names
    combination_assets: list[set[str]] = []
    for combination in clone_combinations:
        if combination.weight < 0:
            raise ValueError("Clone combination weights must be non-negative.")
        unknown_assets = sorted(set(combination.assets) - known_assets)
        if unknown_assets:
            raise ValueError(f"Unknown assets in clone combination: {unknown_assets}.")
        combination_assets.append(set(combination.assets) & clone_asset_names)

    claimed_assets = set().union(*combination_assets) if combination_assets else set()

    expanded: list[tuple[int, list[tuple[int, ...]]]] = []
    for combination, active_assets in zip(clone_combinations, combination_assets):
        if combination.weight == 0:
            continue
        variant_ranges = []
        for asset_name, count in zip(asset_names, variant_counts):
            is_active = asset_name not in claimed_assets or asset_name in active_assets
            variant_ranges.append(range(count) if is_active else (-1,))
        expanded.append((combination.weight, list(itertools.product(*variant_ranges))))

    if not expanded:
        raise ValueError("Clone combinations produced no valid clone rows.")

    # A combination's share is its weight, split evenly across its spawn variants.
    # Integer multiplicities require a common denominator across variant counts.
    # Rows are emitted round-robin across combinations so a truncated prefix
    # (fewer environments than rows) still samples every combination.
    common_multiple = math.lcm(*[len(variants) for _, variants in expanded])
    rows = []
    cursors = [0] * len(expanded)
    for _ in range(common_multiple):
        for index, (weight, variants) in enumerate(expanded):
            for _ in range(weight):
                rows.append(variants[cursors[index] % len(variants)])
                cursors[index] += 1
    return np.asarray(rows, dtype=np.int64)


def _context_rows(
    cfgs: tuple[Any, ...], cfg_rows: dict[int, tuple[int, ...]], populated_rows: set[int]
) -> dict[type[object], tuple[int, ...]]:
    """Route plan rows to the clone contexts registered for this simulation."""
    sim = sim_utils.SimulationContext.instance()
    if sim is None:
        return {}

    physics_context = sim.physics_manager.clone_context_type
    if physics_context is not None and not isinstance(physics_context, type):
        raise TypeError("PhysicsManager.clone_context_type must be a context class.")
    rows_by_context: dict[type[object], set[int]] = {}

    for cfg in cfgs:
        rows = cfg_rows.get(id(cfg))
        if rows is None:
            continue
        references = cfg.cloning_contexts
        if references is None:
            contexts = () if physics_context is None else (physics_context,)
        else:
            contexts = tuple(string_to_callable(value) if isinstance(value, str) else value for value in references)
        if cfg.spawn is not None and has_kit():
            contexts = tuple(dict.fromkeys((*contexts, UsdReplicateContext)))
        if any(not isinstance(context, type) for context in contexts):
            raise TypeError(f"{type(cfg).__name__}.cloning_contexts must contain only context classes.")
        for context_type in contexts:
            rows_by_context.setdefault(context_type, set()).update(rows)

    if UsdReplicateContext in rows_by_context:
        sim.get_or_create_backend(UsdReplicateContext, sim.stage)
    return {
        context_type: tuple(sorted(rows & populated_rows))
        for context_type, rows in rows_by_context.items()
        if rows & populated_rows
    }


def make_clone_plan(
    cfgs: Iterable[Any],
    num_clones: int,
    env_spacing: float,
    global_paths: tuple[str, ...] = (),
    clone_strategy: Callable[[np.ndarray, int], np.ndarray] = sequential,
    valid_set: np.ndarray | None = None,
    env_template: str = DEFAULT_ENV_TEMPLATE,
) -> ClonePlan:
    """Build a :class:`ClonePlan` from asset cfgs.

    Iterates ``cfgs``, identifies env-scoped cfgs with a spawn, expands
    :class:`~isaaclab.sim.MultiAssetSpawnerCfg` / :class:`~isaaclab.sim.MultiUsdFileCfg`
    into per-variant prototype rows, runs ``clone_strategy`` to assign prototypes to
    envs, and returns a self-contained :class:`ClonePlan` with ``cfg_rows`` populated.

    Each input cfg's ``spawn_path`` / ``spawn_paths`` is mutated so the subsequent
    asset constructor spawns the prototype into its first active environment. Every cfg
    is an env-scoped entity with a spawner. Shared assets are declared explicitly through
    ``global_paths`` and are never replicated.

    Args:
        cfgs: Cloneable asset cfgs with resolved env-scoped ``prim_path`` and ``spawn``.
        num_clones: Number of target envs.
        env_spacing: Distance between neighboring grid env origins [m].
        global_paths: Complete shared-asset roots declared by the scene composition root. Defaults to none.
        clone_strategy: Function that assigns prototype combinations to envs. Defaults
            to :func:`~isaaclab.cloner.sequential`.
        valid_set: Optional ``[num_combos, num_groups]`` integer array of valid prototype
            combinations. ``None`` (default) uses the full cartesian product of every
            group's prototype indices.

    Returns:
        A :class:`ClonePlan` whose ``sources``/``destinations``/``clone_mask`` describe
        the flat prototype-to-env mapping, whose ``cfg_rows`` maps each replicated cfg
        to the rows it owns, and whose ``global_paths`` names shared scene assets.
    """

    def set_spawn_paths(spawn_cfg: Any, paths: list[str | None]) -> None:
        if isinstance(spawn_cfg, (sim_utils.MultiAssetSpawnerCfg, sim_utils.MultiUsdFileCfg)):
            spawn_cfg.spawn_paths = paths
        else:
            active = [p for p in paths if p is not None]
            if len(active) == 0:
                spawn_cfg.spawn_path = None
                return
            if len(active) != 1:
                raise ValueError("Single spawner expects exactly one planned source path.")
            spawn_cfg.spawn_path = active[0]

    cfgs = tuple(cfgs)

    # 1) Build per-group records: (cfg, spawn_cfg, destination_template, num_variants).
    groups: list[tuple[Any, Any, str, int]] = []
    for cfg in cfgs:
        matched = match(cfg.prim_path, env_template)
        count = num_spawn_variants(cfg.spawn)
        if count <= 0:
            raise ValueError(f"Spawner at '{cfg.prim_path}' must have at least one variant.")
        groups.append((cfg, cfg.spawn, env_template + matched.suffix, count))
    env_ids = np.arange(num_clones, dtype=np.int64)
    positions, _ = grid_transforms(num_clones, env_spacing)

    # 2) No env-scoped cfgs: emit an empty plan so the scene can still proceed.
    if not groups:
        empty_mask = np.zeros((0, num_clones), dtype=np.bool_)
        return ClonePlan(
            sources=(),
            destinations=(),
            clone_mask=empty_mask,
            env_ids=env_ids,
            positions=positions,
            cfg_rows={},
            global_paths=global_paths,
        )

    # 3) Homogeneous (every cfg is single-variant): emit the simpler env-root plan.
    if valid_set is None and all(count == 1 for _, _, _, count in groups):
        for cfg, spawn_cfg, destination, _ in groups:
            set_spawn_paths(spawn_cfg, [destination.format(0)])
        cfg_rows = {id(cfg): (0,) for cfg, _, _, _ in groups}
        clone_mask = np.ones((1, num_clones), dtype=np.bool_)
        return ClonePlan(
            sources=(env_template.format(0),),
            destinations=(env_template,),
            clone_mask=clone_mask,
            env_ids=env_ids,
            positions=positions,
            cfg_rows=cfg_rows,
            context_rows=_context_rows(cfgs, cfg_rows, {0}),
            global_paths=global_paths,
        )

    # 4) Heterogeneous: enumerate prototype combos, build per-row mask, mutate spawn paths.
    group_sizes = [count for _, _, _, count in groups]

    def validate_combinations(combos: np.ndarray, name: str, expected_rows: int | None = None) -> np.ndarray:
        combos = np.asarray(combos)
        if not np.issubdtype(combos.dtype, np.integer):
            raise ValueError(f"{name} must contain integer prototype indices.")
        combos = combos.astype(np.int64, copy=False)
        if combos.ndim != 2:
            raise ValueError(f"{name} must be a 2-D array, got shape {tuple(combos.shape)}.")
        if combos.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one row.")
        if expected_rows is not None and combos.shape[0] != expected_rows:
            raise ValueError(f"{name} must contain {expected_rows} rows, got {combos.shape[0]}.")
        if combos.shape[1] != len(group_sizes):
            raise ValueError(f"{name} must contain {len(group_sizes)} columns, got {combos.shape[1]}.")
        invalid = (combos < -1) | ((combos >= np.asarray(group_sizes)[None]) & (combos != -1))
        if invalid.any():
            raise ValueError(f"{name} contains prototype indices outside [-1, group_size).")
        return combos

    if valid_set is None:
        all_combos = list(itertools.product(*[range(s) for s in group_sizes]))
        combos = np.asarray(all_combos, dtype=np.int64)
    else:
        combos = validate_combinations(valid_set, "valid_set")
    chosen = validate_combinations(clone_strategy(combos, num_clones), "clone_strategy result", num_clones)

    group_offsets = np.asarray([0] + list(itertools.accumulate(group_sizes[:-1])), dtype=np.int64)
    active = chosen >= 0
    rows = (chosen + group_offsets).reshape(-1)
    cols = np.broadcast_to(np.arange(num_clones)[:, None], chosen.shape).reshape(-1)
    active_flat = active.reshape(-1)

    num_rows = sum(group_sizes)
    clone_mask = np.zeros((num_rows, num_clones), dtype=np.bool_)
    if active_flat.any():
        clone_mask[rows[active_flat], cols[active_flat]] = True

    sources_list: list[str] = []
    destinations_list: list[str] = []
    cfg_rows: dict[int, tuple[int, ...]] = {}
    populated_rows: set[int] = set()
    row = 0
    for cfg, spawn_cfg, destination, count in groups:
        cfg_rows[id(cfg)] = tuple(range(row, row + count))
        group_mask = clone_mask[row : row + count]
        env_ids_assigned = group_mask.argmax(axis=1)
        active_variants = group_mask.any(axis=1)
        populated_rows.update(row + i for i, is_active in enumerate(active_variants) if is_active)
        paths = [
            destination.format(int(env_id)) if is_active else None
            for env_id, is_active in zip(env_ids_assigned, active_variants)
        ]
        for i, path in enumerate(paths):
            destinations_list.append(destination)
            # Inactive prototypes fall back to env-i so the source path stays valid even
            # when the variant has no active environment (matches the legacy behavior).
            sources_list.append(path if path is not None else destination.format(i))
        set_spawn_paths(spawn_cfg, paths)
        row += count

    return ClonePlan(
        sources=tuple(sources_list),
        destinations=tuple(destinations_list),
        clone_mask=clone_mask,
        env_ids=env_ids,
        positions=positions,
        cfg_rows=cfg_rows,
        context_rows=_context_rows(cfgs, cfg_rows, populated_rows),
        global_paths=global_paths,
    )


def clone_plan_from_env_0(
    source: str,
    destination: str,
    num_clones: int,
    positions: np.ndarray | None = None,
    global_paths: tuple[str, ...] = (),
) -> ClonePlan:
    """Build a single-source clone plan that targets every env from one source row.

    Auto-populates :attr:`ClonePlan.cfg_rows` from :data:`~isaaclab.cloner.REPLICATION_QUEUE`,
    including only cfgs whose ``prim_path`` falls under the env-root prefix of
    ``destination``. ``global_paths`` is the complete declaration of shared assets; it is
    never inferred from the stage or replication queue. Must be called *after* all asset
    constructors have run, so their cfgs are already registered in the queue; otherwise
    those assets will be skipped by the subsequent :func:`~isaaclab.cloner.replicate` call.

    Args:
        source: Source prim path (typically ``/World/envs/env_0``).
        destination: Destination template with ``"{}"`` for the env id.
        num_clones: Number of target envs.
        positions: Optional per-env world positions [m], shape ``[num_clones, 3]``.
        global_paths: Complete shared-asset roots for the hand-built scene. Defaults to none.

    Returns:
        A :class:`ClonePlan` with a single source row covering every env.
    """
    from .replicate_session import REPLICATION_QUEUE  # noqa: PLC0415

    queued = tuple(REPLICATION_QUEUE)
    cfg_rows = {id(cfg): (0,) for cfg in queued if match(cfg.prim_path, destination) is not None}
    clone_mask = np.ones((1, num_clones), dtype=np.bool_)
    return ClonePlan(
        sources=(source,),
        destinations=(destination,),
        clone_mask=clone_mask,
        env_ids=np.arange(num_clones, dtype=np.int64),
        positions=positions,
        cfg_rows=cfg_rows,
        context_rows=_context_rows(queued, cfg_rows, {0}),
        global_paths=global_paths,
    )
