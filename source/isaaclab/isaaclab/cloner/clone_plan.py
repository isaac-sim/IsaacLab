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

from .. import sim as sim_utils
from ..sensors.camera.camera_cfg import CameraCfg
from ..utils.string import string_to_callable
from ..utils.version import has_kit
from .cloner_cfg import DEFAULT_ENV_TEMPLATE, CloneCfg, InclusionSet, expand_env_regex_ns
from .cloner_strategies import sequential
from .path import match, under
from .usd import UsdReplicateContext


@dataclass(frozen=True, eq=False)
class ClonePlan:
    """Description of a single replication layout, consumed by :func:`~isaaclab.cloner.replicate`."""

    sources: tuple[Any, ...]
    """Original asset and sensor configurations, including shared assets; no configuration is copied."""

    destinations: np.ndarray
    """Integer variant indices ``[len(sources), num_envs]``; ``-1`` means no replicated instance.

    Column ``j`` selects the variant placed in ``env_ids[j]``. Shared assets have no replicated
    instances and are imported once at their declared prim paths.
    """

    env_ids: np.ndarray | None = None
    """Integer array ``[num_clones]`` of target env ids.

    Optional for plans used only with :func:`~isaaclab.cloner.query.iter_sources` or
    :func:`~isaaclab.cloner.query.path_to_source`; required by :func:`~isaaclab.cloner.replicate`.
    """

    positions: np.ndarray | None = None
    """Per-env world positions [m], shape ``[num_clones, 3]``, or ``None``."""

    clone_template: str = DEFAULT_ENV_TEMPLATE
    """Environment namespace used to resolve source declarations and destination prim paths."""

    context_source_indices: dict[type[object], tuple[int, ...]] = field(default_factory=dict)
    """Clone-context classes to indices into :attr:`sources` for the declarations they consume."""

    @property
    def global_paths(self) -> tuple[str, ...]:
        """Declared shared-asset roots, derived without storing another path manifest."""
        paths = tuple(
            dict.fromkeys(cfg.prim_path for cfg in self.sources if match(cfg.prim_path, self.clone_template) is None)
        )
        return tuple(path for path in paths if not any(path != root and under(path, root) for root in paths))


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


def _set_spawn_paths(spawn_cfg: Any, paths: list[str | None]) -> None:
    if isinstance(spawn_cfg, (sim_utils.MultiAssetSpawnerCfg, sim_utils.MultiUsdFileCfg)):
        spawn_cfg.spawn_path = None
        spawn_cfg.spawn_paths = paths
    else:
        spawn_cfg.spawn_path = paths[0]


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


def _context_source_indices(plan: ClonePlan) -> dict[type[object], tuple[int, ...]]:
    """Route source declarations to registered clone contexts."""
    sim = sim_utils.SimulationContext.instance()
    if sim is None:
        return {}

    physics_context = sim.physics_manager.clone_context_type
    if physics_context is not None and not isinstance(physics_context, type):
        raise TypeError("PhysicsManager.clone_context_type must be a context class.")
    render_contexts = {
        string_to_callable(context) if isinstance(context, str) else context
        for context in sim.render_context.clone_contexts
    }
    for context_type in render_contexts:
        if context_type not in sim.clone_contexts:
            sim.clone_contexts[context_type] = context_type(sim)
    spawn_contexts = render_contexts - {physics_context}
    if has_kit():
        spawn_contexts.add(UsdReplicateContext)
    scene_contexts = {
        context for context in (physics_context, *render_contexts) if context is not None and plan.global_paths
    }
    populated = np.flatnonzero((plan.destinations >= 0).any(axis=1))
    if not len(populated):
        scene_contexts.update(render_contexts)
    source_indices_by_context: dict[type[object], set[int]] = {context: set() for context in scene_contexts}

    for index in populated:
        cfg = plan.sources[index]
        fields = vars(cfg)
        references = fields.get("cloning_contexts", ())
        if references is None:
            contexts = () if physics_context is None else (physics_context,)
        else:
            contexts = tuple(string_to_callable(value) if isinstance(value, str) else value for value in references)
        if isinstance(fields.get("spawn"), sim_utils.SpawnerCfg):
            contexts += tuple(spawn_contexts)
        for context_type in contexts:
            if not isinstance(context_type, type):
                raise TypeError(f"{type(cfg).__name__}.cloning_contexts must contain only context classes.")
            source_indices_by_context.setdefault(context_type, set()).add(int(index))

    if UsdReplicateContext in source_indices_by_context:
        sim.clone_contexts[UsdReplicateContext] = UsdReplicateContext(sim.stage)
    return {context_type: tuple(sorted(indices)) for context_type, indices in source_indices_by_context.items()}


def make_clone_plan(
    cfgs: Iterable[Any],
    num_clones: int,
    env_spacing: float,
    clone_strategy: Callable[[np.ndarray, int], np.ndarray] = sequential,
    valid_set: np.ndarray | None = None,
    env_template: str = DEFAULT_ENV_TEMPLATE,
) -> ClonePlan:
    """Retain declarations and select their variants for each destination environment.

    Planning assigns each spawner's ``spawn_path`` / ``spawn_paths`` before asset construction.
    Configurations without a spawner declare existing assets without adding replication work.
    Configurations outside ``env_template`` declare shared assets, imported once.

    Args:
        cfgs: Flat asset and sensor declarations, including shared assets.
        num_clones: Number of target envs.
        env_spacing: Distance between neighboring grid env origins [m].
        clone_strategy: Function that assigns prototype combinations to envs. Defaults
            to :func:`~isaaclab.cloner.sequential`.
        valid_set: Optional ``[num_combos, num_groups]`` integer array of valid prototype
            combinations. ``None`` (default) uses the full cartesian product of every
            group's prototype indices.

    Returns:
        A plan retaining each declaration once and the selected variant for each environment.
    """
    cfgs = tuple(cfgs)
    sim = sim_utils.SimulationContext.instance()

    groups: list[tuple[int, Any, str, int]] = []
    for index, cfg in enumerate(cfgs):
        cfg.prim_path = expand_env_regex_ns(cfg.prim_path, env_template)
        if isinstance(cfg, CameraCfg) and sim is not None:
            sim.get_or_create_backend(cfg.renderer_cfg)
        matched = match(cfg.prim_path, env_template)
        spawn = getattr(cfg, "spawn", None)
        if spawn is None:
            continue
        if matched is None:
            spawn.spawn_path = cfg.prim_path
            continue
        count = num_spawn_variants(spawn)
        if count <= 0:
            raise ValueError(f"Spawner at '{cfg.prim_path}' must have at least one variant.")
        groups.append((index, spawn, env_template + matched.suffix, count))
    env_ids = np.arange(num_clones, dtype=np.int64)
    positions, _ = grid_transforms(num_clones, env_spacing)
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

    if not groups:
        chosen = np.empty((num_clones, 0), dtype=np.int32)
    elif valid_set is None and all(count == 1 for count in group_sizes):
        chosen = np.zeros((num_clones, len(groups)), dtype=np.int32)
    elif valid_set is None:
        all_combos = list(itertools.product(*[range(s) for s in group_sizes]))
        combos = np.asarray(all_combos, dtype=np.int64)
        chosen = validate_combinations(clone_strategy(combos, num_clones), "clone_strategy result", num_clones)
    else:
        combos = validate_combinations(valid_set, "valid_set")
        chosen = validate_combinations(clone_strategy(combos, num_clones), "clone_strategy result", num_clones)

    destinations = np.full((len(cfgs), num_clones), -1, dtype=np.int32)
    for column, (index, spawn_cfg, template, count) in enumerate(groups):
        destinations[index] = chosen[:, column]
        paths = []
        for variant in range(count):
            selected = np.flatnonzero(destinations[index] == variant)
            paths.append(template.format(int(env_ids[selected[0]])) if len(selected) else None)
        _set_spawn_paths(spawn_cfg, paths)

    plan = ClonePlan(
        sources=cfgs,
        destinations=destinations,
        env_ids=env_ids,
        positions=positions,
        clone_template=env_template,
    )
    plan.context_source_indices.update(_context_source_indices(plan))
    return plan


def clone_plan_from_env_0(
    clone_cfg: CloneCfg,
    asset_cfgs: Iterable[Any],
    num_envs: int,
    env_spacing: float,
    *,
    positions: np.ndarray | None = None,
) -> ClonePlan:
    """Build and publish one homogeneous plan from explicit asset configurations.

    The flat ``asset_cfgs`` sequence is the construction manifest. Environment-scoped
    configurations share one environment prototype; configurations outside that namespace become
    :attr:`ClonePlan.global_paths`. Planning assigns each spawner an exact prototype path
    before callers construct the assets.

    Args:
        clone_cfg: Homogeneous clone policy and environment template.
        asset_cfgs: Flat sequence of prim-authoring configurations.
        num_envs: Number of target environments.
        env_spacing: Distance between neighboring environment origins [m].
        positions: Optional per-environment world positions [m], shape ``[num_envs, 3]``.
            ``None`` uses a centered grid with ``env_spacing``.

    Returns:
        The published :class:`ClonePlan`, selecting one variant of each asset for every environment.

    Raises:
        ValueError: If heterogeneous clone combinations or a multi-variant spawner are supplied.
        RuntimeError: If no simulation is active or it already owns a clone plan.
    """
    if clone_cfg.clone_combinations:
        raise ValueError("clone_plan_from_env_0 requires a homogeneous CloneCfg.")
    sim = sim_utils.SimulationContext.instance()
    if sim is None:
        raise RuntimeError("Clone planning requires an active SimulationContext.")
    if sim.get_clone_plan() is not None:
        raise RuntimeError("A SimulationContext owns exactly one clone lifecycle.")

    asset_cfgs = tuple(asset_cfgs)
    if any(num_spawn_variants(spawn) != 1 for cfg in asset_cfgs if (spawn := getattr(cfg, "spawn", None)) is not None):
        raise ValueError("clone_plan_from_env_0 requires single-variant spawners.")
    destinations = np.full((len(asset_cfgs), num_envs), -1, dtype=np.int32)
    for index, cfg in enumerate(asset_cfgs):
        if isinstance(cfg, CameraCfg):
            sim.get_or_create_backend(cfg.renderer_cfg)
        cfg.prim_path = expand_env_regex_ns(cfg.prim_path, clone_cfg.clone_template)
        matched = match(cfg.prim_path, clone_cfg.clone_template)
        spawn = getattr(cfg, "spawn", None)
        if matched is not None:
            destinations[index] = 0
        if spawn is not None:
            _set_spawn_paths(
                spawn, [cfg.prim_path if matched is None else clone_cfg.clone_template.format(0) + matched.suffix]
            )
    plan = ClonePlan(
        sources=asset_cfgs,
        destinations=destinations,
        env_ids=np.arange(num_envs, dtype=np.int64),
        positions=grid_transforms(num_envs, env_spacing)[0] if positions is None else positions,
        clone_template=clone_cfg.clone_template,
    )
    plan.context_source_indices.update(_context_source_indices(plan))
    sim.set_clone_plan(plan)
    return plan
