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

import torch

import isaaclab.sim as sim_utils
from isaaclab.utils.string import string_to_callable
from isaaclab.utils.version import has_kit

from .cloner_cfg import DEFAULT_ENV_TEMPLATE, InclusionSet
from .cloner_strategies import sequential
from .path import match


@dataclass(frozen=True, eq=False)
class ClonePlan:
    """Description of a single replication layout, consumed by :func:`~isaaclab.cloner.replicate`."""

    sources: tuple[str, ...]
    """Source prim paths, one per replication row."""

    destinations: tuple[str, ...]
    """Destination paths, templated for replicated rows and exact for shared rows."""

    clone_mask: torch.Tensor
    """Bool tensor ``[len(sources), num_clones]``; ``True`` if env ``j`` comes from row ``i``."""

    env_ids: torch.Tensor
    """Long tensor ``[num_clones]`` of target env ids."""

    positions: torch.Tensor
    """Per-env world positions [m], shape ``[num_clones, 3]``."""

    cfg_rows: dict[int, tuple[int, ...]] = field(default_factory=dict)
    """``id(cfg)`` to the row indices the cfg owns."""

    context_rows: dict[type[object], tuple[int, ...]] = field(default_factory=dict)
    """Clone-context types to their routed replication rows."""

    env_template: str = DEFAULT_ENV_TEMPLATE
    """Environment path template used by whole-environment clone contexts."""

    def __post_init__(self) -> None:
        """Validate the flat replication relation at its public boundary."""
        if not isinstance(self.clone_mask, torch.Tensor) or not isinstance(self.env_ids, torch.Tensor):
            raise TypeError("clone_mask and env_ids must be torch tensors.")
        if not isinstance(self.positions, torch.Tensor):
            raise TypeError("positions must be a torch tensor.")
        rows = len(self.sources)
        if len(self.destinations) != rows or self.clone_mask.ndim != 2 or self.clone_mask.shape[0] != rows:
            raise ValueError("sources, destinations, and clone_mask rows must have equal length.")
        if self.clone_mask.dtype != torch.bool:
            raise TypeError("clone_mask must be a bool tensor.")
        if self.env_ids.ndim != 1 or self.clone_mask.shape[1] != len(self.env_ids):
            raise ValueError("clone_mask columns must match the one-dimensional env_ids tensor.")
        if self.env_ids.dtype != torch.long:
            raise TypeError("env_ids must be a torch.long tensor.")
        if self.clone_mask.device != self.env_ids.device or self.positions.device != self.env_ids.device:
            raise ValueError("clone_mask, env_ids, and positions must use the same device.")
        if bool((self.env_ids < 0).any()):
            raise ValueError("env_ids must be nonnegative.")
        if len(torch.unique(self.env_ids)) != len(self.env_ids):
            raise ValueError("env_ids must be unique.")
        if self.positions.shape != (len(self.env_ids), 3):
            raise ValueError("positions must have shape [len(env_ids), 3].")
        for row, (source, destination) in enumerate(zip(self.sources, self.destinations, strict=True)):
            if "{}" in source:
                raise ValueError("ClonePlan sources must be exact prim paths.")
            if destination.count("{}") > 1:
                raise ValueError("A replicated destination must contain exactly one clone slot.")
            if "{}" not in destination and (source != destination or bool(self.clone_mask[row].any())):
                raise ValueError("A shared row requires an exact source == destination and an empty clone mask.")
        routed_rows = (*self.cfg_rows.values(), *self.context_rows.values())
        if any(row < 0 or row >= rows for owned_rows in routed_rows for row in owned_rows):
            raise ValueError("cfg_rows and context_rows must reference existing plan rows.")
        if any(len(set(owned_rows)) != len(owned_rows) for owned_rows in self.context_rows.values()):
            raise ValueError("context_rows cannot route the same plan row more than once.")
        if self.env_template.count("{}") != 1:
            raise ValueError("env_template must contain exactly one clone slot.")
        if any("{}" not in self.destinations[row] for rows in self.context_rows.values() for row in rows):
            raise ValueError("context_rows cannot route exact shared rows.")


def grid_transforms(N: int, spacing: float = 1.0, up_axis: str = "z", device="cpu"):
    """Create a centered grid of transforms for ``N`` instances.

    Computes ``(x, y)`` coordinates in a roughly square grid centered at the origin
    with the provided spacing, places the third coordinate according to ``up_axis``,
    and returns identity orientations. This matches the grid layout used by
    :class:`isaaclab.terrains.TerrainImporter` for consistent environment positioning.

    Args:
        N: Number of instances.
        spacing: Distance between neighboring grid positions [m].
        up_axis: Up axis for positions ("z", "y", or "x").
        device: Torch device for returned tensors.

    Returns:
        A tuple ``(pos, ori)`` where:
            - ``pos`` is a tensor of shape ``(N, 3)`` with positions [m].
            - ``ori`` is a tensor of shape ``(N, 4)`` with identity quaternions in ``(x, y, z, w)``.
    """
    # Match terrain_importer._compute_env_origins_grid layout for consistency
    num_rows = int(math.ceil(N / math.sqrt(N)))
    num_cols = int(math.ceil(N / num_rows))

    # Create meshgrid matching terrain's "ij" indexing
    ii, jj = torch.meshgrid(
        torch.arange(num_rows, device=device, dtype=torch.float32),
        torch.arange(num_cols, device=device, dtype=torch.float32),
        indexing="ij",
    )
    # Flatten and take first N elements
    ii = ii.flatten()[:N]
    jj = jj.flatten()[:N]

    # Match terrain's coordinate system: X from rows (negated), Y from cols
    x = -(ii - (num_rows - 1) / 2) * spacing
    y = (jj - (num_cols - 1) / 2) * spacing
    z0 = torch.zeros(N, device=device)

    # place on plane based on up_axis
    if up_axis.lower() == "z":
        pos = torch.stack([x, y, z0], dim=1)
    elif up_axis.lower() == "y":
        pos = torch.stack([x, z0, y], dim=1)
    else:  # up_axis == "x"
        pos = torch.stack([z0, x, y], dim=1)

    # identity orientations (x,y,z,w): w=1 is index 3
    ori = torch.nn.functional.one_hot(torch.full((N,), 3, device=device), num_classes=4).float()
    return pos, ori


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
    device: str = "cpu",
    *,
    all_asset_names: Sequence[str] | None = None,
) -> torch.Tensor:
    """Build the valid clone-combination variant tensor.

    Each combination contributes rows in proportion to its weight, split evenly
    across its spawn variants and interleaved round-robin, so any prefix of the
    tensor samples every combination.

    Args:
        asset_names: Clone-planned scene asset names, one per tensor column.
        variant_counts: Number of spawn variants per clone-planned asset.
        clone_combinations: Legal clone combinations; assets not mentioned by
            any combination are active in every row. ``None`` uses the full
            cartesian product of variants.
        device: Torch device for the output tensor. Defaults to ``"cpu"``.
        all_asset_names: Optional full scene asset-name list; combination
            entries may reference assets that are not clone-planned.

    Returns:
        A ``[num_valid_combinations, num_assets]`` tensor of source variant
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
        return torch.tensor(list(rows), dtype=torch.long, device=device)

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
    return torch.tensor(rows, dtype=torch.long, device=device)


def _compile_context_rows(
    cfgs: tuple[Any, ...],
    cfg_rows: dict[int, tuple[int, ...]],
    destinations: tuple[str, ...],
    clone_mask: torch.Tensor,
    whole_env_rows: tuple[int, ...],
) -> dict[type[object], tuple[int, ...]]:
    """Compile cfg and simulation-scoped clone-context routing into plan rows."""
    sim = sim_utils.SimulationContext.instance()
    context_roles = {} if sim is None else sim._backend_clone_roles
    populated_rows = {
        row for row, destination in enumerate(destinations) if "{}" in destination and bool(clone_mask[row].any())
    }
    physics_contexts = tuple(context_type for context_type, roles in context_roles.items() if "physics" in roles)

    routed_rows: dict[type[object], set[int]] = {context_type: set() for context_type in context_roles}
    usd_context: type[object] | None = None
    for cfg in cfgs:
        references = getattr(cfg, "cloning_contexts", None)
        contexts = (
            physics_contexts
            if references is None
            else tuple(string_to_callable(value) if isinstance(value, str) else value for value in references)
        )
        if any(not isinstance(context, type) for context in contexts):
            raise TypeError(f"{type(cfg).__name__}.cloning_contexts must contain only context classes.")
        rows = cfg_rows.get(id(cfg))
        if rows is None:
            continue
        automatic_usd = getattr(cfg, "spawn", None) is not None and has_kit()
        if automatic_usd:
            if usd_context is None:
                usd_context = string_to_callable("isaaclab.cloner.usd:UsdReplicateContext")
            contexts = tuple(dict.fromkeys((*contexts, usd_context)))
        explicit_usd = next(
            (
                context
                for context in contexts
                if context.__module__ == "isaaclab.cloner.usd" and context.__qualname__ == "UsdReplicateContext"
            ),
            None,
        )
        if explicit_usd is not None:
            usd_context = explicit_usd
            if sim is None:
                raise RuntimeError("USD cloning requires an active SimulationContext.")
            sim.get_or_create_backend(usd_context, sim.stage, clone_role="scene")
        for context_type in contexts:
            routed_rows.setdefault(context_type, set()).update(rows)

    for context_type, roles in context_roles.items():
        if roles & {"model", "scene"}:
            routed_rows[context_type].update(populated_rows)
        if roles & {"physics", "model", "scene"}:
            routed_rows[context_type].update(whole_env_rows)
    return {context_type: tuple(sorted(rows & populated_rows)) for context_type, rows in routed_rows.items()}


def _finalize_plan(
    sources: tuple[str, ...],
    destinations: tuple[str, ...],
    clone_mask: torch.Tensor,
    env_ids: torch.Tensor,
    positions: torch.Tensor,
    cfg_rows: dict[int, tuple[int, ...]],
    cfgs: tuple[Any, ...],
    global_paths: tuple[str, ...],
    env_template: str,
    whole_env_rows: tuple[int, ...] = (),
) -> ClonePlan:
    """Add exact shared rows and compile the complete clone-context routing."""
    global_paths = tuple(dict.fromkeys(global_paths))
    if any("{}" in path for path in global_paths):
        raise ValueError("Shared scene paths must be exact prim paths without a clone slot.")
    if global_paths:
        sources = (*sources, *global_paths)
        destinations = (*destinations, *global_paths)
        clone_mask = torch.cat(
            [clone_mask, torch.zeros((len(global_paths), len(env_ids)), dtype=torch.bool, device=clone_mask.device)]
        )
    return ClonePlan(
        sources=sources,
        destinations=destinations,
        clone_mask=clone_mask,
        env_ids=env_ids,
        positions=positions,
        cfg_rows=cfg_rows,
        context_rows=_compile_context_rows(cfgs, cfg_rows, destinations, clone_mask, whole_env_rows),
        env_template=env_template,
    )


def make_clone_plan(
    cfgs: Iterable[Any],
    num_clones: int,
    env_spacing: float,
    device: str,
    global_paths: tuple[str, ...] = (),
    clone_strategy: Callable = sequential,
    valid_set: torch.Tensor | None = None,
    env_template: str = DEFAULT_ENV_TEMPLATE,
) -> ClonePlan:
    """Build a :class:`ClonePlan` from asset cfgs.

    Iterates ``cfgs``, identifies env-scoped cfgs with a spawn, expands
    :class:`~isaaclab.sim.MultiAssetSpawnerCfg` / :class:`~isaaclab.sim.MultiUsdFileCfg`
    into per-variant prototype rows, runs ``clone_strategy`` to assign prototypes to
    envs, and returns a self-contained :class:`ClonePlan` with ``cfg_rows`` populated.

    Each input cfg's ``spawn_path`` / ``spawn_paths`` is mutated so the subsequent
    asset constructor spawns the prototype into its first active environment. Every cfg
    is an env-scoped entity with a spawner. Shared assets become exact plan rows with an
    empty clone mask.

    Args:
        cfgs: Cloneable asset cfgs with resolved env-scoped ``prim_path`` and ``spawn``.
        num_clones: Number of target envs.
        env_spacing: Distance between neighboring grid env origins [m].
        device: Torch device for plan tensors.
        global_paths: Complete shared-asset roots declared by the scene composition root. Defaults to none.
        clone_strategy: Function that assigns prototype combinations to envs. Defaults
            to :func:`~isaaclab.cloner.sequential`.
        valid_set: Optional ``[num_combos, num_groups]`` long tensor of valid prototype
            combinations. ``None`` (default) uses the full cartesian product of every
            group's prototype indices.

    Returns:
        A :class:`ClonePlan` whose ``sources``/``destinations``/``clone_mask`` describe
        the flat prototype-to-env mapping and whose ``cfg_rows`` maps each replicated cfg
        to the rows it owns.
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
    env_ids = torch.arange(num_clones, dtype=torch.long, device=device)
    positions, _ = grid_transforms(num_clones, env_spacing, device=device)

    # 2) No env-scoped cfgs: emit an empty plan so the scene can still proceed.
    if not groups:
        empty_mask = torch.zeros((0, num_clones), dtype=torch.bool, device=device)
        return _finalize_plan(
            (),
            (),
            empty_mask,
            env_ids,
            positions,
            {},
            cfgs,
            global_paths,
            env_template,
        )

    # 3) Keep one homogeneous environment root so hand-authored sibling prims remain covered.
    if valid_set is None and all(count == 1 for _, _, _, count in groups):
        for cfg, spawn_cfg, destination, _ in groups:
            set_spawn_paths(spawn_cfg, [destination.format(0)])
        cfg_rows = {id(cfg): (0,) for cfg, _, _, _ in groups}
        return _finalize_plan(
            (env_template.format(0),),
            (env_template,),
            torch.ones((1, num_clones), dtype=torch.bool, device=device),
            env_ids,
            positions,
            cfg_rows,
            cfgs,
            global_paths,
            env_template,
        )

    # 4) Enumerate heterogeneous prototype combinations and build per-row masks.
    group_sizes = [count for _, _, _, count in groups]

    def validate_combo_tensor(combos: torch.Tensor, name: str, expected_rows: int | None = None) -> torch.Tensor:
        if combos.dtype == torch.bool or torch.is_floating_point(combos):
            raise ValueError(f"{name} must contain integer prototype indices.")
        combos = combos.to(device=device, dtype=torch.long)
        if combos.ndim != 2:
            raise ValueError(f"{name} must be a 2-D tensor, got shape {tuple(combos.shape)}.")
        if combos.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one row.")
        if expected_rows is not None and combos.shape[0] != expected_rows:
            raise ValueError(f"{name} must contain {expected_rows} rows, got {combos.shape[0]}.")
        if combos.shape[1] != len(group_sizes):
            raise ValueError(f"{name} must contain {len(group_sizes)} columns, got {combos.shape[1]}.")
        group_sizes_tensor = torch.tensor(group_sizes, dtype=torch.long, device=device).view(1, -1)
        invalid = (combos < -1) | ((combos >= group_sizes_tensor) & (combos != -1))
        if invalid.any():
            raise ValueError(f"{name} contains prototype indices outside [-1, group_size).")
        return combos

    if valid_set is None:
        all_combos = list(itertools.product(*[range(s) for s in group_sizes]))
        combos = torch.tensor(all_combos, dtype=torch.long, device=device)
    else:
        combos = validate_combo_tensor(valid_set, "valid_set")
    chosen = validate_combo_tensor(clone_strategy(combos, num_clones, device), "clone_strategy result", num_clones)

    group_offsets = torch.tensor([0] + list(itertools.accumulate(group_sizes[:-1])), dtype=torch.long, device=device)
    active = chosen >= 0
    rows = (chosen + group_offsets).view(-1)
    cols = torch.arange(num_clones, device=device).view(-1, 1).expand(-1, len(group_sizes)).reshape(-1)
    active_flat = active.view(-1)

    num_rows = sum(group_sizes)
    clone_mask = torch.zeros((num_rows, num_clones), dtype=torch.bool, device=device)
    if active_flat.any():
        clone_mask[rows[active_flat], cols[active_flat]] = True

    sources_list: list[str] = []
    destinations_list: list[str] = []
    cfg_rows: dict[int, tuple[int, ...]] = {}
    row = 0
    for cfg, spawn_cfg, destination, count in groups:
        cfg_rows[id(cfg)] = tuple(range(row, row + count))
        group_mask = clone_mask[row : row + count]
        env_ids_assigned = group_mask.to(torch.int).argmax(dim=1).tolist()
        active = group_mask.any(dim=1).tolist()
        paths = [
            destination.format(env_id) if is_active else None for env_id, is_active in zip(env_ids_assigned, active)
        ]
        for i, path in enumerate(paths):
            destinations_list.append(destination)
            # Inactive prototypes fall back to env-i so the source path stays valid even
            # when the variant has no active environment (matches the legacy behavior).
            sources_list.append(path if path is not None else destination.format(i))
        set_spawn_paths(spawn_cfg, paths)
        row += count

    return _finalize_plan(
        tuple(sources_list),
        tuple(destinations_list),
        clone_mask,
        env_ids,
        positions,
        cfg_rows,
        cfgs,
        global_paths,
        env_template,
    )


def clone_plan_from_env_0(
    source: str,
    destination: str,
    num_clones: int,
    device: str,
    positions: torch.Tensor,
    global_paths: tuple[str, ...] = (),
) -> ClonePlan:
    """Build a whole-environment clone plan after constructing environment zero.

    Auto-populates :attr:`ClonePlan.cfg_rows` from :data:`~isaaclab.cloner.REPLICATION_QUEUE`,
    mapping every env-scoped cfg to the environment-root row. This intentionally covers
    hand-authored sibling prims that are not represented in the queue. ``global_paths`` is the
    complete declaration of shared assets and is never inferred from the stage.

    Args:
        source: Source prim path (typically ``/World/envs/env_0``).
        destination: Destination template with ``"{}"`` for the env id.
        num_clones: Number of target envs.
        device: Torch device for the mask and env id buffers.
        positions: Per-env world positions [m], shape ``[num_clones, 3]``.
        global_paths: Complete shared-asset roots for the hand-built scene. Defaults to none.

    Returns:
        A :class:`ClonePlan` covering the complete environment-zero subtree.
    """
    from .replicate_session import REPLICATION_QUEUE  # noqa: PLC0415

    queued = tuple(REPLICATION_QUEUE)
    global_paths = tuple(dict.fromkeys(global_paths))
    cfg_rows = {id(cfg): (0,) for cfg in queued if match(cfg.prim_path, destination) is not None}
    for global_row, path in enumerate(global_paths, start=1):
        cfg_rows.update({id(cfg): (global_row,) for cfg in queued if cfg.prim_path == path})
    env_ids = torch.arange(num_clones, dtype=torch.long, device=device)
    return _finalize_plan(
        (source,),
        (destination,),
        torch.ones((1, num_clones), dtype=torch.bool, device=device),
        env_ids,
        positions,
        cfg_rows,
        queued,
        global_paths,
        destination,
        whole_env_rows=(0,),
    )
