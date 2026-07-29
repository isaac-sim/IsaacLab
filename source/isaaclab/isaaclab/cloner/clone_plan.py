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

from .cloner_cfg import InclusionSet
from .cloner_strategies import sequential
from .path import split


@dataclass(frozen=True, eq=False)
class ClonePlan:
    """Description of a single replication layout, consumed by :func:`~isaaclab.cloner.replicate`."""

    sources: tuple[str, ...]
    """Source prim paths, one per replication row."""

    destinations: tuple[str, ...]
    """Destination path templates with ``"{}"`` for the env id, one per row."""

    clone_mask: torch.Tensor
    """Bool tensor ``[len(sources), num_clones]``; ``True`` if env ``j`` comes from row ``i``."""

    env_ids: torch.Tensor | None = None
    """Long tensor ``[num_clones]`` of target env ids.

    Optional for plans used only with :func:`~isaaclab.cloner.query.iter_sources` or
    :func:`~isaaclab.cloner.query.path_to_source`; required by :func:`~isaaclab.cloner.replicate`.
    """

    positions: torch.Tensor | None = None
    """Per-env world positions [m], shape ``[num_clones, 3]``, or ``None``."""

    cfg_rows: dict[int, tuple[int, ...]] = field(default_factory=dict)
    """``id(cfg)`` to the row indices the cfg owns."""


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


def make_clone_plan(
    cfgs: Iterable[Any],
    num_clones: int,
    env_spacing: float,
    device: str,
    *,
    clone_strategy: Callable = sequential,
    valid_set: torch.Tensor | None = None,
) -> ClonePlan:
    """Build a :class:`ClonePlan` from asset cfgs.

    Iterates ``cfgs``, identifies env-scoped cfgs with a spawn, expands
    :class:`~isaaclab.sim.MultiAssetSpawnerCfg` / :class:`~isaaclab.sim.MultiUsdFileCfg`
    into per-variant prototype rows, runs ``clone_strategy`` to assign prototypes to
    envs, and returns a self-contained :class:`ClonePlan` with ``cfg_rows`` populated.

    Each input cfg's ``spawn_path`` / ``spawn_paths`` is mutated so the subsequent
    asset constructor spawns the prototype into its first active environment. Cfgs
    whose ``prim_path`` is global (not under the env root ``/World/envs/``) or that
    lack a spawn are skipped — they do not appear in the plan and are not replicated.

    Args:
        cfgs: Asset cfgs with resolved ``prim_path`` (no ``{ENV_REGEX_NS}`` macros).
        num_clones: Number of target envs.
        env_spacing: Distance between neighboring grid env origins [m].
        device: Torch device for plan tensors.
        clone_strategy: Function that assigns prototype combinations to envs. Defaults
            to :func:`~isaaclab.cloner.sequential`.
        valid_set: Optional ``[num_combos, num_groups]`` long tensor of valid prototype
            combinations. ``None`` (default) uses the full cartesian product of every
            group's prototype indices.

    Returns:
        A :class:`ClonePlan` whose ``sources``/``destinations``/``clone_mask`` describe
        the flat prototype-to-env mapping and whose ``cfg_rows`` maps each cfg to the
        rows it owns.
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

    env_root_marker = "/World/envs/"
    env_template = "/World/envs/env_{}"

    # 1) Build per-group records: (cfg, spawn_cfg, destination_template, num_variants).
    groups: list[tuple[Any, Any, str, int]] = []
    for cfg in cfgs:
        if not hasattr(cfg, "prim_path") or not hasattr(cfg, "spawn") or cfg.spawn is None:
            continue
        prim_path = cfg.prim_path
        if env_root_marker not in prim_path:
            continue
        count = num_spawn_variants(cfg.spawn)
        if count <= 0:
            raise ValueError(f"Spawner at '{prim_path}' must have at least one variant.")
        destination = prim_path.replace(".*", "{}")
        groups.append((cfg, cfg.spawn, destination, count))

    env_ids = torch.arange(num_clones, dtype=torch.long, device=device)
    positions, _ = grid_transforms(num_clones, env_spacing, device=device)

    # 2) No env-scoped cfgs: emit an empty plan so the scene can still proceed.
    if not groups:
        empty_mask = torch.zeros((0, num_clones), dtype=torch.bool, device=device)
        return ClonePlan(
            sources=(),
            destinations=(),
            clone_mask=empty_mask,
            env_ids=env_ids,
            positions=positions,
            cfg_rows={},
        )

    # 3) Homogeneous (every cfg is single-variant): emit the simpler env-root plan.
    if valid_set is None and all(count == 1 for _, _, _, count in groups):
        for cfg, spawn_cfg, destination, _ in groups:
            set_spawn_paths(spawn_cfg, [destination.format(0)])
        cfg_rows = {id(cfg): (0,) for cfg, _, _, _ in groups}
        return ClonePlan(
            sources=(env_template.format(0),),
            destinations=(env_template,),
            clone_mask=torch.ones((1, num_clones), dtype=torch.bool, device=device),
            env_ids=env_ids,
            positions=positions,
            cfg_rows=cfg_rows,
        )

    # 4) Heterogeneous: enumerate prototype combos, build per-row mask, mutate spawn paths.
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

    return ClonePlan(
        sources=tuple(sources_list),
        destinations=tuple(destinations_list),
        clone_mask=clone_mask,
        env_ids=env_ids,
        positions=positions,
        cfg_rows=cfg_rows,
    )


def clone_plan_from_env_0(
    source: str,
    destination: str,
    num_clones: int,
    device: str,
    positions: torch.Tensor | None = None,
) -> ClonePlan:
    """Build a single-source clone plan that targets every env from one source row.

    Auto-populates :attr:`ClonePlan.cfg_rows` from :data:`~isaaclab.cloner.REPLICATION_QUEUE`,
    including only cfgs whose ``prim_path`` falls under the env-root prefix of
    ``destination``. Must be called *after* all asset constructors have run, so their cfgs
    are already registered in the queue; otherwise those assets will be skipped by the
    subsequent :func:`~isaaclab.cloner.replicate` call.

    Args:
        source: Source prim path (typically ``/World/envs/env_0``).
        destination: Destination template with ``"{}"`` for the env id.
        num_clones: Number of target envs.
        device: Torch device for the mask and env id buffers.
        positions: Optional per-env world positions [m], shape ``[num_clones, 3]``.

    Returns:
        A :class:`ClonePlan` with a single source row covering every env.
    """
    from .replicate_session import REPLICATION_QUEUE  # noqa: PLC0415

    prefix, _ = split(destination)
    cfg_rows: dict[int, tuple[int, ...]] = {
        id(cfg): (0,) for cfg in REPLICATION_QUEUE if cfg.prim_path.startswith(prefix)
    }
    return ClonePlan(
        sources=(source,),
        destinations=(destination,),
        clone_mask=torch.ones((1, num_clones), dtype=torch.bool, device=device),
        env_ids=torch.arange(num_clones, dtype=torch.long, device=device),
        positions=positions,
        cfg_rows=cfg_rows,
    )
