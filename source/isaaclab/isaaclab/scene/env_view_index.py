# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Centralized environment view index for heterogeneous multi-task scenes.

Design follows Newton's pattern: frozen dataclasses + pure functions + minimal state.

Architecture:
- Layer 1: Pure functions (filter_to_group, get_env_ids, etc.)
- Layer 2: EnvToViewMap wrapper (delegates to pure functions)
- Layer 3: EnvViewIndex - minimal state, exposes raw data for composition
"""

from __future__ import annotations

__all__ = [
    "EnvViewIndex",
    "EnvToViewMap",
    "filter_to_group",
    "resolve_asset_env_ids",
]

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .selector_cfg import SelectorCfg


# ============================================================================
# Layer 1: Pure Functions (Newton-like core)
# ============================================================================


def is_contiguous_slice(indices: list[int]) -> bool:
    """Check if indices form a contiguous sequence."""
    n = len(indices)
    if n > 1:
        for i in range(1, n):
            if indices[i] != indices[i - 1] + 1:
                return False
    return True


@dataclass(frozen=True)
class GroupLayout:
    """Immutable layout data for a group - like Newton's FrequencyLayout.

    Either ``slice`` or ``indices`` is set, never both.
    """

    offset: int
    """First env index in this group."""

    count: int
    """Number of envs in this group."""

    slice: slice | None = None
    """Set if envs are contiguous (zero-copy indexing)."""

    indices: torch.Tensor | None = None
    """Set if envs are non-contiguous (requires gather)."""

    device: str = "cpu"
    """Device for tensor operations."""

    @property
    def is_contiguous(self) -> bool:
        """Whether this group's envs are contiguous."""
        return self.slice is not None


def filter_to_group(layout: GroupLayout, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter env_ids to those in this group. Returns (view_ids, filtered_env_ids)."""
    if layout.slice is not None:
        start, stop = layout.slice.start, layout.slice.stop
        mask = (env_ids >= start) & (env_ids < stop)
        matched = env_ids[mask]
        return matched - start, matched
    else:
        indices = layout.indices
        positions = torch.searchsorted(indices, env_ids)
        valid_mask = (positions < layout.count) & (indices[positions.clamp(max=layout.count - 1)] == env_ids)
        return positions[valid_mask], env_ids[valid_mask]


def get_env_ids(layout: GroupLayout) -> torch.Tensor:
    """Get all env IDs for this group."""
    if layout.slice is not None:
        return torch.arange(layout.slice.start, layout.slice.stop, dtype=torch.long, device=layout.device)
    else:
        return layout.indices


def compute_read_index(
    group_layout: GroupLayout,
    asset_env_ids: torch.Tensor | None,
    group_idx: int,
    asset_group_idxs: tuple[int, ...] | None,
) -> slice | torch.Tensor:
    """Compute read index into asset buffer."""
    if asset_group_idxs is None:
        return group_layout.slice if group_layout.slice is not None else group_layout.indices
    if len(asset_group_idxs) == 1 and asset_group_idxs[0] == group_idx:
        return slice(None)
    if group_layout.slice is not None:
        start = torch.searchsorted(asset_env_ids, group_layout.slice.start).item()
        return slice(start, start + group_layout.count)
    else:
        return torch.searchsorted(asset_env_ids, group_layout.indices)


def build_layout(env_ids: list[int], device: str) -> GroupLayout:
    """Build a GroupLayout from env_ids list.

    Args:
        env_ids: Environment indices (need not be sorted; sorted internally).
        device: Device for tensor operations.
    """
    if len(env_ids) == 0:
        return GroupLayout(offset=0, count=0, slice=slice(0, 0), indices=None, device=device)
    env_ids = sorted(env_ids)
    if is_contiguous_slice(env_ids):
        return GroupLayout(
            offset=env_ids[0],
            count=len(env_ids),
            slice=slice(env_ids[0], env_ids[-1] + 1),
            indices=None,
            device=device,
        )
    else:
        return GroupLayout(
            offset=env_ids[0],
            count=len(env_ids),
            slice=None,
            indices=torch.tensor(env_ids, dtype=torch.long, device=device),
            device=device,
        )


def resolve_asset_env_ids(
    layout: EnvViewIndex,
    asset_name: str,
    env_ids: torch.Tensor | None,
) -> torch.Tensor | None:
    """Map global env_ids to local asset indices for heterogeneous layouts.

    For assets exclusive to a single group, returns 0-based local indices
    within that group. For assets spanning multiple groups, returns indices
    into the asset's combined data buffer (sorted union of all group env_ids).

    Args:
        layout: The EnvViewIndex instance.
        asset_name: Name of the asset.
        env_ids: Global env indices, or None for all envs.

    Returns:
        Local indices for this asset, or None if no envs apply.
    """
    asset_groups = layout.assets.get(asset_name)

    # Homogeneous or unregistered asset: spans all envs
    if not asset_groups:
        return env_ids

    # Single group owns this asset: use fast group-local indices
    if len(asset_groups) == 1:
        if env_ids is None:
            env_ids = torch.arange(layout.num_envs, dtype=torch.long, device=layout._device)
        local, _ = layout[asset_groups[0]].filter(env_ids)
        return local if local.numel() > 0 else None

    # Multi-group asset: find matching global ids, then map to buffer rows
    if env_ids is None:
        env_ids = torch.arange(layout.num_envs, dtype=torch.long, device=layout._device)

    matched_global = []
    for group_name in asset_groups:
        _, matched = layout[group_name].filter(env_ids)
        if matched.numel() > 0:
            matched_global.append(matched)

    if not matched_global:
        return None

    all_matched = torch.cat(matched_global) if len(matched_global) > 1 else matched_global[0]
    asset_env_ids = layout._get_asset_env_ids(asset_name)
    return torch.searchsorted(asset_env_ids, all_matched)


# ============================================================================
# Layer 2: EnvToViewMap (delegates to pure functions)
# ============================================================================


@dataclass
class EnvToViewMap:
    """Maps global env indices to asset-local view indices.

    ``env_ids`` indexes into a full-env output buffer (shape ``(num_envs, ...)``).
    ``view_ids`` indexes into the asset's data buffer (view).
    """

    env_ids: slice | torch.Tensor
    """Index into a full-env ``(num_envs, ...)`` output buffer."""

    view_ids: slice | torch.Tensor
    """Index into the asset's data buffer."""

    layout: GroupLayout = field(repr=False)
    """The underlying GroupLayout - exposed for pure function composition."""

    def filter(self, global_env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Filter env_ids to this group. Returns (view_ids, filtered_env_ids)."""
        return filter_to_group(self.layout, global_env_ids)

    def select(self, tensor: torch.Tensor) -> torch.Tensor:
        """Select rows from an asset-view tensor using :attr:`view_ids`."""
        return tensor[self.view_ids]

    @property
    def count(self) -> int:
        """Number of envs in this group."""
        return self.layout.count


# ============================================================================
# Layer 3: EnvViewIndex (minimal state, exposes raw data for composition)
# ============================================================================


class EnvViewIndex:
    """Centralized environment partitioning for heterogeneous multi-task scenes.

    Minimal API - exposes raw data for caller composition.

    Example::

        # Indexing
        gv = layout["lift", "robot"]
        output[gv.env_ids] = robot.data[gv.view_ids]

        # Composition from raw data
        if (g := layout.assets.get("robot")) and len(g) == 1:
            ...  # asset is exclusive to one group

        # Pure function usage
        view_ids, filtered_env_ids = filter_to_group(gv.layout, env_ids)
    """

    def __init__(self, cfg: SelectorCfg, *, num_envs: int, device: str = "cpu"):
        self._num_envs = num_envs
        self._device = device
        self._homogeneous_layout = GroupLayout(offset=0, count=num_envs, slice=slice(None), indices=None, device=device)
        self._layouts: dict[str, GroupLayout] = {}
        self._group_names: tuple[str, ...] = ()
        self._assets: dict[str, tuple[str, ...]] = {}
        self._cfg = cfg
        self._view_cache: dict[tuple[frozenset[str], str | None], EnvToViewMap] = {}

    # ── fundamental properties ────────────────────────────────────────────

    @property
    def num_envs(self) -> int:
        """Total number of environments."""
        return self._num_envs

    @property
    def group_names(self) -> tuple[str, ...]:
        """Names of all registered groups (immutable)."""
        return self._group_names

    @property
    def assets(self) -> Mapping[str, tuple[str, ...]]:
        """Read-only mapping of asset name → tuple of group names."""
        return MappingProxyType(self._assets)

    def apply_assignment(
        self,
        assignment: torch.Tensor,
        group_names: tuple[str, ...],
        group_assets: dict[str, list[str]] | None = None,
    ) -> None:
        """Populate layout from assignment tensor."""
        assignment = assignment.to(device=self._device, dtype=torch.long)

        self._group_names = group_names
        self._layouts = {}

        for idx, name in enumerate(group_names):
            env_ids = (assignment == idx).nonzero(as_tuple=True)[0].tolist()
            self._layouts[name] = build_layout(env_ids, self._device)

            if group_assets and name in group_assets:
                for asset_name in group_assets[name]:
                    self._add_asset_to_group(asset_name, name)

    def register(self, key: str, env_ids: torch.Tensor) -> None:
        """Register a named environment partition."""
        env_ids_list = env_ids.tolist()

        if any(i < 0 or i >= self._num_envs for i in env_ids_list):
            raise ValueError(f"env_ids for '{key}' out of range [0, {self._num_envs})")
        if len(set(env_ids_list)) != len(env_ids_list):
            raise ValueError(f"env_ids for '{key}' contain duplicates")

        self._layouts[key] = build_layout(env_ids_list, self._device)
        if key not in self._group_names:
            self._group_names = (*self._group_names, key)

    def register_asset(self, asset_name: str, group_key: str) -> None:
        """Register an asset to a group."""
        if group_key in self._layouts:
            self._add_asset_to_group(asset_name, group_key)

    def _add_asset_to_group(self, asset_name: str, group_key: str) -> None:
        current = self._assets.get(asset_name, ())
        if group_key not in current:
            self._assets[asset_name] = (*current, group_key)

    # ── single entry point ────────────────────────────────────────────────

    def __getitem__(self, key: str | None | tuple[str | None, str | None]) -> EnvToViewMap:
        """Get EnvToViewMap: ``layout["lift"]``, ``layout["lift", "robot"]``, or ``layout[None]``."""
        if key is None or isinstance(key, str):
            return self._make_view(key, None)
        return self._make_view(key[0], key[1])

    def get(self, groups: list[str], asset: str | None = None) -> EnvToViewMap:
        """Get a combined :class:`EnvToViewMap` spanning multiple groups, with internal caching.

        For homogeneous layouts (no registered groups), returns a map covering all envs.
        When groups are provided, unions their env IDs and computes ``view_ids`` into
        the asset's data buffer.

        Args:
            groups: Group names to combine.
            asset: Optional asset name for computing ``view_ids``.

        Returns:
            An :class:`EnvToViewMap` whose ``env_ids`` covers the union of the requested
            groups and whose ``view_ids`` indexes into the asset's data buffer.

        Raises:
            KeyError: If any group name is not registered.
        """
        if not self._group_names:
            return EnvToViewMap(env_ids=slice(None), view_ids=slice(None), layout=self._homogeneous_layout)

        cache_key = (frozenset(groups), asset)
        cached = self._view_cache.get(cache_key)
        if cached is not None:
            return cached

        all_ids: list[torch.Tensor] = []
        for name in groups:
            layout = self._layouts.get(name)
            if layout is None:
                raise KeyError(f"unregistered group '{name}'. Available: {list(self._group_names)}")
            all_ids.append(get_env_ids(layout))

        union = torch.cat(all_ids).unique().sort().values if len(all_ids) > 1 else all_ids[0]

        # Build env_ids: use a slice when contiguous, tensor otherwise
        if union.numel() > 0 and is_contiguous_slice(union.tolist()):
            env_ids: slice | torch.Tensor = slice(int(union[0].item()), int(union[-1].item()) + 1)
        else:
            env_ids = union

        # Build view_ids into the asset's data buffer
        if asset is None:
            view_ids: slice | torch.Tensor = slice(None)
        else:
            asset_groups = self._assets.get(asset)
            if asset_groups is None:
                view_ids = env_ids
            else:
                asset_env_ids = self._get_asset_env_ids(asset)
                if isinstance(env_ids, slice):
                    start = torch.searchsorted(asset_env_ids, env_ids.start).item()
                    count = env_ids.stop - env_ids.start
                    view_ids = slice(start, start + count)
                else:
                    view_ids = torch.searchsorted(asset_env_ids, env_ids)

        combined_layout = build_layout(union.tolist(), self._device)
        view = EnvToViewMap(env_ids=env_ids, view_ids=view_ids, layout=combined_layout)
        self._view_cache[cache_key] = view
        return view

    def filter_reset_ids(self, asset_name: str, candidate_env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Intersect reset env IDs with the envs that own ``asset_name``.

        Args:
            asset_name: Name of the asset to filter for.
            candidate_env_ids: Global env indices to filter (from the reset trigger).

        Returns:
            ``(env_ids, view_ids)`` where ``env_ids`` are the subset of
            ``candidate_env_ids`` that contain the asset and ``view_ids``
            are the corresponding indices into the asset's data buffer.
            Both are empty tensors when no envs match.
        """
        asset_groups = self._assets.get(asset_name)

        if not asset_groups:
            return candidate_env_ids, candidate_env_ids

        all_matched: list[torch.Tensor] = []
        for group_name in asset_groups:
            _, matched = self[group_name].filter(candidate_env_ids)
            if matched.numel() > 0:
                all_matched.append(matched)
        if not all_matched:
            empty = torch.tensor([], dtype=torch.long, device=self._device)
            return empty, empty
        env_ids = torch.cat(all_matched) if len(all_matched) > 1 else all_matched[0]

        if len(asset_groups) == 1:
            view_ids, _ = self[asset_groups[0]].filter(env_ids)
        else:
            asset_env_ids = self._get_asset_env_ids(asset_name)
            view_ids = torch.searchsorted(asset_env_ids, env_ids)

        return env_ids, view_ids

    def _make_view(self, group_key: str | None, asset_name: str | None) -> EnvToViewMap:
        if group_key is None:
            return EnvToViewMap(env_ids=slice(None), view_ids=slice(None), layout=self._homogeneous_layout)

        layout = self._layouts.get(group_key)
        if layout is None:
            raise KeyError(f"unregistered group '{group_key}'. Available: {list(self._group_names)}")

        write = layout.slice if layout.slice is not None else layout.indices
        read = self._compute_read(group_key, asset_name, layout)
        return EnvToViewMap(env_ids=write, view_ids=read, layout=layout)

    def _compute_read(self, group_key: str, asset_name: str | None, layout: GroupLayout) -> slice | torch.Tensor:
        if asset_name is None:
            return slice(None)

        asset_groups = self._assets.get(asset_name)
        if asset_groups is None:
            return layout.slice if layout.slice is not None else layout.indices

        group_idx = self._group_names.index(group_key)
        asset_group_idxs = tuple(self._group_names.index(g) for g in asset_groups)

        if len(asset_group_idxs) == 1 and asset_group_idxs[0] == group_idx:
            return slice(None)

        asset_env_ids = self._get_asset_env_ids(asset_name)
        return compute_read_index(layout, asset_env_ids, group_idx, asset_group_idxs)

    def _get_asset_env_ids(self, asset_name: str) -> torch.Tensor:
        asset_groups = self._assets.get(asset_name, ())
        if not asset_groups:
            return torch.arange(self._num_envs, dtype=torch.long, device=self._device)

        tensors = [get_env_ids(self._layouts[g]) for g in asset_groups]
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors).unique().sort().values

    def __repr__(self) -> str:
        groups = ", ".join(f"{n}: {self._layouts[n].count} envs" for n in self._group_names)
        return f"EnvViewIndex(num_envs={self._num_envs}, groups=[{groups or 'homogeneous'}])"
