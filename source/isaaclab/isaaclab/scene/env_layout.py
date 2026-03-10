# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Centralized environment layout for heterogeneous multi-task scenes."""

from __future__ import annotations

from collections.abc import Sequence

import torch


class EnvLayout:
    """Centralized environment partitioning for heterogeneous multi-task scenes.

    One instance per :class:`InteractiveScene`.  All lookup tables, slices and
    masks are cached here — assets and manager terms never instantiate their own
    mapping logic.  They either:

    * query this object by *key* (a string such as the asset or term name), or
    * receive pre-filtered **local** env indices from the orchestration layer
      (scene / manager) and remain completely layout-unaware.

    For homogeneous scenes (single layout) every query degrades to
    ``slice(None)`` / identity with **zero overhead**.

    Example::

        # Setup (done once by InteractiveScene)
        layout = EnvLayout(num_envs=24, device="cuda:0")
        layout.register("lift_cube", [0, 1, 2, 3, 4, 5, 6, 7])
        layout.register("stack_cubes", [8, 9, 10, 11, 12, 13, 14, 15])
        layout.register("ee_pose_cmd", [16, 17, 18, 19, 20, 21, 22, 23])

        # Runtime queries
        layout.global_to_local("lift_cube", torch.tensor([2, 5, 10]))
        # → tensor([2, 5])   (10 is dropped — doesn't belong to lift_cube)

        layout.env_slice("stack_cubes")
        # → slice(8, 16)     (contiguous → zero-copy view)

        layout.scatter("lift_cube", local_reward, fill=0.0)
        # → (24,) tensor with reward in rows 0-7, zeros elsewhere
    """

    def __init__(self, num_envs: int, device: str = "cuda:0"):
        self._num_envs = num_envs
        self._device = device
        # name → canonical id tuple
        self._groups: dict[str, tuple[int, ...]] = {}
        # canonical id tuple → cached tensors (shared when ids are identical)
        self._lookups: dict[tuple[int, ...], torch.Tensor] = {}
        self._slices: dict[tuple[int, ...], slice | torch.Tensor] = {}
        self._masks: dict[tuple[int, ...], torch.Tensor] = {}
        # task-group partition (populated by apply_task_groups)
        self._task_group_partition: dict[str, list[int]] | None = None

    # ── properties ────────────────────────────────────────────────────────

    @property
    def num_envs(self) -> int:
        """Total number of environments across all groups."""
        return self._num_envs

    @property
    def device(self) -> str:
        """Compute device."""
        return self._device

    @property
    def is_heterogeneous(self) -> bool:
        """Whether any partial group has been registered."""
        return len(self._groups) > 0

    @property
    def group_names(self) -> list[str]:
        """Names of all registered groups."""
        return list(self._groups.keys())

    @property
    def task_group_partition(self) -> dict[str, list[int]] | None:
        """The task-group partition, or ``None`` if not configured."""
        return self._task_group_partition

    def apply_task_groups(self, task_groups: dict[str, int] | int) -> None:
        """Partition environments by task groups and register each group.

        Calls :func:`partition_env_ids` internally and registers every
        resulting group.  The partition is stored and accessible via
        :attr:`task_group_partition`.

        Args:
            task_groups: Either an ``int`` for equal-sized anonymous groups,
                or a ``dict[str, int]`` mapping group name to relative weight.

        Raises:
            RuntimeError: If task groups have already been applied.
        """
        if self._task_group_partition is not None:
            raise RuntimeError("Task groups have already been applied to this layout.")
        self._task_group_partition = partition_env_ids(self._num_envs, task_groups)
        for group_name, env_ids in self._task_group_partition.items():
            self.register(group_name, env_ids)

    def resolve_task_group(self, key: str, task_group: str) -> list[int]:
        """Look up the env_ids for a task group, with validation.

        Args:
            key: The name of the entity requesting resolution (for error messages).
            task_group: The task group name to look up.

        Returns:
            The list of environment indices for the requested group.

        Raises:
            ValueError: If no task groups are configured or the name is unknown.
        """
        if self._task_group_partition is None:
            raise ValueError(f"'{key}' sets task_group='{task_group}' but no task_groups are configured on the scene.")
        if task_group not in self._task_group_partition:
            raise ValueError(
                f"'{key}' references unknown task_group='{task_group}'. "
                f"Available groups: {list(self._task_group_partition.keys())}"
            )
        return self._task_group_partition[task_group]

    # ── registration ──────────────────────────────────────────────────────

    def register(self, key: str, env_ids: Sequence[int]) -> None:
        """Register a named environment partition.

        The same key may be re-registered (the old entry is replaced).
        Multiple keys that map to the *same* set of env indices will
        automatically share cached lookup tables and slices.

        Args:
            key: Unique group name (e.g. an asset name or ``"commands/ee_pose"``).
            env_ids: Global environment indices belonging to this group.

        Raises:
            ValueError: If any index is out of ``[0, num_envs)``.
            ValueError: If *env_ids* contains duplicate entries.
        """
        ids = tuple(env_ids)
        if any(i < 0 or i >= self._num_envs for i in ids):
            raise ValueError(f"env_ids for '{key}' out of range [0, {self._num_envs}): got {ids}")
        if len(ids) != len(set(ids)):
            raise ValueError(f"env_ids for '{key}' contain duplicates: {ids}")
        self._groups[key] = ids

    # ── simple queries ────────────────────────────────────────────────────

    def is_partial(self, key: str) -> bool:
        """Whether *key* covers only a subset of environments."""
        return key in self._groups

    def num_envs_for(self, key: str) -> int:
        """Number of environments in a group (all envs if unregistered)."""
        return len(self._groups[key]) if key in self._groups else self._num_envs

    def env_ids(self, key: str) -> tuple[int, ...]:
        """Global env indices for a group (all envs if unregistered)."""
        return self._groups.get(key, tuple(range(self._num_envs)))

    def env_slice(self, key: str) -> slice | torch.Tensor:
        """Fast indexer: ``slice(None)`` if full, ``slice(a, b)`` if
        contiguous, else a long tensor.  Maximises GPU efficiency.
        """
        if key not in self._groups:
            return slice(None)
        ids = self._groups[key]
        if ids not in self._slices:
            self._slices[ids] = _build_slice(ids, self._device)
        return self._slices[ids]

    def mask(self, key: str) -> torch.Tensor:
        """Boolean mask of shape ``(num_envs,)`` — ``True`` for envs in the
        named group.  Returns an all-``True`` mask for unregistered keys.
        """
        if key not in self._groups:
            ids: tuple[int, ...] = tuple(range(self._num_envs))
        else:
            ids = self._groups[key]
        if ids not in self._masks:
            m = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
            m[list(ids)] = True
            self._masks[ids] = m
        return self._masks[ids]

    # ── env-id mapping ────────────────────────────────────────────────────

    def global_to_local(self, key: str, global_ids: torch.Tensor) -> torch.Tensor:
        """Map global env indices to local (0-based) indices for a group.

        Indices that do not belong to the group are **silently dropped**.
        For unregistered (homogeneous) keys the input is returned unchanged.

        Args:
            key: Group name.
            global_ids: 1-D long tensor of global env indices.

        Returns:
            1-D long tensor of local indices.
        """
        if key not in self._groups:
            return global_ids
        lut = self._get_lookup(self._groups[key])
        max_id = lut.shape[0] - 1
        clamped = global_ids.clamp(max=max_id)
        valid = (global_ids <= max_id) & (lut[clamped] >= 0)
        return lut[global_ids[valid]]

    def filter_and_split(self, key: str, global_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(local_ids, matching_global_ids)``.

        Useful when you need both local indices (for term-internal buffers)
        and the corresponding global indices (for scene-wide data like
        :attr:`InteractiveScene.env_origins`).

        Args:
            key: Group name.
            global_ids: 1-D long tensor of global env indices.

        Returns:
            Tuple of (local_ids, matching_global_ids).
        """
        if key not in self._groups:
            return global_ids, global_ids
        lut = self._get_lookup(self._groups[key])
        max_id = lut.shape[0] - 1
        clamped = global_ids.clamp(max=max_id)
        valid = (global_ids <= max_id) & (lut[clamped] >= 0)
        return lut[global_ids[valid]], global_ids[valid]

    # ── tensor alignment ──────────────────────────────────────────────────

    def scatter(self, key: str, local_data: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
        """Scatter local-space data into a full-env tensor.

        Args:
            key: Group name.
            local_data: Tensor of shape ``(num_local_envs, ...)``.
            fill: Fill value for environments not in the group.

        Returns:
            Tensor of shape ``(num_envs, ...)``.
        """
        if key not in self._groups:
            return local_data
        shape = (self._num_envs, *local_data.shape[1:])
        out = local_data.new_full(shape, fill)
        out[self.env_slice(key)] = local_data
        return out

    def gather(self, key: str, full_data: torch.Tensor) -> torch.Tensor:
        """Gather entries from a full-env tensor for a specific group.

        Args:
            key: Group name.
            full_data: Tensor of shape ``(num_envs, ...)``.

        Returns:
            Tensor of shape ``(num_local_envs, ...)``.
        """
        if key not in self._groups:
            return full_data
        return full_data[self.env_slice(key)]

    def cross_slice(self, term_key: str, asset_key: str) -> slice | torch.Tensor:
        """Return indices to align asset data with a term's local buffers.

        Use this when a term manages a subset of envs but the asset it
        references spans a *different* (usually larger) set.

        * Both homogeneous → ``slice(None)``
        * Term partial, asset full → ``env_slice(term_key)``
        * Both partial (same ids) → ``slice(None)``

        Args:
            term_key: Layout key of the term.
            asset_key: Layout key of the asset.

        Returns:
            ``slice(None)`` or a long tensor suitable for indexing.
        """
        if term_key not in self._groups:
            return slice(None)
        if asset_key not in self._groups:
            return self.env_slice(term_key)
        return slice(None)

    # ── metrics ───────────────────────────────────────────────────────────

    def per_group_mean(self, values: torch.Tensor) -> dict[str, float]:
        """Compute per-group mean of a ``(num_envs,)`` tensor.

        Args:
            values: Per-environment values.

        Returns:
            Dict mapping group name to the mean value within that group.
        """
        result: dict[str, float] = {}
        for name in self._groups:
            result[name] = self.gather(name, values).mean().item()
        return result

    # ── validation helpers ────────────────────────────────────────────────

    def validate_no_overlap(self, keys: Sequence[str]) -> None:
        """Assert that listed groups have disjoint env indices.

        Args:
            keys: Group names to check.

        Raises:
            ValueError: If any two groups share env indices.
        """
        seen: set[int] = set()
        for k in keys:
            ids = set(self._groups.get(k, ()))
            overlap = seen & ids
            if overlap:
                raise ValueError(f"Overlapping env_ids {overlap} among groups {list(keys)}")
            seen |= ids

    def validate_full_coverage(self, keys: Sequence[str]) -> None:
        """Assert that the union of listed groups covers all environments.

        Args:
            keys: Group names whose union should equal ``range(num_envs)``.

        Raises:
            ValueError: If some env indices are not covered.
        """
        covered: set[int] = set()
        for k in keys:
            covered |= set(self._groups.get(k, ()))
        missing = set(range(self._num_envs)) - covered
        if missing:
            raise ValueError(f"Env indices {sorted(missing)} not covered by groups {list(keys)}")

    # ── internals ─────────────────────────────────────────────────────────

    def _get_lookup(self, ids: tuple[int, ...]) -> torch.Tensor:
        """Return a cached global→local lookup table for a canonical id set."""
        if ids not in self._lookups:
            t = torch.tensor(ids, device=self._device, dtype=torch.long)
            lut = torch.full((int(t.max().item()) + 1,), -1, dtype=torch.long, device=self._device)
            lut[t] = torch.arange(len(t), device=self._device)
            self._lookups[ids] = lut
        return self._lookups[ids]

    def __repr__(self) -> str:
        groups_info = ", ".join(f"{k}({len(v)} envs)" for k, v in self._groups.items())
        return f"EnvLayout(num_envs={self._num_envs}, groups=[{groups_info or 'homogeneous'}])"


# ── utility functions ─────────────────────────────────────────────────────


def _build_slice(ids: tuple[int, ...], device: str) -> slice | torch.Tensor:
    """Build the most efficient indexer for the given id set."""
    if len(ids) > 0 and ids == tuple(range(ids[0], ids[0] + len(ids))):
        return slice(ids[0], ids[0] + len(ids))
    return torch.tensor(ids, dtype=torch.long, device=device)


def partition_env_ids(
    num_envs: int,
    groups: dict[str, int] | int,
) -> dict[str, list[int]]:
    """Partition environment indices across named groups.

    Args:
        num_envs: Total number of environments.
        groups: Either an ``int`` for equal-sized anonymous groups
            (keys will be ``"group_0"``, ``"group_1"``, ...), or a
            ``dict[str, int]`` mapping group name to desired size.
            Sizes are treated as weights when they don't sum to
            *num_envs*.

    Returns:
        Dict mapping group names to lists of environment indices.

    When *num_envs* is not evenly divisible, each group (except the
    last) is sized by ``round(num_envs * weight / total_weight)`` and
    the last group absorbs whatever remains.

    Example::

        partition_env_ids(24, {"lift": 8, "stack": 8, "reach": 8})
        # {"lift": [0..7], "stack": [8..15], "reach": [16..23]}

        partition_env_ids(24, 3)
        # {"group_0": [0..7], "group_1": [8..15], "group_2": [16..23]}

        # indivisible case: 10 envs, 3 equal-weight groups
        partition_env_ids(10, {"A": 1, "B": 1, "C": 1})
        # {"A": [0..2], "B": [3..5], "C": [6..9]}
        # A and B get 3 envs each (round(10/3) = 3), C gets the remaining 4.
    """
    if isinstance(groups, int):
        groups = {f"group_{i}": 1 for i in range(groups)}

    names = list(groups.keys())
    weights = list(groups.values())
    total_weight = sum(weights)

    result: dict[str, list[int]] = {}
    start = 0
    remaining = num_envs
    for i, (name, w) in enumerate(zip(names, weights)):
        if i == len(names) - 1:
            size = remaining
        else:
            size = round(num_envs * w / total_weight)
            size = min(size, remaining)
        result[name] = list(range(start, start + size))
        start += size
        remaining -= size

    return result
