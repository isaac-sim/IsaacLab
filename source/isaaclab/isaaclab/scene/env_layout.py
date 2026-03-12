# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Centralized environment layout for heterogeneous multi-task scenes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import torch


@dataclass
class _RobotMetaEntry:
    """Mutable accumulator for per-robot metadata.

    Populated incrementally by :class:`ActionManager` (``ee_body``,
    ``joint_patterns``, ``num_joints``) and :class:`CommandManager`
    (``command_name``) during ``_prepare_terms``.
    """

    ee_body: str | None = None
    joint_patterns: list[str] = field(default_factory=list)
    num_joints: int | None = None
    command_name: str | None = None


class RobotSpec(NamedTuple):
    """Per-robot metadata auto-generated from action and command manager configs.

    Instances are produced by :attr:`EnvLayout.robot_specs` and consumed by
    multi-robot observation, reward, and event functions.
    """

    asset_name: str
    """Scene-level asset name (e.g. ``"franka_robot"``)."""
    ee_body: str
    """End-effector body name (e.g. ``"panda_hand"``)."""
    command_name: str
    """Command-manager term name (e.g. ``"franka_ee_pose"``)."""
    joint_patterns: list[str]
    """Joint name regex patterns (e.g. ``["panda_joint.*"]``)."""


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
        layout.apply_task_groups({"lift": 1, "stack": 1, "reach": 1})
        # Registers three groups of 8 envs each.

        # Runtime queries
        layout.global_to_local("lift", torch.tensor([2, 5, 10]))
        # → tensor([2, 5])   (10 is dropped — doesn't belong to "lift")

        layout.env_slice("stack")
        # → slice(8, 16)     (contiguous → zero-copy view)

        layout.scatter("lift", local_reward, fill=0.0)
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
        self._id_tensors: dict[tuple[int, ...], torch.Tensor] = {}
        # task-group partition (populated by apply_task_groups)
        self._task_group_partition: dict[str, list[int]] | None = None
        # entity → group key mappings (centralized registry)
        self._asset_groups: dict[str, str] = {}
        self._term_groups: dict[str, str] = {}
        # cached default ids for unregistered / None keys
        self._all_env_ids: tuple[int, ...] = tuple(range(num_envs))
        # per-task group robot metadata (populated by action/command managers)
        self._robot_meta: dict[str, _RobotMetaEntry] = {}
        self._max_arm_dof: int | None = None
        self._robot_specs_cache: list[RobotSpec] | None = None

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

    # ── entity registry ────────────────────────────────────────────────────

    def register_asset(self, asset_name: str, group_key: str) -> None:
        """Register an asset → group mapping.

        Called by :class:`InteractiveScene` during setup so that managers
        can later resolve layout information by asset name alone.

        Args:
            asset_name: Scene-level name of the asset.
            group_key: The group key (must already be registered via
                :meth:`register` or :meth:`apply_task_groups`).
        """
        self._asset_groups[asset_name] = group_key

    def register_term(self, term_id: str, group_key: str) -> None:
        """Register a manager-term → group mapping.

        Called by managers during ``_prepare_terms`` so that subsequent
        dispatch operations (reset, process_action, …) can resolve
        layout information by term name alone.

        Args:
            term_id: Manager-level term name (the config attribute name).
            group_key: The group key.
        """
        self._term_groups[term_id] = group_key

    def group_for_asset(self, asset_name: str) -> str | None:
        """Return the group key for an asset, or ``None`` if not registered."""
        return self._asset_groups.get(asset_name)

    def group_for_term(self, term_id: str) -> str | None:
        """Return the group key for a manager term, or ``None``."""
        return self._term_groups.get(term_id)

    def resolve_group_key(self, *, task_group: str | None = None, asset_name: str | None = None) -> str | None:
        """Resolve a group key from configuration parameters.

        Priority: *task_group* (if it names a registered group) >
        *asset_name* (via :meth:`group_for_asset`).

        Args:
            task_group: Explicit task-group name from the term config.
            asset_name: Asset name whose group to inherit.

        Returns:
            The resolved group key, or ``None`` when homogeneous.
        """
        if task_group is not None and task_group in self._groups:
            return task_group
        if asset_name is not None:
            return self._asset_groups.get(asset_name)
        return None

    # ── robot metadata registry ───────────────────────────────────────

    def register_robot_meta(
        self,
        asset_name: str,
        *,
        ee_body: str | None = None,
        joint_patterns: list[str] | None = None,
        num_joints: int | None = None,
        command_name: str | None = None,
    ) -> None:
        """Register per-robot metadata for a grouped asset.

        Called by :class:`ActionManager` and :class:`CommandManager` during
        ``_prepare_terms``.  Fields are merged by *asset_name* so that action
        and command managers can each contribute their part independently.

        Args:
            asset_name: Scene-level name of the asset.
            ee_body: End-effector body name (from task-space action configs such as DiffIK).
            joint_patterns: Joint name regex patterns (from action configs).
            num_joints: Resolved joint count for this robot's arm (pushed by :class:`ActionManager`).
            command_name: Manager-level command term name.
        """
        if asset_name not in self._robot_meta:
            self._robot_meta[asset_name] = _RobotMetaEntry()
        meta = self._robot_meta[asset_name]
        if ee_body is not None:
            meta.ee_body = ee_body
        if joint_patterns is not None:
            meta.joint_patterns = joint_patterns
        if num_joints is not None:
            meta.num_joints = num_joints
        if command_name is not None:
            meta.command_name = command_name
        self._max_arm_dof = None
        self._robot_specs_cache = None

    @property
    def robot_specs(self) -> list[RobotSpec]:
        """Auto-generated robot specifications for grouped assets.

        Only assets with *both* action metadata (``ee_body``) and command
        metadata (``command_name``) are included.  The result is cached
        after the first call.

        Returns:
            :class:`RobotSpec` instances for all fully-registered grouped robots,
            ordered by asset registration.
        """
        if self._robot_specs_cache is not None:
            return self._robot_specs_cache
        specs: list[RobotSpec] = []
        for asset_name, meta in self._robot_meta.items():
            if meta.ee_body is not None and meta.command_name is not None and len(meta.joint_patterns) > 0:
                specs.append(RobotSpec(asset_name, meta.ee_body, meta.command_name, meta.joint_patterns))
        self._robot_specs_cache = specs
        return specs

    @property
    def max_arm_dof(self) -> int:
        """Maximum joint count across all registered robots.

        Returns 0 when no robots have ``num_joints`` registered.
        """
        if self._max_arm_dof is not None:
            return self._max_arm_dof
        self._max_arm_dof = max(
            (meta.num_joints for meta in self._robot_meta.values() if meta.num_joints is not None),
            default=0,
        )
        return self._max_arm_dof

    # ── scatter helpers ────────────────────────────────────────────────────

    def scatter_per_robot(
        self,
        feat_dim: int,
        compute_fn: Callable[[RobotSpec], torch.Tensor],
    ) -> torch.Tensor:
        """Scatter per-robot features into a global ``(num_envs, feat_dim)`` tensor.

        Iterates over :attr:`robot_specs`, resolves each robot's group env IDs,
        calls *compute_fn* for that robot, and writes the result into the
        corresponding rows of the output tensor.

        Args:
            feat_dim: Feature width of the output.
            compute_fn: ``(spec) -> Tensor`` of shape ``(group_envs, feat_dim)``.
                The caller typically captures ``env`` via closure.

        Returns:
            Shape ``(num_envs, feat_dim)``.
        """
        out = torch.zeros(self._num_envs, feat_dim, device=self._device)
        for spec in self.robot_specs:
            gids = self.asset_env_ids_t(spec[0])
            if gids is None:
                continue
            out[gids] = compute_fn(spec)
        return out

    def scatter_per_robot_1d(
        self,
        compute_fn: Callable[[RobotSpec], torch.Tensor],
    ) -> torch.Tensor:
        """Like :meth:`scatter_per_robot` but for scalar-per-env outputs.

        Args:
            compute_fn: ``(spec) -> Tensor`` of shape ``(group_envs,)``.

        Returns:
            Shape ``(num_envs,)``.
        """
        out = torch.zeros(self._num_envs, device=self._device)
        for spec in self.robot_specs:
            gids = self.asset_env_ids_t(spec[0])
            if gids is None:
                continue
            out[gids] = compute_fn(spec)
        return out

    # ── registration ──────────────────────────────────────────────────────

    def register(self, key: str, env_ids: Sequence[int]) -> None:
        """Register a named environment partition.

        The same key may be re-registered (the old entry is replaced).
        Multiple keys that map to the *same* set of env indices will
        automatically share cached lookup tables and slices.

        Typically called internally by :meth:`apply_task_groups`.

        Args:
            key: Unique group name (usually a task group name such as ``"lift"``).
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

    def num_envs_for(self, key: str | None) -> int:
        """Number of environments in a group (all envs if *key* is ``None`` or unregistered)."""
        return len(self._groups[key]) if key in self._groups else self._num_envs

    def env_ids(self, key: str | None) -> tuple[int, ...]:
        """Global env indices for a group (all envs if *key* is ``None`` or unregistered)."""
        if key is None:
            return self._all_env_ids
        return self._groups.get(key, self._all_env_ids)

    def env_ids_t(self, key: str | None) -> torch.Tensor:
        """Like :meth:`env_ids` but returns a cached ``torch.long`` tensor.

        Suitable for direct tensor row-indexing (avoids the tuple → multi-dim
        indexing pitfall).
        """
        ids = self.env_ids(key)
        if ids not in self._id_tensors:
            self._id_tensors[ids] = torch.tensor(ids, device=self._device, dtype=torch.long)
        return self._id_tensors[ids]

    def asset_env_ids_t(self, asset_name: str) -> torch.Tensor | None:
        """Cached long tensor of global env IDs for an asset's group, or ``None``."""
        key = self._asset_groups.get(asset_name)
        if key is None:
            return None
        return self.env_ids_t(key)

    def env_slice(self, key: str | None) -> slice | torch.Tensor:
        """Fast indexer: ``slice(None)`` if *key* is ``None`` or full,
        ``slice(a, b)`` if contiguous, else a long tensor.
        """
        if key not in self._groups:
            return slice(None)
        ids = self._groups[key]
        if ids not in self._slices:
            self._slices[ids] = _build_slice(ids, self._device)
        return self._slices[ids]

    def mask(self, key: str | None) -> torch.Tensor:
        """Boolean mask of shape ``(num_envs,)`` — ``True`` for envs in the
        named group.  Returns an all-``True`` mask for unregistered keys.
        """
        if key not in self._groups:
            ids: tuple[int, ...] = self._all_env_ids
        else:
            ids = self._groups[key]
        if ids not in self._masks:
            m = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
            m[list(ids)] = True
            self._masks[ids] = m
        return self._masks[ids]

    # ── env-id mapping ────────────────────────────────────────────────────

    def global_to_local(self, key: str | None, global_ids: torch.Tensor) -> torch.Tensor:
        """Map global env indices to local (0-based) indices for a group.

        Indices that do not belong to the group are **silently dropped**.
        If *key* is ``None`` or unregistered, the input is returned unchanged.

        Args:
            key: Group name, or ``None`` for homogeneous assets.
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

    def local_to_global(self, key: str | None, local_ids: torch.Tensor) -> torch.Tensor:
        """Map local (0-based) indices back to global env indices.

        If *key* is ``None`` or unregistered, the input is returned unchanged.

        Args:
            key: Group name, or ``None`` for homogeneous assets.
            local_ids: 1-D long tensor of group-local env indices.

        Returns:
            1-D long tensor of global indices.
        """
        if key not in self._groups:
            return local_ids
        id_t = self.env_ids_t(key)
        return id_t[local_ids]

    def filter_and_split(self, key: str | None, global_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(local_ids, matching_global_ids)``.

        Useful when you need both local indices (for term-internal buffers)
        and the corresponding global indices (for scene-wide data like
        :attr:`InteractiveScene.env_origins`).

        If *key* is ``None`` or unregistered, both outputs equal *global_ids*.

        Args:
            key: Group name, or ``None`` for homogeneous assets.
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

    def scatter(self, key: str | None, local_data: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
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

    def gather(self, key: str | None, full_data: torch.Tensor) -> torch.Tensor:
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

    def cross_slice(self, term_key: str | None, asset_key: str | None) -> slice | torch.Tensor:
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

    # ── high-level dispatch helpers ────────────────────────────────────────

    def resolve_env_ids(
        self, key: str | None, env_ids: Sequence[int] | slice | None
    ) -> Sequence[int] | torch.Tensor | slice | None:
        """Map global *env_ids* to local indices for a group.

        Handles all edge cases so callers need no branching logic:

        * ``env_ids`` is ``None`` or ``slice`` → returned as-is.
        * *key* is ``None`` or unregistered → *env_ids* returned unchanged.
        * Otherwise → :meth:`global_to_local` is applied; returns ``None``
          when no environments match (caller should skip the term/asset).

        Args:
            key: Group key, or ``None`` for homogeneous entities.
            env_ids: Global env indices, ``slice(None)``, or ``None``.

        Returns:
            Local env indices, ``slice(None)``/``None`` (pass-through),
            or ``None`` if no environments belong to this group.
        """
        if key is None or key not in self._groups:
            return env_ids
        if env_ids is None or isinstance(env_ids, slice):
            return env_ids
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self._device)
        local = self.global_to_local(key, env_ids_t)
        return local if local.numel() > 0 else None

    def resolve_term_env_ids(
        self, term_id: str, env_ids: Sequence[int] | slice | None
    ) -> Sequence[int] | torch.Tensor | slice | None:
        """Convenience: :meth:`resolve_env_ids` keyed by term name."""
        return self.resolve_env_ids(self._term_groups.get(term_id), env_ids)

    def resolve_asset_env_ids(
        self, asset_name: str, env_ids: Sequence[int] | slice | None
    ) -> Sequence[int] | torch.Tensor | slice | None:
        """Convenience: :meth:`resolve_env_ids` keyed by asset name."""
        return self.resolve_env_ids(self._asset_groups.get(asset_name), env_ids)

    def term_env_slice(self, term_id: str) -> slice | torch.Tensor:
        """Return the env slice for a registered term.

        Falls back to ``slice(None)`` when the term is not registered
        or covers all environments.
        """
        return self.env_slice(self._term_groups.get(term_id))

    def term_num_envs(self, term_id: str) -> int:
        """Return the number of environments managed by a registered term."""
        return self.num_envs_for(self._term_groups.get(term_id))

    def term_cross_slice(self, term_id: str, asset_name: str) -> slice | torch.Tensor:
        """Return a cross-slice aligning asset data with a term's buffers."""
        term_key = self._term_groups.get(term_id)
        asset_key = self._asset_groups.get(asset_name)
        return self.cross_slice(term_key, asset_key)

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
