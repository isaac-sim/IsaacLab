# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Centralized environment layout for heterogeneous multi-task scenes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


class RobotInfo:
    """Per-robot metadata accumulator and public descriptor.

    Populated incrementally by :class:`ActionManager` and :class:`CommandManager` during ``_prepare_terms`` via
    :meth:`EnvLayout.register_robot_meta`.  Each ``ActionTerm`` / ``CommandTerm`` contributes its own slice of
    metadata through :meth:`~isaaclab.managers.ManagerTermBase.robot_metadata`.

    Well-known keys (used by the auto-injection in :meth:`ManagerBase._build_per_robot_mdp_term_caches`):

    * ``ee_body`` — end-effector body name (from task-space actions).
    * ``joint_patterns`` — arm joint regex patterns (from actions).
    * ``command_name`` — command-manager term name.

    Additional keys (e.g. ``num_joints``, ``gripper_joint_patterns``, ``body_offset``) can be registered
    freely and are auto-injected into MDP term functions whose signatures declare matching parameter names.
    """

    __slots__ = ("asset_name", "_meta", "_resolved_cfg")

    def __init__(self, asset_name: str):
        self.asset_name: str = asset_name
        self._meta: dict[str, Any] = {}
        self._resolved_cfg: Any = None

    # ── accumulation ──────────────────────────────────────────────

    def update(self, **kwargs: Any) -> None:
        """Merge keyword metadata into this robot's store, invalidating cached cfg."""
        for k, v in kwargs.items():
            if v is not None:
                self._meta[k] = v
        self._resolved_cfg = None

    # ── read access ───────────────────────────────────────────────

    @property
    def ee_body(self) -> str | None:
        return self._meta.get("ee_body")

    @property
    def command_name(self) -> str | None:
        return self._meta.get("command_name")

    @property
    def joint_patterns(self) -> list[str]:
        return self._meta.get("joint_patterns", [])

    @property
    def meta(self) -> dict[str, Any]:
        """Read-only view of all stored metadata."""
        return self._meta

    def resolved_cfg(self, scene: Any) -> Any:
        """Return a cached, resolved :class:`SceneEntityCfg` for this robot.

        The cfg is built from ``asset_name``, ``ee_body``, and ``joint_patterns`` and resolved against *scene*.
        Subsequent calls with the same scene return the cached instance.
        """
        if self._resolved_cfg is not None:
            return self._resolved_cfg
        from isaaclab.managers import SceneEntityCfg

        body_names = [self.ee_body] if self.ee_body else []
        cfg = SceneEntityCfg(name=self.asset_name, body_names=body_names, joint_names=self.joint_patterns)
        cfg.resolve(scene)
        self._resolved_cfg = cfg
        return self._resolved_cfg

    def __repr__(self) -> str:
        return f"RobotInfo({self.asset_name!r}, {self._meta})"


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
        # group name → env-id tuple
        self._env_ids: dict[str, tuple[int, ...]] = {}
        # group name → cached long tensor of env IDs (lazily populated)
        self._env_id_tensors: dict[str | None, torch.Tensor] = {}
        # group name → cached global-to-local lookup table (lazily populated)
        self._lookups: dict[str, torch.Tensor] = {}
        # task-group partition (populated by apply_task_groups)
        self._task_group_partition: dict[str, list[int]] | None = None
        # entity → group key mappings (centralized registry)
        self._asset_groups: dict[str, str] = {}
        self._term_groups: dict[str, str] = {}
        # cached default ids for unregistered / None keys
        self._all_env_ids: tuple[int, ...] = tuple(range(num_envs))
        # per-task robot metadata (populated by action/command managers)
        self._robots: dict[str, RobotInfo] = {}

    # ── properties ────────────────────────────────────────────────────────

    @property
    def num_envs(self) -> int:
        """Total number of environments across all groups."""
        return self._num_envs

    @property
    def is_heterogeneous(self) -> bool:
        """Whether any partial group has been registered."""
        return len(self._env_ids) > 0

    @property
    def group_names(self) -> list[str]:
        """Names of all registered groups."""
        return list(self._env_ids.keys())

    def apply_task_groups(self, task_groups: dict[str, int] | int) -> None:
        """Partition environments by task groups and register each group.

        Calls :func:`partition_env_ids` internally and registers every
        resulting group.

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
        """Resolve a group key from configuration parameters for 'command' terms.

        Priority: *task_group* (if it names a registered group) >
        *asset_name* (via :meth:`group_for_asset`).

        Args:
            task_group: Explicit task-group name from the term config.
            asset_name: Asset name whose group to inherit.

        Returns:
            The resolved group key, or ``None`` when homogeneous.
        """
        if task_group is not None and task_group in self._env_ids:
            return task_group
        if asset_name is not None:
            return self._asset_groups.get(asset_name)
        return None

    # ── robot metadata registry ───────────────────────────────────────

    def register_robot_meta(self, asset_name: str, **kwargs: Any) -> None:
        """Register per-robot metadata for a grouped asset.

        Called by :class:`ActionManager` and :class:`CommandManager` during ``_prepare_terms``.  Fields are merged
        by *asset_name* so that action and command managers can each contribute their part independently.

        Any keyword arguments are accepted; use well-known keys such as ``ee_body``, ``joint_patterns``,
        ``num_joints``, and ``command_name`` for auto-injection by the per-robot dispatch mechanism.

        Args:
            asset_name: Scene-level name of the asset.
            **kwargs: Metadata key-value pairs to merge into this robot's :class:`RobotInfo`.
        """
        if asset_name not in self._robots:
            self._robots[asset_name] = RobotInfo(asset_name)
        self._robots[asset_name].update(**kwargs)

    @property
    def robot_infos(self) -> list[RobotInfo]:
        """Registered :class:`RobotInfo` instances for grouped assets.

        Returns:
            :class:`RobotInfo` instances ordered by registration time.
        """
        return list(self._robots.values())

    # ── scatter helpers ────────────────────────────────────────────────────

    def multi_task_onehot(self) -> torch.Tensor:
        """One-hot encoding of task group for every environment.

        Column *i* is 1.0 for all envs assigned to the *i*-th task group.

        Returns:
            Shape ``(num_envs, num_task_groups)``.
        """
        n_tasks = len(self._env_ids)
        out = torch.zeros(self._num_envs, n_tasks, device=self._device)
        for i, group in enumerate(self._env_ids):
            env_ids = self.env_ids(group)
            out[env_ids, i] = 1.0
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
        self._env_ids[key] = ids

    # ── simple queries ────────────────────────────────────────────────────

    def num_envs_for(self, key: str | None) -> int:
        """Number of environments in a group (all envs if *key* is ``None`` or unregistered)."""
        return len(self._env_ids[key]) if key in self._env_ids else self._num_envs

    def env_ids(self, key: str | None) -> tuple[int, ...]:
        """Global env indices for a group (all envs if *key* is ``None`` or unregistered)."""
        if key is None:
            return self._all_env_ids
        return self._env_ids.get(key, self._all_env_ids)

    def env_ids_t(self, key: str | None) -> torch.Tensor:
        """Like :meth:`env_ids` but returns a cached ``torch.long`` tensor.

        Suitable for direct tensor row-indexing (avoids the tuple → multi-dim
        indexing pitfall).
        """
        if key not in self._env_id_tensors:
            self._env_id_tensors[key] = torch.tensor(self.env_ids(key), device=self._device, dtype=torch.long)
        return self._env_id_tensors[key]

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
        if key not in self._env_ids:
            return slice(None)
        ids = self._env_ids[key]
        if len(ids) > 0 and ids == tuple(range(ids[0], ids[0] + len(ids))):
            return slice(ids[0], ids[0] + len(ids))
        return self.env_ids_t(key)

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
        if key not in self._env_ids:
            return global_ids
        lut = self._get_lookup(key)
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
        if key not in self._env_ids:
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
        if key not in self._env_ids:
            return global_ids, global_ids
        lut = self._get_lookup(key)
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
        if key not in self._env_ids:
            return local_data
        shape = (self._num_envs, *local_data.shape[1:])
        out = local_data.new_full(shape, fill)
        out[self.env_slice(key)] = local_data
        return out

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
        if term_key not in self._env_ids:
            return slice(None)
        if asset_key not in self._env_ids:
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
        if key is None or key not in self._env_ids:
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

    # ── internals ─────────────────────────────────────────────────────────

    def _get_lookup(self, key: str) -> torch.Tensor:
        """Return a cached global-to-local lookup table for a registered group."""
        if key not in self._lookups:
            ids = self._env_ids[key]
            t = torch.tensor(ids, device=self._device, dtype=torch.long)
            lut = torch.full((int(t.max().item()) + 1,), -1, dtype=torch.long, device=self._device)
            lut[t] = torch.arange(len(t), device=self._device)
            self._lookups[key] = lut
        return self._lookups[key]

    def __repr__(self) -> str:
        groups_info = ", ".join(f"{k}({len(v)} envs)" for k, v in self._env_ids.items())
        return f"EnvLayout(num_envs={self._num_envs}, groups=[{groups_info or 'homogeneous'}])"


# ── utility functions ─────────────────────────────────────────────────────


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
