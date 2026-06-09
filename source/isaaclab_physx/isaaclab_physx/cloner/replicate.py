# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from omni.physx import get_physx_replicator_interface
from pxr import Sdf, Usd, UsdUtils

from isaaclab.cloner.replicate_session import REPLICATION_QUEUE


def _select_env_ids(env_ids: torch.Tensor, mapping: torch.Tensor, row: int) -> torch.Tensor:
    """Return the environment ids selected by a replication row."""
    row_mask = mapping[row]
    if row_mask.dtype != torch.bool:
        row_mask = row_mask.to(dtype=torch.bool)
    return env_ids[row_mask]


class PhysxReplicateContext:
    """Queue and run PhysX replication work for one stage."""

    def __init__(self, stage: Usd.Stage):
        """Initialize the context.

        Args:
            stage: USD stage to register with the PhysX replicator.
        """
        self.stage = stage
        self._stage_id = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
        physics_scene_prim = self.stage.GetPrimAtPath("/physicsScene")
        if physics_scene_prim.IsValid():
            physics_scene_prim.CreateAttribute("physxScene:envIdInBoundsBitCount", Sdf.ValueTypeNames.Int).Set(4)
        self._queue: list[tuple[str, str, tuple[int, ...]]] = []

    def queue(self, source: str, destination: str, target_envs: Sequence[int]) -> None:
        """Queue one PhysX source row for replication.

        Args:
            source: Source prim path.
            destination: Destination path template with ``"{}"`` for env id.
            target_envs: Environment ids selected for this source row.
        """
        self._queue.append((source, destination, tuple(int(env_id) for env_id in target_envs)))

    def queue_mapping(
        self,
        sources: Sequence[str],
        destinations: Sequence[str],
        env_ids: torch.Tensor,
        mapping: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        quaternions: torch.Tensor | None = None,
        exclude_self_replication: bool = True,
    ) -> None:
        """Queue replication rows from the current flat clone mapping.

        Args:
            sources: Source prim paths.
            destinations: Destination path templates with ``"{}"`` for env id.
            env_ids: Environment indices.
            mapping: Bool/int mask selecting envs per source.
            positions: Optional per-environment world positions [m], unused by PhysX.
            quaternions: Optional per-environment orientations, unused by PhysX.
            exclude_self_replication: Whether to skip replicating a source prim onto itself
                when it also maps to other environments.
        """
        del positions, quaternions

        if mapping.size(1) <= 1:
            return

        for i, src in enumerate(sources):
            worlds = _select_env_ids(env_ids, mapping, i).tolist()
            if exclude_self_replication:
                pre, _, suf = destinations[i].partition("{}")
                self_id = src.removeprefix(pre).removesuffix(suf)
                if self_id.isdigit():
                    filtered = [w for w in worlds if w != int(self_id)]
                    worlds = filtered if filtered else worlds
            self.queue(src, destinations[i], worlds)

    def replicate(self) -> None:
        """Register the PhysX replicator and run queued rows from ``attach_end_fn``."""
        if not self._queue:
            return

        physx_queue = tuple(self._queue)
        current_worlds: list[int] = []
        current_template: str = ""

        def attach_fn(_stage_id: int):
            return ["/World/template", "/World/envs"]

        def rename_fn(_replicate_path: str, i: int):
            return current_template.format(current_worlds[i])

        def attach_end_fn(_stage_id: int):
            nonlocal current_template
            rep = get_physx_replicator_interface()
            for src, destination, target_envs in physx_queue:
                current_template = destination
                current_worlds[:] = target_envs
                if not current_worlds:
                    continue
                rep.replicate(
                    _stage_id,
                    src,
                    len(current_worlds),
                    # TODO: envIds needs to support heterogeneous setup. for now, we rely on USD collision filtering
                    useEnvIds=False,
                    useFabricForReplication=False,
                )
            rep.unregister_replicator(_stage_id)

        get_physx_replicator_interface().register_replicator(self._stage_id, attach_fn, attach_end_fn, rename_fn)
        self._queue.clear()


def queue_physx_replication(cfg: Any) -> None:
    """Register ``cfg`` for PhysX replication when :func:`~isaaclab.cloner.replicate` next runs.

    Appends ``(cfg, PhysxReplicateContext)`` to
    :data:`~isaaclab.cloner.REPLICATION_QUEUE`. The actual row resolution and dispatch
    happen inside :func:`~isaaclab.cloner.replicate`, so this helper is safe to call from
    any asset constructor — no active session is required.
    """
    REPLICATION_QUEUE.append((cfg, PhysxReplicateContext))


def physx_replicate(
    stage: Usd.Stage,
    sources: Sequence[str],  # e.g. ["/World/Template/A", "/World/Template/B"]
    destinations: Sequence[str],  # e.g. ["/World/envs/env_{}/Robot", "/World/envs/env_{}/Object"]
    env_ids: torch.Tensor,  # env_ids
    mapping: torch.Tensor,  # (num_sources, num_envs) bool; True -> place sources[i] into world=j
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    device: str = "cpu",
    exclude_self_replication: bool = True,
) -> None:
    """Replicate prims via PhysX replicator with per-row mapping.

    Builds per-source destination lists from ``mapping`` and calls PhysX ``replicate``.
    The replicator is registered for the call and then unregistered. Heterogeneous
    rows currently use ``useEnvIds=False`` and rely on USD collision filtering.

    ``attach_fn`` excludes ``/World/template`` and ``/World/envs`` so that PhysX does
    not independently parse prims that the replicator will handle.  The source prim
    receives its physics body as a side-effect of ``rep.replicate()`` (which always
    parses the source internally), so every source must appear in at least one
    ``replicate`` call.

    When ``exclude_self_replication`` is True (default), each source environment is
    removed from its own replication targets so the replicator only creates bodies at
    non-self destinations.  If removing self would leave the world list empty (i.e. the
    source maps only to its own environment), self is kept so that ``rep.replicate()``
    is still called and the source prim gets its physics body.

    Args:
        stage: USD stage.
        sources: Source prim paths (``S``).
        destinations: Destination templates (``S``) with ``"{}"`` for env index.
        env_ids: Environment indices (``[E]``).
        mapping: Bool/int mask (``[S, E]``) selecting envs per source.
        positions: Optional positions (unused, for API compatibility).
        quaternions: Optional orientations (unused, for API compatibility).
        device: Unused legacy argument retained for API compatibility.
        exclude_self_replication: If True, skip replicating a source prim onto itself
            when the source also maps to other environments.  Default is True.
            Self-only sources always keep self so that ``rep.replicate()`` fires.

    Returns:
        None
    """
    del device

    ctx = PhysxReplicateContext(stage)
    ctx.queue_mapping(
        sources,
        destinations,
        env_ids,
        mapping,
        positions=positions,
        quaternions=quaternions,
        exclude_self_replication=exclude_self_replication,
    )
    ctx.replicate()
