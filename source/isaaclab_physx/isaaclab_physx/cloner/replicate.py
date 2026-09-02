# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from omni.physx import get_physx_replicator_interface
from pxr import Sdf, Usd, UsdUtils

from isaaclab import cloner

if TYPE_CHECKING:
    from isaaclab.cloner import ClonePlan


class PhysxReplicateContext:
    """Apply one clone plan through the PhysX replicator."""

    replicate_priority = 0

    def __init__(self, stage: Usd.Stage):
        """Initialize the context.

        Args:
            stage: USD stage to register with the PhysX replicator.
        """
        self.stage = stage
        cache = UsdUtils.StageCache.Get()
        cached_id = cache.GetId(stage)
        self._stage_id = cached_id.ToLongInt() if cached_id.IsValid() else cache.Insert(stage).ToLongInt()
        physics_scene_prim = self.stage.GetPrimAtPath("/physicsScene")
        if physics_scene_prim.IsValid():
            physics_scene_prim.CreateAttribute("physxScene:envIdInBoundsBitCount", Sdf.ValueTypeNames.Int).Set(4)

    def replicate(self, plan: ClonePlan) -> None:
        """Register the PhysX replicator for this context's plan rows.

        Args:
            plan: Replication layout shared by every clone backend.
        """
        if plan.env_ids is None:
            raise ValueError("ClonePlan.env_ids is required for replication.")
        rows = plan.context_rows[type(self)]
        native_rows = set(rows)
        other_rows = {
            row for context, routed in plan.context_rows.items() if context is not type(self) for row in routed
        }
        self._replicate_mapping(
            tuple(plan.sources[row] for row in rows),
            tuple(plan.destinations[row] for row in rows),
            plan.env_ids,
            plan.clone_mask[list(rows)],
            bool(other_rows - native_rows),
            True,
        )

    def _replicate_mapping(
        self,
        sources: Sequence[str],
        destinations: Sequence[str],
        env_ids: torch.Tensor,
        mapping: torch.Tensor,
        has_usd_only_rows: bool,
        exclude_self_replication: bool,
    ) -> None:
        """Register one raw source-to-environment mapping with PhysX."""
        physx_queue: list[tuple[str, str, tuple[int, ...]]] = []

        if mapping.size(1) <= 1:
            return

        native_paths: list[str] = []

        for i, src in enumerate(sources):
            worlds = env_ids[mapping[i].to(dtype=torch.bool)].tolist()
            if has_usd_only_rows:
                native_paths.append(src)
                native_paths.extend(destinations[i].format(int(world)) for world in worlds)
            if exclude_self_replication:
                matched = cloner.path.match(src, destinations[i])
                if matched is not None and matched.instance.isdigit():
                    filtered = [world for world in worlds if world != int(matched.instance)]
                    worlds = filtered if filtered else worlds
            physx_queue.append((src, destinations[i], tuple(map(int, worlds))))

        # Fully-heterogeneous 1:1 layouts have every source mapped only to its own
        # environment (no cross-env replication needed). Calling rep.replicate() once
        # per source with a single self-target is known to trigger intermittent native
        # heap corruption (double-free / SIGABRT) under mGPU, likely due to per-call
        # PhysX-internal allocations summing to a problematic total across processes.
        # For these layouts the source prims are already in their correct env positions
        # and PhysX can parse them from the stage without any replicator registration.
        if all(len(envs) == 1 and src == destination.format(envs[0]) for src, destination, envs in physx_queue):
            return

        current_worlds: list[int] = []
        current_template: str = ""
        prefixes = [cloner.path.split(destination)[0] for destination in destinations]
        env_namespaces = [
            prefix.rstrip("/") if prefix.endswith("/") else prefix.rsplit("/", 1)[0] for prefix in prefixes
        ]
        excluded_paths = (
            list(dict.fromkeys(native_paths))
            if has_usd_only_rows
            else list(dict.fromkeys(("/World/template", *env_namespaces)))
        )

        def attach_fn(_stage_id: int):
            return excluded_paths

        def rename_fn(_replicate_path: str, i: int):
            return current_template.format(current_worlds[i])

        def attach_end_fn(_stage_id: int):
            nonlocal current_template
            replicator = get_physx_replicator_interface()
            for src, destination, target_envs in physx_queue:
                current_template = destination
                current_worlds[:] = target_envs
                if not current_worlds:
                    continue
                replicator.replicate(
                    _stage_id,
                    src,
                    len(current_worlds),
                    useEnvIds=False,
                    useFabricForReplication=False,
                )
            replicator.unregister_replicator(_stage_id)

        get_physx_replicator_interface().register_replicator(self._stage_id, attach_fn, attach_end_fn, rename_fn)


def physx_replicate(
    stage: Usd.Stage,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    device: str = "cpu",
    exclude_self_replication: bool = True,
) -> None:
    """Replicate a raw source-to-environment mapping through PhysX."""
    del positions, quaternions, device
    context = PhysxReplicateContext(stage)
    context._replicate_mapping(sources, destinations, env_ids, mapping, False, exclude_self_replication)
