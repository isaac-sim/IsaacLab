# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from omni.physx import get_physx_replicator_interface
from pxr import Sdf, Usd, UsdUtils

from isaaclab import cloner
from isaaclab.cloner.query import _clone_mapping

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
        self._replicator = None
        self._registered = False

    def replicate(self, plan: ClonePlan) -> None:
        """Register the PhysX replicator for this context's plan rows.

        Args:
            plan: Replication layout shared by every clone backend.
        """
        rows = plan.context_rows[type(self)]
        sources, destinations, mapping = _clone_mapping(plan, rows, whole_env=plan.positions.is_cuda)
        use_env_ids = (
            plan.positions.is_cuda
            and sources == (plan.env_template.format(int(plan.env_ids[0])),)
            and destinations == (plan.env_template,)
        )
        physx_queue: list[tuple[str, str, tuple[int, ...]]] = []

        if mapping.size(1) <= 1:
            return

        native_rows = set(rows)
        has_usd_only_rows = any(
            "{}" in destination and row not in native_rows and bool(plan.clone_mask[row].any())
            for row, destination in enumerate(plan.destinations)
        )
        native_paths: list[str] = []

        for i, src in enumerate(sources):
            worlds = plan.env_ids[mapping[i].to(dtype=torch.bool)].tolist()
            if has_usd_only_rows:
                native_paths.append(src)
                native_paths.extend(destinations[i].format(int(world)) for world in worlds)
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
        env_namespace = plan.env_template.rsplit("/", 1)[0] or "/"
        excluded_paths = list(dict.fromkeys(native_paths)) if has_usd_only_rows else ["/World/template", env_namespace]

        def attach_fn(_stage_id: int):
            return excluded_paths

        def rename_fn(_replicate_path: str, i: int):
            return current_template.format(current_worlds[i])

        def attach_end_fn(_stage_id: int):
            nonlocal current_template
            for src, destination, target_envs in physx_queue:
                current_template = destination
                current_worlds[:] = target_envs
                if not current_worlds:
                    continue
                self._replicator.replicate(
                    _stage_id,
                    src,
                    len(current_worlds),
                    useEnvIds=use_env_ids,
                    useFabricForReplication=False,
                )

        self._replicator = get_physx_replicator_interface()
        self._replicator.register_replicator(self._stage_id, attach_fn, attach_end_fn, rename_fn)
        self._registered = True

    def clear(self) -> None:
        """Unregister this stage's native PhysX replicator."""
        if self._registered:
            self._replicator.unregister_replicator(self._stage_id)
            self._registered = False
            self._replicator = None
