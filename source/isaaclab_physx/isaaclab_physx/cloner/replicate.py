# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from omni.physx import get_physx_replicator_interface
from pxr import Sdf, Usd, UsdUtils

from isaaclab import cloner
from isaaclab.sim import SimulationContext

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

    def replicate(self, plan: ClonePlan, asset_prototype_ids: tuple[int, ...]) -> None:
        """Register the PhysX replicator for this context's source declarations.

        Args:
            plan: Replication layout shared by every clone backend.
            asset_prototype_ids: Asset definitions routed to PhysX.
        """
        usd = SimulationContext.instance().clone_contexts[cloner.UsdReplicateContext]
        self._replicate_instances(
            instances=tuple(instance for instance in usd.instances if instance[0] in asset_prototype_ids),
            env_ids=np.arange(len(plan.topology.world_prototype_layout)),
            has_usd_only_sources=any(
                index not in asset_prototype_ids for index, _, _, world_ids in usd.instances if len(world_ids)
            ),
            exclude_self_replication=True,
        )

    def _replicate_instances(
        self,
        instances: Sequence[tuple[int, str | None, str, np.ndarray]],
        env_ids: np.ndarray,
        has_usd_only_sources: bool,
        exclude_self_replication: bool,
    ) -> None:
        """Register the selected native instance groups with PhysX."""
        physx_queue: list[tuple[str, str, tuple[int, ...]]] = []
        if len(env_ids) <= 1:
            return

        native_paths: list[str] = []

        for _, src, destination, world_ids in instances:
            if not len(world_ids) or world_ids[0] == -1:
                continue
            worlds = tuple(map(int, env_ids[world_ids]))
            if has_usd_only_sources:
                native_paths.append(src)
                native_paths.extend(destination.format(world) for world in worlds)
            if exclude_self_replication:
                matched = cloner.path.match(src, destination)
                if matched is not None and matched.instance.isdigit():
                    filtered = tuple(world for world in worlds if world != int(matched.instance))
                    worlds = filtered if filtered else worlds
            physx_queue.append((src, destination, worlds))

        # Fully-heterogeneous 1:1 layouts have every source mapped only to its own
        # environment (no cross-env replication needed). Calling rep.replicate() once
        # per source with a single self-target is known to trigger intermittent native
        # heap corruption (double-free / SIGABRT) under mGPU, likely due to per-call
        # PhysX-internal allocations summing to a problematic total across processes.
        # For these layouts the source prims are already in their correct env positions
        # and PhysX can parse them from the stage without any replicator registration.
        if all(len(envs) == 1 and src == destination.format(envs[0]) for src, destination, envs in physx_queue):
            return

        physics_scene_prim = self.stage.GetPrimAtPath("/physicsScene")
        if physics_scene_prim.IsValid():
            physics_scene_prim.CreateAttribute("physxScene:envIdInBoundsBitCount", Sdf.ValueTypeNames.Int).Set(4)

        current_worlds: list[int] = []
        current_template: str = ""
        prefixes = [cloner.path.split(destination)[0] for _, destination, _ in physx_queue]
        env_namespaces = [
            prefix.rstrip("/") if prefix.endswith("/") else prefix.rsplit("/", 1)[0] for prefix in prefixes
        ]
        excluded_paths = (
            list(dict.fromkeys(native_paths))
            if has_usd_only_sources
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
    env_ids: np.ndarray,
    mapping: np.ndarray,
    positions: np.ndarray | None = None,
    quaternions: np.ndarray | None = None,
    exclude_self_replication: bool = True,
) -> None:
    """Replicate a raw source-to-environment mapping through PhysX.

    Args:
        stage: USD stage containing the source prims.
        sources: Source prim paths, one per mapping row.
        destinations: Destination templates containing ``"{}"``, one per mapping row.
        env_ids: Integer environment identifiers, shape ``[num_envs]``.
        mapping: Boolean source-to-environment selection, shape ``[len(sources), num_envs]``.
        positions: Optional environment positions [m], shape ``[num_envs, 3]``. Unused by PhysX.
        quaternions: Optional environment orientations in xyzw order, shape ``[num_envs, 4]``. Unused by PhysX.
        exclude_self_replication: Whether to omit a source environment from its own targets.
    """
    del positions, quaternions
    expected_shape = (len(sources), len(env_ids))
    if mapping.shape != expected_shape:
        raise ValueError(f"mapping must have shape {expected_shape}, got {mapping.shape}.")
    context = PhysxReplicateContext(stage)
    context._replicate_instances(
        instances=tuple(
            (index, source, destination, np.flatnonzero(mapping[index]))
            for index, (source, destination) in enumerate(zip(sources, destinations, strict=True))
        ),
        env_ids=env_ids,
        has_usd_only_sources=False,
        exclude_self_replication=exclude_self_replication,
    )
