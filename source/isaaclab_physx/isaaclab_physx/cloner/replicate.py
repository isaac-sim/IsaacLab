# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from omni.physx import get_physx_replicator_interface
from pxr import Sdf, Usd, UsdUtils

from isaaclab import cloner
from isaaclab.assets import DeformableObjectCfg

if TYPE_CHECKING:
    from isaaclab.cloner import ClonePlan


class PhysxReplicateContext(cloner.ReplicateContext):
    """Apply one clone plan through the PhysX replicator."""

    def replicate(self, plan: ClonePlan, asset_prototype_ids: tuple[int, ...]) -> None:
        """Register the PhysX replicator for this context's source declarations.

        Args:
            plan: Replication layout shared by every clone backend.
            asset_prototype_ids: Asset definitions routed to PhysX.
        """
        # PhysX's native replicator does not attach surface or volume deformables, even
        # when those assets are copied through USD. All spawned assets already have USD
        # copies, so let PhysX discover the complete scene without a replicator hook.
        if any(isinstance(cfg, DeformableObjectCfg) for cfg in plan.asset_cfgs):
            return
        sources = cloner.path.get_asset_prototype_paths(plan)
        templates, starts, world_ids, world_starts = cloner.path.get_world_prototype_asset_templates(
            plan, include_world_indices=True
        )
        copies = {}
        for group in np.flatnonzero(np.diff(world_starts[1:])) + 1:
            targets = world_ids[world_starts[group] : world_starts[group + 1]]
            for index in range(*starts[group : group + 2]):
                if (asset := plan.topology.world_prototypes[index]) in asset_prototype_ids:
                    copies.setdefault((sources[asset], templates[index]), []).append(targets)
        copies = ((key, np.concatenate(groups)) for key, groups in copies.items())
        env_ids = np.arange(len(plan.topology.world_prototype_layout))
        usd_only = any(i not in asset_prototype_ids for i, source in enumerate(sources) if source is not None)
        _replicate_instances(self._sim.stage, copies, env_ids, usd_only, exclude_self_replication=True)


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
    pairs = zip(sources, destinations, strict=True)
    copies = ((pair, np.flatnonzero(mapping[index])) for index, pair in enumerate(pairs))
    options = dict(has_usd_only_sources=False, exclude_self_replication=exclude_self_replication)
    _replicate_instances(stage, copies, env_ids, **options)


_EXCLUDED_PATHS: list[str] = []
# PhysX 110.4 caches the first attach callback's list; preserve its identity across registrations.
# Assumes Isaac Lab attaches first and each registration attaches before the next one.
# Multiple replicators are unsupported: https://github.com/NVIDIA-Omniverse/PhysX/issues/590#issuecomment-6436165383


def _replicate_instances(
    stage: Usd.Stage,
    copies: Iterable[tuple[tuple[str, str], np.ndarray]],
    env_ids: np.ndarray,
    has_usd_only_sources: bool,
    exclude_self_replication: bool,
) -> None:
    """Register the selected native instance groups with PhysX."""
    cache = UsdUtils.StageCache.Get()
    cached_id = cache.GetId(stage)
    stage_id = cached_id.ToLongInt() if cached_id.IsValid() else cache.Insert(stage).ToLongInt()
    physx_queue: list[tuple[str, str, tuple[int, ...]]] = []
    if len(env_ids) <= 1:
        return

    for (src, destination), world_ids in copies:
        if not len(world_ids) or world_ids[0] == -1:
            continue
        worlds = tuple(map(int, env_ids[world_ids]))
        if len(worlds) == 1 and src == destination.format(worlds[0]):
            has_usd_only_sources = True
            continue
        if exclude_self_replication:
            matched = cloner.path.match(src, destination)
            if matched is not None and matched.instance.isdigit():
                worlds = tuple(world for world in worlds if world != int(matched.instance)) or worlds
        physx_queue.append((src, destination, worlds))

    if not physx_queue:
        return

    physics_scene_prim = stage.GetPrimAtPath("/physicsScene")
    if physics_scene_prim.IsValid():
        physics_scene_prim.CreateAttribute("physxScene:envIdInBoundsBitCount", Sdf.ValueTypeNames.Int).Set(4)

    current_worlds: tuple[int, ...] = ()
    current_template: str = ""
    excluded_paths = []
    for src, destination, worlds in physx_queue:
        excluded_paths.append(src)
        if has_usd_only_sources:
            excluded_paths.extend(destination.format(world) for world in worlds)
        else:
            prefix = destination.partition("{}")[0]
            excluded_paths.append(prefix.rstrip("/") if prefix.endswith("/") else prefix.rsplit("/", 1)[0])
    _EXCLUDED_PATHS[:] = dict.fromkeys(excluded_paths)

    def attach_fn(_stage_id: int):
        return _EXCLUDED_PATHS

    def rename_fn(_replicate_path: str, i: int):
        return current_template.format(current_worlds[i])

    def attach_end_fn(_stage_id: int):
        nonlocal current_template, current_worlds
        replicator = get_physx_replicator_interface()
        for src, current_template, current_worlds in physx_queue:
            replicator.replicate(_stage_id, src, len(current_worlds), useEnvIds=False, useFabricForReplication=False)
        replicator.unregister_replicator(_stage_id)

    get_physx_replicator_interface().register_replicator(stage_id, attach_fn, attach_end_fn, rename_fn)
