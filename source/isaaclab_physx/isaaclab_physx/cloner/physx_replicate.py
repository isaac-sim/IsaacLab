# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from omni.physx import get_physx_replicator_interface
from pxr import Usd, UsdUtils


def physx_replicate(
    stage: Usd.Stage,
    sources: list[str],  # e.g. ["/World/Template/A", "/World/Template/B"]
    destinations: list[str],  # e.g. ["/World/envs/env_{}/Robot", "/World/envs/env_{}/Object"]
    env_ids: torch.Tensor,  # env_ids
    mapping: torch.Tensor,  # (num_sources, num_envs) bool; True -> place sources[i] into world=j
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    use_fabric: bool = False,
    device: str = "cpu",
) -> None:
    """Replicate prims via PhysX replicator with per-row mapping.

    Builds per-source destination lists from ``mapping`` and calls PhysX ``replicate``.
    Rows covering all environments use ``useEnvIds=True``; partial rows use ``False``.
    The replicator is registered for the call and then unregistered.

    Args:
        stage: USD stage.
        sources: Source prim paths (``S``).
        destinations: Destination templates (``S``) with ``"{}"`` for env index.
        env_ids: Environment indices (``[E]``).
        mapping: Bool/int mask (``[S, E]``) selecting envs per source.
        positions: Optional positions (unused, for API compatibility).
        quaternions: Optional orientations (unused, for API compatibility).
        use_fabric: Use Fabric for replication.
        device: Torch device for determining replication mode.

    Returns:
        None
    """
    # Note: positions and quaternions are unused by PhysX replicator
    # They are included for API compatibility with other backends (e.g., Newton)
    del positions, quaternions

    stage_id = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
    current_worlds: list[int] = []
    current_template: str = ""
    num_envs = mapping.size(1)

    def attach_fn(_stage_id: int):
        return ["/World/envs", *sources]

    def rename_fn(_replicate_path: str, i: int):
        return current_template.format(current_worlds[i])

    def attach_end_fn(_stage_id: int):
        nonlocal current_template
        rep = get_physx_replicator_interface()
        for i, src in enumerate(sources):
            current_worlds[:] = env_ids[mapping[i]].tolist()
            current_template = destinations[i]
            rep.replicate(
                _stage_id,
                src,
                len(current_worlds),
                useEnvIds=(len(current_worlds) == num_envs) and device != "cpu",
                useFabricForReplication=use_fabric,
            )
        # unregister only AFTER all replicate() calls completed
        rep.unregister_replicator(_stage_id)

    get_physx_replicator_interface().register_replicator(stage_id, attach_fn, attach_end_fn, rename_fn)
