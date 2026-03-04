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
    exclude_self_replication: bool = True,
) -> None:
    """Replicate prims via PhysX replicator with per-row mapping.

    Builds per-source destination lists from ``mapping`` and calls PhysX ``replicate``.
    Rows covering all environments use ``useEnvIds=True``; partial rows use ``False``.
    The replicator is registered for the call and then unregistered.

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
        use_fabric: Use Fabric for replication.
        device: Torch device for determining replication mode.
        exclude_self_replication: If True, skip replicating a source prim onto itself
            when the source also maps to other environments.  Default is True.
            Self-only sources always keep self so that ``rep.replicate()`` fires.

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

    if num_envs > 1:
        # Pre-compute effective world lists after self-exclusion.
        # Self is only removed when the source also maps to other environments;
        # if it is the sole destination we must keep it so that rep.replicate()
        # is still called (the source gets its physics body from that call).
        effective_worlds: list[list[int]] = []
        for i, src in enumerate(sources):
            worlds = env_ids[mapping[i]].tolist()
            if exclude_self_replication:
                pre, _, suf = destinations[i].partition("{}")
                self_id = src.removeprefix(pre).removesuffix(suf)
                if self_id.isdigit():
                    filtered = [w for w in worlds if w != int(self_id)]
                    worlds = filtered if filtered else worlds
            effective_worlds.append(worlds)

        def attach_fn(_stage_id: int):
            return ["/World/template", "/World/envs"]

        def rename_fn(_replicate_path: str, i: int):
            return current_template.format(current_worlds[i])

        def attach_end_fn(_stage_id: int):
            nonlocal current_template
            rep = get_physx_replicator_interface()
            for i, src in enumerate(sources):
                current_template = destinations[i]
                current_worlds[:] = effective_worlds[i]
                if not current_worlds:
                    continue
                rep.replicate(
                    _stage_id,
                    src,
                    len(current_worlds),
                    # TODO: envIds needs to support heterogeneous setup. for now, we rely on USD collision filtering
                    useEnvIds=False,  # (len(current_worlds) == num_envs - 1) and device != "cpu",
                    useFabricForReplication=use_fabric,
                )
            # unregister only AFTER all replicate() calls completed
            rep.unregister_replicator(_stage_id)

        get_physx_replicator_interface().register_replicator(stage_id, attach_fn, attach_end_fn, rename_fn)
