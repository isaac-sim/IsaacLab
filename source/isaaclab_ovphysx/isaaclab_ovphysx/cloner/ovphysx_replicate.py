# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OvPhysX replication hook for IsaacLab's cloning pipeline.

Called by :func:`isaaclab.cloner.clone_from_template` in place of the PhysX
or Newton replicators.  Unlike those replicators, ovphysx.PhysX does not exist
yet at this point in the scene setup — it is created lazily on the first
:meth:`~isaaclab_ovphysx.physics.OvPhysxManager.reset` call.

This function records a *pending clone* on :class:`OvPhysxManager`.  When
:meth:`~isaaclab_ovphysx.physics.OvPhysxManager._warmup_and_load` eventually
creates the ``PhysX`` instance and loads the USD stage (which contains only
``env_0`` physics — env_1..N are empty Xform containers), it replays every
pending clone via ``physx.clone(source, targets)`` to create the remaining
environments entirely inside the physics runtime without touching USD.
"""

from __future__ import annotations

import torch
from pxr import Usd


def ovphysx_replicate(
    stage: Usd.Stage,
    sources: list[str],
    destinations: list[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    device: str = "cpu",
) -> None:
    """Record a physics clone for later execution by OvPhysxManager.

    Translates the generic IsaacLab source/destination/mapping representation
    into ``(source_path, [target_paths])`` pairs and registers them on
    :class:`~isaaclab_ovphysx.physics.OvPhysxManager`.  The actual
    ``physx.clone()`` calls happen in ``_warmup_and_load()`` after the USD
    stage has been loaded.

    Positions and orientations are intentionally ignored: after
    ``physx.clone()`` all environments share ``env_0``'s world position.
    The correct grid-spaced positions are written during the first
    :meth:`~isaaclab_ovphysx.assets.articulation.Articulation.reset` call via
    the root-pose tensor binding, exactly as in the PhysX backend.

    Args:
        stage: USD stage (not modified by this function).
        sources: Source prim paths (one per prototype).
        destinations: Destination path templates with ``"{}"`` for env index.
        env_ids: Environment indices tensor.
        mapping: ``(num_sources, num_envs)`` bool tensor; True selects which
            environments receive each source.
        positions: Ignored — positions are set at first reset.
        quaternions: Ignored — orientations are set at first reset.
        device: Torch device (unused; kept for API compatibility).
    """
    # Deferred import to avoid circular dependency at module load time.
    from isaaclab_ovphysx.physics.ovphysx_manager import OvPhysxManager

    for i, src in enumerate(sources):
        active_env_ids = env_ids[mapping[i]].tolist()

        # Exclude the source environment from its own target list.
        # physx.clone() is only needed for *other* envs; the source env_0 is
        # already loaded from USD.  We detect self by matching the source path
        # against the destination template.
        pre, _, suf = destinations[i].partition("{}")
        self_env_id: int | None = None
        candidate = src.removeprefix(pre).removesuffix(suf)
        if candidate.isdigit():
            self_env_id = int(candidate)

        targets = [
            destinations[i].format(e)
            for e in active_env_ids
            if e != self_env_id
        ]

        if targets:
            OvPhysxManager.register_clone(src, targets)
