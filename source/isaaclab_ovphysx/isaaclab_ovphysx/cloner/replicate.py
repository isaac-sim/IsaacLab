# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OvPhysX replication hook for IsaacLab's cloning pipeline.

Called from the scene cloning path in place of immediate PhysX or Newton
replication.  Unlike those replicators, ovphysx.PhysX does not exist yet at
this point in the scene setup — it is created lazily on the first
:meth:`~isaaclab_ovphysx.physics.OvPhysxManager.reset` call.

This function records an active clone recipe on :class:`OvPhysxManager`.  When
:meth:`~isaaclab_ovphysx.physics.OvPhysxManager._warmup_and_load` eventually
creates the ``PhysX`` instance, env-0-only loads replay each recipe via
``physx.clone(source, targets)`` after loading. Full-stage loads instead
materialize or overlay every recipe in serialized USDA before attaching OVStage.
Recipes remain active for the current simulation context so a forced re-warmup
rebuilds the same topology without modifying the live USD stage.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from pxr import Sdf, Usd

from isaaclab import cloner


def _select_env_ids(env_ids: torch.Tensor, mapping: torch.Tensor, row: int) -> torch.Tensor:
    """Return the environment ids selected by a replication row."""
    row_mask = mapping[row]
    if row_mask.dtype != torch.bool:
        row_mask = row_mask.to(dtype=torch.bool)
    return env_ids[row_mask]


class OvPhysxReplicateContext:
    """Queue and run OvPhysX clone operations for one stage."""

    def __init__(self, stage: Usd.Stage):
        """Initialize the context.

        Args:
            stage: USD stage associated with the pending clone operations.
        """
        self.stage = stage
        physics_scene_prim = self.stage.GetPrimAtPath("/physicsScene")
        if physics_scene_prim.IsValid():
            physics_scene_prim.CreateAttribute("physxScene:envIdInBoundsBitCount", Sdf.ValueTypeNames.Int).Set(4)
        self._queue: list[tuple[str, list[str], list[tuple[float, float, float]]]] = []

    def queue(
        self, source: str, targets: Sequence[str], parent_positions: Sequence[tuple[float, float, float]]
    ) -> None:
        """Queue one pending OvPhysX clone operation.

        Args:
            source: Source prim path.
            targets: Destination prim paths.
            parent_positions: Parent Xform positions [m] for each destination.
        """
        self._queue.append((source, list(targets), list(parent_positions)))

    def queue_mapping(
        self,
        sources: Sequence[str],
        destinations: Sequence[str],
        env_ids: torch.Tensor,
        mapping: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        quaternions: torch.Tensor | None = None,
    ) -> None:
        """Queue clone operations from the current flat clone mapping.

        Args:
            sources: Source prim paths.
            destinations: Destination path templates with ``"{}"`` for env id.
            env_ids: Environment indices.
            mapping: Bool/int mask selecting envs per source.
            positions: Optional per-environment world positions [m].
            quaternions: Optional per-environment orientations, unused by OvPhysX.
        """
        del quaternions

        for i, src in enumerate(sources):
            active_env_ids = _select_env_ids(env_ids, mapping, i).tolist()

            self_env_id: int | None = None
            matched = cloner.path.match(src, destinations[i])
            if matched is not None and matched.instance.isdigit():
                self_env_id = int(matched.instance)

            targets: list[str] = []
            parent_positions: list[tuple[float, float, float]] = []
            for env_id in active_env_ids:
                env_id = int(env_id)
                if env_id == self_env_id:
                    continue
                targets.append(destinations[i].format(env_id))
                if positions is not None and env_id < len(positions):
                    pos = positions[env_id]
                    parent_positions.append((float(pos[0]), float(pos[1]), float(pos[2])))
                else:
                    parent_positions.append((0.0, 0.0, 0.0))

            if targets:
                self.queue(src, targets, parent_positions)

    def replicate(self) -> None:
        """Publish all queued clones to :class:`OvPhysxManager`."""
        from isaaclab_ovphysx.physics.ovphysx_manager import OvPhysxManager

        for source, targets, parent_positions in self._queue:
            OvPhysxManager.register_clone(source, targets, parent_positions)
        self._queue.clear()


PHYSICS_CONTEXT = OvPhysxReplicateContext
"""Physics replication context for OvPhysX assets.  OvPhysxReplicateContext authors USD
internally, so USD replication is not separately added.
TODO: decompose into UsdReplicateContext + a pure-physics OvPhysxReplicateContext to match
the physx/newton split."""


def ovphysx_replicate(
    stage: Usd.Stage,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    device: str = "cpu",
) -> None:
    """Record a physics clone for later execution by OvPhysxManager.

    Translates the generic IsaacLab source/destination/mapping representation
    into ``(source_path, [target_paths])`` pairs and registers them on
    :class:`~isaaclab_ovphysx.physics.OvPhysxManager`. During
    ``_warmup_and_load()``, env-0-only loads replay recipes with
    ``physx.clone()`` after OVStage is attached. Full-stage loads materialize
    or overlay recipes in serialized USDA before attaching OVStage instead.

    The ``positions`` parameter contains the 2-D grid world positions for all
    environments.  They are forwarded to the C++ clone plugin so that the
    parent Xform prim for each clone (e.g. ``/World/envs/env_N``) is placed at
    the correct grid location in Fabric. The in-memory OVStage only contains
    ``env_0``; without explicit positions all clone parents would be created at
    the origin, causing all articulations to pile up and the GPU solver to
    diverge on the first warmup step.

    Args:
        stage: USD stage (not modified by this function).
        sources: Source prim paths (one per prototype).
        destinations: Destination path templates with ``"{}"`` for env index.
        env_ids: Environment indices tensor.
        mapping: ``(num_sources, num_envs)`` bool tensor; True selects which
            environments receive each source.
        positions: World (x, y, z) positions [m] for every environment, shape
            ``[num_envs, 3]``. Used to place clone parent Xform prims in
            Fabric at correct grid locations.
        quaternions: Ignored — orientations are set at first reset.
        device: Torch device (unused; kept for API compatibility).
    """
    del device

    ctx = OvPhysxReplicateContext(stage)
    ctx.queue_mapping(
        sources,
        destinations,
        env_ids,
        mapping,
        positions=positions,
        quaternions=quaternions,
    )
    ctx.replicate()
