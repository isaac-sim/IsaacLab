# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OvPhysX replication hook for IsaacLab's cloning pipeline.

Called from the scene cloning path in place of immediate PhysX or Newton
replication.  Unlike those replicators, ovphysx.PhysX does not exist yet at
this point in the scene setup — it is created lazily on the first
:meth:`~isaaclab_ovphysx.physics.OvPhysxManager.reset` call.

This function records a *pending clone* on :class:`OvPhysxManager`.  When
:meth:`~isaaclab_ovphysx.physics.OvPhysxManager._warmup_and_load` eventually
creates the ``PhysX`` instance and loads the USD stage (which contains only
``env_0`` physics — env_1..N are empty Xform containers), it replays every
pending clone via ``physx.clone(source, targets, transforms)`` to create the remaining
environments entirely inside the physics runtime without touching USD.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from pxr import Gf, Sdf, Usd, UsdGeom

from isaaclab.cloner.cloner_utils import split_clone_template

_CloneTransform = tuple[float, float, float, float, float, float, float]


def _select_env_ids(env_ids: torch.Tensor, mapping: torch.Tensor, row: int) -> torch.Tensor:
    """Return the environment ids selected by a replication row."""
    row_mask = mapping[row]
    if row_mask.dtype != torch.bool:
        row_mask = row_mask.to(dtype=torch.bool)
    return env_ids[row_mask]


def _matrix_to_clone_transform(matrix: Gf.Matrix4d) -> _CloneTransform:
    """Convert a USD pose matrix to an OvPhysX xyzw clone transform."""
    matrix = matrix.RemoveScaleShear()
    position = matrix.ExtractTranslation()
    quaternion = matrix.ExtractRotationQuat()
    imaginary = quaternion.GetImaginary()
    return (
        float(position[0]),
        float(position[1]),
        float(position[2]),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
        float(quaternion.GetReal()),
    )


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
        self._queue: list[tuple[str, list[str], list[_CloneTransform]]] = []

    def queue(
        self, source: str, targets: Sequence[str], parent_positions: Sequence[tuple[float, float, float]]
    ) -> None:
        """Queue one pending OvPhysX clone operation.

        Args:
            source: Source prim path.
            targets: Destination prim paths.
            parent_positions: Legacy translation-only world positions [m] for
                whole-environment target roots. Each position becomes a final
                target-root pose with identity rotation.
        """
        target_transforms = [(x, y, z, 0.0, 0.0, 0.0, 1.0) for x, y, z in parent_positions]
        self._queue_transforms(source, targets, target_transforms)

    def _queue_transforms(
        self, source: str, targets: Sequence[str], target_transforms: Sequence[_CloneTransform]
    ) -> None:
        """Queue final target-root world poses for one OvPhysX clone operation."""
        self._queue.append((source, list(targets), list(target_transforms)))

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
            quaternions: Optional per-environment orientations in xyzw order.
        """
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        for i, src in enumerate(sources):
            active_env_ids = _select_env_ids(env_ids, mapping, i).tolist()
            if not active_env_ids:
                continue

            self_env_id: int | None = None
            pre, suf = split_clone_template(destinations[i])
            candidate = src.removeprefix(pre).removesuffix(suf)
            if candidate.isdigit():
                self_env_id = int(candidate)

            source_world = xform_cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(src)).RemoveScaleShear()
            if self_env_id is None:
                source_anchor_world = Gf.Matrix4d(1.0)
            else:
                source_anchor = self.stage.GetPrimAtPath(f"{pre}{self_env_id}")
                source_anchor_world = xform_cache.GetLocalToWorldTransform(source_anchor).RemoveScaleShear()
            source_relative = source_world * source_anchor_world.GetInverse()

            targets: list[str] = []
            target_transforms: list[_CloneTransform] = []
            for env_id in active_env_ids:
                env_id = int(env_id)
                if env_id == self_env_id:
                    continue
                targets.append(destinations[i].format(env_id))

                target_env_world = Gf.Matrix4d(1.0)
                if positions is not None and env_id < len(positions):
                    pos = positions[env_id]
                    target_env_world.SetTranslateOnly(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
                if quaternions is not None and env_id < len(quaternions):
                    quat = quaternions[env_id]
                    target_env_world.SetRotateOnly(
                        Gf.Quatd(
                            float(quat[3]),
                            Gf.Vec3d(float(quat[0]), float(quat[1]), float(quat[2])),
                        )
                    )
                target_transforms.append(_matrix_to_clone_transform(source_relative * target_env_world))

            if targets:
                self._queue_transforms(src, targets, target_transforms)

    def replicate(self) -> None:
        """Publish all queued clones to :class:`OvPhysxManager`."""
        from isaaclab_ovphysx.physics.ovphysx_manager import OvPhysxManager

        for source, targets, target_transforms in self._queue:
            OvPhysxManager._register_clone_transforms(source, targets, target_transforms)
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
    :class:`~isaaclab_ovphysx.physics.OvPhysxManager`.  The actual
    ``physx.clone()`` calls happen in ``_warmup_and_load()`` after OVStage
    has been attached.

    The ``positions`` and ``quaternions`` parameters describe each environment's
    world pose. For nested source rows, the adapter preserves the row's pose
    relative to its source environment and passes the resulting target-root
    world pose to ``physx.clone()``.

    Args:
        stage: USD stage (not modified by this function).
        sources: Source prim paths (one per prototype).
        destinations: Destination path templates with ``"{}"`` for env index.
        env_ids: Environment indices tensor.
        mapping: ``(num_sources, num_envs)`` bool tensor; True selects which
            environments receive each source.
        positions: World (x, y, z) positions [m] for every environment, shape
            ``[num_envs, 3]``.
        quaternions: Optional environment orientations in xyzw order, shape
            ``[num_envs, 4]``.
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
