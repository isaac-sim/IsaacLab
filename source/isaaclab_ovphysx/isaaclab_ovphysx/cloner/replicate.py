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
``physx.clone(source, targets, transforms)`` after loading. Full-stage loads instead
materialize or overlay every recipe in serialized USDA before attaching OVStage.
Recipes remain active for the current simulation context so a forced re-warmup
rebuilds the same topology without modifying the live USD stage.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from pxr import Gf, Sdf, Usd, UsdGeom

from isaaclab import cloner

from isaaclab_ovphysx._clone import CloneTransform, clone_transforms_from_positions


def _select_env_ids(env_ids: torch.Tensor, mapping: torch.Tensor, row: int) -> torch.Tensor:
    """Return the environment ids selected by a replication row."""
    row_mask = mapping[row]
    if row_mask.dtype != torch.bool:
        row_mask = row_mask.to(dtype=torch.bool)
    return env_ids[row_mask]


def _matrix_to_clone_transform(matrix: Gf.Matrix4d) -> CloneTransform:
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


def _pose_tensor_rows(tensor: torch.Tensor | None, name: str, component_count: int) -> list[list[float]] | None:
    """Validate and copy an optional per-environment pose tensor to CPU rows."""
    if tensor is None:
        return None
    if tensor.ndim != 2 or tensor.shape[1] != component_count:
        raise ValueError(f"{name} must have shape [num_envs, {component_count}], got {list(tensor.shape)}.")
    return tensor.detach().cpu().tolist()


def _validate_pose_rows(name: str, rows: list[list[float]] | None, env_ids: Sequence[int]) -> None:
    """Validate that optional pose rows contain every selected environment."""
    if rows is None:
        return
    for env_id in env_ids:
        if env_id < 0 or env_id >= len(rows):
            raise ValueError(f"{name} does not contain selected environment id {env_id}; it has {len(rows)} rows.")


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
        self._queue: list[tuple[str, list[str], list[CloneTransform]]] = []

    def queue(
        self, source: str, targets: Sequence[str], parent_positions: Sequence[tuple[float, float, float]]
    ) -> None:
        """Queue one pending OvPhysX clone operation.

        Args:
            source: Source prim path.
            targets: Destination prim paths.
            parent_positions: Final world positions [m] for each target root;
                orientations are identity.
        """
        target_transforms = clone_transforms_from_positions(parent_positions)
        self._queue_transforms(source, targets, target_transforms)

    def _queue_transforms(
        self, source: str, targets: Sequence[str], target_transforms: Sequence[CloneTransform]
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
            positions: Optional per-environment world positions [m], shape
                ``[num_envs, 3]``.
            quaternions: Optional per-environment orientations in xyzw order,
                shape ``[num_envs, 4]``.

        Raises:
            ValueError: If a provided pose tensor is malformed or lacks a selected
                environment, or if an active source or source anchor prim is invalid.
        """
        positions_list = _pose_tensor_rows(positions, "positions", 3)
        quaternions_list = _pose_tensor_rows(quaternions, "quaternions", 4)
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        for i, src in enumerate(sources):
            active_env_ids = [int(env_id) for env_id in _select_env_ids(env_ids, mapping, i).tolist()]
            if not active_env_ids:
                continue
            _validate_pose_rows("positions", positions_list, active_env_ids)
            _validate_pose_rows("quaternions", quaternions_list, active_env_ids)

            self_env_id: int | None = None
            matched = cloner.path.match(src, destinations[i])
            if matched is not None and matched.instance.isdigit():
                self_env_id = int(matched.instance)

            source_prim = self.stage.GetPrimAtPath(src)
            if not source_prim.IsValid():
                raise ValueError(f"OvPhysX clone source prim is not valid on the stage: {src}")
            source_world = xform_cache.GetLocalToWorldTransform(source_prim).RemoveScaleShear()
            if self_env_id is None:
                source_anchor_world = Gf.Matrix4d(1.0)
            else:
                prefix, _ = cloner.path.split(destinations[i])
                source_anchor_path = f"{prefix}{self_env_id}"
                source_anchor = self.stage.GetPrimAtPath(source_anchor_path)
                if not source_anchor.IsValid():
                    raise ValueError(
                        f"OvPhysX clone source anchor prim is not valid on the stage: {source_anchor_path}"
                    )
                source_anchor_world = xform_cache.GetLocalToWorldTransform(source_anchor).RemoveScaleShear()
            source_relative = source_world * source_anchor_world.GetInverse()

            targets: list[str] = []
            target_transforms: list[CloneTransform] = []
            for env_id in active_env_ids:
                if env_id == self_env_id:
                    continue
                targets.append(destinations[i].format(env_id))

                target_env_world = Gf.Matrix4d(1.0)
                if positions_list is not None:
                    pos = positions_list[env_id]
                    target_env_world.SetTranslateOnly(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
                if quaternions_list is not None:
                    quat = quaternions_list[env_id]
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
    """Queue OvPhysX clone operations from a flat clone mapping.

    The ``positions`` and ``quaternions`` parameters describe each environment's
    world pose. Source-relative rows are composed with the target environment
    poses and queued as final target-root world transforms.

    Args:
        stage: USD stage associated with the clone operations.
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

    Raises:
        ValueError: If a provided pose tensor is malformed or lacks a selected
            environment, or if an active source or source anchor prim is invalid.
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
