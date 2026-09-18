# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OvPhysX clone-context dispatch from the active clone plan."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab import cloner
from isaaclab.physics import PhysicsManager

from isaaclab_ov._clone import CloneRecipe, CloneTransform

if TYPE_CHECKING:
    from isaaclab.cloner import ClonePlan
    from isaaclab.sim import SimulationContext


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


def _physics_topology_counts(source_prim: Usd.Prim) -> tuple[int, tuple[tuple[str, int], ...]]:
    """Count rigid bodies and joint types below a clone source, ignoring geometry."""
    rigid_body_count = 0
    joint_type_counts: dict[str, int] = {}
    for prim in Usd.PrimRange(source_prim):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_count += 1
        if prim.IsA(UsdPhysics.Joint):
            joint_type = prim.GetTypeName()
            joint_type_counts[joint_type] = joint_type_counts.get(joint_type, 0) + 1
    return rigid_body_count, tuple(sorted(joint_type_counts.items()))


def _validate_variant_topology(stage: Usd.Stage, sources: Sequence[str], destinations: Sequence[str]) -> None:
    """Reject variant sources whose body or joint counts differ for one destination."""
    reference_by_destination: dict[str, tuple[str, tuple[int, tuple[tuple[str, int], ...]]]] = {}
    for source, destination in zip(sources, destinations):
        source_prim = stage.GetPrimAtPath(source)
        if not source_prim.IsValid():
            raise ValueError(f"OvPhysX clone source prim is not valid on the stage: {source}")
        topology = _physics_topology_counts(source_prim)
        reference = reference_by_destination.setdefault(destination, (source, topology))
        if topology != reference[1]:
            raise ValueError(
                f"OvPhysX clone variants {reference[0]!r} and {source!r} for {destination!r} have incompatible "
                "rigid-body or joint topology. Geometry may vary, but body counts and joint type/DOF structure "
                "must match."
            )


def _clone_recipes(
    stage: Usd.Stage,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: np.ndarray,
    mapping: np.ndarray,
    positions: np.ndarray | None,
    quaternions: np.ndarray | None,
) -> list[CloneRecipe]:
    """Build OvPhysX clone recipes from one flat mapping."""
    if len(sources) != len(destinations):
        raise ValueError(f"Expected one destination per source, got {len(sources)} and {len(destinations)}.")
    if mapping.shape != (len(sources), len(env_ids)):
        raise ValueError(
            f"mapping must have shape [num_sources, num_envs], got {list(mapping.shape)} for "
            f"{len(sources)} sources and {len(env_ids)} environments."
        )
    if positions is not None and positions.shape != (len(env_ids), 3):
        raise ValueError(f"positions must have shape [num_envs, 3], got {list(positions.shape)}.")
    if quaternions is not None and quaternions.shape != (len(env_ids), 4):
        raise ValueError(f"quaternions must have shape [num_envs, 4], got {list(quaternions.shape)}.")

    columns_by_row = [[] for _ in range(mapping.shape[0])]
    for row, column in np.argwhere(mapping):
        columns_by_row[row].append(column)
    active_rows = [row for row, columns in enumerate(columns_by_row) if columns]
    _validate_variant_topology(
        stage,
        [sources[row] for row in active_rows],
        [destinations[row] for row in active_rows],
    )
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    recipes = []
    for row, source in enumerate(sources):
        columns = columns_by_row[row]
        if not columns:
            continue
        active_env_ids = env_ids[columns]
        matched = cloner.path.match(source, destinations[row])
        self_env_id = int(matched.instance) if matched is not None and matched.instance.isdigit() else None

        source_prim = stage.GetPrimAtPath(source)
        source_world = xform_cache.GetLocalToWorldTransform(source_prim).RemoveScaleShear()
        if self_env_id is None:
            source_anchor_world = Gf.Matrix4d(1.0)
        else:
            prefix, _ = cloner.path.split(destinations[row])
            source_anchor_path = f"{prefix}{self_env_id}"
            source_anchor = stage.GetPrimAtPath(source_anchor_path)
            if not source_anchor.IsValid():
                raise ValueError(f"OvPhysX clone source anchor prim is not valid on the stage: {source_anchor_path}")
            source_anchor_world = xform_cache.GetLocalToWorldTransform(source_anchor).RemoveScaleShear()
        source_relative = source_world * source_anchor_world.GetInverse()

        targets = []
        target_transforms = []
        target_env_ids = []
        for env_id, column in zip(active_env_ids, columns):
            env_id = int(env_id)
            if env_id == self_env_id:
                continue
            targets.append(destinations[row].format(env_id))
            target_env_ids.append(env_id)
            target_env_world = Gf.Matrix4d(1.0)
            if positions is not None:
                target_env_world.SetTranslateOnly(Gf.Vec3d(*map(float, positions[column])))
            if quaternions is not None:
                q = quaternions[column]
                target_env_world.SetRotateOnly(Gf.Quatd(float(q[3]), Gf.Vec3d(*map(float, q[:3]))))
            target_transforms.append(_matrix_to_clone_transform(source_relative * target_env_world))
        # env_0 is retained by the serializer even without a clone call. Other source-only
        # variants still need an empty recipe so their authored source environment survives.
        if targets or self_env_id != 0:
            recipes.append((source, targets, target_transforms, target_env_ids))
    return recipes


class OvPhysxReplicateContext:
    """Apply one clone plan to an OvPhysX simulation."""

    replicate_priority = 0

    def __init__(self, sim_context: SimulationContext):
        """Initialize the context.

        Args:
            sim_context: Simulation context that owns this clone backend.
        """
        self._sim = sim_context
        self.stage = sim_context.stage

    def replicate(self, plan: ClonePlan) -> None:
        """Publish clone operations from this context's plan rows.

        Args:
            plan: Replication layout shared by every clone backend.

        Raises:
            ValueError: If positions are malformed or an active source or source anchor prim is invalid.
        """
        if plan.env_ids is None:
            raise ValueError("ClonePlan.env_ids is required for replication.")
        rows = plan.context_rows[type(self)]
        recipes = _clone_recipes(
            stage=self.stage,
            sources=tuple(plan.sources[row] for row in rows),
            destinations=tuple(plan.destinations[row] for row in rows),
            env_ids=plan.env_ids,
            mapping=plan.clone_mask[list(rows)],
            positions=plan.positions,
            quaternions=None,
        )
        for source, targets, transforms, target_env_ids in recipes:
            self._sim.physics_manager._register_clone_transforms(source, targets, transforms, target_env_ids)


def ovphysx_replicate(
    stage: Usd.Stage,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: np.ndarray,
    mapping: np.ndarray,
    positions: np.ndarray | None = None,
    quaternions: np.ndarray | None = None,
) -> None:
    """Publish OvPhysX clone recipes from one raw source-to-environment mapping.

    Args:
        stage: USD stage containing the source prims.
        sources: Source prim paths, one per mapping row.
        destinations: Destination templates containing ``"{}"``, one per mapping row.
        env_ids: Integer environment identifiers, shape ``[num_envs]``.
        mapping: Boolean source-to-environment selection, shape ``[len(sources), num_envs]``.
        positions: Optional environment positions [m], shape ``[num_envs, 3]``.
        quaternions: Optional environment orientations in xyzw order, shape ``[num_envs, 4]``.

    Raises:
        RuntimeError: If no simulation context is active.
        ValueError: If transforms are malformed or a source or source anchor is invalid.
    """
    recipes = _clone_recipes(
        stage=stage,
        sources=sources,
        destinations=destinations,
        env_ids=env_ids,
        mapping=mapping,
        positions=positions,
        quaternions=quaternions,
    )
    sim = PhysicsManager._sim
    if sim is None:
        raise RuntimeError("OvPhysX replication requires an active SimulationContext.")
    for source, targets, transforms, target_env_ids in recipes:
        sim.physics_manager._register_clone_transforms(source, targets, transforms, target_env_ids)
