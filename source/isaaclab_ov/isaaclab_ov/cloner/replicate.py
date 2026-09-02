# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OvPhysX clone-context dispatch from the active clone plan."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from pxr import Gf, Sdf, Usd, UsdGeom

from isaaclab import cloner
from isaaclab.physics import PhysicsManager

from isaaclab_ov._clone import CloneTransform

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


def _clone_recipes(
    stage: Usd.Stage,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None,
    quaternions: torch.Tensor | None,
) -> list[tuple[str, list[str], list[CloneTransform]]]:
    """Build OvPhysX clone recipes from one flat mapping."""
    if positions is not None and positions.shape != (len(env_ids), 3):
        raise ValueError(f"positions must have shape [num_envs, 3], got {list(positions.shape)}.")
    if quaternions is not None and quaternions.shape != (len(env_ids), 4):
        raise ValueError(f"quaternions must have shape [num_envs, 4], got {list(quaternions.shape)}.")

    columns_by_row = [[] for _ in range(mapping.shape[0])]
    for row, column in mapping.detach().to(dtype=torch.bool).nonzero(as_tuple=False).cpu().tolist():
        columns_by_row[row].append(column)
    env_ids_cpu = env_ids.detach().cpu().tolist()
    positions_cpu = None if positions is None else positions.detach().cpu().tolist()
    quaternions_cpu = None if quaternions is None else quaternions.detach().cpu().tolist()
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    recipes = []
    for row, source in enumerate(sources):
        columns = columns_by_row[row]
        if not columns:
            continue
        active_env_ids = [env_ids_cpu[column] for column in columns]
        matched = cloner.path.match(source, destinations[row])
        self_env_id = int(matched.instance) if matched is not None and matched.instance.isdigit() else None

        source_prim = stage.GetPrimAtPath(source)
        if not source_prim.IsValid():
            raise ValueError(f"OvPhysX clone source prim is not valid on the stage: {source}")
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
        for env_id, column in zip(active_env_ids, columns):
            if env_id == self_env_id:
                continue
            targets.append(destinations[row].format(env_id))
            target_env_world = Gf.Matrix4d(1.0)
            if positions_cpu is not None:
                target_env_world.SetTranslateOnly(Gf.Vec3d(*map(float, positions_cpu[column])))
            if quaternions_cpu is not None:
                q = quaternions_cpu[column]
                target_env_world.SetRotateOnly(Gf.Quatd(float(q[3]), Gf.Vec3d(*map(float, q[:3]))))
            target_transforms.append(_matrix_to_clone_transform(source_relative * target_env_world))
        if targets:
            recipes.append((source, targets, target_transforms))
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
        physics_scene_prim = self.stage.GetPrimAtPath("/physicsScene")
        if physics_scene_prim.IsValid():
            physics_scene_prim.CreateAttribute("physxScene:envIdInBoundsBitCount", Sdf.ValueTypeNames.Int).Set(4)

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
            self.stage,
            tuple(plan.sources[row] for row in rows),
            tuple(plan.destinations[row] for row in rows),
            plan.env_ids,
            plan.clone_mask[list(rows)],
            plan.positions,
            None,
        )
        for source, targets, transforms in recipes:
            self._sim.physics_manager._register_clone_transforms(source, targets, transforms)


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
    """Publish OvPhysX clone recipes from one raw source-to-environment mapping."""
    del device
    recipes = _clone_recipes(stage, sources, destinations, env_ids, mapping, positions, quaternions)
    sim = PhysicsManager._sim
    if sim is None:
        raise RuntimeError("OvPhysX replication requires an active SimulationContext.")
    for source, targets, transforms in recipes:
        sim.physics_manager._register_clone_transforms(source, targets, transforms)
