# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OvPhysX clone-context dispatch from the active clone plan."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pxr import Gf, Sdf, Usd, UsdGeom

from isaaclab import cloner
from isaaclab.cloner.query import _clone_mapping

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
        sources, destinations, mapping = _clone_mapping(plan, plan.context_rows[type(self)], whole_env=True)
        if plan.positions.shape != (len(plan.env_ids), 3):
            raise ValueError(f"positions must have shape [num_envs, 3], got {list(plan.positions.shape)}.")
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        for i, src in enumerate(sources):
            row_mask = mapping[i].to(dtype=torch.bool)
            active_env_ids = [int(env_id) for env_id in plan.env_ids[row_mask].tolist()]
            if not active_env_ids:
                continue
            active_positions = plan.positions[row_mask].detach().cpu().tolist()

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
            for env_id, position in zip(active_env_ids, active_positions, strict=True):
                if env_id == self_env_id:
                    continue
                targets.append(destinations[i].format(env_id))

                target_env_world = Gf.Matrix4d(1.0)
                target_env_world.SetTranslateOnly(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
                target_transforms.append(_matrix_to_clone_transform(source_relative * target_env_world))

            if targets:
                self._sim.physics_manager._register_clone_transforms(src, targets, target_transforms)
