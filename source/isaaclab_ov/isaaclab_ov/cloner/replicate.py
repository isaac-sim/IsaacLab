# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OvPhysX clone-context dispatch from the active clone plan."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
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


def _physics_topology(source_prim: Usd.Prim) -> tuple[tuple, ...]:
    """Describe body/joint identities, connectivity and DOF axes, ignoring geometry."""
    source_path = source_prim.GetPath()
    topology = []
    for prim in Usd.PrimRange(source_prim, Usd.TraverseInstanceProxies()):
        relative_path = str(prim.GetPath().MakeRelativePath(source_path))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            topology.append(("body", relative_path, UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()))
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            topology.append(("articulation", relative_path))
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            axes = ()
            if prim.GetTypeName() == "PhysicsJoint":
                # A D6 axis is locked iff its applied limit has low > high.
                axes = tuple(
                    axis
                    for axis in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ")
                    if not prim.HasAPI(UsdPhysics.LimitAPI, axis)
                    or UsdPhysics.LimitAPI(prim, axis).GetLowAttr().Get()
                    <= UsdPhysics.LimitAPI(prim, axis).GetHighAttr().Get()
                )
            topology.append(
                (
                    "joint",
                    relative_path,
                    prim.GetTypeName(),
                    tuple(str(path.MakeRelativePath(source_path)) for path in joint.GetBody0Rel().GetTargets()),
                    tuple(str(path.MakeRelativePath(source_path)) for path in joint.GetBody1Rel().GetTargets()),
                    joint.GetJointEnabledAttr().Get(),
                    joint.GetExcludeFromArticulationAttr().Get(),
                    prim.GetAttribute("physics:axis").Get(),
                    axes,
                )
            )
    return tuple(sorted(topology))


def _validate_variant_topology(stage: Usd.Stage, sources: Sequence[str], destinations: Sequence[str]) -> None:
    """Reject variant sources whose body/joint topology or DOF layout differs."""
    reference_by_destination: dict[str, tuple[str, tuple[tuple, ...]]] = {}
    for source, destination in zip(sources, destinations):
        source_prim = stage.GetPrimAtPath(source)
        if not source_prim.IsValid():
            raise ValueError(f"OvPhysX clone source prim is not valid on the stage: {source}")
        topology = _physics_topology(source_prim)
        reference = reference_by_destination.setdefault(destination, (source, topology))
        if topology != reference[1]:
            raise ValueError(
                f"OvPhysX clone variants {reference[0]!r} and {source!r} for {destination!r} have incompatible "
                "rigid-body or joint topology. Geometry may vary, but body counts and joint type/DOF structure "
                "must match."
            )


def _clone_recipes(
    stage: Usd.Stage,
    copies: Iterable[tuple[tuple[str, str], np.ndarray]],
    env_ids: np.ndarray,
    positions: np.ndarray | None,
    quaternions: np.ndarray | None,
) -> list[CloneRecipe]:
    """Build OvPhysX clone recipes from the selected native instance groups."""
    if positions is not None and positions.shape != (len(env_ids), 3):
        raise ValueError(f"positions must have shape [num_envs, 3], got {list(positions.shape)}.")
    if quaternions is not None and quaternions.shape != (len(env_ids), 4):
        raise ValueError(f"quaternions must have shape [num_envs, 4], got {list(quaternions.shape)}.")

    copies = list(copies)
    active = [pair for pair, columns in copies if len(columns) and columns[0] != -1]
    _validate_variant_topology(stage, [source for source, _ in active], [template for _, template in active])

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    recipes = []
    for (source, template), columns in copies:
        if not len(columns) or columns[0] == -1:
            continue
        prefix, _, suffix = template.partition("{}")
        env_template = prefix + "{}" + suffix.split("/", 1)[0]
        matched = cloner.path.match(source, env_template)
        self_env_id = int(matched.instance) if matched is not None and matched.instance.isdigit() else None

        source_prim = stage.GetPrimAtPath(source)
        source_world = xform_cache.GetLocalToWorldTransform(source_prim).RemoveScaleShear()
        if self_env_id is None:
            source_anchor_world = Gf.Matrix4d(1.0)
        else:
            source_anchor_path = env_template.format(self_env_id)
            source_anchor = stage.GetPrimAtPath(source_anchor_path)
            if not source_anchor.IsValid():
                raise ValueError(f"OvPhysX clone source anchor prim is not valid on the stage: {source_anchor_path}")
            source_anchor_world = xform_cache.GetLocalToWorldTransform(source_anchor).RemoveScaleShear()
        source_relative = source_world * source_anchor_world.GetInverse()

        targets, target_transforms, target_env_ids = [], [], []
        for env_id, column in zip(env_ids[columns], columns, strict=True):
            env_id = int(env_id)
            destination = template.format(env_id)
            if destination == source:
                continue
            targets.append(destination)
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

    def replicate(self, plan: ClonePlan, asset_prototype_ids: tuple[int, ...]) -> None:
        """Publish clone operations from this context's source declarations.

        Args:
            plan: Replication layout shared by every clone backend.
            asset_prototype_ids: Asset definitions routed to OVPhysX.

        Raises:
            ValueError: If positions are malformed, a source or source anchor is invalid, or variants
                for one destination differ in body/joint topology or effective DOF axes.
        """
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
        for recipe in _clone_recipes(self.stage, copies, env_ids, plan.positions, None):
            self._sim.physics_manager._register_clone_transforms(*recipe)


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
        ValueError: If source/destination lengths or mapping/transform shapes are inconsistent,
            an active source or source anchor is invalid, or variants for one destination differ
            in body/joint topology or effective DOF axes.
    """
    if len(sources) != len(destinations):
        raise ValueError(f"Expected one destination per source, got {len(sources)} and {len(destinations)}.")
    if mapping.shape != (len(sources), len(env_ids)):
        raise ValueError(
            f"mapping must have shape [num_sources, num_envs], got {list(mapping.shape)} for "
            f"{len(sources)} sources and {len(env_ids)} environments."
        )
    pairs = zip(sources, destinations, strict=True)
    copies = ((pair, np.flatnonzero(mapping[index])) for index, pair in enumerate(pairs))
    recipes = _clone_recipes(stage, copies, env_ids, positions, quaternions)
    sim = PhysicsManager._sim
    if sim is None:
        raise RuntimeError("OvPhysX replication requires an active SimulationContext.")
    for recipe in recipes:
        sim.physics_manager._register_clone_transforms(*recipe)
