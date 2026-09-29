# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OvPhysX clone-context dispatch from the active clone plan."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab import cloner
from isaaclab.physics import PhysicsManager

from isaaclab_ov._clone import CloneRecipe

if TYPE_CHECKING:
    from isaaclab.cloner import ClonePlan
    from isaaclab.sim import SimulationContext


def _physics_topology(source_prim: Usd.Prim) -> tuple[tuple, ...]:
    """Describe body/joint identities, connectivity and DOF axes, ignoring geometry."""
    source_path = source_prim.GetPath()
    topology = []
    for prim in Usd.PrimRange(source_prim, Usd.TraverseInstanceProxies()):
        relative_path = str(prim.GetPath().MakeRelativePath(source_path))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            topology.append(("body", relative_path, UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()))
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            enabled = prim.GetAttribute("physxArticulation:articulationEnabled").Get() is not False
            topology.append(("articulation", relative_path, enabled))
        schemas = prim.GetPrimTypeInfo().GetAppliedAPISchemas()
        tendons = tuple(sorted(schema for schema in schemas if schema.startswith("PhysxTendon")))
        if tendons:
            topology.append(("tendon", relative_path, tendons))
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            axes = ()
            if prim.GetTypeName() == "PhysicsJoint":
                # Articulations lock D6 translation; only free rotational axes add tensor DOFs.
                axes = tuple(
                    axis
                    for axis in ("rotX", "rotY", "rotZ")
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

    grouped = {}
    for pair, columns in copies:
        if len(columns) and columns[0] != -1:
            grouped.setdefault(pair, []).append(columns)
    copies = [(pair, np.unique(np.concatenate(columns))) for pair, columns in grouped.items()]
    _validate_variant_topology(stage, [pair[0] for pair, _ in copies], [pair[1] for pair, _ in copies])

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    recipes = []
    for (source, template), columns in copies:
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
            targets.append(destination)
            target_env_ids.append(env_id)
            target_env_world = Gf.Matrix4d(1.0)
            if positions is not None:
                target_env_world.SetTranslateOnly(Gf.Vec3d(*map(float, positions[column])))
            if quaternions is not None:
                q = quaternions[column]
                target_env_world.SetRotateOnly(Gf.Quatd(float(q[3]), Gf.Vec3d(*map(float, q[:3]))))
            pose = (source_relative * target_env_world).RemoveScaleShear()
            rotation = pose.ExtractRotationQuat()
            target_transforms.append((*pose.ExtractTranslation(), *rotation.GetImaginary(), rotation.GetReal()))
        recipes.append((source, targets, target_transforms, target_env_ids, self_env_id))
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
        copies = []
        for group in np.flatnonzero(np.diff(world_starts[1:])) + 1:
            targets = world_ids[world_starts[group] : world_starts[group + 1]]
            for index in range(*starts[group : group + 2]):
                if (asset := plan.topology.world_prototypes[index]) in asset_prototype_ids:
                    copies.append(((sources[asset], templates[index]), targets))
        env_ids = np.arange(len(plan.topology.world_prototype_layout))
        recipes = _clone_recipes(self.stage, copies, env_ids, plan.positions, None)
        self._sim.physics_manager._clone_recipes.extend(recipes)


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
    sim.physics_manager._clone_recipes.extend(recipes)


def _serialize_stage(
    stage: Usd.Stage, recipes: Sequence[CloneRecipe], full_stage: bool
) -> tuple[str, list[CloneRecipe]]:
    """Export complete original worlds once and compile copies from those originals.

    Parsed bodies have native environment ID zero. An original-bearing world must therefore
    import all its assets together; its copies inherit one collision group and environment ID.
    CPU and features without native cloning import every declared world instead.
    """
    # Group worlds by their declared asset memberships, not by completed-stage discovery.
    memberships = {}
    originals = {world for _, _, _, _, world in recipes if world is not None}
    for index, (_, _, _, world_ids, _) in enumerate(recipes):
        for world in world_ids or ():
            memberships.setdefault(world, []).append(index)
    prototypes = {tuple(memberships[world]): world for world in sorted(originals) if world in memberships}
    representatives = {
        world: prototypes.setdefault(tuple(members), world) for world, members in sorted(memberships.items())
    }
    originals.update(prototypes.values())

    # Materialize only originals on GPU. Each native destination is absent from the export.
    layer = stage.Flatten()
    exported = Usd.Stage.Open(layer)
    xforms = UsdGeom.XformCache()
    sources = tuple(Sdf.Path(source) for source, _, _, _, _ in recipes)
    native = {}
    for recipe in sorted(recipes, key=lambda recipe: recipe[0].count("/")):
        source, targets, transforms, world_ids, source_world = recipe
        target_by_world = dict(zip(world_ids, targets, strict=True)) if world_ids is not None else {}
        source_pose = xforms.GetLocalToWorldTransform(stage.GetPrimAtPath(source))
        for index, target in enumerate(targets):
            if target == source:
                continue
            target_path = Sdf.Path(target)
            if any(path.HasPrefix(target_path) for path in sources):
                raise ValueError(f"OvPhysX clone target {target!r} overlaps a clone source.")
            world = world_ids[index] if world_ids is not None else None
            if full_stage or world in originals:
                # References preserve joint relationships and any authored target opinions.
                prim = exported.GetPrimAtPath(target)
                authored = bool(prim)
                prim = prim or exported.DefinePrim(target, "Xform")
                prim.GetReferences().AddInternalReference(source)
                if transforms and not authored:
                    pose = transforms[index]
                    anchor = Gf.Matrix4d().SetRotate(Gf.Quatd(pose[6], Gf.Vec3d(*pose[3:6])))
                    anchor.SetTranslateOnly(Gf.Vec3d(*pose[:3]))
                    world_pose = source_pose * source_pose.RemoveScaleShear().GetInverse() * anchor
                    xform = UsdGeom.Xformable(prim)
                    xform.MakeMatrixXform().Set(world_pose)
                    xform.SetResetXformStack(True)
                continue

            # Clone all members of a world from the same complete original, with one native ID.
            original = representatives[world] if world is not None else source_world
            native_source = target_by_world[original] if world is not None else source
            operation = native.setdefault(
                native_source, (native_source, [], [], [] if world_ids is not None else None, original)
            )
            operation[1].append(target)
            if transforms:
                operation[2].append(transforms[index])
            if world_ids is not None:
                operation[3].append(world)
            with Sdf.ChangeBlock():
                if (spec := layer.GetPrimAtPath(target_path)) is not None:
                    del spec.nameParent.nameChildren[spec.name]
                # Empty clone ancestors make native literal-path lookup enumerate every world.
                spec = layer.GetPrimAtPath(target_path.GetParentPath())
                while spec and not spec.nameChildren and spec.typeName in ("", "Xform"):
                    if spec.HasInfo("apiSchemas") or any(source.HasPrefix(spec.path) for source in sources):
                        break
                    parent = spec.nameParent
                    del (parent.nameChildren if parent else layer.rootPrims)[spec.name]
                    spec = parent

    return layer.ExportToString(), list(native.values())


def _replay_clones(physx, recipes: Sequence[CloneRecipe]) -> None:
    """Apply compiled operations after native stage attachment and before warmup."""
    for source, targets, transforms, env_ids, _ in recipes:
        if targets:
            physx.wait_op(physx.clone(source, targets, transforms or None, env_ids=env_ids))
