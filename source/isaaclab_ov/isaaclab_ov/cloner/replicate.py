# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OV clone-context dispatch and native replication operations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab import cloner
from isaaclab.physics import PhysicsManager

from isaaclab_ov._clone import CloneRecipe

if TYPE_CHECKING:
    import ovrtx
    import ovstage

    from isaaclab.cloner import ClonePlan
    from isaaclab.sim import SimulationContext


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

    # OVPhysX's native metatype omits effective D6 axes and tendon layout. Check only variant sources.
    variants, layouts = {}, {}
    for source, template in grouped:
        variants.setdefault(template, []).append(source)
    for sources in variants.values():
        if len(sources) < 2:
            continue
        for source in sources:
            if source not in layouts:
                root = stage.GetPrimAtPath(source)
                if not root:
                    raise ValueError(f"OvPhysX clone source prim is not valid on the stage: {source}")
                layout = layouts[source] = {}
                for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                    relative = str(prim.GetPath().MakeRelativePath(root.GetPath()))
                    if prim.GetTypeName() == "PhysicsJoint" and UsdPhysics.Joint(prim).GetJointEnabledAttr().Get():
                        # Native articulations always lock translation, regardless of authored limits.
                        axes = []
                        for axis in ("rotX", "rotY", "rotZ"):
                            limit = UsdPhysics.LimitAPI(prim, axis)
                            if not limit or limit.GetLowAttr().Get() <= limit.GetHighAttr().Get():
                                axes.append(axis)
                        layout[relative, "rotation axes"] = tuple(axes)
                    for schema in prim.GetPrimTypeInfo().GetAppliedAPISchemas():
                        if schema.startswith(("PhysxTendonAxisRootAPI:", "PhysxTendonAttachmentRootAPI:")):
                            layout[relative, schema] = None
            if layouts[source] != layouts[sources[0]]:
                raise ValueError(
                    f"OvPhysX variants {sources[0]!r} and {source!r} have incompatible rotation axes or tendon layouts."
                )

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    recipes = []
    for (source, template), columns in copies:
        prefix, _, suffix = template.partition("{}")
        env_template = prefix + "{}" + suffix.split("/", 1)[0]
        matched = cloner.path.match(source, env_template)
        self_env_id = int(matched.instance) if matched is not None and matched.instance.isdigit() else None

        source_prim = stage.GetPrimAtPath(source)
        if not source_prim.IsValid():
            raise ValueError(f"OvPhysX clone source prim is not valid on the stage: {source}")
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
                for one destination differ in effective rotation axes or tendon layout.
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


class OvrtxReplicateContext:
    """Compile routed assets into native copies for OVRTX renderers and borrowed stages."""

    replicate_priority = 100

    def __init__(self, sim_context: SimulationContext):
        self._sim = sim_context

    def replicate(self, plan: ClonePlan, asset_prototype_ids: tuple[int, ...]) -> None:
        """Prepare native operations before camera initialization imports the scene.

        Like OVPhysX recipes, these copies belong to the backend. Renderers must not
        interpret topology or retain asset-id routing to reconstruct them later.
        """
        from isaaclab_ov.renderers.ovrtx_renderer_cfg import OVRTXBackendCfg  # noqa: PLC0415
        from isaaclab_ov.stage import OvstageBackendCfg  # noqa: PLC0415

        if not any(isinstance(cfg, (OVRTXBackendCfg, OvstageBackendCfg)) for cfg, _ in self._sim._backend_registry):
            return
        sources = cloner.path.get_asset_prototype_paths(plan)
        templates, starts, world_ids, world_starts = cloner.path.get_world_prototype_asset_templates(
            plan, include_world_indices=True
        )
        assets = plan.topology.world_prototypes
        copies = {}
        for group in np.flatnonzero(np.diff(world_starts)):
            targets = world_ids[world_starts[group] : world_starts[group + 1]]
            members = [index for index in range(*starts[group : group + 2]) if assets[index] in asset_prototype_ids]
            destinations = [templates[index] for index in members]
            for index, parent in zip(members, cloner.path.get_parent_indices(destinations), strict=True):
                source, template = sources[assets[index]], templates[index]
                if parent != -1:
                    ancestor = members[parent]
                    suffix = cloner.path.relative_to(template, templates[ancestor])
                    if source == sources[assets[ancestor]] + suffix:
                        continue
                copies.setdefault((source, template), []).extend(template.format(int(world)) for world in targets)
        # Native clones cannot overwrite existing prims. Keep self-only sources for export,
        # but omit self-copies and children already carried by the same parent copy.
        native_copies = [
            (source, [target for target in copies[source, template] if target != source])
            for source, template in sorted(copies, key=lambda copy: copy[1].count("/"))
        ]
        env_paths = [plan.env_template.format(world) for world in range(len(plan.topology.world_prototype_layout))]
        for cfg, backend in self._sim._backend_registry:
            if isinstance(cfg, OVRTXBackendCfg):
                backend.clone_copies = native_copies
                backend.clone_env_paths = env_paths
                backend.clone_positions = plan.positions
            elif isinstance(cfg, OvstageBackendCfg):
                backend.populate(self._sim.stage, native_copies, env_paths, plan.positions)


def ovrtx_replicate(
    renderer: ovrtx.Renderer,
    copies: Sequence[tuple[str, Sequence[str]]],
    env_paths: Sequence[str],
    positions: np.ndarray | None = None,
) -> None:
    """Apply prepared subtree copies and environment placement to a native OVRTX scene.

    Args:
        renderer: Native renderer holding the source prims.
        copies: Source paths paired with destination paths, ordered with parents before children.
            Sources with no destinations are retained prototypes; self-copies must be excluded.
        env_paths: Environment-root paths in placement order.
        positions: Optional environment positions [m], shape ``[len(env_paths), 3]``.
    """
    from ovrtx import PrimMode, Semantic  # noqa: PLC0415

    for source, targets in copies:
        if targets:
            renderer.clone_usd(source, targets)
    if positions is not None and env_paths:
        xforms = np.tile(np.eye(4, dtype=np.float64), (len(env_paths), 1, 1))
        xforms[:, 3, :3] = positions
        renderer.write_attribute(
            env_paths, "omni:xform", xforms, semantic=Semantic.XFORM_MAT4x4, prim_mode=PrimMode.MUST_EXIST
        )


def ovstage_replicate(
    stage: ovstage.Stage,
    paths: ovstage.PathDictionary,
    copies: Sequence[tuple[str, Sequence[str]]],
    env_paths: Sequence[str],
    positions: np.ndarray | None = None,
    *,
    ordinal: int,
) -> None:
    """Apply the same prepared copies and placement to an OVStage scene.

    Args:
        stage: Native stage holding the source prims.
        paths: Path dictionary of ``stage``.
        copies: Source paths paired with destination paths, with the same contract as :func:`ovrtx_replicate`.
        env_paths: Environment-root paths in placement order.
        positions: Optional environment positions [m], shape ``[len(env_paths), 3]``.
        ordinal: Write ordinal for the clones and environment placement.
    """
    import ovstage  # noqa: PLC0415

    from isaaclab_ov.stage import xform_tensor_from_numpy  # noqa: PLC0415

    for source, targets in copies:
        if targets:
            stage.clone(source, targets, ordinal=ordinal)
    if positions is not None and env_paths:
        xforms = np.tile(np.eye(4, dtype=np.float64), (len(env_paths), 1, 1))
        xforms[:, 3, :3] = positions
        path_list = paths.create_path_list_from_strings(env_paths)
        try:
            with stage.query_from_path_list(path_list) as query:
                stage.write_attribute(
                    query,
                    "omni:xform",
                    ordinal=ordinal,
                    tensors=xform_tensor_from_numpy(xforms),
                    is_array=False,
                    semantic=ovstage.AttributeSemantic.MATRIX,
                ).wait()
        finally:
            paths.destroy_path_list(path_list)


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
            or an active source or source anchor is invalid.
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
    stage: Usd.Stage, recipes: Sequence[CloneRecipe], full_stage: bool, plan: ClonePlan | None = None
) -> tuple[str, list[CloneRecipe]]:
    """Export complete original worlds once and compile copies from those originals.

    Parsed bodies have native environment ID zero. An original-bearing world must therefore
    import all its assets together; its copies inherit one collision group and environment ID.
    CPU and features without native cloning import every declared world instead.
    """
    # Group worlds by their declared asset memberships, not by completed-stage discovery.
    sources = tuple(Sdf.Path(source) for source, _, _, _, _ in recipes)
    memberships = {}
    originals = {world for _, _, _, _, world in recipes if world is not None}
    if plan is not None and not full_stage:
        # USD-only physics keeps native ID zero. Coverage requires the same copy in each destination world.
        asset_sources = cloner.path.get_asset_prototype_paths(plan)
        templates, starts, world_ids, world_starts = cloner.path.get_world_prototype_asset_templates(
            plan, include_world_indices=True
        )
        for group in np.flatnonzero(np.diff(world_starts[1:])) + 1:
            worlds = world_ids[world_starts[group] : world_starts[group + 1]]
            for index in range(*starts[group : group + 2]):
                uncovered = set(worlds) - originals
                if not uncovered:
                    break
                source, template = asset_sources[plan.topology.world_prototypes[index]], templates[index]
                for root, targets, *_ in recipes:
                    suffix = cloner.path.relative_to(source, root)
                    if suffix is not None:
                        copied_paths = {target + suffix for target in targets}
                        uncovered = {world for world in uncovered if template.format(world) not in copied_paths}
                if not uncovered:
                    continue
                prim = stage.GetPrimAtPath(source)
                if prim and any(
                    child.HasAPI(UsdPhysics.RigidBodyAPI) or child.HasAPI(UsdPhysics.CollisionAPI)
                    for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
                ):
                    originals.update(uncovered)
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
