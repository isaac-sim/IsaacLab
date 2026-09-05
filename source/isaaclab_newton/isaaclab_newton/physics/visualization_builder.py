# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
from newton import ModelBuilder
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx
from newton._src.utils import topological_sort_undirected

from pxr import Usd, UsdPhysics

from isaaclab.scene_data.deformable_discovery import DeformableStageEntry, discover_deformables_on_stage
from isaaclab.sim.utils.transforms import resolve_prim_pose

from isaaclab_newton.cloner.newton_clone_utils import (
    _restore_visible_colliders_without_visual_shapes,
    build_source_builders,
    replicate_builder_mapping,
)
from isaaclab_newton.physics.visualization_deformables import add_shadow_deformables_to_builder
from isaaclab_newton.renderers.visual_material import import_builder_visual_material_paths


def _deformable_ignore_paths(
    stage: Usd.Stage,
    sources: Sequence[str] | None = None,
    entries: Sequence[DeformableStageEntry] | None = None,
) -> list[str]:
    """Collect deformable prim paths for ``add_usd`` ignore lists.

    Shadow deformables are added explicitly by :func:`add_shadow_deformables_to_builder`.
    Ignoring them during USD import prevents double-allocating particles in the
    shadow ``particle_q`` buffer (which breaks geometry mapping offsets).

    Args:
        stage: USD stage to scan.
        sources: Optional clone-source roots to restrict the ignore list. When ``None``,
            every discovered deformable path is ignored (standalone imports).
        entries: Optional pre-discovered deformable entries to avoid a second stage walk.
    """
    source_prefixes: tuple[str, ...] | None = None
    if sources is not None:
        source_prefixes = tuple(f"{source.rstrip('/')}/" for source in sources)

    if entries is None:
        entries = discover_deformables_on_stage(stage)

    ignore_paths: list[str] = []

    for entry in entries:
        for path in (entry.root_path, entry.sim_mesh_path, entry.vis_mesh_path):
            if source_prefixes is None:
                ignore_paths.append(path)
                continue

            under_source = any(
                path == source.rstrip("/") or path.startswith(prefix)
                for source, prefix in zip(sources, source_prefixes, strict=True)
            )
            if under_source:
                ignore_paths.append(path)

    # Preserve order while dropping duplicates.
    return list(dict.fromkeys(ignore_paths))


def _reversed_joint_ignore_paths(
    stage: Usd.Stage,
    roots: Sequence[str] | None = None,
    excluded_roots: Sequence[str] = (),
) -> list[str]:
    """Return one exact path expression for joints incompatible with the shadow import.

    Newton cannot import a joint whose authored body order opposes its rooted
    articulation graph. The shadow model can omit those constraints because the
    active physics backend supplies all body poses, while retaining supported joints
    for joint visualization. Loop constraints in an affected graph are omitted too;
    removing their reversed tree edges would otherwise leave them orphaned. Combining
    the paths into one expression also avoids Python's regular-expression cache limit
    on large cloned stages.

    Args:
        stage: USD stage to scan.
        roots: Optional prim roots that restrict the scan.
        excluded_roots: Prim roots to omit from the scan.

    Returns:
        A one-element ignore-expression list, or an empty list when no reversed joints
        are found.
    """

    def _is_under(path: str, root: str) -> bool:
        root = root.rstrip("/") or "/"
        return root == "/" or path == root or path.startswith(f"{root}/")

    def _resolve_body_path(target: Any) -> str:
        prim = stage.GetPrimAtPath(target)
        while prim:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return str(prim.GetPath())
            prim = prim.GetParent()
        return str(target)

    if roots is None:
        prim_ranges = (stage.Traverse(),)
    else:
        prim_ranges = tuple(
            Usd.PrimRange(root_prim)
            for root in roots
            if (root_prim := stage.GetPrimAtPath(root)) and root_prim.IsValid()
        )

    joints: list[tuple[str, str | None, str | None, bool]] = []
    visited_paths: set[str] = set()
    for prim_range in prim_ranges:
        for prim in prim_range:
            if not prim.IsA(UsdPhysics.Joint):
                continue
            path = str(prim.GetPath())
            if path in visited_paths or any(_is_under(path, root) for root in excluded_roots):
                continue
            visited_paths.add(path)
            joint = UsdPhysics.Joint(prim)
            if joint.GetJointEnabledAttr().Get() is False:
                continue
            body0_targets = joint.GetBody0Rel().GetTargets()
            body1_targets = joint.GetBody1Rel().GetTargets()
            body0 = _resolve_body_path(body0_targets[0]) if body0_targets else None
            body1 = _resolve_body_path(body1_targets[0]) if body1_targets else None
            if body0 is not None or body1 is not None:
                joints.append((path, body0, body1, joint.GetExcludeFromArticulationAttr().Get() is True))

    body_ids = {
        body_path: index
        for index, body_path in enumerate(
            dict.fromkeys(body_path for _, body0, body1, _ in joints for body_path in (body0, body1) if body_path)
        )
    }
    component_parents = list(range(len(body_ids)))

    def _find(body_id: int) -> int:
        while component_parents[body_id] != body_id:
            component_parents[body_id] = component_parents[component_parents[body_id]]
            body_id = component_parents[body_id]
        return body_id

    def _union(body0: str, body1: str) -> None:
        root0 = _find(body_ids[body0])
        root1 = _find(body_ids[body1])
        if root0 != root1:
            component_parents[root1] = root0

    for _, body0, body1, excluded in joints:
        if body0 is not None and body1 is not None and not excluded:
            _union(body0, body1)

    components: dict[int, list[tuple[str, str | None, str | None]]] = defaultdict(list)
    for path, body0, body1, excluded in joints:
        body_path = body0 if body0 is not None else body1
        if body_path is not None and not excluded:
            components[_find(body_ids[body_path])].append((path, body0, body1))

    reversed_paths: list[str] = []
    affected_components: set[int] = set()
    for component, component_joints in components.items():
        joint_edges: list[tuple[int, int]] = []
        joint_paths: list[str] = []
        body_pairs: set[tuple[int, int]] = set()
        for path, body0, body1 in component_joints:
            body_pair = (body_ids[body0] if body0 is not None else -1, body_ids[body1] if body1 is not None else -1)
            if body_pair in body_pairs:
                continue
            body_pairs.add(body_pair)
            joint_edges.append(body_pair)
            joint_paths.append(path)
        try:
            _, reversed_joint_indices = topological_sort_undirected(joint_edges, use_dfs=True, ensure_single_root=True)
        except ValueError:
            # Preserve Newton's canonical error for malformed non-tree topologies.
            continue
        if reversed_joint_indices:
            affected_components.add(component)
        reversed_paths.extend(joint_paths[index] for index in reversed_joint_indices)

    for path, body0, body1, excluded in joints:
        if not excluded:
            continue
        body_paths = (body_path for body_path in (body0, body1) if body_path in body_ids)
        if any(_find(body_ids[body_path]) in affected_components for body_path in body_paths):
            reversed_paths.append(path)

    if not reversed_paths:
        return []
    alternatives = "|".join(re.escape(path) for path in dict.fromkeys(reversed_paths))
    return [f"^(?:{alternatives})$"]


def build_visualization_builder_from_stage_envs(
    stage: Usd.Stage,
    env_paths: Sequence[tuple[int, str]],
    clone_plan: Any | None,
    *,
    up_axis: str = "Z",
    device: str = "cpu",
) -> tuple[ModelBuilder, tuple[list, list]]:
    """Build a Newton shadow visualization builder from a USD stage.

    Cloned scenes use the clone plan to preserve per-environment world layout and
    labels. Standalone scenes without a clone plan are imported as one world, then
    deformables are added as shadow particles so OVRTX registry bindings stay populated.

    Args:
        stage: USD stage to import into the shadow builder.
        env_paths: Sorted ``(env_id, env_prim_path)`` pairs.
        clone_plan: Optional clone plan; ``None`` imports the stage as a standalone world.
        up_axis: Up axis for the :class:`~newton.ModelBuilder`.
        device: Warp device for volume sim-to-visual remap tables.

    Returns:
        A tuple of the populated :class:`~newton.ModelBuilder` and shadow-deformable
        metadata ``(shadow_entities, registry_groups)``.
    """
    schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]
    builder = ModelBuilder(up_axis=up_axis)
    # Discover once and reuse via ``entries=`` below (ignore paths + shadow add).
    deformable_entries = discover_deformables_on_stage(stage)
    if clone_plan is None:
        # Ignore deformables during USD import; add them as shadow particles below so
        # SceneData mapping and OVRTX registry receive the same particle offsets.
        ignore_paths = [
            *_deformable_ignore_paths(stage, entries=deformable_entries),
            *_reversed_joint_ignore_paths(stage),
        ]
        import_result = builder.add_usd(
            stage,
            schema_resolvers=schema_resolvers,
            ignore_paths=ignore_paths or None,
            bodies_follow_joint_ordering=False,
            skip_mesh_approximation=True,
        )
        _restore_visible_colliders_without_visual_shapes(builder, stage, import_result["path_shape_map"])
        import_builder_visual_material_paths(builder, stage)
        shadow_entities, registry_groups = add_shadow_deformables_to_builder(
            builder, stage, env_paths, device=device, entries=deformable_entries, clone_plan=clone_plan
        )
        builder.shape_collision_filter_pairs = []
        builder.shape_collision_group[:] = [0] * builder.shape_count
        return builder, (shadow_entities, registry_groups)

    if not env_paths:
        raise ValueError("clone plan requires at least one environment path")

    env_path_by_id = dict(env_paths)

    sources = tuple(clone_plan.sources)
    destinations = tuple(clone_plan.destinations)
    env_ids = clone_plan.env_ids
    mapping = clone_plan.clone_mask

    poses = [resolve_prim_pose(stage.GetPrimAtPath(env_path_by_id[int(env_id)])) for env_id in env_ids]
    positions = np.asarray([pos for pos, _ in poses], dtype=np.float32)
    quaternions = np.asarray([quat for _, quat in poses], dtype=np.float32)
    # Ignore every deformable on the stage for the world import — not only those under
    # clone sources. Otherwise a non-env deformable (e.g. ``/World/Assets/Cloth``) is
    # imported here and added again by ``add_shadow_deformables_to_builder``.
    deformable_ignore_paths = _deformable_ignore_paths(stage, entries=deformable_entries)
    global_reversed_joint_ignore_paths = _reversed_joint_ignore_paths(stage, excluded_roots=("/World/envs", *sources))
    import_result = builder.add_usd(
        stage,
        ignore_paths=["/World/envs", *sources, *deformable_ignore_paths, *global_reversed_joint_ignore_paths],
        schema_resolvers=schema_resolvers,
        bodies_follow_joint_ordering=False,
        skip_mesh_approximation=True,
    )
    _restore_visible_colliders_without_visual_shapes(builder, stage, import_result["path_shape_map"])
    import_builder_visual_material_paths(builder, stage)
    source_deformable_ignore_paths = _deformable_ignore_paths(stage, sources, entries=deformable_entries)
    source_reversed_joint_ignore_paths = _reversed_joint_ignore_paths(stage, roots=sources)
    source_builders = build_source_builders(
        stage,
        sources,
        lambda: ModelBuilder(up_axis=up_axis),
        schema_resolvers,
        ignore_paths=[*source_deformable_ignore_paths, *source_reversed_joint_ignore_paths] or None,
        bodies_follow_joint_ordering=False,
        skip_mesh_approximation=True,
    )
    global_builder = builder
    builder = ModelBuilder(up_axis=up_axis)  # Preserve Newton's compact empty filter store.
    for visual_builder in (global_builder, *source_builders.values()):
        visual_builder.shape_collision_filter_pairs = []
        visual_builder.shape_collision_group[:] = [0] * visual_builder.shape_count
    builder.add_builder(global_builder)
    replicate_builder_mapping(
        builder=builder,
        sources=sources,
        mapping=mapping,
        positions=positions,
        quaternions=quaternions,
        source_builders=source_builders,
        destinations=destinations,
        env_ids=env_ids,
    )
    shadow_entities, registry_groups = add_shadow_deformables_to_builder(
        builder, stage, env_paths, device=device, entries=deformable_entries, clone_plan=clone_plan
    )
    return builder, (shadow_entities, registry_groups)
