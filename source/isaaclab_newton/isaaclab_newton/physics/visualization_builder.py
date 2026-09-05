# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
from newton import ModelBuilder
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx

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


def _joint_ignore_paths(stage: Usd.Stage) -> list[str]:
    """Return exact path expressions for joints omitted from the shadow model.

    The active physics backend supplies every rigid-body pose to the visualization
    state, so the shadow model needs bodies and shapes but not their constraints.
    Omitting joints also lets renderers display legacy assets whose authored joint
    direction is unsupported by Newton's simulation importer.
    """
    return [f"^{re.escape(str(prim.GetPath()))}$" for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]


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
    joint_ignore_paths = _joint_ignore_paths(stage)

    if clone_plan is None:
        # Ignore deformables during USD import; add them as shadow particles below so
        # SceneData mapping and OVRTX registry receive the same particle offsets.
        ignore_paths = [*_deformable_ignore_paths(stage, entries=deformable_entries), *joint_ignore_paths]
        import_result = builder.add_usd(
            stage,
            schema_resolvers=schema_resolvers,
            ignore_paths=ignore_paths or None,
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
    import_result = builder.add_usd(
        stage,
        ignore_paths=["/World/envs", *sources, *deformable_ignore_paths, *joint_ignore_paths],
        schema_resolvers=schema_resolvers,
        skip_mesh_approximation=True,
    )
    _restore_visible_colliders_without_visual_shapes(builder, stage, import_result["path_shape_map"])
    import_builder_visual_material_paths(builder, stage)
    source_deformable_ignore_paths = _deformable_ignore_paths(stage, sources, entries=deformable_entries)
    source_builders = build_source_builders(
        stage,
        sources,
        lambda: ModelBuilder(up_axis=up_axis),
        schema_resolvers,
        ignore_paths=[*source_deformable_ignore_paths, *joint_ignore_paths] or None,
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
