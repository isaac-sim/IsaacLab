# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
from newton import ModelBuilder
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx

from pxr import Usd

from isaaclab.cloner import ClonePlan
from isaaclab.scene_data.deformable_discovery import discover_deformables_on_stage

from isaaclab_newton.cloner.newton_clone_utils import (
    _restore_visible_colliders_without_visual_shapes,
    build_source_builders,
    replicate_builder_mapping,
)
from isaaclab_newton.physics.visualization_deformables import add_shadow_deformables_to_builder
from isaaclab_newton.renderers.visual_material import import_builder_visual_material_paths


def build_visualization_builder_from_plan(
    stage: Usd.Stage,
    plan: ClonePlan,
    rows: tuple[int, ...],
    *,
    up_axis: str = "Z",
    device: str = "cpu",
) -> tuple[ModelBuilder, tuple[list, list]]:
    """Build a Newton visualization model from declared clone sources and shared roots.

    Args:
        stage: USD stage containing the declared source assets.
        plan: Complete replication layout.
        rows: Plan rows routed to the Newton visualization model.
        up_axis: Model up axis.
        device: Warp device for sim-to-visual deformable remap tables.

    Returns:
        Populated builder and shadow-deformable metadata ``(entities, registry_groups)``.
    """
    sources = tuple(plan.sources[row] for row in rows)
    schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]
    entries = discover_deformables_on_stage(stage, root_paths=(*sources, *plan.global_paths))
    ignore_paths = list(
        dict.fromkeys(path for entry in entries for path in (entry.root_path, entry.sim_mesh_path, entry.vis_mesh_path))
    )
    global_builder = ModelBuilder(up_axis=up_axis)
    for root_path in plan.global_paths:
        result = global_builder.add_usd(
            stage,
            root_path=root_path,
            schema_resolvers=schema_resolvers,
            ignore_paths=[*plan.sources, *ignore_paths],
            skip_mesh_approximation=True,
        )
        _restore_visible_colliders_without_visual_shapes(global_builder, stage, result["path_shape_map"])
    import_builder_visual_material_paths(global_builder, stage)
    source_builders = build_source_builders(
        stage,
        sources,
        lambda: ModelBuilder(up_axis=up_axis),
        schema_resolvers,
        ignore_paths=ignore_paths,
        skip_mesh_approximation=True,
    )
    builder = ModelBuilder(up_axis=up_axis)  # Preserve Newton's compact empty filter store.
    for visual_builder in (global_builder, *source_builders.values()):
        visual_builder.shape_collision_filter_pairs = []
        visual_builder.shape_collision_group[:] = [0] * visual_builder.shape_count
    builder.add_builder(global_builder)
    quaternions = np.zeros((len(plan.env_ids), 4), dtype=np.float32)
    quaternions[:, 3] = 1.0
    replicate_builder_mapping(
        builder=builder,
        sources=sources,
        mapping=plan.clone_mask[list(rows)],
        positions=plan.positions,
        quaternions=quaternions,
        source_builders=source_builders,
        destinations=tuple(plan.destinations[row] for row in rows),
        env_ids=plan.env_ids,
    )
    return builder, add_shadow_deformables_to_builder(builder, stage, entries, plan, device=device)
