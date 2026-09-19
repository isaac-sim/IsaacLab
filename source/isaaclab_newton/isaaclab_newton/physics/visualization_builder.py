# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
from newton import ModelBuilder
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx

from pxr import Usd, UsdPhysics

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
    stage: Usd.Stage, plan: ClonePlan, *, up_axis: str = "Z", device: str = "cpu"
) -> tuple[ModelBuilder, tuple[list, list]]:
    """Build the renderer's Newton resource from declared clone sources and shared roots."""
    schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]
    entries = discover_deformables_on_stage(stage, root_paths=(*plan.sources, *plan.global_paths))
    ignore_paths = list(
        dict.fromkeys(path for entry in entries for path in (entry.root_path, entry.sim_mesh_path, entry.vis_mesh_path))
    )
    global_builder = ModelBuilder(up_axis=up_axis)
    for root_path in plan.global_paths:
        result = global_builder.add_usd(
            stage,
            root_path=root_path,
            ignore_paths=ignore_paths,
            schema_resolvers=schema_resolvers,
            skip_mesh_approximation=True,
        )
        _restore_visible_colliders_without_visual_shapes(global_builder, stage, result["path_shape_map"])
    import_builder_visual_material_paths(global_builder, stage)
    source_builders = build_source_builders(
        stage,
        plan.sources,
        lambda: ModelBuilder(up_axis=up_axis),
        schema_resolvers,
        ignore_paths=ignore_paths,
        skip_mesh_approximation=True,
    )
    builder = ModelBuilder(up_axis=up_axis)
    for source_builder in (global_builder, *source_builders.values()):
        source_builder.shape_collision_filter_pairs = []
        source_builder.shape_collision_group[:] = [0] * source_builder.shape_count
        # Bind importer-generated joint labels to actual rigid-body prims before replication.
        for index, label in enumerate(source_builder.body_label):
            prim = stage.GetPrimAtPath(label)
            if prim.IsValid() and prim.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(prim)
                targets = (*joint.GetBody1Rel().GetTargets(), *joint.GetBody0Rel().GetTargets())
                for target in targets:
                    target_prim = stage.GetPrimAtPath(target)
                    if target_prim.IsValid() and target_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        source_builder.body_label[index] = str(target)
                        break
    builder.add_builder(global_builder)
    quaternions = np.zeros((len(plan.env_ids), 4), dtype=np.float32)
    quaternions[:, 3] = 1.0
    replicate_builder_mapping(
        builder=builder,
        sources=plan.sources,
        mapping=plan.clone_mask,
        positions=plan.positions,
        quaternions=quaternions,
        source_builders=source_builders,
        destinations=plan.destinations,
        env_ids=plan.env_ids,
    )
    geometry = add_shadow_deformables_to_builder(builder, stage, (), device=device, entries=entries, clone_plan=plan)
    return builder, geometry
