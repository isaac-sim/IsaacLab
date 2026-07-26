# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from newton import ModelBuilder
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx

from pxr import Usd

from isaaclab.sim.utils.transforms import resolve_prim_pose

from isaaclab_newton.cloner.newton_clone_utils import (
    _restore_visible_colliders_without_visual_shapes,
    build_source_builders,
    rename_builder_labels,
    replicate_builder_mapping,
)


def build_visualization_builder_from_stage_envs(
    stage: Usd.Stage,
    env_paths: Sequence[tuple[int, str]],
    clone_plan: Any | None,
    *,
    up_axis: str = "Z",
) -> ModelBuilder:
    """Build a Newton shadow visualization builder from a USD stage.

    Cloned scenes use the clone plan to preserve per-environment world layout and
    labels. Standalone scenes without a clone plan are imported as one world.
    """
    schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]
    builder = ModelBuilder(up_axis=up_axis)
    if clone_plan is None:
        import_result = builder.add_usd(stage, schema_resolvers=schema_resolvers)
        _restore_visible_colliders_without_visual_shapes(builder, stage, import_result["path_shape_map"])
        return builder
    if not env_paths:
        raise ValueError("clone plan requires at least one environment path")

    env_path_by_id = dict(env_paths)

    sources = tuple(clone_plan.sources)
    destinations = tuple(clone_plan.destinations)
    env_ids = clone_plan.env_ids.detach().cpu()
    mapping = clone_plan.clone_mask.detach().cpu()

    poses = [resolve_prim_pose(stage.GetPrimAtPath(env_path_by_id[int(env_id)])) for env_id in env_ids.tolist()]
    positions = torch.tensor([pos for pos, _ in poses], dtype=torch.float32)
    quaternions = torch.tensor([quat for _, quat in poses], dtype=torch.float32)
    import_result = builder.add_usd(stage, ignore_paths=["/World/envs", *sources], schema_resolvers=schema_resolvers)
    _restore_visible_colliders_without_visual_shapes(builder, stage, import_result["path_shape_map"])
    source_builders = build_source_builders(
        stage,
        sources,
        lambda: ModelBuilder(up_axis=up_axis),
        schema_resolvers,
        simplify_meshes=False,
    )
    replicate_builder_mapping(builder, sources, mapping, positions, quaternions, source_builders)
    rename_builder_labels(builder, sources, destinations, env_ids, mapping)
    return builder
