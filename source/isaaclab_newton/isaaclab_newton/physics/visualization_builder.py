# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import warp as wp
from newton import Axis, ModelBuilder
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx

from pxr import Usd, UsdGeom

from isaaclab.sim.utils.transforms import resolve_prim_pose

from isaaclab_newton.cloner.newton_clone_utils import (
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
    """Build the Newton shadow visualization builder from cloned USD environments.

    When ``clone_plan`` is ``None`` (no published ClonePlan -- common on the
    PhysX backend during early renderer initialization, or for scenes that do
    not publish a plan), falls back to an env_0 prototype-replicate path so the
    PhysX-backed visualizer still gets a populated Newton ``Model``. The
    ClonePlan-aware path produces a more accurate model for heterogeneous
    environments; this fallback covers the homogeneous-env case that pre-existed
    PR #6119.
    """
    if clone_plan is None:
        return _build_from_env_0_prototype(stage, env_paths, up_axis=up_axis)

    env_path_by_id = dict(env_paths)

    sources = tuple(clone_plan.sources)
    destinations = tuple(clone_plan.destinations)
    env_ids = clone_plan.env_ids.detach().cpu()
    mapping = clone_plan.clone_mask.detach().cpu()

    poses = [resolve_prim_pose(stage.GetPrimAtPath(env_path_by_id[int(env_id)])) for env_id in env_ids.tolist()]
    positions = torch.tensor([pos for pos, _ in poses], dtype=torch.float32)
    quaternions = torch.tensor([quat for _, quat in poses], dtype=torch.float32)
    schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]
    builder = ModelBuilder(up_axis=up_axis)
    builder.add_usd(stage, ignore_paths=["/World/envs", *sources], schema_resolvers=schema_resolvers)
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


def _build_from_env_0_prototype(
    stage: Usd.Stage,
    env_paths: Sequence[tuple[int, str]],
    *,
    up_axis: str = "Z",
) -> ModelBuilder:
    """Pre-#6119 fallback: build the visualization model from env_0 as a prototype.

    Used when no ``ClonePlan`` is published (e.g. early in the PhysX-backend
    lifecycle, or for scenes that do not publish a plan). Walks the
    ``/World/envs/env_<id>`` convention, builds ``env_0`` as a prototype, then
    replicates it across every env via :meth:`ModelBuilder.add_builder` with the
    correct relative transform. Label arrays are rewritten so each replicated
    world references its own env prim path (otherwise the PhysX ->
    ``state.body_q`` sync resolves every match to world 0 and worlds 1..N never
    receive fresh poses).
    """
    up_axis_enum = Axis.from_string(up_axis) if isinstance(up_axis, str) else up_axis
    schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]

    builder = ModelBuilder(up_axis=up_axis_enum)

    if not env_paths:
        # No ``/World/envs/env_<id>`` prims -- ingest the whole stage as a single world.
        builder.add_usd(stage, schema_resolvers=schema_resolvers)
        return builder

    # Ingest stage-level (non-env) geometry first so visualization sees ground
    # planes, ceilings, fixed props, etc.
    builder.add_usd(
        stage,
        ignore_paths=[r"/World/envs($|/.*)"],
        schema_resolvers=schema_resolvers,
    )

    sorted_envs = sorted(env_paths, key=lambda x: x[0])
    proto_env_path = sorted_envs[0][1]
    proto = ModelBuilder(up_axis=up_axis_enum)
    proto.add_usd(
        stage,
        root_path=proto_env_path,
        schema_resolvers=schema_resolvers,
    )

    xform_cache = UsdGeom.XformCache()

    label_attrs = ("body_label", "articulation_label", "joint_label", "shape_label")
    label_starts = {attr: len(getattr(builder, attr)) for attr in label_attrs}

    proto_world_gf = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(proto_env_path))
    proto_translation = proto_world_gf.ExtractTranslation()
    proto_rotation = proto_world_gf.ExtractRotationQuat()
    proto_world_tf = wp.transform(
        (proto_translation[0], proto_translation[1], proto_translation[2]),
        (
            proto_rotation.GetImaginary()[0],
            proto_rotation.GetImaginary()[1],
            proto_rotation.GetImaginary()[2],
            proto_rotation.GetReal(),
        ),
    )
    proto_world_tf_inv = wp.transform_inverse(proto_world_tf)

    for _, env_path in sorted_envs:
        world_xform = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(env_path))
        translation = world_xform.ExtractTranslation()
        rotation = world_xform.ExtractRotationQuat()
        env_world_tf = wp.transform(
            (translation[0], translation[1], translation[2]),
            (
                rotation.GetImaginary()[0],
                rotation.GetImaginary()[1],
                rotation.GetImaginary()[2],
                rotation.GetReal(),
            ),
        )
        relative_tf = wp.transform_multiply(env_world_tf, proto_world_tf_inv)
        builder.begin_world()
        builder.add_builder(proto, xform=relative_tf)
        if env_path != proto_env_path:
            for attr in label_attrs:
                labels = getattr(builder, attr)
                for i in range(label_starts[attr], len(labels)):
                    labels[i] = labels[i].replace(proto_env_path, env_path, 1)
        for attr in label_attrs:
            label_starts[attr] = len(getattr(builder, attr))
        builder.end_world()

    return builder
