# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
import warp as wp
from newton import ModelBuilder, solvers

from pxr import Usd

from isaaclab.cloner.cloner_utils import replace_path_prefix
from isaaclab.sim.utils.newton_model_utils import replace_newton_builder_shape_colors


def build_source_builders(
    stage: Usd.Stage,
    sources: Sequence[str],
    create_builder: Callable[[], ModelBuilder],
    schema_resolvers: Sequence[Any],
    *,
    ignore_paths: Sequence[str] | None = None,
    simplify_meshes: bool = True,
) -> dict[str, ModelBuilder]:
    """Build one Newton builder for each clone source prim path."""
    builders: dict[str, ModelBuilder] = {}
    for source in sources:
        builder = create_builder()
        solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(
            stage,
            root_path=source,
            load_visual_shapes=True,
            skip_mesh_approximation=True,
            schema_resolvers=schema_resolvers,
            ignore_paths=ignore_paths,
        )
        if simplify_meshes:
            builder.approximate_meshes("convex_hull", keep_visual_shapes=True)
        replace_newton_builder_shape_colors(builder, stage)
        builders[source] = builder
    return builders


def replicate_builder_mapping(
    builder: ModelBuilder,
    sources: Sequence[str],
    mapping: torch.Tensor,
    positions: torch.Tensor,
    quaternions: torch.Tensor,
    source_builders: dict[str, ModelBuilder],
    *,
    source_site_indices: dict[int, dict[str, list[int]]] | None = None,
    env_root_sites: dict[str, wp.transform] | None = None,
    per_world_builder_hooks: Sequence[Callable[[ModelBuilder, int, list[float], list[float]], None]] = (),
    post_replicate_hooks: Sequence[Callable[[ModelBuilder], None]] = (),
) -> tuple[dict[str, list[list[int]]], list[wp.transform]]:
    """Replicate source builders into per-env Newton worlds."""
    source_site_indices = source_site_indices or {}
    env_root_sites = env_root_sites or {}
    num_worlds = mapping.size(1)
    local_site_map: dict[str, list[list[int]]] = {}
    world_xforms: list[wp.transform] = []
    source_world_indices = mapping.to(dtype=torch.int64).argmax(dim=1)

    for col in range(num_worlds):
        builder.begin_world()
        world_xform = wp.transform(positions[col], quaternions[col])
        world_xforms.append(world_xform)

        for label, xform in env_root_sites.items():
            site_idx = builder.add_site(body=-1, xform=wp.transform_multiply(world_xform, xform), label=label)
            local_site_map.setdefault(label, [[] for _ in range(num_worlds)])[col].append(site_idx)

        for row in torch.nonzero(mapping[:, col], as_tuple=True)[0].tolist():
            source_builder = source_builders[sources[int(row)]]
            offset = builder.shape_count
            source_col = int(source_world_indices[int(row)])
            source_xform = wp.transform(positions[source_col], quaternions[source_col])
            builder.add_builder(
                source_builder, xform=wp.transform_multiply(world_xform, wp.transform_inverse(source_xform))
            )

            for label, source_shape_indices in source_site_indices.get(id(source_builder), {}).items():
                local_indices = local_site_map.setdefault(label, [[] for _ in range(num_worlds)])[col]
                local_indices.extend(offset + shape_idx for shape_idx in source_shape_indices)

        for hook in per_world_builder_hooks:
            hook(builder, col, positions[col].tolist(), quaternions[col].tolist())
        builder.end_world()

    for hook in post_replicate_hooks:
        hook(builder)
    return local_site_map, world_xforms


_BUILTIN_LABEL_TYPES: tuple[str, ...] = (
    "body",
    "joint",
    "shape",
    "articulation",
    "constraint_mimic",
    "equality_constraint",
)


def rename_builder_labels(
    builder: ModelBuilder,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
) -> list[tuple[str, int]]:
    """Rewrite source-root labels to per-env destination roots and return Fabric body bindings."""
    fabric_body_bindings: list[tuple[str, int]] = []
    bound_body_indices: set[int] = set()

    for source_index, source in enumerate(sources):
        source_root = source.rstrip("/")
        world_cols = torch.nonzero(mapping[source_index], as_tuple=True)[0].tolist()
        world_roots = {int(env_ids[col]): destinations[source_index].format(int(env_ids[col])) for col in world_cols}

        def _rename_pair(values, worlds, *, collect_body_bindings: bool = False):
            for index, (value, world) in enumerate(zip(values, worlds, strict=True)):
                if world is None:
                    continue
                world_root = world_roots.get(int(world))
                if isinstance(value, str) and world_root is not None:
                    renamed_value = replace_path_prefix(value, source_root, world_root)
                    if renamed_value != value:
                        values[index] = renamed_value
                        if collect_body_bindings:
                            fabric_body_bindings.append((renamed_value, index))
                            bound_body_indices.add(index)

        for labels, worlds, collect_body_bindings in (
            (builder.body_label, builder.body_world, True),
            (builder.joint_label, builder.joint_world, False),
            (builder.shape_label, builder.shape_world, False),
            (builder.articulation_label, builder.articulation_world, False),
            (builder.constraint_mimic_label, builder.constraint_mimic_world, False),
        ):
            _rename_pair(labels, worlds, collect_body_bindings=collect_body_bindings)

        if "mujoco:equality_constraint_label" not in builder.custom_attributes:
            _rename_pair(builder.equality_constraint_label, builder.equality_constraint_world)

        custom_attrs = builder.custom_attributes.values()
        worlds_by_freq = {attr.frequency: attr.values for attr in custom_attrs if attr.references == "world"}
        for attr in custom_attrs:
            if attr.dtype is str and attr.values and (worlds := worlds_by_freq.get(attr.frequency)):
                _rename_pair(attr.values, worlds)

    fabric_body_bindings.extend(
        (label, index) for index, label in enumerate(builder.body_label) if index not in bound_body_indices
    )
    return fabric_body_bindings
