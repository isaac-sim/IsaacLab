# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warp as wp
from newton import ModelBuilder, solvers
from newton.usd import SchemaResolverNewton, SchemaResolverPhysx

from pxr import Usd

from isaaclab_newton.physics import NewtonManager


def newton_replicate(
    stage: Usd.Stage,
    sources: list[str],
    destinations: list[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    device: str = "cpu",
    up_axis: str = "Z",
    simplify_meshes: bool = True,
):
    """Replicate prims into a Newton ``ModelBuilder`` using a per-source mapping."""

    if positions is None:
        positions = torch.zeros((mapping.size(1), 3), device=mapping.device, dtype=torch.float32)
    if quaternions is None:
        quaternions = torch.zeros((mapping.size(1), 4), device=mapping.device, dtype=torch.float32)
        quaternions[:, 3] = 1.0

    # load empty stage
    builder = ModelBuilder(up_axis=up_axis)
    stage_info = builder.add_usd(stage, ignore_paths=["/World/envs"] + sources)

    schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]

    # The prototype is built from env_0 in absolute world coordinates.
    # add_builder xforms are deltas from env_0 so positions don't get double-counted.
    env0_pos = positions[0]
    protos: dict[str, ModelBuilder] = {}
    for src_path in sources:
        p = ModelBuilder(up_axis=up_axis)
        solvers.SolverMuJoCo.register_custom_attributes(p)
        p.add_usd(
            stage,
            root_path=src_path,
            load_visual_shapes=True,
            skip_mesh_approximation=True,
            schema_resolvers=schema_resolvers,
        )
        if simplify_meshes:
            p.approximate_meshes("convex_hull", keep_visual_shapes=True)
        protos[src_path] = p

    # create a separate world for each environment (heterogeneous spawning)
    # Newton assigns sequential world IDs (0, 1, 2, ...), so we need to track the mapping
    newton_world_to_env_id = {}
    for col, env_id in enumerate(env_ids.tolist()):
        # begin a new world context (Newton assigns world ID = col)
        builder.begin_world()
        newton_world_to_env_id[col] = env_id

        # add all active sources for this world
        delta_pos = (positions[col] - env0_pos).tolist()
        for row in torch.nonzero(mapping[:, col], as_tuple=True)[0].tolist():
            builder.add_builder(
                protos[sources[row]],
                xform=wp.transform(delta_pos, quaternions[col].tolist()),
            )

        # end the world context
        builder.end_world()

    # per-source, per-world renaming (strict prefix swap), compact style preserved
    for i, src_path in enumerate(sources):
        src_prefix_len = len(src_path.rstrip("/"))
        swap = lambda name, new_root: new_root + name[src_prefix_len:]  # noqa: E731
        world_cols = torch.nonzero(mapping[i], as_tuple=True)[0].tolist()
        # Map Newton world IDs (sequential) to destination paths using env_ids
        world_roots = {int(env_ids[c]): destinations[i].format(int(env_ids[c])) for c in world_cols}

        for t in ("body", "joint", "shape", "articulation"):
            labels = getattr(builder, f"{t}_label", None)
            if labels is None:
                labels = getattr(builder, f"{t}_key")
            worlds_arr = getattr(builder, f"{t}_world")
            for k, w in enumerate(worlds_arr):
                world_id = int(w)
                if world_id in world_roots and labels[k].startswith(src_path):
                    labels[k] = swap(labels[k], world_roots[world_id])

    NewtonManager.set_builder(builder)
    NewtonManager._num_envs = mapping.size(1)
    return builder, stage_info
