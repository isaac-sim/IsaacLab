# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warp as wp
from newton import ModelBuilder, solvers
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx

from pxr import Usd


def _build_and_label(
    stage: Usd.Stage,
    sources: list[str],
    destinations: list[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    up_axis: str = "Z",
    simplify_meshes: bool = True,
) -> tuple[ModelBuilder, object]:
    """Build a Newton model builder from clone mapping inputs and rename labels.

    Args:
        stage: USD stage containing source assets.
        sources: Source prim paths used for cloning.
        destinations: Destination prim path templates.
        env_ids: Environment ids for destination worlds.
        mapping: Boolean source-to-environment mapping matrix.
        positions: Optional per-environment world positions.
        quaternions: Optional per-environment orientations in xyzw order.
        up_axis: Up axis for the Newton model builder.
        simplify_meshes: Whether to run convex-hull mesh approximation.

    Returns:
        Tuple of the populated Newton model builder and stage metadata returned by ``add_usd``.
    """
    if positions is None:
        positions = torch.zeros((mapping.size(1), 3), device=mapping.device, dtype=torch.float32)
    if quaternions is None:
        quaternions = torch.zeros((mapping.size(1), 4), device=mapping.device, dtype=torch.float32)
        quaternions[:, 3] = 1.0

    schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]

    builder = ModelBuilder(up_axis=up_axis)
    stage_info = builder.add_usd(
        stage,
        ignore_paths=["/World/envs"] + sources,
        schema_resolvers=schema_resolvers,
    )

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
    for col, _ in enumerate(env_ids.tolist()):
        # begin a new world context (Newton assigns world ID = col)
        builder.begin_world()
        # add all active sources for this world
        delta_pos = (positions[col] - env0_pos).tolist()
        for row in torch.nonzero(mapping[:, col], as_tuple=True)[0].tolist():
            builder.add_builder(
                protos[sources[row]],
                xform=wp.transform(delta_pos, quaternions[col].tolist()),
            )
        # end the world context
        builder.end_world()

    # Rename builder labels from source roots to destination roots
    for i, src_path in enumerate(sources):
        n = len(src_path.rstrip("/"))
        world_cols = torch.nonzero(mapping[i], as_tuple=True)[0].tolist()
        world_roots = {int(env_ids[c]): destinations[i].format(int(env_ids[c])) for c in world_cols}
        for t in ("body", "joint", "shape", "articulation"):
            labels = getattr(builder, f"{t}_label", None) or getattr(builder, f"{t}_key")
            for k, w in enumerate(getattr(builder, f"{t}_world")):
                wid = int(w)
                if wid in world_roots and labels[k].startswith(src_path):
                    labels[k] = world_roots[wid] + labels[k][n:]

    return builder, stage_info


def newton_physics_replicate(
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
    """Replicate prims into a Newton ``ModelBuilder`` using a per-source mapping.

    Args:
        stage: USD stage containing source assets.
        sources: Source prim paths used for cloning.
        destinations: Destination prim path templates.
        env_ids: Environment ids for destination worlds.
        mapping: Boolean source-to-environment mapping matrix.
        positions: Optional per-environment world positions.
        quaternions: Optional per-environment orientations in xyzw order.
        device: Device used by the finalized Newton model builder.
        up_axis: Up axis for the Newton model builder.
        simplify_meshes: Whether to run convex-hull mesh approximation.

    Returns:
        Tuple of the populated Newton model builder and stage metadata.
    """
    from isaaclab_newton.physics import NewtonManager

    builder, stage_info = _build_and_label(
        stage, sources, destinations, env_ids, mapping,
        positions, quaternions, up_axis, simplify_meshes,
    )
    NewtonManager.set_builder(builder)
    NewtonManager._num_envs = mapping.size(1)
    return builder, stage_info


def newton_visualizer_prebuild(
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
    """Replicate a clone plan into a finalized Newton model/state for visualization.

    Unlike :func:`newton_physics_replicate`, this path does not mutate ``NewtonManager`` and is intended
    for prebuilding visualizer-only artifacts that can be consumed by scene data providers.

    Args:
        stage: USD stage containing source assets.
        sources: Source prim paths used for cloning.
        destinations: Destination prim path templates.
        env_ids: Environment ids for destination worlds.
        mapping: Boolean source-to-environment mapping matrix.
        positions: Optional per-environment world positions.
        quaternions: Optional per-environment orientations in xyzw order.
        device: Device used by the finalized Newton model.
        up_axis: Up axis for the Newton model builder.
        simplify_meshes: Whether to run convex-hull mesh approximation.

    Returns:
        Tuple of finalized Newton model and state.
    """
    builder, _ = _build_and_label(
        stage, sources, destinations, env_ids, mapping,
        positions, quaternions, up_axis, simplify_meshes,
    )
    model = builder.finalize(device=device)
    state = model.state()
    return model, state
