# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warp as wp
from newton import ModelBuilder, solvers
from newton.usd import SchemaResolverNewton, SchemaResolverPhysx

from pxr import Usd, UsdGeom

from isaaclab_newton.physics import NewtonManager


def _build_builder_from_mapping(
    stage: Usd.Stage,
    sources: list[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None,
    quaternions: torch.Tensor | None,
    up_axis: str,
    simplify_meshes: bool,
    register_custom_attributes: bool,
) -> tuple[ModelBuilder, object]:
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

    # Build one local prototype per source. These are added into each world according to mapping.
    protos: dict[str, ModelBuilder] = {}
    for src_path in sources:
        p = ModelBuilder(up_axis=up_axis)
        if register_custom_attributes:
            solvers.SolverMuJoCo.register_custom_attributes(p)
        inverse_env_xform = _get_inverse_env_xform(stage, src_path)
        p.add_usd(
            stage,
            root_path=src_path,
            load_visual_shapes=True,
            skip_mesh_approximation=True,
            xform=inverse_env_xform,
            schema_resolvers=schema_resolvers,
        )
        if simplify_meshes:
            p.approximate_meshes("convex_hull", keep_visual_shapes=True)
        protos[src_path] = p

    # Newton world IDs are sequential by begin_world/end_world order.
    for col, _env_id in enumerate(env_ids.tolist()):
        builder.begin_world()
        for row in torch.nonzero(mapping[:, col], as_tuple=True)[0].tolist():
            builder.add_builder(
                protos[sources[row]],
                xform=wp.transform(positions[col].tolist(), quaternions[col].tolist()),
            )
        builder.end_world()

    return builder, stage_info


def _rename_builder_labels(
    builder: ModelBuilder, sources: list[str], destinations: list[str], env_ids: torch.Tensor, mapping: torch.Tensor
) -> None:
    # Per-source, per-world renaming (strict prefix swap).
    for i, src_path in enumerate(sources):
        src_prefix_len = len(src_path.rstrip("/"))
        swap = lambda name, new_root: new_root + name[src_prefix_len:]  # noqa: E731
        world_cols = torch.nonzero(mapping[i], as_tuple=True)[0].tolist()
        # Map Newton world index (column) to the destination root path for that world's env id.
        world_roots = {c: destinations[i].format(int(env_ids[c])) for c in world_cols}

        for t in ("body", "joint", "shape", "articulation"):
            labels = getattr(builder, f"{t}_label", None)
            if labels is None:
                labels = getattr(builder, f"{t}_key")
            worlds_arr = getattr(builder, f"{t}_world")
            for k, w in enumerate(worlds_arr):
                world_id = int(w)
                if world_id in world_roots and labels[k].startswith(src_path):
                    labels[k] = swap(labels[k], world_roots[world_id])


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
    """Replicate prims into a Newton ``ModelBuilder`` using a per-source mapping."""
    builder, stage_info = _build_builder_from_mapping(
        stage=stage,
        sources=sources,
        env_ids=env_ids,
        mapping=mapping,
        positions=positions,
        quaternions=quaternions,
        up_axis=up_axis,
        simplify_meshes=simplify_meshes,
        register_custom_attributes=True,
    )
    _rename_builder_labels(builder, sources, destinations, env_ids, mapping)

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
    """
    builder, _ = _build_builder_from_mapping(
        stage=stage,
        sources=sources,
        env_ids=env_ids,
        mapping=mapping,
        positions=positions,
        quaternions=quaternions,
        up_axis=up_axis,
        simplify_meshes=simplify_meshes,
        register_custom_attributes=False,
    )
    _rename_builder_labels(builder, sources, destinations, env_ids, mapping)
    model = builder.finalize(device=device)
    state = model.state()
    return model, state


def _get_inverse_env_xform(stage: Usd.Stage, src_path: str):
    """Get the inverse transform of src_path to convert world→local."""
    xform_cache = UsdGeom.XformCache()
    world_xform = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(src_path))

    # Get the inverse of the world transform
    inv_xform = world_xform.GetInverse()

    # Extract translation and rotation from inverse
    inv_translation = inv_xform.ExtractTranslation()
    inv_rotation = inv_xform.ExtractRotationQuat()

    inv_pos = (inv_translation[0], inv_translation[1], inv_translation[2])
    inv_quat = (
        inv_rotation.GetImaginary()[0],
        inv_rotation.GetImaginary()[1],
        inv_rotation.GetImaginary()[2],
        inv_rotation.GetReal(),
    )

    return wp.transform(inv_pos, inv_quat)
