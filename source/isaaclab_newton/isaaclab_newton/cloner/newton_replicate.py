# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warp as wp
from newton import ModelBuilder, solvers

from pxr import Usd, UsdGeom

from isaaclab_newton.physics import NewtonManager


def _proto_env_mappings(
    sources: list[str],
    destinations: list[str],
    mapping: torch.Tensor,
    env_ids: torch.Tensor,
) -> list[tuple[str, str, dict[int, int]]]:
    """Map each prototype source to its destination template and per-world env IDs.

    Returns one ``(src_prefix, dest_template, world_to_env)`` tuple per source,
    where *world_to_env* maps Newton world IDs to environment IDs for worlds
    that contain that source.  Used by both :func:`_cl_inject_sites` (to
    translate sensor body patterns into prototype-local paths) and the rename
    loop (to rewrite labels from prototype paths to per-env paths).
    """
    result: list[tuple[str, str, dict[int, int]]] = []
    for i, src_path in enumerate(sources):
        src_prefix = src_path.rstrip("/")
        dest_template = destinations[i]
        world_cols = torch.nonzero(mapping[i], as_tuple=True)[0].tolist()
        world_to_env = {c: int(env_ids[c]) for c in world_cols}
        result.append((src_prefix, dest_template, world_to_env))
    return result


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
    builder = NewtonManager.create_builder(up_axis=up_axis)
    stage_info = builder.add_usd(stage, ignore_paths=["/World/envs"] + sources)

    # build a prototype for each source
    proto_builders: dict[str, ModelBuilder] = {}
    for src_path in sources:
        p = NewtonManager.create_builder(up_axis=up_axis)
        solvers.SolverMuJoCo.register_custom_attributes(p)
        inverse_env_xform = get_inverse_env_xform(stage, src_path)
        p.add_usd(
            stage,
            root_path=src_path,
            load_visual_shapes=True,
            skip_mesh_approximation=True,
            xform=inverse_env_xform,
        )
        if simplify_meshes:
            p.approximate_meshes("convex_hull", keep_visual_shapes=True)
        proto_builders[src_path] = p

    # Shared mapping used by both site injection and renaming
    proto_env_map = _proto_env_mappings(sources, destinations, mapping, env_ids)

    # Inject registered sites into prototypes (and global sites into main builder)
    NewtonManager._cl_inject_sites(builder, proto_builders, proto_env_map)

    # create a separate world for each environment (heterogeneous spawning)
    for col, env_id in enumerate(env_ids.tolist()):
        builder.begin_world()

        for row in torch.nonzero(mapping[:, col], as_tuple=True)[0].tolist():
            builder.add_builder(
                proto_builders[sources[row]],
                xform=wp.transform(positions[col].tolist(), quaternions[col].tolist()),
            )

        builder.end_world()

    # per-source, per-world renaming (strict prefix swap)
    for src_prefix, dest_template, world_to_env in proto_env_map:
        src_len = len(src_prefix)
        for t in ("body", "joint", "shape", "articulation"):
            labels = getattr(builder, f"{t}_label", None)
            if labels is None:
                labels = getattr(builder, f"{t}_key")
            worlds_arr = getattr(builder, f"{t}_world")
            for k, w in enumerate(worlds_arr):
                if w in world_to_env and labels[k].startswith(src_prefix):
                    labels[k] = dest_template.format(world_to_env[w]) + labels[k][src_len:]

    NewtonManager.set_builder(builder)
    NewtonManager._num_envs = mapping.size(1)
    return builder, stage_info


def get_inverse_env_xform(stage, src_path: str):
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
