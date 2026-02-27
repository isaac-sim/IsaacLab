# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import warp as wp
import newton
from pxr import Usd, UsdGeom, UsdPhysics


def newton_replicate(
    stage: Usd.Stage,
    sources: list[str],
    destinations: list[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    up_axis: str = "Z",
    simplify_meshes: bool = True,
):
    """Replicate prims into a Newton ``ModelBuilder`` using a per-source mapping."""
    from isaaclab_newton.physics import NewtonManager
    from newton import ModelBuilder, solvers

    if positions is None:
        positions = torch.zeros((mapping.size(1), 3), device=mapping.device, dtype=torch.float32)
    if quaternions is None:
        quaternions = torch.zeros((mapping.size(1), 4), device=mapping.device, dtype=torch.float32)
        quaternions[:, 3] = 1.0

    # load empty stage
    builder = ModelBuilder(up_axis=up_axis)
    builder.rigid_contact_margin = 0.001
    builder.default_shape_cfg.contact_margin = 0.001
    stage_info = builder.add_usd(stage, ignore_paths=["/World/envs"] + sources)

    # build a prototype for each source
    protos: dict[str, ModelBuilder] = {}
    for src_path in sources:
        p = ModelBuilder(up_axis=up_axis)
        p.rigid_contact_margin = 0.001
        p.default_shape_cfg.contact_margin = 0.001
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
            # Check if SDF patterns are configured — skip approximation for matching shapes
            import re
            from isaaclab.physics import PhysicsManager

            sdf_patterns = None
            cfg = PhysicsManager._cfg
            if cfg is not None:
                sdf_pats = getattr(cfg, "sdf_shape_patterns", None)
                if sdf_pats is not None and (
                    getattr(cfg, "sdf_max_resolution", None) is not None
                    or getattr(cfg, "sdf_target_voxel_size", None) is not None
                ):
                    sdf_patterns = [re.compile(pat) for pat in sdf_pats]

            # Split shapes by USD collision approximation: convexDecomposition → coacd, everything else → convex_hull
            decomp_indices = []
            hull_indices = []
            for i in range(len(p.shape_type)):
                if p.shape_type[i] != newton.GeoType.MESH:
                    continue
                key = p.shape_key[i] if i < len(p.shape_key) else ""
                # Skip shapes matching SDF patterns — SDF uses original mesh, no convex approx needed
                if sdf_patterns is not None and any(pat.search(key) for pat in sdf_patterns):
                    continue
                prim = stage.GetPrimAtPath(key)
                approx = ""
                if prim and prim.IsValid():
                    if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                        approx = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or ""
                if approx == "convexDecomposition":
                    decomp_indices.append(i)
                else:
                    hull_indices.append(i)
            if decomp_indices:
                p.approximate_meshes("coacd", shape_indices=decomp_indices, keep_visual_shapes=True, threshold=0.1, mcts_iterations=1, mcts_max_depth=3)
            if hull_indices:
                p.approximate_meshes("convex_hull", shape_indices=hull_indices, keep_visual_shapes=True)
        protos[src_path] = p

    # create a separate world for each environment (heterogeneous spawning)
    # Newton assigns sequential world IDs (0, 1, 2, ...), so we need to track the mapping
    newton_world_to_env_id = {}
    for col, env_id in enumerate(env_ids.tolist()):
        # begin a new world context (Newton assigns world ID = col)
        builder.begin_world()
        newton_world_to_env_id[col] = env_id

        # add all active sources for this world
        for row in torch.nonzero(mapping[:, col], as_tuple=True)[0].tolist():
            builder.add_builder(
                protos[sources[row]],
                xform=wp.transform(positions[col].tolist(), quaternions[col].tolist()),
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
            keys, worlds_arr = getattr(builder, f"{t}_key"), getattr(builder, f"{t}_world")
            for k, w in enumerate(worlds_arr):
                if w in world_roots and keys[k].startswith(src_path):
                    keys[k] = swap(keys[k], world_roots[w])

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
