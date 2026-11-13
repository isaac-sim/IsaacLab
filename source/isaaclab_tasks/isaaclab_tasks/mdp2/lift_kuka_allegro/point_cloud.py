# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Surface point clouds of the objects a scene spawned, sampled once in each object's frame."""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner


def sample_object_point_cloud(asset, num_points: int) -> torch.Tensor:
    """Return ``num_points`` surface points of the object of each world, ``(num_worlds, num_points, 3)``.

    Each object that a multi-asset spawner can choose is sampled once: area-weighted surface samples, twice as many
    as needed, thinned by farthest point sampling. Worlds that spawned the same object get the same points.
    """
    import trimesh  # noqa: PLC0415

    from pxr import UsdGeom  # noqa: PLC0415

    from isaaclab.utils.mesh import (  # noqa: PLC0415
        PRIMITIVE_MESH_TYPES,
        create_trimesh_from_geom_mesh,
        create_trimesh_from_geom_shape,
    )

    points = torch.zeros(asset.num_instances, num_points, 3, device=asset.device)
    stage = sim_utils.get_current_stage()
    xform_cache = UsdGeom.XformCache()
    plan = sim_utils.SimulationContext.instance().get_clone_plan()
    source_paths = cloner.path.get_asset_prototype_paths(plan)

    for index in cloner.path.get_asset_prototypes(plan, asset.cfg.prim_path):
        env_ids, _ = cloner.query.get_asset_prototype_unique_world_index(plan.topology, index)
        if not len(env_ids):
            continue
        root = stage.GetPrimAtPath(source_paths[index])
        root_from_world = xform_cache.GetLocalToWorldTransform(root).GetInverse()
        prims = sim_utils.get_all_matching_child_prims(
            source_paths[index], predicate=lambda prim: prim.GetTypeName() in [*PRIMITIVE_MESH_TYPES, "Mesh"]
        )

        # Sample each shape, then thin the samples of all shapes together.
        samples = []
        for prim in prims:
            is_mesh = prim.GetTypeName() == "Mesh"
            mesh = create_trimesh_from_geom_mesh(prim) if is_mesh else create_trimesh_from_geom_shape(prim)
            surface, _ = trimesh.sample.sample_surface(mesh, 2 * num_points)
            surface = torch.from_numpy(surface.astype(np.float32))
            surface = surface[_farthest_points(surface, num_points)]

            # Express the samples in the object's frame. USD matrices act on row vectors.
            matrix = xform_cache.GetLocalToWorldTransform(prim) * root_from_world
            matrix = torch.tensor([[matrix[r][c] for c in range(4)] for r in range(4)], dtype=torch.float32)
            surface = torch.cat([surface, torch.ones(len(surface), 1)], dim=1) @ matrix
            samples.append(surface[:, :3])
        samples = torch.cat(samples)
        samples = samples[_farthest_points(samples, num_points)]

        scale = root.GetAttribute("xformOp:scale")
        if scale and scale.Get() is not None:
            samples = samples * torch.tensor(scale.Get(), dtype=torch.float32)
        points[torch.as_tensor(env_ids, device=asset.device)] = samples.to(asset.device)
    return points


def _farthest_points(points: torch.Tensor, count: int) -> torch.Tensor:
    """Return the indices of ``count`` points spread as far apart as possible, starting from a random one."""
    if count >= len(points):
        return torch.arange(len(points))

    distances = torch.cdist(points, points)
    chosen = torch.zeros(count, dtype=torch.long)
    nearest = torch.full((len(points),), float("inf"))
    farthest = torch.randint(len(points), (1,))
    for k in range(count):
        chosen[k] = farthest
        nearest = torch.minimum(nearest, distances[farthest].view(-1))
        farthest = torch.argmax(nearest)
    return chosen
