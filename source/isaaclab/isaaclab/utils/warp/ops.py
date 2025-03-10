# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapping around warp kernels for compatibility with torch tensors."""

# needed to import for allowing type-hinting: torch.Tensor | None
from __future__ import annotations

import numpy as np
import torch

import warp as wp

# disable warp module initialization messages
wp.config.quiet = True
# initialize the warp module
wp.init()

from . import kernels


def raycast_mesh(
    ray_starts: torch.Tensor,
    ray_directions: torch.Tensor,
    mesh: wp.Mesh,
    max_dist: float = 1e6,
    return_distance: bool = False,
    return_normal: bool = False,
    return_face_id: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Performs ray-casting against a mesh.

    Note that the `ray_starts` and `ray_directions`, and `ray_hits` should have compatible shapes
    and data types to ensure proper execution. Additionally, they all must be in the same frame.

    Args:
        ray_starts: The starting position of the rays. Shape (N, 3).
        ray_directions: The ray directions for each ray. Shape (N, 3).
        mesh: The warp mesh to ray-cast against.
        max_dist: The maximum distance to ray-cast. Defaults to 1e6.
        return_distance: Whether to return the distance of the ray until it hits the mesh. Defaults to False.
        return_normal: Whether to return the normal of the mesh face the ray hits. Defaults to False.
        return_face_id: Whether to return the face id of the mesh face the ray hits. Defaults to False.

    Returns:
        The ray hit position. Shape (N, 3).
            The returned tensor contains :obj:`float('inf')` for missed hits.
        The ray hit distance. Shape (N,).
            Will only return if :attr:`return_distance` is True, else returns None.
            The returned tensor contains :obj:`float('inf')` for missed hits.
        The ray hit normal. Shape (N, 3).
            Will only return if :attr:`return_normal` is True else returns None.
            The returned tensor contains :obj:`float('inf')` for missed hits.
        The ray hit face id. Shape (N,).
            Will only return if :attr:`return_face_id` is True else returns None.
            The returned tensor contains :obj:`int(-1)` for missed hits.
    """
    # extract device and shape information
    shape = ray_starts.shape
    device = ray_starts.device
    # device of the mesh
    torch_device = wp.device_to_torch(mesh.device)
    # reshape the tensors
    ray_starts = ray_starts.to(torch_device).view(-1, 3).contiguous()
    ray_directions = ray_directions.to(torch_device).view(-1, 3).contiguous()
    num_rays = ray_starts.shape[0]
    # create output tensor for the ray hits
    ray_hits = torch.full((num_rays, 3), float("inf"), device=torch_device).contiguous()

    # map the memory to warp arrays
    ray_starts_wp = wp.from_torch(ray_starts, dtype=wp.vec3)
    ray_directions_wp = wp.from_torch(ray_directions, dtype=wp.vec3)
    ray_hits_wp = wp.from_torch(ray_hits, dtype=wp.vec3)

    if return_distance:
        ray_distance = torch.full((num_rays,), float("inf"), device=torch_device).contiguous()
        ray_distance_wp = wp.from_torch(ray_distance, dtype=wp.float32)
    else:
        ray_distance = None
        ray_distance_wp = wp.empty((1,), dtype=wp.float32, device=torch_device)

    if return_normal:
        ray_normal = torch.full((num_rays, 3), float("inf"), device=torch_device).contiguous()
        ray_normal_wp = wp.from_torch(ray_normal, dtype=wp.vec3)
    else:
        ray_normal = None
        ray_normal_wp = wp.empty((1,), dtype=wp.vec3, device=torch_device)

    if return_face_id:
        ray_face_id = torch.ones((num_rays,), dtype=torch.int32, device=torch_device).contiguous() * (-1)
        ray_face_id_wp = wp.from_torch(ray_face_id, dtype=wp.int32)
    else:
        ray_face_id = None
        ray_face_id_wp = wp.empty((1,), dtype=wp.int32, device=torch_device)

    # launch the warp kernel
    wp.launch(
        kernel=kernels.raycast_mesh_kernel,
        dim=num_rays,
        inputs=[
            mesh.id,
            ray_starts_wp,
            ray_directions_wp,
            ray_hits_wp,
            ray_distance_wp,
            ray_normal_wp,
            ray_face_id_wp,
            float(max_dist),
            int(return_distance),
            int(return_normal),
            int(return_face_id),
        ],
        device=mesh.device,
    )
    # NOTE: Synchronize is not needed anymore, but we keep it for now. Check with @dhoeller.
    wp.synchronize()

    if return_distance:
        ray_distance = ray_distance.to(device).view(shape[0], shape[1])
    if return_normal:
        ray_normal = ray_normal.to(device).view(shape)
    if return_face_id:
        ray_face_id = ray_face_id.to(device).view(shape[0], shape[1])

    return ray_hits.to(device).view(shape), ray_distance, ray_normal, ray_face_id


def multi_raycast_mesh(
    ray_starts: torch.Tensor,
    ray_directions: torch.Tensor,
    env_offsets: torch.Tensor,
    mesh_ids: torch.Tensor,
    rays_per_env: int,
    max_dist: float = 1e6,
    return_distance: bool = False,
    return_normal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Performs batched ray-casting using multiple meshes across environments.

    This function launches the warp kernel `multi_mesh_raycast_kernel` to process a batch of rays
    grouped by environment. Each ray is cast against all meshes in its environment and the closest
    hit is recorded.

    Args:
        ray_starts: The starting positions of the rays. Shape (N, 3), where N is the total number of rays.
        ray_directions: The normalized ray direction vectors. Shape (N, 3).
        env_offsets: Offsets delineating the mesh grouping per environment. Shape (num_envs + 1,). Each pair of
            consecutive entries defines the range of mesh indices for that environment.
        mesh_ids: Sorted warp mesh IDs corresponding to the meshes in each environment.
        rays_per_env: The number of rays processed per environment.
        max_dist: The maximum distance to search for ray intersections. Defaults to 1e6.
        return_distance: Whether to return the distance from the ray start to the hit point. Defaults to False.
        return_normal: Whether to return the normal of the mesh face that is hit. Defaults to False.

    Returns:
        The ray hit positions. Shape (N, 3), with float('inf') for missed hits.
        The ray hit distances. Shape (N,), with float('inf') for missed hits.
            Returned only if return_distance is True; else, returns None.
        The ray hit normals. Shape (N, 3), with float('inf') for missed hits.
            Returned only if return_normal is True; else, returns None.
    """
    total_rays = ray_starts.shape[0]
    device = ray_starts.device
    out_hits = torch.full((total_rays, 3), float("inf"), device=device)
    if return_distance:
        out_distance = torch.full((total_rays,), float("inf"), device=device)
    else:
        out_distance = None
    if return_normal:
        out_normal = torch.full((total_rays, 3), float("inf"), device=device)
    else:
        out_normal = None

    # convert tensors to warp arrays once
    ray_starts_wp = wp.from_torch(ray_starts.contiguous(), dtype=wp.vec3)
    ray_directions_wp = wp.from_torch(ray_directions.contiguous(), dtype=wp.vec3)
    out_hits_wp = wp.from_torch(out_hits, dtype=wp.vec3)
    if return_distance:
        out_distance_wp = wp.from_torch(out_distance, dtype=wp.float32)
    else:
        out_distance_wp = wp.empty((1,), dtype=wp.float32, device=device.type)
    if return_normal:
        out_normal_wp = wp.from_torch(out_normal, dtype=wp.vec3)
    else:
        out_normal_wp = wp.empty((1,), dtype=wp.vec3, device=device.type)

    mesh_ids_wp = wp.from_torch(mesh_ids.contiguous(), dtype=wp.uint64)
    env_offsets_wp = wp.from_torch(env_offsets.contiguous(), dtype=wp.int32)

    wp.launch(
        kernel=kernels.multi_mesh_raycast_kernel,
        dim=total_rays,
        inputs=[
            env_offsets_wp,
            mesh_ids_wp,
            ray_starts_wp,
            ray_directions_wp,
            out_hits_wp,
            out_distance_wp,
            out_normal_wp,
            rays_per_env,
            float(max_dist),
            int(return_distance),
            int(return_normal),
        ],
        device=device.type,
    )
    wp.synchronize()

    # note: the kernel does not support infinity, so it is replaced here
    if return_distance:
        mask = out_distance == max_dist
        out_distance[mask] = float("inf")
        out_hits[mask] = float("inf")
        out_distance = out_distance.to(device)
    out_hits = out_hits.to(device)

    if return_normal:
        out_normal = out_normal.to(device)
        return out_hits, out_distance, out_normal
    else:
        return out_hits, out_distance


def convert_to_warp_mesh(points: np.ndarray, indices: np.ndarray, device: str) -> wp.Mesh:
    """Create a warp mesh object with a mesh defined from vertices and triangles.

    Args:
        points: The vertices of the mesh. Shape is (N, 3), where N is the number of vertices.
        indices: The triangles of the mesh as references to vertices for each triangle.
            Shape is (M, 3), where M is the number of triangles / faces.
        device: The device to use for the mesh.

    Returns:
        The warp mesh object.
    """
    return wp.Mesh(
        points=wp.array(points.astype(np.float32), dtype=wp.vec3, device=device),
        indices=wp.array(indices.astype(np.int32).flatten(), dtype=wp.int32, device=device),
    )
