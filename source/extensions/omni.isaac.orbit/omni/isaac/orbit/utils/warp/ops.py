# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Wrapping around warp kernels for compatibility with torch tensors."""

import numpy as np
import torch

import warp as wp

from . import kernels


def raycast_mesh(ray_starts: torch.Tensor, ray_directions: torch.Tensor, mesh: wp.Mesh) -> torch.Tensor:
    """Performs ray-casting against a mesh.

    Note that the `ray_starts` and `ray_directions`, and `ray_hits` should have compatible shapes
    and data types to ensure proper execution. Additionally, they all must be in the same frame.

    Args:
        ray_start: The starting position of the rays. Shape (N, 3).
        ray_directions: The ray directions for each ray. Shape (N, 3).
        mesh: The warp mesh to ray-cast against.

    Returns:
        The ray hit position. Shape (N, 3). The returned tensor contains :obj:`float('inf')` for missed hits.
    """
    # extract device and shape information
    shape = ray_starts.shape
    device = ray_starts.device
    # device of the mesh
    mesh_device = wp.device_to_torch(mesh.device)
    # reshape the tensors
    ray_starts = ray_starts.to(mesh_device).view(-1, 3)
    ray_directions = ray_directions.to(mesh_device).view(-1, 3)
    num_rays = ray_starts.shape[0]
    # create output tensor for the ray hits
    ray_hits = torch.full((num_rays, 3), float("inf"), device=mesh_device)

    # map the memory to warp arrays
    ray_starts_wp = wp.from_torch(ray_starts, dtype=wp.vec3)
    ray_directions_wp = wp.from_torch(ray_directions, dtype=wp.vec3)
    ray_hits_wp = wp.from_torch(ray_hits, dtype=wp.vec3)

    # launch the warp kernel
    wp.launch(
        kernel=kernels.raycast_mesh_kernel,
        dim=num_rays,
        inputs=[mesh.id, ray_starts_wp, ray_directions_wp, ray_hits_wp],
        device=mesh.device,
    )
    # NOTE: Synchronize is not needed anymore, but we keep it for now. Check with @dhoeller.
    wp.synchronize()

    return ray_hits.to(device).view(shape)


def convert_to_warp_mesh(points: np.ndarray, indices: np.ndarray, device: str) -> wp.Mesh:
    """Create a warp mesh object with a mesh defined from vertices and triangles.

    Args:
        points: The vertices of the mesh. Shape is :math:`(N, 3)`, where :math:`N`
            is the number of vertices.
        indices: The triangles of the mesh as references to vertices for each triangle.
            Shape is :math:`(M, 3)`, where :math:`M` is the number of triangles / faces.
        device: The device to use for the mesh.

    Returns:
        The warp mesh object.
    """
    # create warp mesh
    return wp.Mesh(
        points=wp.array(points.astype(np.float32), dtype=wp.vec3, device=device),
        indices=wp.array(indices.astype(np.int32).flatten(), dtype=wp.int32, device=device),
    )
