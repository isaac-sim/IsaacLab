# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

# Cache of all-True env masks keyed by (n_envs, device) to avoid per-call allocations in
# raycast_dynamic_meshes. Populated lazily on first call with a given (n_envs, device) pair.
_all_env_mask_cache: dict[tuple[int, str], wp.array] = {}


# Tile size for the spatial-axis split in :func:`_uint8_spatial_mean`. Tuned on L40;
# robust across modern NVIDIA arches (Ampere/Ada/Hopper) at R256.
_UINT8_SUM_TILE_HW: int = 32

# Cache of int32 partials scratch tensors keyed by (src.shape, device, channel_dim). Avoids
# per-call allocation in :func:`_uint8_spatial_mean`. channel_dim is part of the key so
# BCHW and BHWC inputs with otherwise-identical shape get separate scratch slots. Typically
# holds one entry per training run (camera resolution, device, and layout are all fixed).
_uint8_sum_partials_cache: dict[tuple[tuple[int, ...], str, int], torch.Tensor] = {}


def _uint8_spatial_mean(src: torch.Tensor, scale: float, channel_dim: int = 3) -> torch.Tensor:
    """Per-(batch, channel) mean of a uint8 image scaled by ``1 / scale``.

    Equivalent to ``src.sum(dim=spatial_dims, dtype=int64).float() / scale`` where
    ``spatial_dims`` is the pair of non-batch, non-channel axes. The int64
    promotion is safe at any resolution; the per-tile Warp accumulator stays
    int32 (overflow-safe up to ~16M values per tile).

    Args:
        src: Input image. Shape is ``(B, H, W, C)`` (BHWC) or ``(B, C, H, W)`` (BCHW),
            dtype ``torch.uint8``, contiguous.
        scale: Multiplier for the per-channel sum. Pass ``H * W * 255`` to get the mean
            of ``src / 255``.
        channel_dim: Resolved positive position of the channel axis -- ``1`` (BCHW) or
            ``3`` (BHWC). Defaults to ``3`` (BHWC) for back-compat with internal callers.

    Returns:
        Per-(batch, channel) mean as float32. Shape is ``(B, C)``.
    """
    if channel_dim == 1:
        b, c, h, _ = src.shape
    else:
        b, h, _, c = src.shape
    device_str = str(src.device)
    cache_key = (src.shape, device_str, channel_dim)
    partials = _uint8_sum_partials_cache.get(cache_key)
    if partials is None:
        num_tiles = (h + _UINT8_SUM_TILE_HW - 1) // _UINT8_SUM_TILE_HW
        # C innermost: adjacent threads stride-1 along src's contiguous trailing dim (BHWC fast path).
        partials = torch.empty((b, num_tiles, c), dtype=torch.int32, device=src.device)
        _uint8_sum_partials_cache[cache_key] = partials

    src_wp = wp.from_torch(src, dtype=wp.uint8)
    partials_wp = wp.from_torch(partials, dtype=wp.int32)
    wp.launch(
        kernel=kernels.spatial_sum_uint8_tiled,
        dim=partials.shape,
        inputs=[src_wp, partials_wp, _UINT8_SUM_TILE_HW, channel_dim],
        device=device_str,
    )
    return partials.sum(dim=1, dtype=torch.int64).float() / scale


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


def raycast_single_mesh(
    ray_starts: torch.Tensor,
    ray_directions: torch.Tensor,
    mesh_id: int,
    max_dist: float = 1e6,
    return_distance: bool = False,
    return_normal: bool = False,
    return_face_id: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Performs ray-casting against a mesh.

    Note that the :attr:`ray_starts` and :attr:`ray_directions`, and :attr:`ray_hits` should have compatible shapes
    and data types to ensure proper execution. Additionally, they all must be in the same frame.

    Args:
        ray_starts: The starting position of the rays. Shape (B, N, 3).
        ray_directions: The ray directions for each ray. Shape (B, N, 3).
        mesh_id: The warp mesh id to ray-cast against.
        max_dist: The maximum distance to ray-cast. Defaults to 1e6.
        return_distance: Whether to return the distance of the ray until it hits the mesh. Defaults to False.
        return_normal: Whether to return the normal of the mesh face the ray hits. Defaults to False.
        return_face_id: Whether to return the face id of the mesh face the ray hits. Defaults to False.

    Returns:
        The ray hit position. Shape (B, N, 3).
            The returned tensor contains :obj:`float('inf')` for missed hits.
        The ray hit distance. Shape (B, N,).
            Will only return if :attr:`return_distance` is True, else returns None.
            The returned tensor contains :obj:`float('inf')` for missed hits.
        The ray hit normal. Shape (B, N, 3).
            Will only return if :attr:`return_normal` is True else returns None.
            The returned tensor contains :obj:`float('inf')` for missed hits.
        The ray hit face id. Shape (B, N,).
            Will only return if :attr:`return_face_id` is True else returns None.
            The returned tensor contains :obj:`int(-1)` for missed hits.
    """
    # cast mesh id into array
    mesh_ids = wp.array2d(
        [[mesh_id] for _ in range(ray_starts.shape[0])], dtype=wp.uint64, device=wp.device_from_torch(ray_starts.device)
    )
    ray_hits, ray_distance, ray_normal, ray_face_id, ray_mesh_id = raycast_dynamic_meshes(
        ray_starts=ray_starts,
        ray_directions=ray_directions,
        mesh_ids_wp=mesh_ids,
        max_dist=max_dist,
        return_distance=return_distance,
        return_normal=return_normal,
        return_face_id=return_face_id,
    )

    return ray_hits, ray_distance, ray_normal, ray_face_id


def raycast_dynamic_meshes(
    ray_starts: torch.Tensor,
    ray_directions: torch.Tensor,
    mesh_ids_wp: wp.Array,
    mesh_positions_w: torch.Tensor | None = None,
    mesh_orientations_w: torch.Tensor | None = None,
    max_dist: float = 1e6,
    return_distance: bool = False,
    return_normal: bool = False,
    return_face_id: bool = False,
    return_mesh_id: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Performs ray-casting against multiple, dynamic meshes.

    Note that the :attr:`ray_starts` and :attr:`ray_directions`, and :attr:`ray_hits` should have compatible shapes
    and data types to ensure proper execution. Additionally, they all must be in the same frame.

    If mesh positions and rotations are provided, they need to have to have the same shape as the
    number of meshes.

    Args:
        ray_starts: The starting position of the rays. Shape (B, N, 3).
        ray_directions: The ray directions for each ray. Shape (B, N, 3).
        mesh_ids_wp: The warp mesh ids to ray-cast against. Length (B, M).
        mesh_positions_w: The world positions of the meshes. Shape (B, M, 3).
        mesh_orientations_w: The world orientation as quaternion (x, y, z, w) format. Shape (B, M, 4).
        max_dist: The maximum distance to ray-cast. Defaults to 1e6.
        return_distance: Whether to return the distance of the ray until it hits the mesh. Defaults to False.
        return_normal: Whether to return the normal of the mesh face the ray hits. Defaults to False.
        return_face_id: Whether to return the face id of the mesh face the ray hits. Defaults to False.
        return_mesh_id: Whether to return the mesh id of the mesh face the ray hits. Defaults to False.
                        NOTE: the type of the returned tensor is torch.int16, so you can't have more than 32767 meshes.

    Returns:
        The ray hit position. Shape (B, N, 3).
            The returned tensor contains :obj:`float('inf')` for missed hits.
        The ray hit distance. Shape (B, N,).
            Will only return if :attr:`return_distance` is True, else returns None.
            The returned tensor contains :obj:`float('inf')` for missed hits.
        The ray hit normal. Shape (B, N, 3).
            Will only return if :attr:`return_normal` is True else returns None.
            The returned tensor contains :obj:`float('inf')` for missed hits.
        The ray hit face id. Shape (B, N,).
            Will only return if :attr:`return_face_id` is True else returns None.
            The returned tensor contains :obj:`int(-1)` for missed hits.
        The ray hit mesh id. Shape (B, N,).
            Will only return if :attr:`return_mesh_id` is True else returns None.
            The returned tensor contains :obj:`-1` for missed hits.
    """
    # extract device and shape information
    shape = ray_starts.shape
    device = ray_starts.device

    # device of the mesh
    torch_device = wp.device_to_torch(mesh_ids_wp.device)
    n_meshes = mesh_ids_wp.shape[1]

    n_envs = ray_starts.shape[0]
    n_rays_per_env = ray_starts.shape[1]

    # reshape the tensors
    ray_starts = ray_starts.to(torch_device).view(n_envs, n_rays_per_env, 3).contiguous()
    ray_directions = ray_directions.to(torch_device).view(n_envs, n_rays_per_env, 3).contiguous()

    # create output tensor for the ray hits
    ray_hits = torch.full((n_envs, n_rays_per_env, 3), float("inf"), device=torch_device).contiguous()

    # map the memory to warp arrays
    ray_starts_wp = wp.from_torch(ray_starts, dtype=wp.vec3)
    ray_directions_wp = wp.from_torch(ray_directions, dtype=wp.vec3)
    ray_hits_wp = wp.from_torch(ray_hits, dtype=wp.vec3)
    # required to check if a closer hit is reported, returned only if return_distance is true
    ray_distance = torch.full(
        (
            n_envs,
            n_rays_per_env,
        ),
        float("inf"),
        device=torch_device,
    ).contiguous()
    ray_distance_wp = wp.from_torch(ray_distance, dtype=wp.float32)

    if return_normal:
        ray_normal = torch.full((n_envs, n_rays_per_env, 3), float("inf"), device=torch_device).contiguous()
        ray_normal_wp = wp.from_torch(ray_normal, dtype=wp.vec3)
    else:
        ray_normal = None
        ray_normal_wp = wp.empty((1, 1), dtype=wp.vec3, device=torch_device)

    if return_face_id:
        ray_face_id = torch.ones(
            (
                n_envs,
                n_rays_per_env,
            ),
            dtype=torch.int32,
            device=torch_device,
        ).contiguous() * (-1)
        ray_face_id_wp = wp.from_torch(ray_face_id, dtype=wp.int32)
    else:
        ray_face_id = None
        ray_face_id_wp = wp.empty((1, 1), dtype=wp.int32, device=torch_device)

    if return_mesh_id:
        ray_mesh_id = -torch.ones((n_envs, n_rays_per_env), dtype=torch.int16, device=torch_device).contiguous()
        ray_mesh_id_wp = wp.from_torch(ray_mesh_id, dtype=wp.int16)
    else:
        ray_mesh_id = None
        ray_mesh_id_wp = wp.empty((1, 1), dtype=wp.int16, device=torch_device)

    ##
    # Call the warp kernels
    ###
    if mesh_positions_w is None and mesh_orientations_w is None:
        # Static mesh case, no need to pass in positions and rotations.
        # launch the warp kernel
        wp.launch(
            kernel=kernels.raycast_static_meshes_kernel,
            dim=[n_meshes, n_envs, n_rays_per_env],
            inputs=[
                mesh_ids_wp,
                ray_starts_wp,
                ray_directions_wp,
                ray_hits_wp,
                ray_distance_wp,
                ray_normal_wp,
                ray_face_id_wp,
                ray_mesh_id_wp,
                float(max_dist),
                int(return_normal),
                int(return_face_id),
                int(return_mesh_id),
            ],
            device=torch_device,
        )
    else:
        # dynamic mesh case
        if mesh_positions_w is None:
            mesh_positions_wp_w = wp.zeros((n_envs, n_meshes), dtype=wp.vec3, device=torch_device)
        else:
            mesh_positions_w = mesh_positions_w.to(torch_device).view(n_envs, n_meshes, 3).contiguous()
            mesh_positions_wp_w = wp.from_torch(mesh_positions_w, dtype=wp.vec3)

        if mesh_orientations_w is None:
            # Note (zrene): This is a little bit ugly, since it requires to initialize torch memory first
            # But I couldn't find a better way to initialize a quaternion identity in warp
            # wp.zeros(1, dtype=wp.quat, device=torch_device) gives all zero quaternion
            quat_identity = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=torch_device).repeat(
                n_envs, n_meshes, 1
            )
            mesh_quat_wp_w = wp.from_torch(quat_identity, dtype=wp.quat)
        else:
            # mesh orientations are already in xyzw format
            mesh_orientations_w = mesh_orientations_w.to(dtype=torch.float32, device=torch_device).contiguous()
            mesh_quat_wp_w = wp.from_torch(mesh_orientations_w, dtype=wp.quat)

        # All environments active when called through this public API.
        # Cache the mask by (n_envs, device) to avoid a per-call allocation.
        cache_key = (n_envs, str(torch_device))
        if cache_key not in _all_env_mask_cache:
            _all_env_mask_cache[cache_key] = wp.from_torch(torch.ones(n_envs, dtype=torch.bool, device=torch_device))
        all_env_mask = _all_env_mask_cache[cache_key]

        # launch the warp kernel
        wp.launch(
            kernel=kernels.raycast_dynamic_meshes_kernel,
            dim=[n_meshes, n_envs, n_rays_per_env],
            inputs=[
                all_env_mask,
                mesh_ids_wp,
                ray_starts_wp,
                ray_directions_wp,
                ray_hits_wp,
                ray_distance_wp,
                ray_normal_wp,
                ray_face_id_wp,
                ray_mesh_id_wp,
                mesh_positions_wp_w,
                mesh_quat_wp_w,
                float(max_dist),
                int(return_normal),
                int(return_face_id),
                int(return_mesh_id),
            ],
            device=torch_device,
        )
    ##
    # Cleanup and convert back to torch tensors
    ##

    # NOTE: Synchronize is not needed anymore, but we keep it for now. Check with @dhoeller.
    wp.synchronize()

    if return_distance:
        ray_distance = ray_distance.to(device).view(shape[:2])
    if return_normal:
        ray_normal = ray_normal.to(device).view(shape)
    if return_face_id:
        ray_face_id = ray_face_id.to(device).view(shape[:2])
    if return_mesh_id:
        ray_mesh_id = ray_mesh_id.to(device).view(shape[:2])

    return ray_hits.to(device).view(shape), ray_distance, ray_normal, ray_face_id, ray_mesh_id


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


def normalize_image_uint8(
    src: torch.Tensor,
    channel_dim: int = -1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``(src / 255.0) - mean(src / 255.0, spatial_dims, keepdim=True)`` via a fused Warp kernel.

    Equivalent to the pure-PyTorch expression to within float32 precision. Pass an ``out``
    tensor to reuse storage across steps.

    Supports both image layouts via ``channel_dim``:

    - BHWC (``channel_dim=-1`` or ``3``, the default): mean is taken over axes (1, 2).
    - BCHW (``channel_dim=-3`` or ``1``): mean is taken over axes (2, 3).

    Note:
        Most callers should go through :func:`isaaclab.utils.images.normalize_camera_image`,
        which dispatches non-uint8 / non-RGB inputs to a PyTorch fallback.

    Args:
        src: Input uint8 image tensor. Shape is ``(B, H, W, C)`` or ``(B, C, H, W)``.
            Must be contiguous.
        channel_dim: Position of the channel axis. Must resolve to ``1`` (BCHW) or ``3`` (BHWC).
            Negative values are supported (``-1`` == BHWC, ``-3`` == BCHW). Defaults to ``-1``.
        out: Optional pre-allocated float32 output. Same shape as ``src``, contiguous, on the
            same device. If omitted, a fresh tensor is allocated. Defaults to None.

            .. warning::

                If you pass the same ``out`` tensor across calls that happen on either
                side of an environment-step boundary (i.e., the result of one call is
                still being read by the RL trainer when the next call is made), the
                returned observation will alias the latest call's output and the
                trainer will see overwritten data. Use a ping-pong of two ``out``
                buffers, or omit ``out`` entirely, when the result lifetime crosses
                ``env.step()`` boundaries.

    Returns:
        The normalized float32 tensor. Same object as ``out`` when provided.

    Raises:
        ValueError: If ``src`` is not 4D uint8, not contiguous, ``channel_dim`` does not
            resolve to 1 or 3, or ``out``'s shape / dtype / device does not match.
    """
    if src.dtype != torch.uint8 or src.ndim != 4:
        raise ValueError(f"src must be a 4D uint8 tensor; got dtype={src.dtype}, ndim={src.ndim}")
    if not src.is_contiguous():
        raise ValueError("src must be contiguous (Warp kernel reads it as a 4D wp.array)")

    # Resolve negative channel_dim to its positive index in [1, src.ndim - 1].
    resolved_channel_dim = channel_dim + src.ndim if channel_dim < 0 else channel_dim
    if resolved_channel_dim not in (1, 3):
        raise ValueError(
            f"channel_dim must resolve to 1 (BCHW) or 3 (BHWC) for 4D input;"
            f" got channel_dim={channel_dim} -> {resolved_channel_dim}"
        )

    if out is None:
        out = torch.empty(src.shape, dtype=torch.float32, device=src.device)
    elif out.shape != src.shape or out.dtype != torch.float32 or out.device != src.device:
        raise ValueError(
            f"out shape/dtype/device mismatch: expected {tuple(src.shape)}/float32/{src.device},"
            f" got {tuple(out.shape)}/{out.dtype}/{out.device}"
        )
    elif not out.is_contiguous():
        raise ValueError("out must be contiguous")

    # Spatial dims = the two non-batch, non-channel axes; mean is shape (B, C) for both layouts.
    spatial_dims = tuple(d for d in (1, 2, 3) if d != resolved_channel_dim)
    spatial_size = src.shape[spatial_dims[0]] * src.shape[spatial_dims[1]]
    mean = _uint8_spatial_mean(src, spatial_size * 255.0, channel_dim=resolved_channel_dim)

    src_wp = wp.from_torch(src, dtype=wp.uint8)
    mean_wp = wp.from_torch(mean, dtype=wp.float32)
    out_wp = wp.from_torch(out, dtype=wp.float32)
    wp.launch(
        kernel=kernels.normalize_image_uint8,
        dim=src.shape,
        inputs=[src_wp, mean_wp, out_wp, resolved_channel_dim],
        device=str(src.device),
    )
    return out
