# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom kernels for warp."""

from typing import Any

import warp as wp


@wp.kernel(enable_backward=False)
def raycast_mesh_kernel(
    mesh: wp.uint64,
    ray_starts: wp.array(dtype=wp.vec3),
    ray_directions: wp.array(dtype=wp.vec3),
    ray_hits: wp.array(dtype=wp.vec3),
    ray_distance: wp.array(dtype=wp.float32),
    ray_normal: wp.array(dtype=wp.vec3),
    ray_face_id: wp.array(dtype=wp.int32),
    max_dist: float = 1e6,
    return_distance: int = False,
    return_normal: int = False,
    return_face_id: int = False,
):
    """Performs ray-casting against a mesh.

    This function performs ray-casting against the given mesh using the provided ray start positions
    and directions. The resulting ray hit positions are stored in the :obj:`ray_hits` array.

    Note that the `ray_starts`, `ray_directions`, and `ray_hits` arrays should have compatible shapes
    and data types to ensure proper execution. Additionally, they all must be in the same frame.

    The function utilizes the `mesh_query_ray` method from the `wp` module to perform the actual ray-casting
    operation. The maximum ray-cast distance is set to `1e6` units.

    Args:
        mesh: The input mesh. The ray-casting is performed against this mesh on the device specified by the
            `mesh`'s `device` attribute.
        ray_starts: The input ray start positions. Shape is (N, 3).
        ray_directions: The input ray directions. Shape is (N, 3).
        ray_hits: The output ray hit positions. Shape is (N, 3).
        ray_distance: The output ray hit distances. Shape is (N,), if `return_distance` is True. Otherwise,
            this array is not used.
        ray_normal: The output ray hit normals. Shape is (N, 3), if `return_normal` is True. Otherwise,
            this array is not used.
        ray_face_id: The output ray hit face ids. Shape is (N,), if `return_face_id` is True. Otherwise,
            this array is not used.
        max_dist: The maximum ray-cast distance. Defaults to 1e6.
        return_distance: Whether to return the ray hit distances. Defaults to False.
        return_normal: Whether to return the ray hit normals. Defaults to False`.
        return_face_id: Whether to return the ray hit face ids. Defaults to False.
    """
    # get the thread id
    tid = wp.tid()

    t = float(0.0)  # hit distance along ray
    u = float(0.0)  # hit face barycentric u
    v = float(0.0)  # hit face barycentric v
    sign = float(0.0)  # hit face sign
    n = wp.vec3()  # hit face normal
    f = int(0)  # hit face index

    # ray cast against the mesh and store the hit position
    hit_success = wp.mesh_query_ray(mesh, ray_starts[tid], ray_directions[tid], max_dist, t, u, v, sign, n, f)
    # if the ray hit, store the hit data
    if hit_success:
        ray_hits[tid] = ray_starts[tid] + t * ray_directions[tid]
        if return_distance == 1:
            ray_distance[tid] = t
        if return_normal == 1:
            ray_normal[tid] = n
        if return_face_id == 1:
            ray_face_id[tid] = f


@wp.kernel(enable_backward=False)
def multi_mesh_raycast_kernel(
    env_offsets: wp.array(dtype=wp.int32),
    mesh_ids: wp.array(dtype=wp.uint64),
    ray_starts: wp.array(dtype=wp.vec3),
    ray_directions: wp.array(dtype=wp.vec3),
    out_hits: wp.array(dtype=wp.vec3),
    out_distance: wp.array(dtype=wp.float32),
    out_normal: wp.array(dtype=wp.vec3),
    rays_per_env: int,
    max_dist: float = 1e6,
    return_distance: int = False,
    return_normal: int = False,
):
    """
    Performs batched ray-casting against multiple meshes across environments.

    This kernel processes a batch of rays, where each ray is cast against all meshes within its associated
    environment. Meshes are grouped by environment using the provided environment offsets and sorted mesh IDs.
    For each ray, the kernel iterates through the relevant meshes to determine the closest intersection. If an
    intersection is found within the maximum distance, the corresponding hit data (position, and optionally,
    distance and surface normal) is recorded. If no intersection occurs, the ray hit position is computed as
    ray_start + max_dist * ray_direction.

    Args:
        env_offsets (wp.array(dtype=wp.int32)): Offsets for each environment. This array has size (num_envs + 1)
            and delineates the start and end indices in the sorted `mesh_ids` array for each environment.
        mesh_ids (wp.array(dtype=wp.uint64)): Sorted warp mesh IDs corresponding to the meshes in each environment.
        ray_starts (wp.array(dtype=wp.vec3)): Flattened array of ray start positions. Shape is (N, 3),
            where N is the total number of rays.
        ray_directions (wp.array(dtype=wp.vec3)): Flattened array of normalized ray direction vectors. Shape is (N, 3).
        out_hits (wp.array(dtype=wp.vec3)): Output array for storing computed ray hit positions. Shape is (N, 3).
        out_distance (wp.array(dtype=wp.float32)): Output array for storing the distance from the ray start to the hit point.
            Shape is (N,). This array is used only if `return_distance` is set to True.
        out_normal (wp.array(dtype=wp.vec3)): Output array for storing surface normals at the hit point.
            Shape is (N, 3). This array is used only if `return_normal` is set to True.
        rays_per_env (int): The number of rays to be processed per environment.
        max_dist (float): The maximum distance to search for ray intersections. Defaults to 1e6.
        return_distance (int): Flag indicating whether to output the hit distance in `out_distance`. Defaults to False.
        return_normal (int): Flag indicating whether to output the surface normal at the hit point in `out_normal`. Defaults to False.
    """
    tid = wp.tid()
    env = tid // rays_per_env
    best_distance = max_dist
    best_hit = ray_starts[tid] + max_dist * ray_directions[tid]
    best_normal = wp.vec3(0.0, 0.0, 0.0)

    # get the range of mesh indices for this environment
    start_idx = env_offsets[env]
    end_idx = env_offsets[env + 1]

    for i in range(start_idx, end_idx):
        mesh_id = mesh_ids[i]
        t = float(0.0)  # hit distance along ray
        u = float(0.0)  # hit face barycentric u
        v = float(0.0)  # hit face barycentric v
        sign = float(0.0)  # hit face sign
        n = wp.vec3(0.0, 0.0, 0.0)  # hit face normal
        f = int(0)  # hit face index
        hit_success = wp.mesh_query_ray(mesh_id, ray_starts[tid], ray_directions[tid], max_dist, t, u, v, sign, n, f)
        if hit_success and (t < best_distance):
            best_distance = t
            best_hit = ray_starts[tid] + t * ray_directions[tid]
            best_normal = n

    out_hits[tid] = best_hit
    if return_distance == 1:
        out_distance[tid] = best_distance
    if return_normal == 1:
        out_normal[tid] = best_normal


@wp.kernel(enable_backward=False)
def reshape_tiled_image(
    tiled_image_buffer: Any,
    batched_image: Any,
    image_height: int,
    image_width: int,
    num_channels: int,
    num_tiles_x: int,
):
    """Reshapes a tiled image into a batch of images.

    This function reshapes the input tiled image buffer into a batch of images. The input image buffer
    is assumed to be tiled in the x and y directions. The output image is a batch of images with the
    specified height, width, and number of channels.

    Args:
        tiled_image_buffer: The input image buffer. Shape is (height * width * num_channels * num_cameras,).
        batched_image: The output image. Shape is (num_cameras, height, width, num_channels).
        image_width: The width of the image.
        image_height: The height of the image.
        num_channels: The number of channels in the image.
        num_tiles_x: The number of tiles in x-direction.
    """
    # get the thread id
    camera_id, height_id, width_id = wp.tid()

    # resolve the tile indices
    tile_x_id = camera_id % num_tiles_x
    tile_y_id = camera_id // num_tiles_x
    # compute the start index of the pixel in the tiled image buffer
    pixel_start = (
        num_channels * num_tiles_x * image_width * (image_height * tile_y_id + height_id)
        + num_channels * tile_x_id * image_width
        + num_channels * width_id
    )

    # copy the pixel values into the batched image
    for i in range(num_channels):
        batched_image[camera_id, height_id, width_id, i] = batched_image.dtype(tiled_image_buffer[pixel_start + i])


# uint32 -> int32 conversion is required for non-colored segmentation annotators
wp.overload(
    reshape_tiled_image,
    {"tiled_image_buffer": wp.array(dtype=wp.uint32), "batched_image": wp.array(dtype=wp.uint32, ndim=4)},
)
# uint8 is used for 4 channel annotators
wp.overload(
    reshape_tiled_image,
    {"tiled_image_buffer": wp.array(dtype=wp.uint8), "batched_image": wp.array(dtype=wp.uint8, ndim=4)},
)
# float32 is used for single channel annotators
wp.overload(
    reshape_tiled_image,
    {"tiled_image_buffer": wp.array(dtype=wp.float32), "batched_image": wp.array(dtype=wp.float32, ndim=4)},
)
