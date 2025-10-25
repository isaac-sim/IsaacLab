# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
def raycast_static_meshes_kernel(
    mesh: wp.array2d(dtype=wp.uint64),
    ray_starts: wp.array2d(dtype=wp.vec3),
    ray_directions: wp.array2d(dtype=wp.vec3),
    ray_hits: wp.array2d(dtype=wp.vec3),
    ray_distance: wp.array2d(dtype=wp.float32),
    ray_normal: wp.array2d(dtype=wp.vec3),
    ray_face_id: wp.array2d(dtype=wp.int32),
    ray_mesh_id: wp.array2d(dtype=wp.int16),
    max_dist: float = 1e6,
    return_normal: int = False,
    return_face_id: int = False,
    return_mesh_id: int = False,
):
    """Performs ray-casting against multiple static meshes.

    This function performs ray-casting against the given meshes using the provided ray start positions
    and directions. The resulting ray hit positions are stored in the :obj:`ray_hits` array.

    The function utilizes the ``mesh_query_ray`` method from the ``wp`` module to perform the actual ray-casting
    operation. The maximum ray-cast distance is set to ``1e6`` units.

    .. note::
        That the ``ray_starts``, ``ray_directions``, and ``ray_hits`` arrays should have compatible shapes
        and data types to ensure proper execution. Additionally, they all must be in the same frame.

        This kernel differs from the :meth:`raycast_dynamic_meshes_kernel` in that it does not take into
        account the mesh's position and rotation. This kernel is useful for ray-casting against static meshes
        that are not expected to move.

    Args:
        mesh: The input mesh. The ray-casting is performed against this mesh on the device specified by the
            `mesh`'s `device` attribute.
        ray_starts: The input ray start positions. Shape is (B, N, 3).
        ray_directions: The input ray directions. Shape is (B, N, 3).
        ray_hits: The output ray hit positions. Shape is (B, N, 3).
        ray_distance: The output ray hit distances. Shape is (B, N,), if ``return_distance`` is True. Otherwise,
            this array is not used.
        ray_normal: The output ray hit normals. Shape is (B, N, 3), if ``return_normal`` is True. Otherwise,
            this array is not used.
        ray_face_id: The output ray hit face ids. Shape is (B, N,), if ``return_face_id`` is True. Otherwise,
            this array is not used.
        ray_mesh_id: The output ray hit mesh ids. Shape is (B, N,), if ``return_mesh_id`` is True. Otherwise,
            this array is not used.
        max_dist: The maximum ray-cast distance. Defaults to 1e6.
        return_normal: Whether to return the ray hit normals. Defaults to False`.
        return_face_id: Whether to return the ray hit face ids. Defaults to False.
        return_mesh_id: Whether to return the mesh id. Defaults to False.
    """
    # get the thread id
    tid_mesh_id, tid_env, tid_ray = wp.tid()

    direction = ray_directions[tid_env, tid_ray]
    start_pos = ray_starts[tid_env, tid_ray]

    # ray cast against the mesh and store the hit position
    mesh_query_ray_t = wp.mesh_query_ray(mesh[tid_env, tid_mesh_id], start_pos, direction, max_dist)

    # if the ray hit, store the hit data
    if mesh_query_ray_t.result:

        # check if hit distance is less than the current hit distance, only then update the memory
        if mesh_query_ray_t.t < ray_distance[tid_env, tid_ray]:

            # convert back to world space and update the hit data
            ray_hits[tid_env, tid_ray] = start_pos + mesh_query_ray_t.t * direction

            # update the hit distance
            ray_distance[tid_env, tid_ray] = mesh_query_ray_t.t

            # update the normal and face id if requested
            if return_normal == 1:
                ray_normal[tid_env, tid_ray] = mesh_query_ray_t.normal
            if return_face_id == 1:
                ray_face_id[tid_env, tid_ray] = mesh_query_ray_t.face
            if return_mesh_id == 1:
                ray_mesh_id[tid_env, tid_ray] = wp.int16(tid_mesh_id)


@wp.kernel
def raycast_dynamic_meshes_kernel(
    mesh: wp.array2d(dtype=wp.uint64),
    ray_starts: wp.array2d(dtype=wp.vec3),
    ray_directions: wp.array2d(dtype=wp.vec3),
    ray_hits: wp.array2d(dtype=wp.vec3),
    ray_distance: wp.array2d(dtype=wp.float32),
    ray_normal: wp.array2d(dtype=wp.vec3),
    ray_face_id: wp.array2d(dtype=wp.int32),
    ray_mesh_id: wp.array2d(dtype=wp.int16),
    mesh_positions: wp.array2d(dtype=wp.vec3),
    mesh_rotations: wp.array2d(dtype=wp.quat),
    max_dist: float = 1e6,
    return_normal: int = False,
    return_face_id: int = False,
    return_mesh_id: int = False,
):
    """Performs ray-casting against multiple meshes.

    This function performs ray-casting against the given meshes using the provided ray start positions
    and directions. The resulting ray hit positions are stored in the :obj:`ray_hits` array.

    The function utilizes the ``mesh_query_ray`` method from the ``wp`` module to perform the actual ray-casting
    operation. The maximum ray-cast distance is set to ``1e6`` units.


    Note:
        That the ``ray_starts``, ``ray_directions``, and ``ray_hits`` arrays should have compatible shapes
        and data types to ensure proper execution. Additionally, they all must be in the same frame.

        All arguments are expected to be batched with the first dimension (B, batch) being the number of envs
        and the second dimension (N, num_rays) being the number of rays. For Meshes, W is the number of meshes.

    Args:
        mesh: The input mesh. The ray-casting is performed against this mesh on the device specified by the
            `mesh`'s `device` attribute.
        ray_starts: The input ray start positions. Shape is (B, N, 3).
        ray_directions: The input ray directions. Shape is (B, N, 3).
        ray_hits: The output ray hit positions. Shape is (B, N, 3).
        ray_distance: The output ray hit distances. Shape is (B, N,), if ``return_distance`` is True. Otherwise,
            this array is not used.
        ray_normal: The output ray hit normals. Shape is (B, N, 3), if ``return_normal`` is True. Otherwise,
            this array is not used.
        ray_face_id: The output ray hit face ids. Shape is (B, N,), if ``return_face_id`` is True. Otherwise,
            this array is not used.
        ray_mesh_id: The output ray hit mesh ids. Shape is (B, N,), if ``return_mesh_id`` is True. Otherwise,
            this array is not used.
        mesh_positions: The input mesh positions in world frame. Shape is (W, 3).
        mesh_rotations: The input mesh rotations in world frame. Shape is (W, 4).
        max_dist: The maximum ray-cast distance. Defaults to 1e6.
        return_normal: Whether to return the ray hit normals. Defaults to False`.
        return_face_id: Whether to return the ray hit face ids. Defaults to False.
        return_mesh_id: Whether to return the mesh id. Defaults to False.
    """
    # get the thread id
    tid_mesh_id, tid_env, tid_ray = wp.tid()

    mesh_pose = wp.transform(mesh_positions[tid_env, tid_mesh_id], mesh_rotations[tid_env, tid_mesh_id])
    mesh_pose_inv = wp.transform_inverse(mesh_pose)
    direction = wp.transform_vector(mesh_pose_inv, ray_directions[tid_env, tid_ray])
    start_pos = wp.transform_point(mesh_pose_inv, ray_starts[tid_env, tid_ray])

    # ray cast against the mesh and store the hit position
    mesh_query_ray_t = wp.mesh_query_ray(mesh[tid_env, tid_mesh_id], start_pos, direction, max_dist)
    # if the ray hit, store the hit data
    if mesh_query_ray_t.result:

        # check if hit distance is less than the current hit distance, only then update the memory
        if mesh_query_ray_t.t < ray_distance[tid_env, tid_ray]:

            # convert back to world space and update the hit data
            hit_pos = start_pos + mesh_query_ray_t.t * direction
            ray_hits[tid_env, tid_ray] = wp.transform_point(mesh_pose, hit_pos)

            # update the hit distance
            ray_distance[tid_env, tid_ray] = mesh_query_ray_t.t

            # update the normal and face id if requested
            if return_normal == 1:
                n = wp.transform_vector(mesh_pose, mesh_query_ray_t.normal)
                ray_normal[tid_env, tid_ray] = n
            if return_face_id == 1:
                ray_face_id[tid_env, tid_ray] = mesh_query_ray_t.face
            if return_mesh_id == 1:
                ray_mesh_id[tid_env, tid_ray] = wp.int16(tid_mesh_id)


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
