# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom kernels for warp."""

from typing import Any

import warp as wp

##
# Raycasting
##


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
        return_normal: Whether to return the ray hit normals. Defaults to False.
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
def raycast_mesh_masked_kernel(
    # input
    mesh: wp.uint64,
    env_mask: wp.array(dtype=wp.bool),
    ray_starts: wp.array2d(dtype=wp.vec3f),
    ray_directions: wp.array2d(dtype=wp.vec3f),
    max_dist: wp.float32,
    return_distance: int,
    return_normal: int,
    # output
    ray_hits: wp.array2d(dtype=wp.vec3f),
    ray_distance: wp.array2d(dtype=wp.float32),
    ray_normal: wp.array2d(dtype=wp.vec3f),
):
    """Ray-cast against a single static mesh for masked environments.

    Extends :func:`raycast_mesh_kernel` with environment masking and optional distance/normal output,
    for use in multi-environment sensor pipelines.

    Launch with ``dim=(num_envs, num_rays)``.

    Args:
        mesh: Warp mesh id to ray-cast against.
        env_mask: Boolean mask for which environments to update. Shape is (num_envs,).
        ray_starts: World-frame ray start positions [m]. Shape is (num_envs, num_rays).
        ray_directions: World-frame unit ray directions. Shape is (num_envs, num_rays).
        max_dist: Maximum ray-cast distance [m].
        return_distance: Whether to write hit distances to ``ray_distance`` (1) or skip (0).
        return_normal: Whether to write surface normals to ``ray_normal`` (1) or skip (0).
        ray_hits: Output ray hit positions [m]. Shape is (num_envs, num_rays).
            Pre-filled with inf for missed hits; unchanged on miss.
        ray_distance: Output hit distances [m]. Shape is (num_envs, num_rays).
            Written only when ``return_distance`` is 1; pre-filled with inf for missed hits.
        ray_normal: Output surface normals at hit positions. Shape is (num_envs, num_rays).
            Written only when ``return_normal`` is 1; pre-filled with inf for missed hits.
    """
    env, ray = wp.tid()
    if not env_mask[env]:
        return

    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3f()
    f = int(0)

    hit = wp.mesh_query_ray(mesh, ray_starts[env, ray], ray_directions[env, ray], max_dist, t, u, v, sign, n, f)
    if hit:
        ray_hits[env, ray] = ray_starts[env, ray] + t * ray_directions[env, ray]
        if return_distance == 1:
            ray_distance[env, ray] = t
        if return_normal == 1:
            ray_normal[env, ray] = n


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

    .. warning::
        **Known race condition:** When two meshes are equidistant to the same ray, the
        ``atomic_min`` + equality-check pattern used for closest-hit resolution is not fully
        thread-safe.  Two threads may both pass the equality check and write different output
        fields (e.g., ``ray_hits`` from mesh A, ``ray_normal`` from mesh B).  In practice this
        is rare (requires exact floating-point tie) and the position output is still correct,
        but normals, face IDs, and mesh IDs may be inconsistent for the affected ray.
        See `warp#1058 <https://github.com/NVIDIA/warp/issues/1058>`_ for progress on a
        thread-safe fix.

    Args:
        mesh: The input mesh. The ray-casting is performed against this mesh on the device specified by the
            `mesh`'s `device` attribute.
        ray_starts: The input ray start positions. Shape is (B, N, 3).
        ray_directions: The input ray directions. Shape is (B, N, 3).
        ray_hits: The output ray hit positions. Shape is (B, N, 3).
        ray_distance: The closest hit distance buffer. Shape is (B, N). Updated via ``atomic_min`` for every
            thread that records a hit; used to resolve closest-hit among multiple meshes.
        ray_normal: The output ray hit normals. Shape is (B, N, 3), if ``return_normal`` is True. Otherwise,
            this array is not used.
        ray_face_id: The output ray hit face ids. Shape is (B, N,), if ``return_face_id`` is True. Otherwise,
            this array is not used.
        ray_mesh_id: The output ray hit mesh ids. Shape is (B, N,), if ``return_mesh_id`` is True. Otherwise,
            this array is not used.
        max_dist: The maximum ray-cast distance. Defaults to 1e6.
        return_normal: Whether to return the ray hit normals. Defaults to False.
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
        wp.atomic_min(ray_distance, tid_env, tid_ray, mesh_query_ray_t.t)
        # TODO(warp#1058): Use the return value of atomic_min to avoid the non-thread-safe
        # equality check below. Currently warp atomic_min returns wrong values on GPU, so we
        # fall back to a racy read-back. When two meshes tie on distance, normals/face-ids/
        # mesh-ids may be written by different threads. The hit *position* is still correct
        # because all tying threads compute the same world-space point.
        if mesh_query_ray_t.t == ray_distance[tid_env, tid_ray]:
            # convert back to world space and update the hit data
            ray_hits[tid_env, tid_ray] = start_pos + mesh_query_ray_t.t * direction

            # update the normal and face id if requested
            if return_normal == 1:
                ray_normal[tid_env, tid_ray] = mesh_query_ray_t.normal
            if return_face_id == 1:
                ray_face_id[tid_env, tid_ray] = mesh_query_ray_t.face
            if return_mesh_id == 1:
                ray_mesh_id[tid_env, tid_ray] = wp.int16(tid_mesh_id)


@wp.kernel(enable_backward=False)
def raycast_dynamic_meshes_kernel(
    env_mask: wp.array(dtype=wp.bool),
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
    """Performs ray-casting against multiple dynamic meshes.

    This function performs ray-casting against the given meshes using the provided ray start positions
    and directions. The resulting ray hit positions are stored in the :obj:`ray_hits` array.

    The function utilizes the ``mesh_query_ray`` method from the ``wp`` module to perform the actual ray-casting
    operation. The maximum ray-cast distance is set to ``1e6`` units.

    Note:
        That the ``ray_starts``, ``ray_directions``, and ``ray_hits`` arrays should have compatible shapes
        and data types to ensure proper execution. Additionally, they all must be in the same frame.

        All arguments are expected to be batched with the first dimension (B, batch) being the number of envs
        and the second dimension (N, num_rays) being the number of rays. For Meshes, W is the number of meshes.

    .. warning::
        **Known race condition:** When two meshes are equidistant to the same ray, the
        ``atomic_min`` + equality-check pattern used for closest-hit resolution is not fully
        thread-safe.  Two threads may both pass the equality check and write different output
        fields (e.g., ``ray_hits`` from mesh A, ``ray_normal`` from mesh B).  In practice this
        is rare (requires exact floating-point tie) and the position output is still correct,
        but normals, face IDs, and mesh IDs may be inconsistent for the affected ray.
        See `warp#1058 <https://github.com/NVIDIA/warp/issues/1058>`_ for progress on a
        thread-safe fix.

    Args:
        env_mask: Boolean mask selecting which environments to process. Shape is (B,).
        mesh: The input mesh. The ray-casting is performed against this mesh on the device specified by the
            `mesh`'s `device` attribute.
        ray_starts: The input ray start positions. Shape is (B, N, 3).
        ray_directions: The input ray directions. Shape is (B, N, 3).
        ray_hits: The output ray hit positions. Shape is (B, N, 3).
        ray_distance: The closest hit distance buffer. Shape is (B, N). Updated via ``atomic_min`` for every
            thread that records a hit; used to resolve closest-hit among multiple meshes.
        ray_normal: The output ray hit normals. Shape is (B, N, 3), if ``return_normal`` is True. Otherwise,
            this array is not used.
        ray_face_id: The output ray hit face ids. Shape is (B, N,), if ``return_face_id`` is True. Otherwise,
            this array is not used.
        ray_mesh_id: The output ray hit mesh ids. Shape is (B, N,), if ``return_mesh_id`` is True. Otherwise,
            this array is not used.
        mesh_positions: The input mesh positions in world frame. Shape is (B, W, 3).
        mesh_rotations: The input mesh rotations in world frame. Shape is (B, W, 4).
        max_dist: The maximum ray-cast distance. Defaults to 1e6.
        return_normal: Whether to return the ray hit normals. Defaults to False.
        return_face_id: Whether to return the ray hit face ids. Defaults to False.
        return_mesh_id: Whether to return the mesh id. Defaults to False.
    """
    # get the thread id
    tid_mesh_id, tid_env, tid_ray = wp.tid()
    if not env_mask[tid_env]:
        return

    mesh_pose = wp.transform(mesh_positions[tid_env, tid_mesh_id], mesh_rotations[tid_env, tid_mesh_id])
    mesh_pose_inv = wp.transform_inverse(mesh_pose)
    direction = wp.transform_vector(mesh_pose_inv, ray_directions[tid_env, tid_ray])
    start_pos = wp.transform_point(mesh_pose_inv, ray_starts[tid_env, tid_ray])

    # ray cast against the mesh and store the hit position
    mesh_query_ray_t = wp.mesh_query_ray(mesh[tid_env, tid_mesh_id], start_pos, direction, max_dist)
    # if the ray hit, store the hit data
    if mesh_query_ray_t.result:
        wp.atomic_min(ray_distance, tid_env, tid_ray, mesh_query_ray_t.t)
        # TODO(warp#1058): Use the return value of atomic_min to avoid the non-thread-safe
        # equality check below. Currently warp atomic_min returns wrong values on GPU, so we
        # fall back to a racy read-back. When two meshes tie on distance, normals/face-ids/
        # mesh-ids may be written by different threads. The hit *position* is still correct
        # because all tying threads compute the same world-space point.
        if mesh_query_ray_t.t == ray_distance[tid_env, tid_ray]:
            # convert back to world space and update the hit data
            hit_pos = start_pos + mesh_query_ray_t.t * direction
            ray_hits[tid_env, tid_ray] = wp.transform_point(mesh_pose, hit_pos)

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

##
# Wrench Composer — Dual-Buffer Architecture
##


@wp.kernel
def set_forces_to_dual_buffers_index(
    env_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
    forces: wp.array2d(dtype=wp.vec3f),
    torques: wp.array2d(dtype=wp.vec3f),
    positions: wp.array2d(dtype=wp.vec3f),
    global_force_w: wp.array2d(dtype=wp.vec3f),
    global_torque_w: wp.array2d(dtype=wp.vec3f),
    global_force_at_com_w: wp.array2d(dtype=wp.vec3f),
    local_force_b: wp.array2d(dtype=wp.vec3f),
    local_torque_b: wp.array2d(dtype=wp.vec3f),
    is_global: bool,
):
    """Set forces/torques into dual buffers using index selection (overwrites).

    Dispatched with ``dim=(len(env_ids), len(body_ids))``.

    When ``is_global`` is True, forces/torques are written to the world-frame buffers.
    Forces with ``positions`` go to ``global_force_w`` with torque ``cross(P, F)`` accumulated
    into ``global_torque_w``; forces without positions go to ``global_force_at_com_w``.
    When ``is_global`` is False, values go to ``local_force_b`` / ``local_torque_b``.

    Any of ``forces``, ``torques``, or ``positions`` may be ``None`` (null array).
    """
    tid_env, tid_body = wp.tid()
    ei = env_ids[tid_env]
    bi = body_ids[tid_body]

    if is_global:
        if torques:
            global_torque_w[ei, bi] = torques[tid_env, tid_body]
        if forces:
            if positions:
                global_force_w[ei, bi] = forces[tid_env, tid_body]
                if torques:
                    global_torque_w[ei, bi] = global_torque_w[ei, bi] + wp.cross(
                        positions[tid_env, tid_body], forces[tid_env, tid_body]
                    )
                else:
                    global_torque_w[ei, bi] = wp.cross(positions[tid_env, tid_body], forces[tid_env, tid_body])
            else:
                global_force_at_com_w[ei, bi] = forces[tid_env, tid_body]
    else:
        if torques:
            local_torque_b[ei, bi] = torques[tid_env, tid_body]
        if forces:
            local_force_b[ei, bi] = forces[tid_env, tid_body]
            if positions:
                if torques:
                    local_torque_b[ei, bi] = local_torque_b[ei, bi] + wp.cross(
                        positions[tid_env, tid_body], forces[tid_env, tid_body]
                    )
                else:
                    local_torque_b[ei, bi] = wp.cross(positions[tid_env, tid_body], forces[tid_env, tid_body])


@wp.kernel
def add_forces_to_dual_buffers_index(
    env_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
    forces: wp.array2d(dtype=wp.vec3f),
    torques: wp.array2d(dtype=wp.vec3f),
    positions: wp.array2d(dtype=wp.vec3f),
    global_force_w: wp.array2d(dtype=wp.vec3f),
    global_torque_w: wp.array2d(dtype=wp.vec3f),
    global_force_at_com_w: wp.array2d(dtype=wp.vec3f),
    local_force_b: wp.array2d(dtype=wp.vec3f),
    local_torque_b: wp.array2d(dtype=wp.vec3f),
    is_global: bool,
):
    """Add forces/torques into dual buffers using index selection (accumulates).

    Same routing logic as :func:`set_forces_to_dual_buffers_index` but uses ``+=`` instead of ``=``.
    Dispatched with ``dim=(len(env_ids), len(body_ids))``.
    """
    tid_env, tid_body = wp.tid()
    ei = env_ids[tid_env]
    bi = body_ids[tid_body]

    if is_global:
        if forces:
            if positions:
                global_force_w[ei, bi] = global_force_w[ei, bi] + forces[tid_env, tid_body]
                global_torque_w[ei, bi] = global_torque_w[ei, bi] + wp.cross(
                    positions[tid_env, tid_body], forces[tid_env, tid_body]
                )
            else:
                global_force_at_com_w[ei, bi] = global_force_at_com_w[ei, bi] + forces[tid_env, tid_body]
        if torques:
            global_torque_w[ei, bi] = global_torque_w[ei, bi] + torques[tid_env, tid_body]
    else:
        if forces:
            local_force_b[ei, bi] = local_force_b[ei, bi] + forces[tid_env, tid_body]
            if positions:
                local_torque_b[ei, bi] = local_torque_b[ei, bi] + wp.cross(
                    positions[tid_env, tid_body], forces[tid_env, tid_body]
                )
        if torques:
            local_torque_b[ei, bi] = local_torque_b[ei, bi] + torques[tid_env, tid_body]


@wp.kernel
def set_forces_to_dual_buffers_mask(
    env_mask: wp.array(dtype=wp.bool),
    body_mask: wp.array(dtype=wp.bool),
    forces: wp.array2d(dtype=wp.vec3f),
    torques: wp.array2d(dtype=wp.vec3f),
    positions: wp.array2d(dtype=wp.vec3f),
    global_force_w: wp.array2d(dtype=wp.vec3f),
    global_torque_w: wp.array2d(dtype=wp.vec3f),
    global_force_at_com_w: wp.array2d(dtype=wp.vec3f),
    local_force_b: wp.array2d(dtype=wp.vec3f),
    local_torque_b: wp.array2d(dtype=wp.vec3f),
    is_global: bool,
):
    """Set forces/torques into dual buffers using mask selection (overwrites).

    Same routing logic as :func:`set_forces_to_dual_buffers_index` but threads are gated by
    ``env_mask[tid_env] and body_mask[tid_body]``, and indices are direct (no indirection array).
    Dispatched with ``dim=(num_envs, num_bodies)``.
    """
    tid_env, tid_body = wp.tid()

    if env_mask[tid_env] and body_mask[tid_body]:
        if is_global:
            if torques:
                global_torque_w[tid_env, tid_body] = torques[tid_env, tid_body]
            if forces:
                if positions:
                    global_force_w[tid_env, tid_body] = forces[tid_env, tid_body]
                    if torques:
                        global_torque_w[tid_env, tid_body] = global_torque_w[tid_env, tid_body] + wp.cross(
                            positions[tid_env, tid_body], forces[tid_env, tid_body]
                        )
                    else:
                        global_torque_w[tid_env, tid_body] = wp.cross(
                            positions[tid_env, tid_body], forces[tid_env, tid_body]
                        )
                else:
                    global_force_at_com_w[tid_env, tid_body] = forces[tid_env, tid_body]
        else:
            if torques:
                local_torque_b[tid_env, tid_body] = torques[tid_env, tid_body]
            if forces:
                local_force_b[tid_env, tid_body] = forces[tid_env, tid_body]
                if positions:
                    if torques:
                        local_torque_b[tid_env, tid_body] = local_torque_b[tid_env, tid_body] + wp.cross(
                            positions[tid_env, tid_body], forces[tid_env, tid_body]
                        )
                    else:
                        local_torque_b[tid_env, tid_body] = wp.cross(
                            positions[tid_env, tid_body], forces[tid_env, tid_body]
                        )


@wp.kernel
def add_forces_to_dual_buffers_mask(
    env_mask: wp.array(dtype=wp.bool),
    body_mask: wp.array(dtype=wp.bool),
    forces: wp.array2d(dtype=wp.vec3f),
    torques: wp.array2d(dtype=wp.vec3f),
    positions: wp.array2d(dtype=wp.vec3f),
    global_force_w: wp.array2d(dtype=wp.vec3f),
    global_torque_w: wp.array2d(dtype=wp.vec3f),
    global_force_at_com_w: wp.array2d(dtype=wp.vec3f),
    local_force_b: wp.array2d(dtype=wp.vec3f),
    local_torque_b: wp.array2d(dtype=wp.vec3f),
    is_global: bool,
):
    """Add forces/torques into dual buffers using mask selection (accumulates).

    Same routing logic as :func:`add_forces_to_dual_buffers_index` but threads are gated by
    ``env_mask[tid_env] and body_mask[tid_body]``.
    Dispatched with ``dim=(num_envs, num_bodies)``.
    """
    tid_env, tid_body = wp.tid()

    if env_mask[tid_env] and body_mask[tid_body]:
        if is_global:
            if forces:
                if positions:
                    global_force_w[tid_env, tid_body] = global_force_w[tid_env, tid_body] + forces[tid_env, tid_body]
                    global_torque_w[tid_env, tid_body] = global_torque_w[tid_env, tid_body] + wp.cross(
                        positions[tid_env, tid_body], forces[tid_env, tid_body]
                    )
                else:
                    global_force_at_com_w[tid_env, tid_body] = (
                        global_force_at_com_w[tid_env, tid_body] + forces[tid_env, tid_body]
                    )
            if torques:
                global_torque_w[tid_env, tid_body] = global_torque_w[tid_env, tid_body] + torques[tid_env, tid_body]
        else:
            if forces:
                local_force_b[tid_env, tid_body] = local_force_b[tid_env, tid_body] + forces[tid_env, tid_body]
                if positions:
                    local_torque_b[tid_env, tid_body] = local_torque_b[tid_env, tid_body] + wp.cross(
                        positions[tid_env, tid_body], forces[tid_env, tid_body]
                    )
            if torques:
                local_torque_b[tid_env, tid_body] = local_torque_b[tid_env, tid_body] + torques[tid_env, tid_body]


@wp.kernel
def add_raw_wrench_buffers(
    src_gf: wp.array2d(dtype=wp.vec3f),
    src_gt: wp.array2d(dtype=wp.vec3f),
    src_gfc: wp.array2d(dtype=wp.vec3f),
    src_lf: wp.array2d(dtype=wp.vec3f),
    src_lt: wp.array2d(dtype=wp.vec3f),
    dst_gf: wp.array2d(dtype=wp.vec3f),
    dst_gt: wp.array2d(dtype=wp.vec3f),
    dst_gfc: wp.array2d(dtype=wp.vec3f),
    dst_lf: wp.array2d(dtype=wp.vec3f),
    dst_lt: wp.array2d(dtype=wp.vec3f),
):
    """Element-wise add all five source wrench buffers into destination buffers.

    Dispatched with ``dim=(num_envs, num_bodies)``. Each ``src_*`` / ``dst_*`` pair corresponds
    to one of the five input buffers (global_force_w, global_torque_w, global_force_at_com_w,
    local_force_b, local_torque_b).
    """
    tid_env, tid_body = wp.tid()
    dst_gf[tid_env, tid_body] = dst_gf[tid_env, tid_body] + src_gf[tid_env, tid_body]
    dst_gt[tid_env, tid_body] = dst_gt[tid_env, tid_body] + src_gt[tid_env, tid_body]
    dst_gfc[tid_env, tid_body] = dst_gfc[tid_env, tid_body] + src_gfc[tid_env, tid_body]
    dst_lf[tid_env, tid_body] = dst_lf[tid_env, tid_body] + src_lf[tid_env, tid_body]
    dst_lt[tid_env, tid_body] = dst_lt[tid_env, tid_body] + src_lt[tid_env, tid_body]


@wp.kernel
def compose_wrench_to_body_frame(
    global_force_w: wp.array2d(dtype=wp.vec3f),
    global_torque_w: wp.array2d(dtype=wp.vec3f),
    global_force_at_com_w: wp.array2d(dtype=wp.vec3f),
    local_force_b: wp.array2d(dtype=wp.vec3f),
    local_torque_b: wp.array2d(dtype=wp.vec3f),
    com_pos_w: wp.array2d(dtype=wp.vec3f),
    link_quat_w: wp.array2d(dtype=wp.quatf),
    out_force_b: wp.array2d(dtype=wp.vec3f),
    out_torque_b: wp.array2d(dtype=wp.vec3f),
):
    """Compose global and local wrench buffers into a single body-frame output.

    Global torques store the moment of positional forces about the world origin: ``cross(P, F)``.
    This kernel corrects to be about the body's CoM via ``cross(P, F) - cross(com_pos_w, F) =
    cross(P - com_pos_w, F)``, then rotates both force and torque into the body frame using
    ``quat_rotate_inv(link_quat_w, ...)``, and adds local-frame values.

    Dispatched with ``dim=(num_envs, num_bodies)``.
    """
    tid_env, tid_body = wp.tid()
    total_force_w = global_force_w[tid_env, tid_body] + global_force_at_com_w[tid_env, tid_body]
    corrected_torque_w = global_torque_w[tid_env, tid_body] - wp.cross(
        com_pos_w[tid_env, tid_body], global_force_w[tid_env, tid_body]
    )
    out_force_b[tid_env, tid_body] = (
        wp.quat_rotate_inv(link_quat_w[tid_env, tid_body], total_force_w) + local_force_b[tid_env, tid_body]
    )
    out_torque_b[tid_env, tid_body] = (
        wp.quat_rotate_inv(link_quat_w[tid_env, tid_body], corrected_torque_w) + local_torque_b[tid_env, tid_body]
    )


@wp.kernel
def reset_wrench_composer_index(
    env_ids: wp.array(dtype=wp.int32),
    global_force_w: wp.array2d(dtype=wp.vec3f),
    global_torque_w: wp.array2d(dtype=wp.vec3f),
    global_force_at_com_w: wp.array2d(dtype=wp.vec3f),
    local_force_b: wp.array2d(dtype=wp.vec3f),
    local_torque_b: wp.array2d(dtype=wp.vec3f),
    out_force_b: wp.array2d(dtype=wp.vec3f),
    out_torque_b: wp.array2d(dtype=wp.vec3f),
):
    """Zero all 7 wrench composer buffers at the specified environment indices.

    Dispatched with ``dim=(len(env_ids), num_bodies)``.
    """
    tid_env, tid_body = wp.tid()
    ei = env_ids[tid_env]
    z = wp.vec3f(0.0)
    global_force_w[ei, tid_body] = z
    global_torque_w[ei, tid_body] = z
    global_force_at_com_w[ei, tid_body] = z
    local_force_b[ei, tid_body] = z
    local_torque_b[ei, tid_body] = z
    out_force_b[ei, tid_body] = z
    out_torque_b[ei, tid_body] = z


@wp.kernel
def reset_wrench_composer_mask(
    env_mask: wp.array(dtype=wp.bool),
    global_force_w: wp.array2d(dtype=wp.vec3f),
    global_torque_w: wp.array2d(dtype=wp.vec3f),
    global_force_at_com_w: wp.array2d(dtype=wp.vec3f),
    local_force_b: wp.array2d(dtype=wp.vec3f),
    local_torque_b: wp.array2d(dtype=wp.vec3f),
    out_force_b: wp.array2d(dtype=wp.vec3f),
    out_torque_b: wp.array2d(dtype=wp.vec3f),
):
    """Zero all 7 wrench composer buffers for environments matching the mask.

    Dispatched with ``dim=(num_envs, num_bodies)``.
    """
    tid_env, tid_body = wp.tid()
    if env_mask[tid_env]:
        z = wp.vec3f(0.0)
        global_force_w[tid_env, tid_body] = z
        global_torque_w[tid_env, tid_body] = z
        global_force_at_com_w[tid_env, tid_body] = z
        local_force_b[tid_env, tid_body] = z
        local_torque_b[tid_env, tid_body] = z
        out_force_b[tid_env, tid_body] = z
        out_torque_b[tid_env, tid_body] = z
