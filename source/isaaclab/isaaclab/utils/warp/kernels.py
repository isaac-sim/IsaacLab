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
        wp.atomic_min(ray_distance, tid_env, tid_ray, mesh_query_ray_t.t)
        # check if hit distance is less than the current hit distance, only then update the memory
        # TODO, in theory we could use the output of atomic_min to avoid the non-thread safe next comparison
        # however, warp atomic_min is returning the wrong values on gpu currently.
        # FIXME https://github.com/NVIDIA/warp/issues/1058
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
        wp.atomic_min(ray_distance, tid_env, tid_ray, mesh_query_ray_t.t)
        # check if hit distance is less than the current hit distance, only then update the memory
        # TODO, in theory we could use the output of atomic_min to avoid the non-thread safe next comparison
        # however, warp atomic_min is returning the wrong values on gpu currently.
        # FIXME https://github.com/NVIDIA/warp/issues/1058
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
# Wrench Composer
##


@wp.func
def cast_to_link_frame(position: wp.vec3f, link_position: wp.vec3f, is_global: bool) -> wp.vec3f:
    """Casts a position to the link frame of the body.

    Args:
        position: The position to cast.
        link_position: The link frame position.
        is_global: Whether the position is in the global frame.

    Returns:
        The position in the link frame of the body.
    """
    if is_global:
        return position - link_position
    else:
        return position


@wp.func
def cast_force_to_link_frame(force: wp.vec3f, link_quat: wp.quatf, is_global: bool) -> wp.vec3f:
    """Casts a force to the link frame of the body.

    Args:
        force: The force to cast.
        link_quat: The link frame quaternion.
        is_global: Whether the force is applied in the global frame.
    Returns:
        The force in the link frame of the body.
    """
    if is_global:
        return wp.quat_rotate_inv(link_quat, force)
    else:
        return force


@wp.func
def cast_torque_to_link_frame(torque: wp.vec3f, link_quat: wp.quatf, is_global: bool) -> wp.vec3f:
    """Casts a torque to the link frame of the body.

    Args:
        torque: The torque to cast.
        link_quat: The link frame quaternion.
        is_global: Whether the torque is applied in the global frame.

    Returns:
        The torque in the link frame of the body.
    """
    if is_global:
        return wp.quat_rotate_inv(link_quat, torque)
    else:
        return torque


@wp.kernel
def add_forces_and_torques_at_position_index(
    env_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
    forces: wp.array2d(dtype=wp.vec3f),
    torques: wp.array2d(dtype=wp.vec3f),
    positions: wp.array2d(dtype=wp.vec3f),
    link_poses: wp.array2d(dtype=wp.transformf),
    is_global: bool,
    composed_forces_b: wp.array2d(dtype=wp.vec3f),
    composed_torques_b: wp.array2d(dtype=wp.vec3f),
):
    """Add forces and torques to the composed wrench at user-provided positions using index selection.

    When is_global is False, the user-provided positions offset the force application relative to
    the link frame. When is_global is True, positions are in the global frame. Results are
    accumulated (added) into the composed buffers.

    .. note::
        Expects partial data from the user (indexed by env_ids/body_ids).

    Args:
        env_ids: Input array of environment indices. Shape is (num_selected_envs,).
        body_ids: Input array of body indices. Shape is (num_selected_bodies,).
        forces: Input array of forces to apply. Shape is (num_selected_envs, num_selected_bodies).
            Can be None if not provided.
        torques: Input array of torques to apply. Shape is (num_selected_envs, num_selected_bodies).
            Can be None if not provided.
        positions: Input array of position offsets for force application.
            Shape is (num_selected_envs, num_selected_bodies). Can be None if not provided.
        link_poses: Input array of link frame poses in world frame.
            Shape is (num_envs, num_bodies).
        is_global: Input flag indicating whether forces/torques/positions are in the global frame.
        composed_forces_b: Output array where forces in the link frame are accumulated.
            Shape is (num_envs, num_bodies).
        composed_torques_b: Output array where torques in the link frame are accumulated.
            Shape is (num_envs, num_bodies).
    """
    # get the thread id
    tid_env, tid_body = wp.tid()

    # add the forces to the composed force, if the positions are provided, also adds a torque to the composed torque.
    if forces:
        # add the forces to the composed force
        composed_forces_b[env_ids[tid_env], body_ids[tid_body]] += cast_force_to_link_frame(
            forces[tid_env, tid_body],
            wp.transform_get_rotation(link_poses[env_ids[tid_env], body_ids[tid_body]]),
            is_global,
        )
        # if there is a position offset, add a torque to the composed torque.
        if positions:
            composed_torques_b[env_ids[tid_env], body_ids[tid_body]] += wp.skew(
                cast_to_link_frame(
                    positions[tid_env, tid_body],
                    wp.transform_get_translation(link_poses[env_ids[tid_env], body_ids[tid_body]]),
                    is_global,
                )
            ) @ cast_force_to_link_frame(
                forces[tid_env, tid_body],
                wp.transform_get_rotation(link_poses[env_ids[tid_env], body_ids[tid_body]]),
                is_global,
            )
    if torques:
        composed_torques_b[env_ids[tid_env], body_ids[tid_body]] += cast_torque_to_link_frame(
            torques[tid_env, tid_body],
            wp.transform_get_rotation(link_poses[env_ids[tid_env], body_ids[tid_body]]),
            is_global,
        )


@wp.kernel
def set_forces_and_torques_at_position_index(
    env_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
    forces: wp.array2d(dtype=wp.vec3f),
    torques: wp.array2d(dtype=wp.vec3f),
    positions: wp.array2d(dtype=wp.vec3f),
    link_poses: wp.array2d(dtype=wp.transformf),
    is_global: bool,
    composed_forces_b: wp.array2d(dtype=wp.vec3f),
    composed_torques_b: wp.array2d(dtype=wp.vec3f),
):
    """Set forces and torques to the composed wrench at user-provided positions using index selection.

    When is_global is False, the user-provided positions offset the force application relative to
    the link frame. When is_global is True, positions are in the global frame. Results are
    overwritten (set) in the composed buffers.

    .. note::
        Expects partial data from the user (indexed by env_ids/body_ids).

    Args:
        env_ids: Input array of environment indices. Shape is (num_selected_envs,).
        body_ids: Input array of body indices. Shape is (num_selected_bodies,).
        forces: Input array of forces to apply. Shape is (num_selected_envs, num_selected_bodies).
            Can be None if not provided.
        torques: Input array of torques to apply. Shape is (num_selected_envs, num_selected_bodies).
            Can be None if not provided.
        positions: Input array of position offsets for force application.
            Shape is (num_selected_envs, num_selected_bodies). Can be None if not provided.
        link_poses: Input array of link frame poses in world frame.
            Shape is (num_envs, num_bodies).
        is_global: Input flag indicating whether forces/torques/positions are in the global frame.
        composed_forces_b: Output array where forces in the link frame are written.
            Shape is (num_envs, num_bodies).
        composed_torques_b: Output array where torques in the link frame are written.
            Shape is (num_envs, num_bodies).
    """
    # get the thread id
    tid_env, tid_body = wp.tid()

    # set the torques to the composed torque
    if torques:
        composed_torques_b[env_ids[tid_env], body_ids[tid_body]] = cast_torque_to_link_frame(
            torques[tid_env, tid_body],
            wp.transform_get_rotation(link_poses[env_ids[tid_env], body_ids[tid_body]]),
            is_global,
        )
    # set the forces to the composed force, if the positions are provided, adds a torque to the composed torque
    # from the force at that position.
    if forces:
        # set the forces to the composed force
        composed_forces_b[env_ids[tid_env], body_ids[tid_body]] = cast_force_to_link_frame(
            forces[tid_env, tid_body],
            wp.transform_get_rotation(link_poses[env_ids[tid_env], body_ids[tid_body]]),
            is_global,
        )
        # if there is a position offset, set the torque from the force at that position.
        if positions:
            composed_torques_b[env_ids[tid_env], body_ids[tid_body]] = wp.skew(
                cast_to_link_frame(
                    positions[tid_env, tid_body],
                    wp.transform_get_translation(link_poses[env_ids[tid_env], body_ids[tid_body]]),
                    is_global,
                )
            ) @ cast_force_to_link_frame(
                forces[tid_env, tid_body],
                wp.transform_get_rotation(link_poses[env_ids[tid_env], body_ids[tid_body]]),
                is_global,
            )


@wp.kernel
def add_forces_and_torques_at_position_mask(
    env_mask: wp.array(dtype=wp.bool),
    body_mask: wp.array(dtype=wp.bool),
    forces: wp.array2d(dtype=wp.vec3f),
    torques: wp.array2d(dtype=wp.vec3f),
    positions: wp.array2d(dtype=wp.vec3f),
    link_poses: wp.array2d(dtype=wp.transformf),
    is_global: bool,
    composed_forces_b: wp.array2d(dtype=wp.vec3f),
    composed_torques_b: wp.array2d(dtype=wp.vec3f),
):
    """Add forces and torques to the composed wrench at user-provided positions using mask selection.

    When is_global is False, the user-provided positions offset the force application relative to
    the link frame. When is_global is True, positions are in the global frame. Results are
    accumulated (added) into the composed buffers. Only entries where both env_mask and body_mask
    are True are processed.

    .. note::
        Expects full data from the user (num_envs x num_bodies).

    Args:
        env_mask: Input boolean mask for environments. Shape is (num_envs,).
        body_mask: Input boolean mask for bodies. Shape is (num_bodies,).
        forces: Input array of forces to apply. Shape is (num_envs, num_bodies).
            Can be None if not provided.
        torques: Input array of torques to apply. Shape is (num_envs, num_bodies).
            Can be None if not provided.
        positions: Input array of position offsets for force application.
            Shape is (num_envs, num_bodies). Can be None if not provided.
        link_poses: Input array of link frame poses in world frame.
            Shape is (num_envs, num_bodies).
        is_global: Input flag indicating whether forces/torques/positions are in the global frame.
        composed_forces_b: Output array where forces in the link frame are accumulated.
            Shape is (num_envs, num_bodies).
        composed_torques_b: Output array where torques in the link frame are accumulated.
            Shape is (num_envs, num_bodies).
    """
    # get the thread id
    tid_env, tid_body = wp.tid()

    if env_mask[tid_env] and body_mask[tid_body]:
        # add the forces to the composed force, if the positions are provided, also adds a torque to the composed
        # torque.
        if forces:
            # add the forces to the composed force
            composed_forces_b[tid_env, tid_body] += cast_force_to_link_frame(
                forces[tid_env, tid_body], wp.transform_get_rotation(link_poses[tid_env, tid_body]), is_global
            )
            # if there is a position offset, add a torque to the composed torque.
            if positions:
                composed_torques_b[tid_env, tid_body] += wp.skew(
                    cast_to_link_frame(
                        positions[tid_env, tid_body],
                        wp.transform_get_translation(link_poses[tid_env, tid_body]),
                        is_global,
                    )
                ) @ cast_force_to_link_frame(
                    forces[tid_env, tid_body], wp.transform_get_rotation(link_poses[tid_env, tid_body]), is_global
                )
        if torques:
            composed_torques_b[tid_env, tid_body] += cast_torque_to_link_frame(
                torques[tid_env, tid_body], wp.transform_get_rotation(link_poses[tid_env, tid_body]), is_global
            )


@wp.kernel
def set_forces_and_torques_at_position_mask(
    env_mask: wp.array(dtype=wp.bool),
    body_mask: wp.array(dtype=wp.bool),
    forces: wp.array2d(dtype=wp.vec3f),
    torques: wp.array2d(dtype=wp.vec3f),
    positions: wp.array2d(dtype=wp.vec3f),
    link_poses: wp.array2d(dtype=wp.transformf),
    is_global: bool,
    composed_forces_b: wp.array2d(dtype=wp.vec3f),
    composed_torques_b: wp.array2d(dtype=wp.vec3f),
):
    """Set forces and torques to the composed wrench at user-provided positions using mask selection.

    When is_global is False, the user-provided positions offset the force application relative to
    the link frame. When is_global is True, positions are in the global frame. Results are
    overwritten (set) in the composed buffers. Only entries where both env_mask and body_mask
    are True are processed.

    .. note::
        Expects full data from the user (num_envs x num_bodies).

    Args:
        env_mask: Input boolean mask for environments. Shape is (num_envs,).
        body_mask: Input boolean mask for bodies. Shape is (num_bodies,).
        forces: Input array of forces to apply. Shape is (num_envs, num_bodies).
            Can be None if not provided.
        torques: Input array of torques to apply. Shape is (num_envs, num_bodies).
            Can be None if not provided.
        positions: Input array of position offsets for force application.
            Shape is (num_envs, num_bodies). Can be None if not provided.
        link_poses: Input array of link frame poses in world frame.
            Shape is (num_envs, num_bodies).
        is_global: Input flag indicating whether forces/torques/positions are in the global frame.
        composed_forces_b: Output array where forces in the link frame are written.
            Shape is (num_envs, num_bodies).
        composed_torques_b: Output array where torques in the link frame are written.
            Shape is (num_envs, num_bodies).
    """
    # get the thread id
    tid_env, tid_body = wp.tid()

    # set the torques to the composed torque
    if env_mask[tid_env] and body_mask[tid_body]:
        if torques:
            composed_torques_b[tid_env, tid_body] = cast_torque_to_link_frame(
                torques[tid_env, tid_body], wp.transform_get_rotation(link_poses[tid_env, tid_body]), is_global
            )
        # set the forces to the composed force, if the positions are provided, adds a torque to the composed torque
        # from the force at that position.
        if forces:
            # set the forces to the composed force
            composed_forces_b[tid_env, tid_body] = cast_force_to_link_frame(
                forces[tid_env, tid_body], wp.transform_get_rotation(link_poses[tid_env, tid_body]), is_global
            )
            # if there is a position offset, set the torque from the force at that position.
            if positions:
                composed_torques_b[tid_env, tid_body] = wp.skew(
                    cast_to_link_frame(
                        positions[tid_env, tid_body],
                        wp.transform_get_translation(link_poses[tid_env, tid_body]),
                        is_global,
                    )
                ) @ cast_force_to_link_frame(
                    forces[tid_env, tid_body], wp.transform_get_rotation(link_poses[tid_env, tid_body]), is_global
                )


@wp.kernel
def reset_wrench_composer_index(
    env_ids: wp.array(dtype=wp.int32),
    composed_forces_b: wp.array2d(dtype=wp.vec3f),
    composed_torques_b: wp.array2d(dtype=wp.vec3f),
):
    """Reset the composed force and torque to zero at the specified environment indices.

    Args:
        env_ids: Input array of environment indices to reset. Shape is (num_selected_envs,).
        composed_forces_b: Output array where forces are zeroed. Shape is (num_envs, num_bodies).
        composed_torques_b: Output array where torques are zeroed. Shape is (num_envs, num_bodies).
    """

    # get the thread id
    tid_env, tid_body = wp.tid()

    # reset the composed force and torque
    composed_forces_b[env_ids[tid_env], tid_body] = wp.vec3f(0.0)
    composed_torques_b[env_ids[tid_env], tid_body] = wp.vec3f(0.0)


@wp.kernel
def reset_wrench_composer_mask(
    env_mask: wp.array(dtype=wp.bool),
    composed_forces_b: wp.array2d(dtype=wp.vec3f),
    composed_torques_b: wp.array2d(dtype=wp.vec3f),
):
    """Reset the composed force and torque to zero for environments matching the mask.

    Args:
        env_mask: Input boolean mask for environments. Shape is (num_envs,).
        composed_forces_b: Output array where forces are zeroed. Shape is (num_envs, num_bodies).
        composed_torques_b: Output array where torques are zeroed. Shape is (num_envs, num_bodies).
    """
    # get the thread id
    tid_env, tid_body = wp.tid()

    # reset the composed force and torque
    if env_mask[tid_env]:
        composed_forces_b[tid_env, tid_body] = wp.vec3f(0.0)
        composed_torques_b[tid_env, tid_body] = wp.vec3f(0.0)
