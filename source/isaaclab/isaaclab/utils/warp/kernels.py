# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
def cast_to_com_frame(position: wp.vec3f, com_pose_w: wp.transformf, is_global: bool) -> wp.vec3f:
    """Casts a position to the com frame of the body. In Newton, the com frame and the link frame are aligned.

    Args:
        position: The position to cast.
        com_pose_w: The com pose in the world frame.
        is_global: Whether the position is in the global frame.

    Returns:
        The position in the com frame of the body.
    """
    if is_global:
        return position - wp.transform_get_translation(com_pose_w)
    else:
        return position


@wp.func
def cast_force_to_com_frame(force: wp.vec3f, com_pose_w: wp.transformf, is_global: bool) -> wp.vec3f:
    """Casts a force to the com frame of the body. In Newton, the com frame and the link frame are aligned.

    Args:
        force: The force to cast.
        com_pose_w: The com pose in the world frame.
        is_global: Whether the force is applied in the global frame.
    Returns:
        The force in the com frame of the body.
    """
    if is_global:
        return wp.quat_rotate_inv(wp.transform_get_rotation(com_pose_w), force)
    else:
        return force


@wp.func
def cast_torque_to_com_frame(torque: wp.vec3f, com_pose_w: wp.transformf, is_global: bool) -> wp.vec3f:
    """Casts a torque to the com frame of the body. In Newton, the com frame and the link frame are aligned.

    Args:
        torque: The torque to cast.
        com_pose_w: The com pose in the world frame.
        is_global: Whether the torque is applied in the global frame.

    Returns:
        The torque in the com frame of the body.
    """
    if is_global:
        return wp.quat_rotate_inv(wp.transform_get_rotation(com_pose_w), torque)
    else:
        return torque


@wp.kernel
def add_forces_and_torques_at_position(
    env_mask: wp.array(dtype=wp.bool),
    body_mask: wp.array(dtype=wp.bool),
    forces: wp.array2d(dtype=wp.vec3f),
    torques: wp.array2d(dtype=wp.vec3f),
    positions: wp.array2d(dtype=wp.vec3f),
    com_poses: wp.array2d(dtype=wp.transformf),
    composed_forces_b: wp.array2d(dtype=wp.vec3f),
    composed_torques_b: wp.array2d(dtype=wp.vec3f),
    is_global: bool,
):
    """Adds forces and torques to the composed force and torque at the user-provided positions.
    When is_global is False, the user-provided positions are offsetting the application of the force relatively to the
    com frame of the body. When is_global is True, the user-provided positions are the global positions of the force
    application.

    Args:
        env_mask: The environment mask.
        body_mask: The body mask.
        forces: The forces.
        torques: The torques.
        positions: The positions.
        com_poses: The com poses.
        composed_forces_b: The composed forces.
        composed_torques_b: The composed torques.
        is_global: Whether the forces and torques are applied in the global frame.
    """
    # get the thread id
    tid_env, tid_body = wp.tid()

    if env_mask[tid_env] and body_mask[tid_body]:
        # add the forces to the composed force, if the positions are provided, also adds a torque to the composed torque.
        if forces:
            # add the forces to the composed force
            composed_forces_b[tid_env, tid_body] += cast_force_to_com_frame(
                forces[tid_env, tid_body], com_poses[tid_env, tid_body], is_global
            )
            # if there is a position offset, add a torque to the composed torque.
            if positions:
                composed_torques_b[tid_env, tid_body] += wp.cross(
                    cast_to_com_frame(positions[tid_env, tid_body], com_poses[tid_env, tid_body], is_global),
                    cast_force_to_com_frame(forces[tid_env, tid_body], com_poses[tid_env, tid_body], is_global),
                )
        if torques:
            composed_torques_b[tid_env, tid_body] += cast_torque_to_com_frame(
                torques[tid_env, tid_body], com_poses[tid_env, tid_body], is_global
            )


@wp.kernel
def set_forces_and_torques_at_position(
    env_mask: wp.array(dtype=wp.bool),
    body_mask: wp.array(dtype=wp.bool),
    forces: wp.array2d(dtype=wp.vec3f),
    torques: wp.array2d(dtype=wp.vec3f),
    positions: wp.array2d(dtype=wp.vec3f),
    com_poses: wp.array2d(dtype=wp.transformf),
    composed_forces_b: wp.array2d(dtype=wp.vec3f),
    composed_torques_b: wp.array2d(dtype=wp.vec3f),
    is_global: bool,
):
    """Sets forces and torques to the composed force and torque at the user-provided positions.
    When is_global is False, the user-provided positions are offsetting the application of the force relatively
    to the com frame of the body. When is_global is True, the user-provided positions are the global positions
    of the force application.

    Args:
        env_mask: The environment mask.
        body_mask: The body mask.
        forces: The forces.
        torques: The torques.
        positions: The positions.
        com_poses: The com poses.
        composed_forces_b: The composed forces.
        composed_torques_b: The composed torques.
        is_global: Whether the forces and torques are applied in the global frame.
    """
    # get the thread id
    tid_env, tid_body = wp.tid()

    if env_mask[tid_env] and body_mask[tid_body]:
        # set the torques to the composed torque
        if torques:
            composed_torques_b[tid_env, tid_body] = cast_torque_to_com_frame(
                torques[tid_env, tid_body], com_poses[tid_env, tid_body], is_global
            )
        # set the forces to the composed force, if the positions are provided, adds a torque to the composed torque
        # from the force at that position.
        if forces:
            # set the forces to the composed force
            composed_forces_b[tid_env, tid_body] = cast_force_to_com_frame(
                forces[tid_env, tid_body], com_poses[tid_env, tid_body], is_global
            )
            # if there is a position offset, set the torque from the force at that position.
            if positions:
                composed_torques_b[tid_env, tid_body] = wp.cross(
                    cast_to_com_frame(positions[tid_env, tid_body], com_poses[tid_env, tid_body], is_global),
                    cast_force_to_com_frame(forces[tid_env, tid_body], com_poses[tid_env, tid_body], is_global),
                )
