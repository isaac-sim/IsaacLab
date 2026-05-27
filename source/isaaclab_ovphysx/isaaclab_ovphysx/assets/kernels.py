# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp

vec13f = wp.types.vector(length=13, dtype=wp.float32)

"""
Shared @wp.func helpers.
"""


@wp.func
def get_link_vel_from_root_com_vel_func(
    com_vel: wp.spatial_vectorf,
    link_pose: wp.transformf,
    body_com_pose: wp.transformf,
):
    """Compute link velocity from center-of-mass velocity.

    Transforms a COM spatial velocity into a link-frame velocity by projecting
    the angular velocity contribution from the COM offset relative to the link frame.

    Args:
        com_vel: COM spatial velocity (angular, linear).
        link_pose: Link pose in world frame.
        body_com_pose: COM pose in body (link) frame.

    Returns:
        Link spatial velocity (angular, linear).
    """
    projected_vel = wp.cross(
        wp.spatial_bottom(com_vel),
        wp.quat_rotate(wp.transform_get_rotation(link_pose), -wp.transform_get_translation(body_com_pose)),
    )
    return wp.spatial_vector(wp.spatial_top(com_vel) + projected_vel, wp.spatial_bottom(com_vel))


@wp.func
def get_com_pose_from_link_pose_func(
    link_pose: wp.transformf,
    body_com_pose: wp.transformf,
):
    """Compute COM pose in world frame from link pose and body-frame COM offset.

    Args:
        link_pose: Link pose in world frame.
        body_com_pose: COM pose in body (link) frame.

    Returns:
        COM pose in world frame.
    """
    return link_pose * body_com_pose


@wp.func
def concat_pose_and_vel_to_state_func(
    pose: wp.transformf,
    vel: wp.spatial_vectorf,
) -> vec13f:
    """Concatenate a pose and velocity into a 13-element state vector.

    The state vector layout is [pos(3), quat(4), ang_vel(3), lin_vel(3)].

    Args:
        pose: Pose as a transform (position + quaternion).
        vel: Spatial velocity (angular, linear).

    Returns:
        13-element state vector.
    """
    return vec13f(
        pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6], vel[0], vel[1], vel[2], vel[3], vel[4], vel[5]
    )


@wp.func
def compute_heading_w_func(
    forward_vec: wp.vec3f,
    quat: wp.quatf,
):
    """Compute heading angle (yaw) in world frame from a forward vector and orientation.

    Rotates the forward vector by the quaternion and computes atan2(y, x).

    Args:
        forward_vec: Forward direction vector in body frame.
        quat: Orientation quaternion.

    Returns:
        Heading angle in radians.
    """
    forward_w = wp.quat_rotate(quat, forward_vec)
    return wp.atan2(forward_w[1], forward_w[0])


@wp.func
def set_state_transforms_func(
    state: vec13f,
    transform: wp.transformf,
) -> vec13f:
    """Set the pose portion (first 7 elements) of a 13-element state vector.

    Overwrites elements [0..6] (position + quaternion) with the given transform,
    leaving the velocity portion [7..12] unchanged.

    Args:
        state: 13-element state vector to modify.
        transform: New pose (position + quaternion).

    Returns:
        Updated 13-element state vector.
    """
    state[0] = transform[0]
    state[1] = transform[1]
    state[2] = transform[2]
    state[3] = transform[3]
    state[4] = transform[4]
    state[5] = transform[5]
    state[6] = transform[6]
    return state


@wp.func
def set_state_velocities_func(
    state: vec13f,
    velocity: wp.spatial_vectorf,
) -> vec13f:
    """Set the velocity portion (last 6 elements) of a 13-element state vector.

    Overwrites elements [7..12] (angular + linear velocity) with the given spatial velocity,
    leaving the pose portion [0..6] unchanged.

    Args:
        state: 13-element state vector to modify.
        velocity: New spatial velocity (angular, linear).

    Returns:
        Updated 13-element state vector.
    """
    state[7] = velocity[0]
    state[8] = velocity[1]
    state[9] = velocity[2]
    state[10] = velocity[3]
    state[11] = velocity[4]
    state[12] = velocity[5]
    return state


@wp.func
def get_link_velocity_in_com_frame_func(
    link_velocity_w: wp.spatial_vectorf,
    link_pose_w: wp.transformf,
    body_com_pose_b: wp.transformf,
):
    """Compute COM velocity from link velocity by accounting for the COM offset.

    Transforms a link-frame spatial velocity into a COM-frame velocity by adding
    the cross-product contribution of the COM offset rotated into the world frame.

    Args:
        link_velocity_w: Link spatial velocity in world frame (angular, linear).
        link_pose_w: Link pose in world frame.
        body_com_pose_b: COM pose in body (link) frame.

    Returns:
        COM spatial velocity in world frame (angular, linear).
    """
    return wp.spatial_vector(
        wp.spatial_top(link_velocity_w)
        + wp.cross(
            wp.spatial_bottom(link_velocity_w),
            wp.quat_rotate(wp.transform_get_rotation(link_pose_w), wp.transform_get_translation(body_com_pose_b)),
        ),
        wp.spatial_bottom(link_velocity_w),
    )


@wp.func
def get_com_pose_in_link_frame_func(
    com_pose_w: wp.transformf,
    com_pose_b: wp.transformf,
):
    """Compute link pose in world frame from COM pose by inverting the body-frame COM offset.

    This is the inverse of ``get_com_pose_from_link_pose_func``. Given the COM pose in
    world frame and the COM offset in body frame, it recovers the link pose in world frame.

    Args:
        com_pose_w: COM pose in world frame.
        com_pose_b: COM pose in body (link) frame.

    Returns:
        Link pose in world frame.
    """
    T2 = wp.transform(
        wp.quat_rotate(
            wp.quat_inverse(wp.transform_get_rotation(com_pose_b)), -wp.transform_get_translation(com_pose_b)
        ),
        wp.quat_inverse(wp.transform_get_rotation(com_pose_b)),
    )
    link_pose_w = com_pose_w * T2
    return link_pose_w


"""
Root-level @wp.kernel (1D — used by RigidObject + Articulation).
"""


@wp.kernel
def get_root_link_vel_from_root_com_vel(
    com_vel: wp.array(dtype=wp.spatial_vectorf),
    link_pose: wp.array(dtype=wp.transformf),
    body_com_pose_b: wp.array2d(dtype=wp.transformf),
    link_vel: wp.array(dtype=wp.spatial_vectorf),
):
    """Compute root link velocity from root center-of-mass velocity.

    This kernel transforms the root COM velocity into link-frame velocity by projecting
    the angular velocity contribution from the COM offset.

    Args:
        com_vel: Input array of root COM spatial velocities. Shape is (num_envs,).
        link_pose: Input array of root link poses in world frame. Shape is (num_envs,).
        body_com_pose_b: Input array of body COM poses in body frame. Shape is (num_envs, num_bodies).
            Only the first body (index 0) is used for the root.
        link_vel: Output array where root link velocities are written. Shape is (num_envs,).
    """
    i = wp.tid()
    link_vel[i] = get_link_vel_from_root_com_vel_func(com_vel[i], link_pose[i], body_com_pose_b[i, 0])


@wp.kernel
def get_root_com_pose_from_root_link_pose(
    link_pose: wp.array(dtype=wp.transformf),
    body_com_pose_b: wp.array2d(dtype=wp.transformf),
    com_pose_w: wp.array(dtype=wp.transformf),
):
    """Compute root COM pose from root link pose.

    This kernel transforms the root link pose to the root COM pose using the body COM offset.

    Args:
        link_pose: Input array of root link poses in world frame. Shape is (num_envs,).
        body_com_pose_b: Input array of body COM poses in body frame. Shape is (num_envs, num_bodies).
            Only the first body (index 0) is used for the root.
        com_pose_w: Output array where root COM poses are written. Shape is (num_envs,).
    """
    i = wp.tid()
    com_pose_w[i] = get_com_pose_from_link_pose_func(link_pose[i], body_com_pose_b[i, 0])


@wp.kernel
def concat_root_pose_and_vel_to_state(
    pose: wp.array(dtype=wp.transformf),
    vel: wp.array(dtype=wp.spatial_vectorf),
    state: wp.array(dtype=vec13f),
):
    """Concatenate root pose and velocity into a 13-element state vector.

    This kernel combines a 7-element pose (pos + quat) and a 6-element velocity
    (angular + linear) into a single 13-element state vector.

    Args:
        pose: Input array of root poses in world frame. Shape is (num_envs,).
        vel: Input array of root spatial velocities. Shape is (num_envs,).
        state: Output array where concatenated state vectors are written. Shape is (num_envs,).
    """
    i = wp.tid()
    state[i] = concat_pose_and_vel_to_state_func(pose[i], vel[i])


@wp.kernel
def split_state_to_root_pose_and_vel(
    state: wp.array2d(dtype=wp.float32),
    pose: wp.array(dtype=wp.transformf),
    vel: wp.array(dtype=wp.spatial_vectorf),
):
    """Split a 13-element state vector into root pose and velocity.

    This kernel extracts a 7-element pose (pos + quat) and a 6-element velocity
    (angular + linear) from a 13-element state vector.

    Args:
        state: Input array of root states. Shape is (num_envs, 13).
        pose: Output array where root poses are written. Shape is (num_envs,).
        vel: Output array where root spatial velocities are written. Shape is (num_envs,).
    """
    i = wp.tid()
    # Extract pose: [pos(3), quat(4)] = state[0:7]
    pose[i] = wp.transform(
        wp.vec3f(state[i, 0], state[i, 1], state[i, 2]), wp.quatf(state[i, 3], state[i, 4], state[i, 5], state[i, 6])
    )
    # Extract velocity: [ang_vel(3), lin_vel(3)] = state[7:13]
    vel[i] = wp.spatial_vector(
        wp.vec3f(state[i, 7], state[i, 8], state[i, 9]),  # angular velocity
        wp.vec3f(state[i, 10], state[i, 11], state[i, 12]),  # linear velocity
    )


"""
Body-level @wp.kernel (2D — used by Articulation + RigidObjectCollection).
"""


@wp.kernel
def get_body_link_vel_from_body_com_vel(
    body_com_vel: wp.array2d(dtype=wp.spatial_vectorf),
    body_link_pose: wp.array2d(dtype=wp.transformf),
    body_com_pose: wp.array2d(dtype=wp.transformf),
    body_link_vel: wp.array2d(dtype=wp.spatial_vectorf),
):
    """Compute body link velocities from body COM velocities for all bodies.

    This kernel transforms COM velocities into link-frame velocities by projecting
    the angular velocity contribution from the COM offset, for each body in each environment.

    Args:
        body_com_vel: Input array of body COM spatial velocities. Shape is (num_envs, num_bodies).
        body_link_pose: Input array of body link poses in world frame. Shape is (num_envs, num_bodies).
        body_com_pose: Input array of body COM poses in body frame. Shape is (num_envs, num_bodies).
        body_link_vel: Output array where body link velocities are written. Shape is (num_envs, num_bodies).
    """
    i, j = wp.tid()
    body_link_vel[i, j] = get_link_vel_from_root_com_vel_func(
        body_com_vel[i, j], body_link_pose[i, j], body_com_pose[i, j]
    )


@wp.kernel
def get_body_com_pose_from_body_link_pose(
    body_link_pose: wp.array2d(dtype=wp.transformf),
    body_com_pose_b: wp.array2d(dtype=wp.transformf),
    body_com_pose_w: wp.array2d(dtype=wp.transformf),
):
    """Compute body COM poses from body link poses for all bodies.

    This kernel transforms link poses to COM poses using the body COM offset in the body frame.

    Args:
        body_link_pose: Input array of body link poses in world frame. Shape is (num_envs, num_bodies).
        body_com_pose_b: Input array of body COM poses in body frame. Shape is (num_envs, num_bodies).
        body_com_pose_w: Output array where body COM poses in world frame are written.
            Shape is (num_envs, num_bodies).
    """
    i, j = wp.tid()
    body_com_pose_w[i, j] = get_com_pose_from_link_pose_func(body_link_pose[i, j], body_com_pose_b[i, j])


@wp.kernel
def concat_body_pose_and_vel_to_state(
    pose: wp.array2d(dtype=wp.transformf),
    vel: wp.array2d(dtype=wp.spatial_vectorf),
    state: wp.array2d(dtype=vec13f),
):
    """Concatenate body pose and velocity into 13-element state vectors for all bodies.

    This kernel combines a 7-element pose (pos + quat) and a 6-element velocity
    (angular + linear) into a single 13-element state vector, for each body in each environment.

    Args:
        pose: Input array of body poses in world frame. Shape is (num_envs, num_bodies).
        vel: Input array of body spatial velocities. Shape is (num_envs, num_bodies).
        state: Output array where concatenated state vectors are written.
            Shape is (num_envs, num_bodies).
    """
    i, j = wp.tid()
    state[i, j] = concat_pose_and_vel_to_state_func(pose[i, j], vel[i, j])


"""
Derived property kernels.
"""


@wp.kernel
def quat_apply_inverse_1D_kernel(
    gravity: wp.array(dtype=wp.vec3f),
    quat: wp.array(dtype=wp.quatf),
    projected_gravity: wp.array(dtype=wp.vec3f),
):
    """Apply inverse quaternion rotation to gravity vectors (1D).

    This kernel rotates gravity vectors into the local frame of each environment
    using the inverse of the provided quaternion.

    Args:
        gravity: Input array of gravity vectors in world frame. Shape is (num_envs,).
        quat: Input array of quaternions representing orientations. Shape is (num_envs,).
        projected_gravity: Output array where projected gravity vectors are written.
            Shape is (num_envs,).
    """
    i = wp.tid()
    projected_gravity[i] = wp.quat_rotate_inv(quat[i], gravity[i])


@wp.kernel
def root_heading_w(
    forward_vec: wp.array(dtype=wp.vec3f),
    quat: wp.array(dtype=wp.quatf),
    heading_w: wp.array(dtype=wp.float32),
):
    """Compute root heading angle in the world frame.

    This kernel computes the heading angle (yaw) by rotating the forward vector
    by the root quaternion and computing atan2 of the resulting x and y components.

    Args:
        forward_vec: Input array of forward direction vectors. Shape is (num_envs,).
        quat: Input array of root quaternions. Shape is (num_envs,).
        heading_w: Output array where heading angles (radians) are written. Shape is (num_envs,).
    """
    i = wp.tid()
    heading_w[i] = compute_heading_w_func(forward_vec[i], quat[i])


@wp.kernel
def quat_apply_inverse_2D_kernel(
    vec: wp.array2d(dtype=wp.vec3f),
    quat: wp.array2d(dtype=wp.quatf),
    result: wp.array2d(dtype=wp.vec3f),
):
    """Apply inverse quaternion rotation to vectors (2D).

    This kernel rotates vectors into the local frame of each body in each environment
    using the inverse of the provided quaternion.

    Args:
        vec: Input array of vectors in world frame. Shape is (num_envs, num_bodies).
        quat: Input array of quaternions representing orientations. Shape is (num_envs, num_bodies).
        result: Output array where rotated vectors are written. Shape is (num_envs, num_bodies).
    """
    i, j = wp.tid()
    result[i, j] = wp.quat_rotate_inv(quat[i, j], vec[i, j])


@wp.kernel
def body_heading_w(
    forward_vec: wp.array2d(dtype=wp.vec3f),
    quat: wp.array2d(dtype=wp.quatf),
    heading_w: wp.array2d(dtype=wp.float32),
):
    """Compute body heading angles in the world frame for all bodies.

    This kernel computes heading angles (yaw) by rotating forward vectors
    by body quaternions and computing atan2 of the resulting x and y components.

    Args:
        forward_vec: Input array of forward direction vectors. Shape is (num_envs, num_bodies).
        quat: Input array of body quaternions. Shape is (num_envs, num_bodies).
        heading_w: Output array where heading angles (radians) are written.
            Shape is (num_envs, num_bodies).
    """
    i, j = wp.tid()
    heading_w[i, j] = compute_heading_w_func(forward_vec[i, j], quat[i, j])


"""
Root-level write kernels (1D — used by RigidObject + Articulation).
"""


@wp.kernel
def set_root_link_pose_to_sim_index(
    data: wp.array(dtype=wp.transformf),
    env_ids: wp.array(dtype=wp.int32),
    root_link_pose_w: wp.array(dtype=wp.transformf),
):
    """Write root link pose data to simulation buffers.

    This kernel scatters root link poses from the partial input array into the cached
    world-frame buffer at the specified environment indices.

    Args:
        data: Input array of root link poses. Shape is (num_selected_envs,).
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        root_link_pose_w: Output array where root link poses are written. Shape is (num_envs,).
    """
    i = wp.tid()
    root_link_pose_w[env_ids[i]] = data[i]


@wp.kernel
def set_root_com_pose_to_sim_index(
    data: wp.array(dtype=wp.transformf),
    body_com_pose_b: wp.array2d(dtype=wp.transformf),
    env_ids: wp.array(dtype=wp.int32),
    root_com_pose_w: wp.array(dtype=wp.transformf),
    root_link_pose_w: wp.array(dtype=wp.transformf),
):
    """Write root COM pose data to simulation buffers.

    This kernel scatters root COM poses from the partial input array into the cached
    world-frame buffer at the specified environment indices and derives the
    corresponding link pose via the body-frame COM offset.

    Args:
        data: Input array of root COM poses. Shape is (num_selected_envs,).
        body_com_pose_b: Input array of body COM poses in body frame. Shape is
            (num_envs, num_bodies). Only the first body (index 0) is used for the root.
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        root_com_pose_w: Output array where root COM poses are written. Shape is (num_envs,).
        root_link_pose_w: Output array where root link poses (derived from COM) are written.
            Shape is (num_envs,).
    """
    i = wp.tid()
    root_com_pose_w[env_ids[i]] = data[i]
    # Get the com pose in the link frame
    root_link_pose_w[env_ids[i]] = get_com_pose_in_link_frame_func(
        root_com_pose_w[env_ids[i]], body_com_pose_b[env_ids[i], 0]
    )


@wp.kernel
def set_root_com_velocity_to_sim_index(
    data: wp.array(dtype=wp.spatial_vectorf),
    env_ids: wp.array(dtype=wp.int32),
    num_bodies: wp.int32,
    root_com_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    body_acc_w: wp.array2d(dtype=wp.spatial_vectorf),
):
    """Write root COM velocity data to simulation buffers.

    This kernel scatters root COM velocities from the partial input array into the cached
    world-frame buffer at the specified environment indices and zeros the body acceleration
    buffer to prevent reporting stale values.

    Args:
        data: Input array of root COM spatial velocities. Shape is (num_selected_envs,).
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        num_bodies: Input scalar number of bodies per environment.
        root_com_velocity_w: Output array where root COM velocities are written. Shape is (num_envs,).
        body_acc_w: Output array where body accelerations are zeroed. Shape is
            (num_envs, num_bodies).
    """
    i = wp.tid()
    root_com_velocity_w[env_ids[i]] = data[i]
    # Make the acceleration zero to prevent reporting old values
    for j in range(num_bodies):
        body_acc_w[env_ids[i], j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def set_root_link_velocity_to_sim_index(
    data: wp.array(dtype=wp.spatial_vectorf),
    body_com_pose_b: wp.array2d(dtype=wp.transformf),
    link_pose_w: wp.array(dtype=wp.transformf),
    env_ids: wp.array(dtype=wp.int32),
    num_bodies: wp.int32,
    root_link_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    root_com_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    body_acc_w: wp.array2d(dtype=wp.spatial_vectorf),
):
    """Write root link velocity data to simulation buffers.

    This kernel scatters root link velocities from the partial input array into the cached
    world-frame buffer at the specified environment indices, derives the corresponding
    COM velocity via the lever-arm transform, and zeros the body acceleration buffer.

    Args:
        data: Input array of root link spatial velocities. Shape is (num_selected_envs,).
        body_com_pose_b: Input array of body COM poses in body frame. Shape is
            (num_envs, num_bodies). Only the first body (index 0) is used for the root.
        link_pose_w: Input array of root link poses in world frame. Shape is (num_envs,).
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        num_bodies: Input scalar number of bodies per environment.
        root_link_velocity_w: Output array where root link velocities are written.
            Shape is (num_envs,).
        root_com_velocity_w: Output array where root COM velocities (derived from link)
            are written. Shape is (num_envs,).
        body_acc_w: Output array where body accelerations are zeroed.
            Shape is (num_envs, num_bodies).
    """
    i = wp.tid()
    root_link_velocity_w[env_ids[i]] = data[i]
    # Get the link velocity in the com frame
    root_com_velocity_w[env_ids[i]] = get_link_velocity_in_com_frame_func(
        root_link_velocity_w[env_ids[i]], link_pose_w[env_ids[i]], body_com_pose_b[env_ids[i], 0]
    )
    # Make the acceleration zero to prevent reporting old values
    for j in range(num_bodies):
        body_acc_w[env_ids[i], j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


"""
Body-level write kernels (2D — used by RigidObjectCollection).
"""


@wp.kernel
def set_body_link_pose_to_sim(
    data: wp.array2d(dtype=wp.transformf),
    env_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
    from_mask: bool,
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    body_link_state_w: wp.array2d(dtype=vec13f),
    body_state_w: wp.array2d(dtype=vec13f),
):
    """Write body link pose data to simulation buffers.

    This kernel writes body link poses from the input array to the output buffers
    and optionally updates the corresponding state vectors, for each body in each environment.

    Args:
        data: Input array of body link poses. Shape is (num_envs, num_bodies) or
            (num_selected_envs, num_selected_bodies) depending on from_mask.
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        body_ids: Input array of body indices to write to. Shape is (num_selected_bodies,).
        from_mask: Input flag indicating whether to use masked indexing.
        body_link_pose_w: Output array where body link poses are written.
            Shape is (num_envs, num_bodies).
        body_link_state_w: Output array where body link states are updated (pose portion).
            Shape is (num_envs, num_bodies). Can be None if not needed.
        body_state_w: Output array where body states are updated (pose portion).
            Shape is (num_envs, num_bodies). Can be None if not needed.
    """
    i, j = wp.tid()
    if from_mask:
        body_link_pose_w[env_ids[i], body_ids[j]] = data[env_ids[i], body_ids[j]]
        if body_link_state_w:
            body_link_state_w[env_ids[i], body_ids[j]] = set_state_transforms_func(
                body_link_state_w[env_ids[i], body_ids[j]], data[env_ids[i], body_ids[j]]
            )
        if body_state_w:
            body_state_w[env_ids[i], body_ids[j]] = set_state_transforms_func(
                body_state_w[env_ids[i], body_ids[j]], data[env_ids[i], body_ids[j]]
            )
    else:
        body_link_pose_w[env_ids[i], body_ids[j]] = data[i, j]
        if body_link_state_w:
            body_link_state_w[env_ids[i], body_ids[j]] = set_state_transforms_func(
                body_link_state_w[env_ids[i], body_ids[j]], data[i, j]
            )
        if body_state_w:
            body_state_w[env_ids[i], body_ids[j]] = set_state_transforms_func(
                body_state_w[env_ids[i], body_ids[j]], data[i, j]
            )


@wp.kernel
def set_body_com_pose_to_sim(
    data: wp.array2d(dtype=wp.transformf),
    body_com_pose_b: wp.array2d(dtype=wp.transformf),
    env_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
    from_mask: bool,
    body_com_pose_w: wp.array2d(dtype=wp.transformf),
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    body_com_state_w: wp.array2d(dtype=vec13f),
    body_link_state_w: wp.array2d(dtype=vec13f),
    body_state_w: wp.array2d(dtype=vec13f),
):
    """Write body COM pose data to simulation buffers.

    This kernel writes body COM poses from the input array to the output buffers,
    computes the corresponding link poses from the COM poses, and optionally updates
    the corresponding state vectors, for each body in each environment.

    Args:
        data: Input array of body COM poses. Shape is (num_envs, num_bodies) or
            (num_selected_envs, num_selected_bodies) depending on from_mask.
        body_com_pose_b: Input array of body COM poses in body frame. Shape is
            (num_envs, num_bodies).
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        body_ids: Input array of body indices to write to. Shape is (num_selected_bodies,).
        from_mask: Input flag indicating whether to use masked indexing.
        body_com_pose_w: Output array where body COM poses are written.
            Shape is (num_envs, num_bodies).
        body_link_pose_w: Output array where body link poses (derived from COM) are written.
            Shape is (num_envs, num_bodies).
        body_com_state_w: Output array where body COM states are updated (pose portion).
            Shape is (num_envs, num_bodies). Can be None if not needed.
        body_link_state_w: Output array where body link states are updated (pose portion).
            Shape is (num_envs, num_bodies). Can be None if not needed.
        body_state_w: Output array where body states are updated (pose portion).
            Shape is (num_envs, num_bodies). Can be None if not needed.
    """
    i, j = wp.tid()
    if from_mask:
        body_com_pose_w[env_ids[i], body_ids[j]] = data[env_ids[i], body_ids[j]]
        if body_com_state_w:
            body_com_state_w[env_ids[i], body_ids[j]] = set_state_transforms_func(
                body_com_state_w[env_ids[i], body_ids[j]], data[env_ids[i], body_ids[j]]
            )
    else:
        body_com_pose_w[env_ids[i], body_ids[j]] = data[i, j]
        if body_com_state_w:
            body_com_state_w[env_ids[i], body_ids[j]] = set_state_transforms_func(
                body_com_state_w[env_ids[i], body_ids[j]], data[i, j]
            )
    # Get the link pose from com pose
    body_link_pose_w[env_ids[i], body_ids[j]] = get_com_pose_in_link_frame_func(
        body_com_pose_w[env_ids[i], body_ids[j]], body_com_pose_b[env_ids[i], body_ids[j]]
    )
    if body_link_state_w:
        body_link_state_w[env_ids[i], body_ids[j]] = set_state_transforms_func(
            body_link_state_w[env_ids[i], body_ids[j]], body_link_pose_w[env_ids[i], body_ids[j]]
        )
    if body_state_w:
        body_state_w[env_ids[i], body_ids[j]] = set_state_transforms_func(
            body_state_w[env_ids[i], body_ids[j]], body_link_pose_w[env_ids[i], body_ids[j]]
        )


@wp.kernel
def set_body_com_velocity_to_sim(
    data: wp.array2d(dtype=wp.spatial_vectorf),
    env_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
    from_mask: bool,
    body_com_velocity_w: wp.array2d(dtype=wp.spatial_vectorf),
    body_acc_w: wp.array2d(dtype=wp.spatial_vectorf),
    body_state_w: wp.array2d(dtype=vec13f),
    body_com_state_w: wp.array2d(dtype=vec13f),
):
    """Write body COM velocity data to simulation buffers.

    This kernel writes body COM velocities from the input array to the output buffers,
    optionally updates the corresponding state vectors, and zeros out the body
    acceleration buffer, for each body in each environment.

    Args:
        data: Input array of body COM spatial velocities. Shape is (num_envs, num_bodies) or
            (num_selected_envs, num_selected_bodies) depending on from_mask.
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        body_ids: Input array of body indices to write to. Shape is (num_selected_bodies,).
        from_mask: Input flag indicating whether to use masked indexing.
        body_com_velocity_w: Output array where body COM velocities are written.
            Shape is (num_envs, num_bodies).
        body_acc_w: Output array where body accelerations are zeroed.
            Shape is (num_envs, num_bodies).
        body_state_w: Output array where body states are updated (velocity portion).
            Shape is (num_envs, num_bodies). Can be None if not needed.
        body_com_state_w: Output array where body COM states are updated (velocity portion).
            Shape is (num_envs, num_bodies). Can be None if not needed.
    """
    i, j = wp.tid()
    if from_mask:
        body_com_velocity_w[env_ids[i], body_ids[j]] = data[env_ids[i], body_ids[j]]
        if body_state_w:
            body_state_w[env_ids[i], body_ids[j]] = set_state_velocities_func(
                body_state_w[env_ids[i], body_ids[j]], data[env_ids[i], body_ids[j]]
            )
        if body_com_state_w:
            body_com_state_w[env_ids[i], body_ids[j]] = set_state_velocities_func(
                body_com_state_w[env_ids[i], body_ids[j]], data[env_ids[i], body_ids[j]]
            )
    else:
        body_com_velocity_w[env_ids[i], body_ids[j]] = data[i, j]
        if body_state_w:
            body_state_w[env_ids[i], body_ids[j]] = set_state_velocities_func(
                body_state_w[env_ids[i], body_ids[j]], data[i, j]
            )
        if body_com_state_w:
            body_com_state_w[env_ids[i], body_ids[j]] = set_state_velocities_func(
                body_com_state_w[env_ids[i], body_ids[j]], data[i, j]
            )
    # Make the acceleration zero to prevent reporting old values
    body_acc_w[env_ids[i], body_ids[j]] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def set_body_link_velocity_to_sim(
    data: wp.array2d(dtype=wp.spatial_vectorf),
    body_com_pose_b: wp.array2d(dtype=wp.transformf),
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    env_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
    from_mask: bool,
    body_link_velocity_w: wp.array2d(dtype=wp.spatial_vectorf),
    body_com_velocity_w: wp.array2d(dtype=wp.spatial_vectorf),
    body_acc_w: wp.array2d(dtype=wp.spatial_vectorf),
    body_link_state_w: wp.array2d(dtype=vec13f),
    body_state_w: wp.array2d(dtype=vec13f),
    body_com_state_w: wp.array2d(dtype=vec13f),
):
    """Write body link velocity data to simulation buffers.

    This kernel writes body link velocities from the input array to the output buffers,
    computes the corresponding COM velocities from the link velocities, optionally updates
    the corresponding state vectors, and zeros out the body acceleration buffer.

    Args:
        data: Input array of body link spatial velocities. Shape is (num_envs, num_bodies)
            or (num_selected_envs, num_selected_bodies) depending on from_mask.
        body_com_pose_b: Input array of body COM poses in body frame. Shape is
            (num_envs, num_bodies).
        body_link_pose_w: Input array of body link poses in world frame. Shape is
            (num_envs, num_bodies).
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        body_ids: Input array of body indices to write to. Shape is (num_selected_bodies,).
        from_mask: Input flag indicating whether to use masked indexing.
        body_link_velocity_w: Output array where body link velocities are written.
            Shape is (num_envs, num_bodies).
        body_com_velocity_w: Output array where body COM velocities (derived from link)
            are written. Shape is (num_envs, num_bodies).
        body_acc_w: Output array where body accelerations are zeroed.
            Shape is (num_envs, num_bodies).
        body_link_state_w: Output array where body link states are updated (velocity portion).
            Shape is (num_envs, num_bodies). Can be None if not needed.
        body_state_w: Output array where body states are updated (velocity portion).
            Shape is (num_envs, num_bodies). Can be None if not needed.
        body_com_state_w: Output array where body COM states are updated (velocity portion).
            Shape is (num_envs, num_bodies). Can be None if not needed.
    """
    i, j = wp.tid()
    if from_mask:
        body_link_velocity_w[env_ids[i], body_ids[j]] = data[env_ids[i], body_ids[j]]
        if body_link_state_w:
            body_link_state_w[env_ids[i], body_ids[j]] = set_state_velocities_func(
                body_link_state_w[env_ids[i], body_ids[j]], data[env_ids[i], body_ids[j]]
            )
    else:
        body_link_velocity_w[env_ids[i], body_ids[j]] = data[i, j]
        if body_link_state_w:
            body_link_state_w[env_ids[i], body_ids[j]] = set_state_velocities_func(
                body_link_state_w[env_ids[i], body_ids[j]], data[i, j]
            )
    # Get the link velocity in the com frame
    body_com_velocity_w[env_ids[i], body_ids[j]] = get_link_velocity_in_com_frame_func(
        body_link_velocity_w[env_ids[i], body_ids[j]],
        body_link_pose_w[env_ids[i], body_ids[j]],
        body_com_pose_b[env_ids[i], body_ids[j]],
    )
    if body_com_state_w:
        body_com_state_w[env_ids[i], body_ids[j]] = set_state_velocities_func(
            body_com_state_w[env_ids[i], body_ids[j]], body_com_velocity_w[env_ids[i], body_ids[j]]
        )
    if body_state_w:
        body_state_w[env_ids[i], body_ids[j]] = set_state_velocities_func(
            body_state_w[env_ids[i], body_ids[j]], body_com_velocity_w[env_ids[i], body_ids[j]]
        )
    # Make the acceleration zero to prevent reporting old values
    body_acc_w[env_ids[i], body_ids[j]] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


"""
Generic buffer-writing kernels (used by Articulation + RigidObject + RigidObjectCollection).
"""


@wp.kernel
def write_2d_data_to_buffer_with_indices(
    in_data: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=wp.int32),
    joint_ids: wp.array(dtype=wp.int32),
    out_data: wp.array2d(dtype=wp.float32),
):
    """Write 2D float data to a buffer at specified indices.

    This kernel copies float data from a partial input array to an output buffer at the
    specified environment and joint/body indices.

    Args:
        in_data: Input array containing float data. Shape is (num_selected_envs, num_selected_joints).
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        joint_ids: Input array of joint/body indices to write to. Shape is (num_selected_joints,).
        out_data: Output array where data is written. Shape is (num_envs, num_joints).
    """
    i, j = wp.tid()
    out_data[env_ids[i], joint_ids[j]] = in_data[i, j]


@wp.kernel
def write_body_inertia_to_buffer_index(
    in_data: wp.array3d(dtype=wp.float32),
    env_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
    out_data: wp.array3d(dtype=wp.float32),
):
    """Write body inertia data to a buffer at specified indices.

    This kernel copies 3x3 inertia tensor data (stored as 9 floats) from a partial input
    array to an output buffer at the specified environment and body indices.

    Args:
        in_data: Input array containing inertia data. Shape is (num_selected_envs, num_selected_bodies, 9).
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        body_ids: Input array of body indices to write to. Shape is (num_selected_bodies,).
        out_data: Output array where inertia data is written. Shape is (num_envs, num_bodies, 9).
    """
    i, j = wp.tid()
    for k in range(9):
        out_data[env_ids[i], body_ids[j], k] = in_data[i, j, k]


@wp.kernel
def write_body_com_pose_to_buffer_index(
    in_data: wp.array2d(dtype=wp.transformf),
    env_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
    out_data: wp.array2d(dtype=wp.transformf),
):
    """Write body COM pose data to a buffer at specified indices.

    This kernel copies body COM pose data from a partial input array to an output buffer
    at the specified environment and body indices.

    Args:
        in_data: Input array containing body COM poses. Shape is (num_selected_envs, num_selected_bodies).
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        body_ids: Input array of body indices to write to. Shape is (num_selected_bodies,).
        out_data: Output array where body COM poses are written. Shape is (num_envs, num_bodies).
    """
    i, j = wp.tid()
    out_data[env_ids[i], body_ids[j]] = in_data[i, j]


@wp.kernel
def derive_body_acceleration_from_body_com_velocities(
    body_com_vel: wp.array2d(dtype=wp.spatial_vectorf),
    dt: wp.float32,
    prev_body_com_vel: wp.array2d(dtype=wp.spatial_vectorf),
    body_acc: wp.array2d(dtype=wp.spatial_vectorf),
):
    """Derive body acceleration from body COM velocities.

    This kernel derives body acceleration from body COM velocities using finite differencing.

    Args:
        body_com_vel: Input array of body COM velocities. Shape is (num_envs, num_bodies).
        dt: Input time step (scalar) used for finite differencing.
        prev_body_com_vel: Input/output array of previous body COM velocities. Shape is (num_envs, num_bodies).
        body_acc: Output array where body accelerations are written. Shape is (num_envs, num_bodies).
    """
    i, j = wp.tid()
    # Compute the acceleration
    body_acc[i, j] = (body_com_vel[i, j] - prev_body_com_vel[i, j]) / dt
    # Update the previous body COM velocity
    prev_body_com_vel[i, j] = body_com_vel[i, j]


@wp.kernel
def _body_wrench_to_world(
    force_b: wp.array(dtype=wp.vec3f, ndim=2),
    torque_b: wp.array(dtype=wp.vec3f, ndim=2),
    poses: wp.array(dtype=wp.transformf, ndim=2),
    wrench_out: wp.array(dtype=wp.float32, ndim=3),
):
    """Rotate body-frame force/torque to world frame and pack into a flat output array.

    Output layout per ``(i, j)`` slice (9 floats total):

    * ``[0:3]`` -- world-frame force ``[N]``
    * ``[3:6]`` -- world-frame torque ``[N*m]``
    * ``[6:9]`` -- world-frame link position ``[m]``

    Args:
        force_b: Body-frame applied forces ``[N]``. Shape is ``(N, L)``.
        torque_b: Body-frame applied torques ``[N*m]``. Shape is ``(N, L)``.
        poses: Link poses in world frame. Shape is ``(N, L)``.
        wrench_out: Output packed wrench array. Shape is ``(N, L, 9)``.
    """
    i, j = wp.tid()
    q = wp.transform_get_rotation(poses[i, j])
    f_w = wp.quat_rotate(q, force_b[i, j])
    t_w = wp.quat_rotate(q, torque_b[i, j])
    wrench_out[i, j, 0] = f_w[0]
    wrench_out[i, j, 1] = f_w[1]
    wrench_out[i, j, 2] = f_w[2]
    wrench_out[i, j, 3] = t_w[0]
    wrench_out[i, j, 4] = t_w[1]
    wrench_out[i, j, 5] = t_w[2]
    p_w = wp.transform_get_translation(poses[i, j])
    wrench_out[i, j, 6] = p_w[0]
    wrench_out[i, j, 7] = p_w[1]
    wrench_out[i, j, 8] = p_w[2]


@wp.kernel
def _scatter_rows_partial(
    dst: wp.array2d(dtype=wp.float32),
    src: wp.array2d(dtype=wp.float32),
    ids: wp.array(dtype=wp.int32),
):
    """Scatter a partial row-indexed source array into a larger destination array.

    For each thread ``(i, j)`` writes ``dst[ids[i], j] = src[i, j]``.

    Args:
        dst: Destination array of shape ``(N, C)`` to scatter values into.
        src: Source array of shape ``(K, C)`` containing the values to scatter.
        ids: Row indices into ``dst`` for each row of ``src``. Shape is ``(K,)``.
    """
    i, j = wp.tid()
    dst[ids[i], j] = src[i, j]


"""
Native-mask scatter kernels (mirrors Newton; the OVPhysX wheel's ``binding.write`` natively
supports a boolean mask via the ``mask=`` argument, so the ``*_mask`` setters update the cache
in-place and pass the mask straight through to the wheel without a ``torch.nonzero`` round-trip).
"""


@wp.kernel
def set_root_link_pose_to_sim_mask(
    data: wp.array(dtype=wp.transformf),
    env_mask: wp.array(dtype=wp.bool),
    root_link_pose_w: wp.array(dtype=wp.transformf),
):
    """Mask-scatter root link poses into the cache; rows where ``env_mask[i]`` is False are untouched."""
    i = wp.tid()
    if env_mask[i]:
        root_link_pose_w[i] = data[i]


@wp.kernel
def set_root_com_pose_to_sim_mask(
    data: wp.array(dtype=wp.transformf),
    body_com_pose_b: wp.array2d(dtype=wp.transformf),
    env_mask: wp.array(dtype=wp.bool),
    root_com_pose_w: wp.array(dtype=wp.transformf),
    root_link_pose_w: wp.array(dtype=wp.transformf),
):
    """Mask-scatter root COM poses into the cache and derive the corresponding link poses."""
    i = wp.tid()
    if env_mask[i]:
        root_com_pose_w[i] = data[i]
        # link_pose = com_pose * inverse(com_pose_b)
        root_link_pose_w[i] = wp.transform_multiply(root_com_pose_w[i], wp.transform_inverse(body_com_pose_b[i, 0]))


@wp.kernel
def set_root_com_velocity_to_sim_mask(
    data: wp.array(dtype=wp.spatial_vectorf),
    env_mask: wp.array(dtype=wp.bool),
    num_bodies: wp.int32,
    root_com_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    body_acc_w: wp.array2d(dtype=wp.spatial_vectorf),
):
    """Mask-scatter root COM velocities into the cache and zero the dependent body acceleration."""
    i = wp.tid()
    if env_mask[i]:
        root_com_velocity_w[i] = data[i]
        for j in range(num_bodies):
            body_acc_w[i, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def set_root_link_velocity_to_sim_mask(
    data: wp.array(dtype=wp.spatial_vectorf),
    body_com_pose_b: wp.array2d(dtype=wp.transformf),
    link_pose_w: wp.array(dtype=wp.transformf),
    env_mask: wp.array(dtype=wp.bool),
    num_bodies: wp.int32,
    root_link_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    root_com_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    body_acc_w: wp.array2d(dtype=wp.spatial_vectorf),
):
    """Mask-scatter root link velocities into the cache and derive the corresponding COM velocities
    via the lever-arm transform: ``com_lin = link_lin + omega x rot(link_rot, com_offset)``.
    """
    i = wp.tid()
    if env_mask[i]:
        root_link_velocity_w[i] = data[i]
        ang = wp.spatial_bottom(data[i])
        lever = wp.quat_rotate(
            wp.transform_get_rotation(link_pose_w[i]), wp.transform_get_translation(body_com_pose_b[i, 0])
        )
        com_lin = wp.spatial_top(data[i]) + wp.cross(ang, lever)
        root_com_velocity_w[i] = wp.spatial_vector(com_lin, ang)
        for j in range(num_bodies):
            body_acc_w[i, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def write_2d_data_to_buffer_with_mask(
    in_data: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    body_mask: wp.array(dtype=wp.bool),
    out_data: wp.array2d(dtype=wp.float32),
):
    """Mask-scatter 2D float data into the cache where both ``env_mask[i]`` and ``body_mask[j]`` are True."""
    i, j = wp.tid()
    if env_mask[i] and body_mask[j]:
        out_data[i, j] = in_data[i, j]


@wp.kernel
def write_body_inertia_to_buffer_mask(
    in_data: wp.array3d(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    body_mask: wp.array(dtype=wp.bool),
    out_data: wp.array3d(dtype=wp.float32),
):
    """Mask-scatter body inertia (3x3 = 9 floats per body) into the cache."""
    i, j = wp.tid()
    if env_mask[i] and body_mask[j]:
        for k in range(9):
            out_data[i, j, k] = in_data[i, j, k]


@wp.kernel
def write_body_com_pose_to_buffer_mask(
    in_data: wp.array2d(dtype=wp.transformf),
    env_mask: wp.array(dtype=wp.bool),
    body_mask: wp.array(dtype=wp.bool),
    out_data: wp.array2d(dtype=wp.transformf),
):
    """Mask-scatter body COM poses (transformf) into the cache."""
    i, j = wp.tid()
    if env_mask[i] and body_mask[j]:
        out_data[i, j] = in_data[i, j]


"""
Articulation-only kernels (used by isaaclab_ovphysx.assets.articulation).
"""


@wp.kernel
def _copy_first_body(
    body_vel: wp.array(dtype=wp.spatial_vectorf, ndim=2),
    root_vel: wp.array(dtype=wp.spatial_vectorf),
):
    """Copy the first body's spatial velocity to the root velocity buffer.

    For single rigid-body assets, index 0 is always the root body.  This
    kernel extracts that slice without allocating an intermediate buffer.

    Args:
        body_vel: Body spatial velocities ``[m/s, rad/s]``. Shape is
            ``(num_envs, num_bodies)`` with dtype ``wp.spatial_vectorf``.
        root_vel: Output root spatial velocities ``[m/s, rad/s]``. Shape is
            ``(num_envs,)`` with dtype ``wp.spatial_vectorf``.
    """
    i = wp.tid()
    root_vel[i] = body_vel[i, 0]


@wp.kernel
def _compose_root_com_pose(
    link_pose: wp.array(dtype=wp.transformf),
    com_pose_b: wp.array(dtype=wp.transformf, ndim=2),
    com_pose_w: wp.array(dtype=wp.transformf),
):
    """Compose root link pose with the body-frame COM offset to get the world-frame COM pose.

    Implements the forward transform:

        ``com_pose_w = link_pose_w * com_pose_b[0]``

    where ``*`` denotes ``wp.transform_multiply``.  Only the first body
    (index ``0``) is used; for articulations this is the base link body.

    Args:
        link_pose: Root link poses in world frame ``[m, -]``. Shape is
            ``(num_envs,)`` with dtype ``wp.transformf``.
        com_pose_b: Body-frame COM offsets ``[m, -]`` from the
            ``RIGID_BODY_COM_POSE`` binding. Shape is ``(num_envs, num_bodies)``
            with dtype ``wp.transformf``.
        com_pose_w: Output world-frame root COM poses ``[m, -]``. Shape is
            ``(num_envs,)`` with dtype ``wp.transformf``.
    """
    i = wp.tid()
    com_pose_w[i] = wp.transform_multiply(link_pose[i], com_pose_b[i, 0])


@wp.kernel
def _projected_gravity(
    gravity_vec_w: wp.array(dtype=wp.vec3f),
    root_pose: wp.array(dtype=wp.transformf),
    out: wp.array(dtype=wp.vec3f),
):
    """Project the world-frame gravity direction into the root body frame.

    Applies the inverse of the root orientation quaternion to the world-frame
    gravity vector, yielding the gravity direction expressed in the body frame.
    The magnitude is preserved (unit vector in, unit vector out if input is a
    unit vector).

    Args:
        gravity_vec_w: Gravity direction per instance in world frame ``[-]``
            (typically the normalised ``(0, 0, -1)`` gravitational acceleration
            direction). Shape is ``(num_envs,)`` with dtype ``wp.vec3f``.
        root_pose: Root link poses in world frame ``[m, -]``. Only the
            rotation component is used. Shape is ``(num_envs,)`` with dtype
            ``wp.transformf``.
        out: Output gravity direction in body frame ``[-]``. Shape is
            ``(num_envs,)`` with dtype ``wp.vec3f``.
    """
    i = wp.tid()
    q = wp.transform_get_rotation(root_pose[i])
    out[i] = wp.quat_rotate_inv(q, gravity_vec_w[i])


@wp.kernel
def _compute_heading(
    forward_vec_b: wp.array(dtype=wp.vec3f),
    root_pose: wp.array(dtype=wp.transformf),
    out: wp.array(dtype=wp.float32),
):
    """Compute the yaw heading angle by rotating a body-frame forward vector to world frame.

    Rotates ``forward_vec_b`` by the root orientation quaternion and then computes the
    heading as ``atan2(forward_w.y, forward_w.x)`` ``[rad]``, i.e. the signed angle
    from the world X-axis to the projected forward direction in the XY plane.

    Args:
        forward_vec_b: Forward direction in body frame per instance ``[-]``.
            Shape is ``(num_envs,)`` with dtype ``wp.vec3f``.
        root_pose: Root link poses in world frame ``[m, -]``. Only the rotation
            component is used. Shape is ``(num_envs,)`` with dtype ``wp.transformf``.
        out: Output heading angles ``[rad]`` in ``[-π, π]``. Shape is
            ``(num_envs,)`` with dtype ``wp.float32``.
    """
    i = wp.tid()
    q = wp.transform_get_rotation(root_pose[i])
    forward = wp.quat_rotate(q, forward_vec_b[i])
    out[i] = wp.atan2(forward[1], forward[0])


@wp.kernel
def _world_vel_to_body_lin(
    root_pose: wp.array(dtype=wp.transformf),
    vel_w: wp.array(dtype=wp.spatial_vectorf),
    out: wp.array(dtype=wp.vec3f),
):
    """Rotate the world-frame linear velocity component into the root body frame.

    Extracts the linear velocity from the top three components of the spatial
    velocity vector (``wp.spatial_top``) and rotates it by the inverse of the
    root orientation quaternion.

    Args:
        root_pose: Root link poses in world frame ``[m, -]``. Only the rotation
            component is used. Shape is ``(num_envs,)`` with dtype ``wp.transformf``.
        vel_w: Root spatial velocities in world frame ``[m/s, rad/s]``.
            Shape is ``(num_envs,)`` with dtype ``wp.spatial_vectorf``.
        out: Output linear velocity in body frame ``[m/s]``. Shape is
            ``(num_envs,)`` with dtype ``wp.vec3f``.
    """
    i = wp.tid()
    q = wp.transform_get_rotation(root_pose[i])
    lin = wp.spatial_top(vel_w[i])
    out[i] = wp.quat_rotate_inv(q, lin)


@wp.kernel
def _world_vel_to_body_ang(
    root_pose: wp.array(dtype=wp.transformf),
    vel_w: wp.array(dtype=wp.spatial_vectorf),
    out: wp.array(dtype=wp.vec3f),
):
    """Rotate the world-frame angular velocity component into the root body frame.

    Extracts the angular velocity from the bottom three components of the spatial
    velocity vector (``wp.spatial_bottom``) and rotates it by the inverse of the
    root orientation quaternion.

    Args:
        root_pose: Root link poses in world frame ``[m, -]``. Only the rotation
            component is used. Shape is ``(num_envs,)`` with dtype ``wp.transformf``.
        vel_w: Root spatial velocities in world frame ``[m/s, rad/s]``.
            Shape is ``(num_envs,)`` with dtype ``wp.spatial_vectorf``.
        out: Output angular velocity in body frame ``[rad/s]``. Shape is
            ``(num_envs,)`` with dtype ``wp.vec3f``.
    """
    i = wp.tid()
    q = wp.transform_get_rotation(root_pose[i])
    ang = wp.spatial_bottom(vel_w[i])
    out[i] = wp.quat_rotate_inv(q, ang)


@wp.kernel
def write_joint_position_limit_to_buffer_index(
    in_data: wp.array3d(dtype=wp.float32),
    env_ids: wp.array(dtype=wp.int32),
    joint_ids: wp.array(dtype=wp.int32),
    out_data: wp.array(dtype=wp.vec2f, ndim=2),
):
    """Write joint position-limit data to a vec2f buffer at specified indices.

    This kernel copies ``[lower, upper]`` limit pairs from a partial float32 input
    array into the output ``wp.vec2f`` buffer at the specified environment and joint
    indices.

    Args:
        in_data: Input array containing limit pairs ``[lower, upper]`` [m or rad].
            Shape is (num_selected_envs, num_selected_joints, 2).
        env_ids: Input array of environment indices to write to.
            Shape is (num_selected_envs,).
        joint_ids: Input array of joint indices to write to.
            Shape is (num_selected_joints,).
        out_data: Output array where limit data is written. Shape is
            (num_envs, num_joints) with dtype ``wp.vec2f``.
    """
    i, j = wp.tid()
    out_data[env_ids[i], joint_ids[j]] = wp.vec2f(in_data[i, j, 0], in_data[i, j, 1])


@wp.kernel
def write_joint_position_limit_to_buffer_mask(
    in_data: wp.array3d(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    joint_mask: wp.array(dtype=wp.bool),
    out_data: wp.array(dtype=wp.vec2f, ndim=2),
):
    """Mask-scatter joint position-limit data into the vec2f cache buffer.

    Copies ``[lower, upper]`` limit pairs where both ``env_mask[i]`` and
    ``joint_mask[j]`` are True.

    Args:
        in_data: Input array containing limit pairs ``[lower, upper]`` [m or rad].
            Shape is (num_envs, num_joints, 2).
        env_mask: Boolean environment mask. Shape is (num_envs,).
        joint_mask: Boolean joint mask. Shape is (num_joints,).
        out_data: Output array where limit data is written. Shape is
            (num_envs, num_joints) with dtype ``wp.vec2f``.
    """
    i, j = wp.tid()
    if env_mask[i] and joint_mask[j]:
        out_data[i, j] = wp.vec2f(in_data[i, j, 0], in_data[i, j, 1])


@wp.kernel
def write_joint_friction_to_buffer_index(
    in_data: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=wp.int32),
    joint_ids: wp.array(dtype=wp.int32),
    out_data: wp.array3d(dtype=wp.float32),
):
    """Write joint friction coefficient to all three components of the friction buffer.

    Broadcasts a single friction value into the static (index 0), dynamic (index 1),
    and viscous (index 2) components of the ``(N, D, 3)`` friction properties buffer
    at the specified environment and joint indices.

    Args:
        in_data: Input friction coefficients [dimensionless]. Shape is
            (num_selected_envs, num_selected_joints).
        env_ids: Input array of environment indices to write to.
            Shape is (num_selected_envs,).
        joint_ids: Input array of joint indices to write to.
            Shape is (num_selected_joints,).
        out_data: Output friction properties buffer. Shape is (num_envs, num_joints, 3).
    """
    i, j = wp.tid()
    val = in_data[i, j]
    out_data[env_ids[i], joint_ids[j], 0] = val
    out_data[env_ids[i], joint_ids[j], 1] = val
    out_data[env_ids[i], joint_ids[j], 2] = val


@wp.kernel
def resolve_view_ids(
    env_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
    num_query_envs: wp.int32,
    num_total_envs: wp.int32,
    view_ids: wp.array(dtype=wp.int32),
) -> None:
    """Resolve flat view indices from environment and body index pairs.

    Computes flat view indices from (env_id, body_id) pairs using body-major ordering:
    ``view_id = body_id * num_total_envs + env_id``. The output array is laid out in
    column-major order over the (env, body) grid.

    Args:
        env_ids: Input environment indices. Shape is (num_query_envs,).
        body_ids: Input body indices. Shape is (num_query_bodies,).
        num_query_envs: Total number of queried environments.
        num_total_envs: Total number of environments in the simulation.
        view_ids: Output flat view indices. Shape is (num_query_bodies * num_query_envs,).
    """
    i, j = wp.tid()
    view_ids[j * num_query_envs + i] = body_ids[j] * num_total_envs + env_ids[i]


@wp.kernel
def write_joint_friction_to_buffer_mask(
    in_data: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    joint_mask: wp.array(dtype=wp.bool),
    out_data: wp.array3d(dtype=wp.float32),
):
    """Mask-scatter joint friction coefficient into all three components of the friction buffer.

    Broadcasts a single friction value into the static (index 0), dynamic (index 1),
    and viscous (index 2) components where both ``env_mask[i]`` and ``joint_mask[j]``
    are True.

    Args:
        in_data: Input friction coefficients [dimensionless]. Shape is
            (num_envs, num_joints).
        env_mask: Boolean environment mask. Shape is (num_envs,).
        joint_mask: Boolean joint mask. Shape is (num_joints,).
        out_data: Output friction properties buffer. Shape is (num_envs, num_joints, 3).
    """
    i, j = wp.tid()
    if env_mask[i] and joint_mask[j]:
        val = in_data[i, j]
        out_data[i, j, 0] = val
        out_data[i, j, 1] = val
        out_data[i, j, 2] = val
