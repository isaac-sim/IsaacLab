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
def update_wrench_with_force_and_torque(
    force: wp.vec3f,
    torque: wp.vec3f,
) -> wp.spatial_vectorf:
    return wp.spatial_vector(force, torque, wp.float32)


@wp.func
def get_link_vel_from_root_com_vel_func(
    com_vel: wp.spatial_vectorf,
    link_pose: wp.transformf,
    body_com_pos_b: wp.vec3f,
):
    """Compute link velocity from center-of-mass velocity.

    Transforms a COM spatial velocity into a link-frame velocity by projecting
    the angular velocity contribution from the COM offset relative to the link frame.

    Args:
        com_vel: COM spatial velocity (angular, linear).
        link_pose: Link pose in world frame.
        body_com_pos_b: COM position in body (link) frame.

    Returns:
        Link spatial velocity (angular, linear).
    """
    projected_vel = wp.cross(
        wp.spatial_bottom(com_vel),
        wp.quat_rotate(wp.transform_get_rotation(link_pose), -body_com_pos_b),
    )
    return wp.spatial_vector(wp.spatial_top(com_vel) + projected_vel, wp.spatial_bottom(com_vel))


@wp.func
def get_com_pose_from_link_pose_func(
    link_pose: wp.transformf,
    body_com_pos_b: wp.vec3f,
):
    """Compute COM pose in world frame from link pose and body-frame COM offset.

    Args:
        link_pose: Link pose in world frame.
        body_com_pos_b: COM position in body (link) frame.

    Returns:
        COM pose in world frame.
    """
    return link_pose * wp.transformf(body_com_pos_b, wp.quatf(0.0, 0.0, 0.0, 1.0))


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
    body_com_pos_b: wp.array2d(dtype=wp.vec3f),
    link_vel: wp.array(dtype=wp.spatial_vectorf),
):
    """Compute root link velocity from root center-of-mass velocity.

    This kernel transforms the root COM velocity into link-frame velocity by projecting
    the angular velocity contribution from the COM offset.

    Args:
        com_vel: Input array of root COM spatial velocities. Shape is (num_envs,).
        link_pose: Input array of root link poses in world frame. Shape is (num_envs,).
        body_com_pos_b: Input array of body COM positions in body frame. Shape is (num_envs, num_bodies).
            Only the first body (index 0) is used for the root.
        link_vel: Output array where root link velocities are written. Shape is (num_envs,).
    """
    i = wp.tid()
    link_vel[i] = get_link_vel_from_root_com_vel_func(com_vel[i], link_pose[i], body_com_pos_b[i, 0])


@wp.kernel
def get_root_com_pose_from_root_link_pose(
    link_pose: wp.array(dtype=wp.transformf),
    body_com_pos_b: wp.array2d(dtype=wp.vec3f),
    com_pose_w: wp.array(dtype=wp.transformf),
):
    """Compute root COM pose from root link pose.

    This kernel transforms the root link pose to the root COM pose using the body COM offset.

    Args:
        link_pose: Input array of root link poses in world frame. Shape is (num_envs,).
        body_com_pos_b: Input array of body COM positions in body frame. Shape is (num_envs, num_bodies).
            Only the first body (index 0) is used for the root.
        com_pose_w: Output array where root COM poses are written. Shape is (num_envs,).
    """
    i = wp.tid()
    com_pose_w[i] = get_com_pose_from_link_pose_func(link_pose[i], body_com_pos_b[i, 0])


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
    body_com_pos_b: wp.array2d(dtype=wp.vec3f),
    body_link_vel: wp.array2d(dtype=wp.spatial_vectorf),
):
    """Compute body link velocities from body COM velocities for all bodies.

    This kernel transforms COM velocities into link-frame velocities by projecting
    the angular velocity contribution from the COM offset, for each body in each environment.

    Args:
        body_com_vel: Input array of body COM spatial velocities. Shape is (num_envs, num_bodies).
        body_link_pose: Input array of body link poses in world frame. Shape is (num_envs, num_bodies).
        body_com_pos_b: Input array of body COM positions in body frame. Shape is (num_envs, num_bodies).
        body_link_vel: Output array where body link velocities are written. Shape is (num_envs, num_bodies).
    """
    i, j = wp.tid()
    body_link_vel[i, j] = get_link_vel_from_root_com_vel_func(
        body_com_vel[i, j], body_link_pose[i, j], body_com_pos_b[i, j]
    )


@wp.kernel
def get_body_com_pose_from_body_link_pose(
    body_link_pose: wp.array2d(dtype=wp.transformf),
    body_com_pos_b: wp.array2d(dtype=wp.vec3f),
    body_com_pose_w: wp.array2d(dtype=wp.transformf),
):
    """Compute body COM poses from body link poses for all bodies.

    This kernel transforms link poses to COM poses using the body COM offset in the body frame.

    Args:
        body_link_pose: Input array of body link poses in world frame. Shape is (num_envs, num_bodies).
        body_com_pos_b: Input array of body COM positions in body frame. Shape is (num_envs, num_bodies).
        body_com_pose_w: Output array where body COM poses in world frame are written.
            Shape is (num_envs, num_bodies).
    """
    i, j = wp.tid()
    body_com_pose_w[i, j] = get_com_pose_from_link_pose_func(body_link_pose[i, j], body_com_pos_b[i, j])


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
    root_link_state_w: wp.array(dtype=vec13f),
    root_state_w: wp.array(dtype=vec13f),
):
    """Write root link pose data to simulation buffers.

    This kernel writes root link poses from the input array to the output buffers
    and optionally updates the corresponding state vectors.

    Args:
        data: Input array of root link poses. Shape is (num_selected_envs,).
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        root_link_pose_w: Output array where root link poses are written. Shape is (num_envs,).
        root_link_state_w: Output array where root link states are updated (pose portion).
            Shape is (num_envs,). Can be None if not needed.
        root_state_w: Output array where root states are updated (pose portion).
            Shape is (num_envs,). Can be None if not needed.
    """
    i = wp.tid()
    root_link_pose_w[env_ids[i]] = data[i]
    if root_link_state_w:
        root_link_state_w[env_ids[i]] = set_state_transforms_func(root_link_state_w[env_ids[i]], data[i])
    if root_state_w:
        root_state_w[env_ids[i]] = set_state_transforms_func(root_state_w[env_ids[i]], data[i])


@wp.kernel
def set_root_link_pose_to_sim_mask(
    data: wp.array(dtype=wp.transformf),
    env_mask: wp.array(dtype=wp.bool),
    root_link_pose_w: wp.array(dtype=wp.transformf),
    root_link_state_w: wp.array(dtype=vec13f),
    root_state_w: wp.array(dtype=vec13f),
):
    """Write root link pose data to simulation buffers.

    This kernel writes root link poses from the input array to the output buffers
    and optionally updates the corresponding state vectors.

    Args:
        data: Input array of root link poses. Shape is (num_instances,).
        env_mask: Input array of environment mask. Shape is (num_instances,).
        root_link_pose_w: Output array where root link poses are written. Shape is (num_envs,).
        root_link_state_w: Output array where root link states are updated (pose portion).
            Shape is (num_envs,). Can be None if not needed.
        root_state_w: Output array where root states are updated (pose portion).
            Shape is (num_envs,). Can be None if not needed.
    """
    i = wp.tid()
    if env_mask[i]:
        root_link_pose_w[i] = data[i]
        if root_link_state_w:
            root_link_state_w[i] = set_state_transforms_func(root_link_state_w[i], data[i])
        if root_state_w:
            root_state_w[i] = set_state_transforms_func(root_state_w[i], data[i])


@wp.kernel
def set_root_com_pose_to_sim_index(
    data: wp.array(dtype=wp.transformf),
    body_com_pose_b: wp.array2d(dtype=wp.transformf),
    env_ids: wp.array(dtype=wp.int32),
    root_com_pose_w: wp.array(dtype=wp.transformf),
    root_link_pose_w: wp.array(dtype=wp.transformf),
    root_com_state_w: wp.array(dtype=vec13f),
    root_link_state_w: wp.array(dtype=vec13f),
    root_state_w: wp.array(dtype=vec13f),
):
    """Write root COM pose data to simulation buffers.

    This kernel writes root COM poses from the input array to the output buffers,
    computes the corresponding link pose from the COM pose, and optionally updates
    the corresponding state vectors.

    Args:
        data: Input array of root COM poses. Shape is (num_selected_envs,).
        body_com_pose_b: Input array of body COM poses in body frame. Shape is
            (num_envs, num_bodies). Only the first body (index 0) is used for the root.
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        root_com_pose_w: Output array where root COM poses are written. Shape is (num_envs,).
        root_link_pose_w: Output array where root link poses (derived from COM) are written.
            Shape is (num_envs,).
        root_com_state_w: Output array where root COM states are updated (pose portion).
            Shape is (num_envs,). Can be None if not needed.
        root_link_state_w: Output array where root link states are updated (pose portion).
            Shape is (num_envs,). Can be None if not needed.
        root_state_w: Output array where root states are updated (pose portion).
            Shape is (num_envs,). Can be None if not needed.
    """
    i = wp.tid()
    root_com_pose_w[env_ids[i]] = data[i]
    if root_com_state_w:
        root_com_state_w[env_ids[i]] = set_state_transforms_func(root_com_state_w[env_ids[i]], data[i])
    # Get the com pose in the link frame
    root_link_pose_w[env_ids[i]] = get_com_pose_in_link_frame_func(
        root_com_pose_w[env_ids[i]], body_com_pose_b[env_ids[i], 0]
    )
    if root_link_state_w:
        root_link_state_w[env_ids[i]] = set_state_transforms_func(
            root_link_state_w[env_ids[i]], root_link_pose_w[env_ids[i]]
        )
    if root_state_w:
        root_state_w[env_ids[i]] = set_state_transforms_func(root_state_w[env_ids[i]], root_link_pose_w[env_ids[i]])


@wp.kernel
def set_root_com_pose_to_sim_mask(
    data: wp.array(dtype=wp.transformf),
    body_com_pose_b: wp.array2d(dtype=wp.transformf),
    env_mask: wp.array(dtype=wp.bool),
    root_com_pose_w: wp.array(dtype=wp.transformf),
    root_link_pose_w: wp.array(dtype=wp.transformf),
    root_com_state_w: wp.array(dtype=vec13f),
    root_link_state_w: wp.array(dtype=vec13f),
    root_state_w: wp.array(dtype=vec13f),
):
    """Write root COM pose data to simulation buffers.

    This kernel writes root COM poses from the input array to the output buffers,
    computes the corresponding link pose from the COM pose, and optionally updates
    the corresponding state vectors.

    Args:
        data: Input array of root COM poses. Shape is (num_instances,).
        body_com_pose_b: Input array of body COM poses in body frame. Shape is
            (num_envs, num_bodies). Only the first body (index 0) is used for the root.
        env_mask: Input array of environment mask. Shape is (num_instances,).
        root_com_pose_w: Output array where root COM poses are written. Shape is (num_envs,).
        root_link_pose_w: Output array where root link poses (derived from COM) are written.
            Shape is (num_envs,).
        root_com_state_w: Output array where root COM states are updated (pose portion).
            Shape is (num_envs,). Can be None if not needed.
        root_link_state_w: Output array where root link states are updated (pose portion).
            Shape is (num_envs,). Can be None if not needed.
        root_state_w: Output array where root states are updated (pose portion).
            Shape is (num_envs,). Can be None if not needed.
    """
    i = wp.tid()
    if env_mask[i]:
        root_com_pose_w[i] = data[i]
        if root_com_state_w:
            root_com_state_w[i] = set_state_transforms_func(root_com_state_w[i], data[i])
        # Get the com pose in the link frame
        root_link_pose_w[i] = get_com_pose_in_link_frame_func(root_com_pose_w[i], body_com_pose_b[i, 0])
        if root_link_state_w:
            root_link_state_w[i] = set_state_transforms_func(root_link_state_w[i], root_link_pose_w[i])
        if root_state_w:
            root_state_w[i] = set_state_transforms_func(root_state_w[i], root_link_pose_w[i])


@wp.kernel
def set_root_com_velocity_to_sim_index(
    data: wp.array(dtype=wp.spatial_vectorf),
    env_ids: wp.array(dtype=wp.int32),
    num_bodies: wp.int32,
    root_com_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    body_acc_w: wp.array2d(dtype=wp.spatial_vectorf),
    root_state_w: wp.array(dtype=vec13f),
    root_com_state_w: wp.array(dtype=vec13f),
):
    """Write root COM velocity data to simulation buffers.

    This kernel writes root COM velocities from the input array to the output buffers,
    optionally updates the corresponding state vectors, and zeros out the body
    acceleration buffer to prevent reporting stale values.

    Args:
        data: Input array of root COM spatial velocities. Shape is (num_selected_envs,).
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        num_bodies: Input scalar number of bodies per environment.
        root_com_velocity_w: Output array where root COM velocities are written. Shape is (num_envs,).
        body_acc_w: Output array where body accelerations are zeroed. Shape is
            (num_envs, num_bodies).
        root_state_w: Output array where root states are updated (velocity portion).
            Shape is (num_envs,). Can be None if not needed.
        root_com_state_w: Output array where root COM states are updated (velocity portion).
            Shape is (num_envs,). Can be None if not needed.
    """
    i = wp.tid()
    root_com_velocity_w[env_ids[i]] = data[i]
    if root_state_w:
        root_state_w[env_ids[i]] = set_state_velocities_func(root_state_w[env_ids[i]], data[i])
    if root_com_state_w:
        root_com_state_w[env_ids[i]] = set_state_velocities_func(root_com_state_w[env_ids[i]], data[i])
    # Make the acceleration zero to prevent reporting old values
    for j in range(num_bodies):
        body_acc_w[env_ids[i], j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def set_root_com_velocity_to_sim_mask(
    data: wp.array(dtype=wp.spatial_vectorf),
    env_mask: wp.array(dtype=wp.bool),
    num_bodies: wp.int32,
    root_com_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    body_acc_w: wp.array2d(dtype=wp.spatial_vectorf),
    root_state_w: wp.array(dtype=vec13f),
    root_com_state_w: wp.array(dtype=vec13f),
):
    """Write root COM velocity data to simulation buffers.

    This kernel writes root COM velocities from the input array to the output buffers,
    optionally updates the corresponding state vectors, and zeros out the body
    acceleration buffer to prevent reporting stale values.

    Args:
        data: Input array of root COM spatial velocities. Shape is (num_instances,).
        env_mask: Input array of environment mask. Shape is (num_instances,).
        num_bodies: Input scalar number of bodies per environment.
        root_com_velocity_w: Output array where root COM velocities are written. Shape is (num_envs,).
        body_acc_w: Output array where body accelerations are zeroed. Shape is
            (num_envs, num_bodies).
        root_state_w: Output array where root states are updated (velocity portion).
            Shape is (num_envs,). Can be None if not needed.
        root_com_state_w: Output array where root COM states are updated (velocity portion).
            Shape is (num_envs,). Can be None if not needed.
    """
    i = wp.tid()
    if env_mask[i]:
        root_com_velocity_w[i] = data[i]
        if root_state_w:
            root_state_w[i] = set_state_velocities_func(root_state_w[i], data[i])
        if root_com_state_w:
            root_com_state_w[i] = set_state_velocities_func(root_com_state_w[i], data[i])
        # Make the acceleration zero to prevent reporting old values
        for j in range(num_bodies):
            body_acc_w[i, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


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
    root_link_state_w: wp.array(dtype=vec13f),
    root_state_w: wp.array(dtype=vec13f),
    root_com_state_w: wp.array(dtype=vec13f),
):
    """Write root link velocity data to simulation buffers.

    This kernel writes root link velocities from the input array to the output buffers,
    computes the corresponding COM velocity from the link velocity, optionally updates
    the corresponding state vectors, and zeros out the body acceleration buffer.

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
        root_link_state_w: Output array where root link states are updated (velocity portion).
            Shape is (num_envs,). Can be None if not needed.
        root_state_w: Output array where root states are updated (velocity portion).
            Shape is (num_envs,). Can be None if not needed.
        root_com_state_w: Output array where root COM states are updated (velocity portion).
            Shape is (num_envs,). Can be None if not needed.
    """
    i = wp.tid()
    root_link_velocity_w[env_ids[i]] = data[i]
    if root_link_state_w:
        root_link_state_w[env_ids[i]] = set_state_velocities_func(root_link_state_w[env_ids[i]], data[i])
    # Get the link velocity in the com frame
    root_com_velocity_w[env_ids[i]] = get_link_velocity_in_com_frame_func(
        root_link_velocity_w[env_ids[i]], link_pose_w[env_ids[i]], body_com_pose_b[env_ids[i], 0]
    )
    if root_com_state_w:
        root_com_state_w[env_ids[i]] = set_state_velocities_func(
            root_com_state_w[env_ids[i]], root_com_velocity_w[env_ids[i]]
        )
    if root_state_w:
        root_state_w[env_ids[i]] = set_state_velocities_func(root_state_w[env_ids[i]], root_com_velocity_w[env_ids[i]])
    # Make the acceleration zero to prevent reporting old values
    for j in range(num_bodies):
        body_acc_w[env_ids[i], j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


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
    root_link_state_w: wp.array(dtype=vec13f),
    root_state_w: wp.array(dtype=vec13f),
    root_com_state_w: wp.array(dtype=vec13f),
):
    """Write root link velocity data to simulation buffers.

    This kernel writes root link velocities from the input array to the output buffers,
    computes the corresponding COM velocity from the link velocity, optionally updates
    the corresponding state vectors, and zeros out the body acceleration buffer.

    Args:
        data: Input array of root link spatial velocities. Shape is (num_instances,).
        body_com_pose_b: Input array of body COM poses in body frame. Shape is
            (num_envs, num_bodies). Only the first body (index 0) is used for the root.
        link_pose_w: Input array of root link poses in world frame. Shape is (num_envs,).
        env_mask: Input array of environment mask. Shape is (num_instances,).
        num_bodies: Input scalar number of bodies per environment.
        root_link_velocity_w: Output array where root link velocities are written.
            Shape is (num_envs,).
        root_com_velocity_w: Output array where root COM velocities (derived from link)
            are written. Shape is (num_envs,).
        body_acc_w: Output array where body accelerations are zeroed.
            Shape is (num_envs, num_bodies).
        root_link_state_w: Output array where root link states are updated (velocity portion).
            Shape is (num_envs,). Can be None if not needed.
        root_state_w: Output array where root states are updated (velocity portion).
            Shape is (num_envs,). Can be None if not needed.
        root_com_state_w: Output array where root COM states are updated (velocity portion).
            Shape is (num_envs,). Can be None if not needed.
    """
    i = wp.tid()
    if env_mask[i]:
        root_link_velocity_w[i] = data[i]
        if root_link_state_w:
            root_link_state_w[i] = set_state_velocities_func(root_link_state_w[i], data[i])
        # Get the link velocity in the com frame
        root_com_velocity_w[i] = get_link_velocity_in_com_frame_func(
            root_link_velocity_w[i], link_pose_w[i], body_com_pose_b[i, 0]
        )
        if root_com_state_w:
            root_com_state_w[i] = set_state_velocities_func(root_com_state_w[i], root_com_velocity_w[i])
        if root_state_w:
            root_state_w[i] = set_state_velocities_func(root_state_w[i], root_com_velocity_w[i])
        # Make the acceleration zero to prevent reporting old values
        for j in range(num_bodies):
            body_acc_w[i, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


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

    This kernel copies float data from an input array to an output buffer at the specified
    environment and joint/body indices.

    Args:
        in_data: Input array containing float data. Shape is (num_selected_envs, num_selected_joints).
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        joint_ids: Input array of joint/body indices to write to. Shape is (num_selected_joints,).
        out_data: Output array where data is written. Shape is (num_envs, num_joints).
    """
    i, j = wp.tid()
    out_data[env_ids[i], joint_ids[j]] = in_data[i, j]


@wp.kernel
def write_2d_data_to_buffer_with_mask(
    in_data: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    joint_mask: wp.array(dtype=wp.bool),
    out_data: wp.array2d(dtype=wp.float32),
):
    """Write 2D float data to a buffer at specified indices.

    This kernel copies float data from an input array to an output buffer at the specified
    environment and joint/body indices.

    Args:
        in_data: Input array containing float data. Shape is (num_instances, num_joints).
        env_mask: Input array of environment mask. Shape is (num_instances,).
        joint_mask: Input array of joint/body mask. Shape is (num_instances, num_joints).
        out_data: Output array where data is written. Shape is (num_instances, num_joints).
    """
    i, j = wp.tid()
    if env_mask[i] and joint_mask[j]:
        out_data[i, j] = in_data[i, j]


@wp.kernel
def write_body_inertia_to_buffer_index(
    in_data: wp.array3d(dtype=wp.float32),
    env_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
    out_data: wp.array3d(dtype=wp.float32),
):
    """Write body inertia data to a buffer at specified indices.

    This kernel copies 3x3 inertia tensor data (stored as 9 floats) from an input array
    to an output buffer at the specified environment and body indices.

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
def write_body_inertia_to_buffer_mask(
    in_data: wp.array3d(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    body_mask: wp.array(dtype=wp.bool),
    out_data: wp.array3d(dtype=wp.float32),
):
    """Write body inertia data to a buffer at specified indices.

    This kernel copies 3x3 inertia tensor data (stored as 9 floats) from an input array
    to an output buffer at the specified environment and body indices.

    Args:
        in_data: Input array containing inertia data. Shape is (num_selected_envs, num_selected_bodies, 9).
        env_mask: Input array of environment mask. Shape is (num_selected_envs,).
        body_mask: Input array of body mask. Shape is (num_selected_bodies,).
        out_data: Output array where inertia data is written. Shape is (num_envs, num_bodies, 9).
    """
    i, j = wp.tid()
    if env_mask[i] and body_mask[j]:
        for k in range(9):
            out_data[i, j, k] = in_data[i, j, k]


@wp.kernel
def write_single_body_inertia_to_buffer(
    in_data: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=wp.int32),
    from_mask: bool,
    out_data: wp.array2d(dtype=wp.float32),
):
    """Write body inertia data to a buffer at specified indices.

    This kernel copies 3x3 inertia tensor data (stored as 9 floats) from an input array
    to an output buffer at the specified environment and body indices.

    Args:
        in_data: Input array containing inertia data. Shape is (num_envs, 9) or
            (num_selected_envs, 9) depending on from_mask.
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        from_mask: Input flag indicating whether to use masked indexing.
        out_data: Output array where inertia data is written. Shape is (num_envs, 9).
    """
    i = wp.tid()
    if from_mask:
        for k in range(9):
            out_data[env_ids[i], k] = in_data[env_ids[i], k]
    else:
        for k in range(9):
            out_data[env_ids[i], k] = in_data[i, k]


@wp.kernel
def write_body_com_position_to_buffer_index(
    in_data: wp.array2d(dtype=wp.vec3f),
    env_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
    out_data: wp.array2d(dtype=wp.vec3f),
):
    """Write body COM position data to a buffer at specified indices.

    This kernel copies body COM position data from an input array to an output buffer at the
    specified environment and body indices.

    Args:
        in_data: Input array containing body COM positions. Shape is (num_selected_envs, num_selected_bodies).
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        body_ids: Input array of body indices to write to. Shape is (num_selected_bodies,).
        out_data: Output array where body COM positions are written. Shape is (num_envs, num_bodies).
    """
    i, j = wp.tid()
    out_data[env_ids[i], body_ids[j]] = in_data[i, j]


@wp.kernel
def write_body_com_position_to_buffer_mask(
    in_data: wp.array2d(dtype=wp.vec3f),
    env_mask: wp.array(dtype=wp.bool),
    body_mask: wp.array(dtype=wp.bool),
    out_data: wp.array2d(dtype=wp.vec3f),
):
    """Write body COM position data to a buffer at specified masks.

    This kernel copies body COM position data from an input array to an output buffer at the
    specified environment and body masks.

    Args:
        in_data: Input array containing body COM positions. Shape is (num_instances, num_bodies).
        env_mask: Input array of environment mask. Shape is (num_instances,).
        body_mask: Input array of body mask. Shape is (num_bodies).
        out_data: Output array where body COM positions are written. Shape is (num_instances, num_bodies).
    """
    i, j = wp.tid()
    if env_mask[i] and body_mask[j]:
        out_data[i, j] = in_data[i, j]


@wp.kernel
def split_transform_to_pos_1d(
    transform: wp.array(dtype=wp.transformf),
    pos: wp.array(dtype=wp.vec3f),
):
    """Split a 1D transform array into a position array.

    This kernel splits a 1D transform array into a position array.

    Args:
        transform: Input array of transforms. Shape is (num_envs, 7).
        pos: Output array where positions are written. Shape is (num_envs, 3).
    """
    i = wp.tid()
    pos[i] = wp.transform_get_translation(transform[i])


@wp.kernel
def split_transform_to_quat_1d(
    transform: wp.array(dtype=wp.transformf),
    quat: wp.array(dtype=wp.quatf),
):
    """Split a 1D transform array into a quaternion array.

    This kernel splits a 1D transform array into a quaternion array.

    Args:
        transform: Input array of transforms. Shape is (num_envs, 7).
        quat: Output array where quaternions are written. Shape is (num_envs, 4).
    """
    i = wp.tid()
    quat[i] = wp.transform_get_rotation(transform[i])


@wp.kernel
def split_transform_to_pos_2d(
    transform: wp.array2d(dtype=wp.transformf),
    pos: wp.array2d(dtype=wp.vec3f),
):
    """Split a 2D transform array into a position array.

    This kernel splits a 2D transform array into a position array.

    Args:
        transform: Input array of transforms. Shape is (num_envs, num_bodies, 7).
        pos: Output array where positions are written. Shape is (num_envs, num_bodies, 3).
    """
    i, j = wp.tid()
    pos[i, j] = wp.transform_get_translation(transform[i, j])


@wp.kernel
def split_transform_to_quat_2d(
    transform: wp.array2d(dtype=wp.transformf),
    quat: wp.array2d(dtype=wp.quatf),
):
    """Split a 2D transform array into a quaternion array.

    This kernel splits a 2D transform array into a quaternion array.

    Args:
        transform: Input array of transforms. Shape is (num_envs, num_bodies, 7).
        quat: Output array where quaternions are written. Shape is (num_envs, num_bodies, 4).
    """
    i, j = wp.tid()
    quat[i, j] = wp.transform_get_rotation(transform[i, j])


@wp.kernel
def split_spatial_vector_to_top_1d(
    spatial_vector: wp.array(dtype=wp.spatial_vectorf),
    top_part: wp.array(dtype=wp.vec3f),
):
    """Split a 1D spatial vector array into a top part array.

    This kernel splits a 1D spatial vector array into a top part array.

    Args:
        spatial_vector: Input array of spatial vectors. Shape is (num_envs, 6).
        top_part: Output array where top parts are written. Shape is (num_envs, 3).
    """
    i = wp.tid()
    top_part[i] = wp.spatial_top(spatial_vector[i])


@wp.kernel
def split_spatial_vector_to_bottom_1d(
    spatial_vector: wp.array(dtype=wp.spatial_vectorf),
    bottom_part: wp.array(dtype=wp.vec3f),
):
    """Split a 1D spatial vector array into a bottom part array.

    This kernel splits a 1D spatial vector array into a bottom part array.

    Args:
        spatial_vector: Input array of spatial vectors. Shape is (num_envs, 6).
        bottom_part: Output array where bottom parts are written. Shape is (num_envs, 3).
    """
    i = wp.tid()
    bottom_part[i] = wp.spatial_bottom(spatial_vector[i])


@wp.kernel
def split_spatial_vector_to_top_2d(
    spatial_vector: wp.array2d(dtype=wp.spatial_vectorf),
    top_part: wp.array2d(dtype=wp.vec3f),
):
    """Split a 2D spatial vector array into a top part array.

    This kernel splits a 2D spatial vector array into a top part array.

    Args:
        spatial_vector: Input array of spatial vectors. Shape is (num_envs, num_bodies, 6).
        top_part: Output array where top parts are written. Shape is (num_envs, num_bodies, 3).
    """
    i, j = wp.tid()
    top_part[i, j] = wp.spatial_top(spatial_vector[i, j])


@wp.kernel
def split_spatial_vector_to_bottom_2d(
    spatial_vector: wp.array2d(dtype=wp.spatial_vectorf),
    bottom_part: wp.array2d(dtype=wp.vec3f),
):
    """Split a 2D spatial vector array into a bottom part array.

    This kernel splits a 2D spatial vector array into a bottom part array.

    Args:
        spatial_vector: Input array of spatial vectors. Shape is (num_envs, num_bodies, 6).
        bottom_part: Output array where bottom parts are written. Shape is (num_envs, num_bodies, 3).
    """
    i, j = wp.tid()
    bottom_part[i, j] = wp.spatial_bottom(spatial_vector[i, j])


@wp.kernel
def make_dummy_body_com_pose_b(
    body_com_pos_b: wp.array2d(dtype=wp.vec3f),
    body_com_pose_b: wp.array2d(dtype=wp.transformf),
):
    """Make a dummy body COM pose in body frame.

    This kernel makes a dummy body COM pose in body frame.

    Args:
        body_com_pos_b: Input array of body COM positions in body frame. Shape is (num_envs, num_bodies).
        body_com_pose_b: Output array where body COM poses are written. Shape is (num_envs, num_bodies).
    """
    i, j = wp.tid()
    # Concatenate the position and a unit quaternion
    body_com_pose_b[i, j] = wp.transformf(body_com_pos_b[i, j], wp.quatf(0.0, 0.0, 0.0, 1.0))


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
def update_wrench_array_with_force_and_torque(
    forces: wp.array2d(dtype=wp.vec3f),
    torques: wp.array2d(dtype=wp.vec3f),
    wrench: wp.array2d(dtype=wp.spatial_vectorf),
    env_ids: wp.array(dtype=wp.bool),
    body_ids: wp.array(dtype=wp.bool),
):
    env_index, body_index = wp.tid()
    if env_ids[env_index] and body_ids[body_index]:
        wrench[env_index, body_index] = update_wrench_with_force_and_torque(
            forces[env_index, body_index],
            torques[env_index, body_index],
        )
