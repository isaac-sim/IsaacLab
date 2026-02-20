# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp

"""
Articulation-specific warp functions.
"""


@wp.func
def compute_soft_joint_pos_limits_func(
    joint_pos_limits: wp.vec2f,
    soft_limit_factor: wp.float32,
):
    """Compute the soft joint position limits.

    Args:
        joint_pos_limits: The joint position limits.
        soft_limit_factor: The soft limit factor.

    Returns:
        The soft joint position limits.
    """
    joint_pos_mean = (joint_pos_limits[0] + joint_pos_limits[1]) / 2.0
    joint_pos_range = joint_pos_limits[1] - joint_pos_limits[0]
    return wp.vec2f(
        joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor,
        joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor,
    )


"""
Articulation-specific warp kernels.
"""


@wp.kernel
def get_joint_acc_from_joint_vel(
    joint_vel: wp.array2d(dtype=wp.float32),
    prev_joint_vel: wp.array2d(dtype=wp.float32),
    dt: wp.float32,
    joint_acc: wp.array2d(dtype=wp.float32),
):
    """Compute the joint acceleration from the joint velocity using finite differencing.

    This kernel computes the joint acceleration by taking the difference between the current
    and previous joint velocities, divided by the time step. It also updates the previous
    joint velocity buffer with the current values.

    Args:
        joint_vel: Input array of current joint velocities. Shape is (num_envs, num_joints).
        prev_joint_vel: Input/output array of previous joint velocities. Shape is (num_envs, num_joints).
            This buffer is updated with the current joint velocities after computing acceleration.
        dt: Input time step (scalar) used for finite differencing.
        joint_acc: Output array where joint accelerations are written. Shape is (num_envs, num_joints).
    """
    i, j = wp.tid()
    joint_acc[i, j] = (joint_vel[i, j] - prev_joint_vel[i, j]) / dt
    prev_joint_vel[i, j] = joint_vel[i, j]


@wp.kernel
def write_joint_vel_data_index(
    in_data: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=wp.int32),
    joint_ids: wp.array(dtype=wp.int32),
    joint_vel: wp.array2d(dtype=wp.float32),
    prev_joint_vel: wp.array2d(dtype=wp.float32),
    joint_acc: wp.array2d(dtype=wp.float32),
):
    """Write joint velocity data to the output buffers.

    This kernel writes joint velocity data from the input array to the output buffers.
    It also updates the previous joint velocity buffer and resets the joint acceleration to 0.0.

    Args:
        in_data: Input array containing joint velocity data. Shape is (num_selected_envs, num_selected_joints).
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        joint_ids: Input array of joint indices to write to. Shape is (num_selected_joints,).
        joint_vel: Output array where joint velocities are written. Shape is (num_envs, num_joints).
        prev_joint_vel: Output array where previous joint velocities are written. Shape is
            (num_envs, num_joints).
        joint_acc: Output array where joint accelerations are reset to 0.0. Shape is
            (num_envs, num_joints).
    """
    i, j = wp.tid()
    joint_vel[env_ids[i], joint_ids[j]] = in_data[i, j]
    prev_joint_vel[env_ids[i], joint_ids[j]] = in_data[i, j]
    joint_acc[env_ids[i], joint_ids[j]] = 0.0


@wp.kernel
def write_joint_vel_data_mask(
    in_data: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    joint_mask: wp.array(dtype=wp.bool),
    joint_vel: wp.array2d(dtype=wp.float32),
    prev_joint_vel: wp.array2d(dtype=wp.float32),
    joint_acc: wp.array2d(dtype=wp.float32),
):
    """Write joint velocity data to the output buffers.

    This kernel writes joint velocity data from the input array to the output buffers.
    It also updates the previous joint velocity buffer and resets the joint acceleration to 0.0.

    Args:
        in_data: Input array containing joint velocity data. Shape is (num_envs, num_joints).
        env_mask: Input array of environment mask. Shape is (num_envs,).
        joint_mask: Input array of joint mask. Shape is (num_joints,).
        joint_vel: Output array where joint velocities are written. Shape is (num_envs, num_joints).
        prev_joint_vel: Output array where previous joint velocities are written. Shape is
            (num_envs, num_joints).
        joint_acc: Output array where joint accelerations are reset to 0.0. Shape is
            (num_envs, num_joints).
    """
    i, j = wp.tid()
    if env_mask[i] and joint_mask[j]:
        joint_vel[i, j] = in_data[i, j]
        prev_joint_vel[i, j] = in_data[i, j]
        joint_acc[i, j] = 0.0


@wp.kernel
def write_joint_limit_data_to_buffer_index(
    in_data: wp.array2d(dtype=wp.vec2f),
    soft_limit_factor: wp.float32,
    env_ids: wp.array(dtype=wp.int32),
    joint_ids: wp.array(dtype=wp.int32),
    joint_pos_limits_lower: wp.array2d(dtype=wp.float32),
    joint_pos_limits_upper: wp.array2d(dtype=wp.float32),
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    default_joint_pos: wp.array2d(dtype=wp.float32),
    clamped_defaults: wp.array(dtype=wp.int32),
):
    """Write joint limit data to the output buffers and compute soft limits.

    This kernel writes joint position limits from the input array to the output buffer,
    computes soft joint position limits, and clamps default joint positions if they
    fall outside the limits.

    Args:
        in_data: Input array containing joint position limits as vec2f (lower, upper).
            Shape is (num_selected_envs, num_selected_joints).
        soft_limit_factor: Input scalar factor for computing soft limits (typically 0.0-1.0).
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        joint_ids: Input array of joint indices to write to. Shape is (num_selected_joints,).
        joint_pos_limits_lower: Output array where joint position limits lower are written. Shape is
            (num_envs, num_joints).
        joint_pos_limits_upper: Output array where joint position limits upper are written. Shape is
            (num_envs, num_joints).
        soft_joint_pos_limits: Output array where soft joint position limits are written.
            Shape is (num_envs, num_joints).
        default_joint_pos: Input/output array of default joint positions. If any values fall
            outside the limits, they are clamped. Shape is (num_envs, num_joints).
        clamped_defaults: Output 1-element array flag indicating whether any default joint
            positions were clamped. Non-zero if any clamping occurred. Shape is (1,).
    """
    i, j = wp.tid()
    joint_pos_limits_lower[env_ids[i], joint_ids[j]] = in_data[i, j][0]
    joint_pos_limits_upper[env_ids[i], joint_ids[j]] = in_data[i, j][1]
    if (
        default_joint_pos[env_ids[i], joint_ids[j]] < joint_pos_limits_lower[env_ids[i], joint_ids[j]]
    ) or default_joint_pos[env_ids[i], joint_ids[j]] > joint_pos_limits_upper[env_ids[i], joint_ids[j]]:
        wp.atomic_add(clamped_defaults, 0, 1)
        default_joint_pos[env_ids[i], joint_ids[j]] = wp.clamp(
            default_joint_pos[env_ids[i], joint_ids[j]],
            joint_pos_limits_lower[env_ids[i], joint_ids[j]],
            joint_pos_limits_upper[env_ids[i], joint_ids[j]],
        )
    soft_joint_pos_limits[env_ids[i], joint_ids[j]] = compute_soft_joint_pos_limits_func(
        wp.vec2f(joint_pos_limits_lower[env_ids[i], joint_ids[j]], joint_pos_limits_upper[env_ids[i], joint_ids[j]]),
        soft_limit_factor,
    )


@wp.kernel
def write_joint_limit_data_to_buffer_mask(
    in_data: wp.array2d(dtype=wp.vec2f),
    soft_limit_factor: wp.float32,
    env_mask: wp.array(dtype=wp.bool),
    joint_mask: wp.array(dtype=wp.bool),
    joint_pos_limits_lower: wp.array2d(dtype=wp.float32),
    joint_pos_limits_upper: wp.array2d(dtype=wp.float32),
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    default_joint_pos: wp.array2d(dtype=wp.float32),
    clamped_defaults: wp.array(dtype=wp.int32),
):
    """Write joint limit data to the output buffers and compute soft limits.

    This kernel writes joint position limits from the input array to the output buffer,
    computes soft joint position limits, and clamps default joint positions if they
    fall outside the limits.

    Args:
        in_data: Input array containing joint position limits as vec2f (lower, upper).
            Shape is (num_envs, num_joints).
        soft_limit_factor: Input scalar factor for computing soft limits (typically 0.0-1.0).
        env_mask: Input array of environment mask. Shape is (num_envs,).
        joint_mask: Input array of joint mask. Shape is (num_joints,).
        joint_pos_limits_lower: Output array where joint position limits lower are written. Shape is
            (num_envs, num_joints).
        joint_pos_limits_upper: Output array where joint position limits upper are written. Shape is
            (num_envs, num_joints).
        soft_joint_pos_limits: Output array where soft joint position limits are written.
            Shape is (num_envs, num_joints).
        default_joint_pos: Input/output array of default joint positions. If any values fall
            outside the limits, they are clamped. Shape is (num_envs, num_joints).
        clamped_defaults: Output 1-element array flag indicating whether any default joint
            positions were clamped. Non-zero if any clamping occurred. Shape is (1,).
    """
    i, j = wp.tid()
    if env_mask[i] and joint_mask[j]:
        joint_pos_limits_lower[i, j] = in_data[i, j][0]
        joint_pos_limits_upper[i, j] = in_data[i, j][1]
        if (default_joint_pos[i, j] < joint_pos_limits_lower[i, j]) or default_joint_pos[i, j] > joint_pos_limits_upper[
            i, j
        ]:
            wp.atomic_add(clamped_defaults, 0, 1)
            default_joint_pos[i, j] = wp.clamp(
                default_joint_pos[i, j],
                joint_pos_limits_lower[i, j],
                joint_pos_limits_upper[i, j],
            )
        soft_joint_pos_limits[i, j] = compute_soft_joint_pos_limits_func(
            wp.vec2f(joint_pos_limits_lower[i, j], joint_pos_limits_upper[i, j]), soft_limit_factor
        )


@wp.kernel
def write_joint_friction_data_to_buffer(
    in_friction: wp.array2d(dtype=wp.float32),
    in_dynamic_friction: wp.array2d(dtype=wp.float32),
    in_viscous_friction: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=wp.int32),
    joint_ids: wp.array(dtype=wp.int32),
    from_mask: bool,
    out_friction: wp.array2d(dtype=wp.float32),
    out_dynamic_friction: wp.array2d(dtype=wp.float32),
    out_viscous_friction: wp.array2d(dtype=wp.float32),
    friction_props: wp.array3d(dtype=wp.float32),
):
    """Write joint friction data to the output buffers.

    This kernel writes joint friction coefficients from input arrays to output buffers
    and updates the friction properties array used by the physics simulation.

    Args:
        in_friction: Input array containing joint friction coefficients. Shape is
            (num_envs, num_joints) or (num_selected_envs, num_selected_joints) depending
            on from_mask. Can be None if not provided.
        in_dynamic_friction: Input array containing joint dynamic friction coefficients.
            Shape is (num_envs, num_joints) or (num_selected_envs, num_selected_joints).
            Can be None if not provided.
        in_viscous_friction: Input array containing joint viscous friction coefficients.
            Shape is (num_envs, num_joints) or (num_selected_envs, num_selected_joints).
            Can be None if not provided.
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        joint_ids: Input array of joint indices to write to. Shape is (num_selected_joints,).
        from_mask: Input flag indicating whether to use masked indexing. If True, indices from
            env_ids and joint_ids are used to index into input arrays. If False, input arrays
            are indexed directly using the thread indices.
        out_friction: Output array where joint friction coefficients are written. Shape is
            (num_envs, num_joints).
        out_dynamic_friction: Output array where joint dynamic friction coefficients are written.
            Shape is (num_envs, num_joints).
        out_viscous_friction: Output array where joint viscous friction coefficients are written.
            Shape is (num_envs, num_joints).
        friction_props: Output array where friction properties are written for the physics
            simulation. Shape is (num_envs, num_joints, 3) where the last dimension contains
            [friction, dynamic_friction, viscous_friction].
    """
    i, j = wp.tid()
    # First update the output buffers
    if from_mask:
        out_friction[env_ids[i], joint_ids[j]] = in_friction[env_ids[i], joint_ids[j]]
        if in_dynamic_friction:
            out_dynamic_friction[env_ids[i], joint_ids[j]] = in_dynamic_friction[env_ids[i], joint_ids[j]]
        if in_viscous_friction:
            out_viscous_friction[env_ids[i], joint_ids[j]] = in_viscous_friction[env_ids[i], joint_ids[j]]
    else:
        out_friction[env_ids[i], joint_ids[j]] = in_friction[i, j]
        if in_dynamic_friction:
            out_dynamic_friction[env_ids[i], joint_ids[j]] = in_dynamic_friction[i, j]
        if in_viscous_friction:
            out_viscous_friction[env_ids[i], joint_ids[j]] = in_viscous_friction[i, j]
    # Then update the friction properties
    friction_props[env_ids[i], joint_ids[j], 0] = out_friction[env_ids[i], joint_ids[j]]
    if in_dynamic_friction:
        friction_props[env_ids[i], joint_ids[j], 1] = out_dynamic_friction[env_ids[i], joint_ids[j]]
    if in_viscous_friction:
        friction_props[env_ids[i], joint_ids[j], 2] = out_viscous_friction[env_ids[i], joint_ids[j]]


@wp.kernel
def write_joint_friction_param_to_buffer(
    in_data: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=wp.int32),
    joint_ids: wp.array(dtype=wp.int32),
    buffer_index: wp.int32,
    from_mask: bool,
    out_data: wp.array2d(dtype=wp.float32),
    out_buffer: wp.array3d(dtype=wp.float32),
):
    """Write a joint friction parameter to the output buffers.

    This kernel writes a single joint friction parameter (e.g., dynamic or viscous friction)
    from the input array to both a 2D output array and a specific slice of a 3D buffer array.

    Args:
        in_data: Input array containing joint friction parameter values. Shape is
            (num_envs, num_joints) or (num_selected_envs, num_selected_joints) depending
            on from_mask.
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        joint_ids: Input array of joint indices to write to. Shape is (num_selected_joints,).
        buffer_index: Input scalar index specifying which slice of the 3D buffer to write to.
            Typically 0 for friction, 1 for dynamic friction, or 2 for viscous friction.
        from_mask: Input flag indicating whether to use masked indexing. If True, indices from
            env_ids and joint_ids are used to index into in_data. If False, in_data is indexed
            directly using the thread indices.
        out_data: Output array where friction parameter values are written. Shape is
            (num_envs, num_joints).
        out_buffer: Output 3D array where friction parameter values are written to the specified
            slice. Shape is (num_envs, num_joints, num_friction_params).
    """
    i, j = wp.tid()
    if from_mask:
        out_data[env_ids[i], joint_ids[j]] = in_data[env_ids[i], joint_ids[j]]
        out_buffer[env_ids[i], joint_ids[j], buffer_index] = in_data[env_ids[i], joint_ids[j]]
    else:
        out_data[env_ids[i], joint_ids[j]] = in_data[i, j]
        out_buffer[env_ids[i], joint_ids[j], buffer_index] = in_data[i, j]


@wp.kernel
def float_data_to_buffer_with_indices(
    in_data: wp.float32,
    env_ids: wp.array(dtype=wp.int32),
    joint_ids: wp.array(dtype=wp.int32),
    out_data: wp.array2d(dtype=wp.float32),
):
    """Write a scalar float value to a 2D buffer at specified indices.

    This kernel broadcasts a single scalar float value to all specified (env_id, joint_id)
    locations in the output buffer.

    Args:
        in_data: Input scalar float value to broadcast.
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        joint_ids: Input array of joint indices to write to. Shape is (num_selected_joints,).
        out_data: Output array where the scalar value is written. Shape is (num_envs, num_joints).
    """
    i, j = wp.tid()
    out_data[env_ids[i], joint_ids[j]] = in_data


@wp.kernel
def float_data_to_buffer_with_mask(
    in_data: wp.float32,
    env_mask: wp.array(dtype=wp.bool),
    joint_mask: wp.array(dtype=wp.bool),
    out_data: wp.array2d(dtype=wp.float32),
):
    """Write a scalar float value to a 2D buffer at specified mask.

    This kernel broadcasts a single scalar float value to all the positions that are marked as True in the environment
    and joint masks.

    Args:
        in_data: Input scalar float value to broadcast.
        env_mask: Input array of environment mask. Shape is (num_envs,).
        joint_mask: Input array of joint mask. Shape is (num_joints,).
        out_data: Output array where the scalar value is written. Shape is (num_envs, num_joints).
    """
    i, j = wp.tid()
    if env_mask[i] and joint_mask[j]:
        out_data[i, j] = in_data


@wp.kernel
def update_soft_joint_pos_limits(
    joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    soft_limit_factor: wp.float32,
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
):
    """Update soft joint position limits based on hard limits and a soft limit factor.

    This kernel computes soft joint position limits from hard joint position limits using
    a soft limit factor. Soft limits are typically used to provide a safety margin before
    reaching the hard limits.

    Args:
        joint_pos_limits: Input array of hard joint position limits as vec2f (lower, upper).
            Shape is (num_envs, num_joints).
        soft_limit_factor: Input scalar factor for computing soft limits (typically 0.0-1.0).
            A value of 1.0 means soft limits equal hard limits, while smaller values create
            a tighter range.
        soft_joint_pos_limits: Output array where soft joint position limits are written.
            Shape is (num_envs, num_joints).
    """
    i, j = wp.tid()
    soft_joint_pos_limits[i, j] = compute_soft_joint_pos_limits_func(joint_pos_limits[i, j], soft_limit_factor)


@wp.kernel
def update_default_joint_values(
    source: wp.array(dtype=wp.float32),
    ids: wp.array(dtype=wp.int32),
    target: wp.array2d(dtype=wp.float32),
):
    """Update default joint values from a source array using joint indices.

    This kernel writes values from a 1D source array to specific joint indices in a 2D
    target array for all environments.

    Args:
        source: Input array containing joint values to write. Shape is (num_joints,).
        ids: Input array of joint indices specifying which joints to update. Shape is
            (num_selected_joints,).
        target: Output array where joint values are written. Shape is (num_envs, num_joints).
            Values are written to target[i, ids[j]] for all environments i.
    """
    i, j = wp.tid()
    target[i, ids[j]] = source[j]


@wp.kernel
def update_targets(
    source_joint_positions: wp.array2d(dtype=wp.float32),
    source_joint_velocities: wp.array2d(dtype=wp.float32),
    source_joint_efforts: wp.array2d(dtype=wp.float32),
    joint_indices: wp.array(dtype=wp.int32),
    target_joint_positions: wp.array2d(dtype=wp.float32),
    target_joint_velocities: wp.array2d(dtype=wp.float32),
    target_joint_efforts: wp.array2d(dtype=wp.float32),
):
    """Update joint target values from source arrays using joint indices.

    This kernel copies joint positions, velocities, and efforts from source arrays to
    target arrays, remapping joint indices using the provided joint_indices array.
    Only non-None source arrays are processed.

    Args:
        source_joint_positions: Input array of source joint positions. Shape is
            (num_envs, num_selected_joints). Can be None if not provided.
        source_joint_velocities: Input array of source joint velocities. Shape is
            (num_envs, num_selected_joints). Can be None if not provided.
        source_joint_efforts: Input array of source joint efforts. Shape is
            (num_envs, num_selected_joints). Can be None if not provided.
        joint_indices: Input array of joint indices for remapping. Shape is
            (num_selected_joints,). Specifies which joints in the target arrays to update.
        target_joint_positions: Output array where joint positions are written. Shape is
            (num_envs, num_joints).
        target_joint_velocities: Output array where joint velocities are written. Shape is
            (num_envs, num_joints).
        target_joint_efforts: Output array where joint efforts are written. Shape is
            (num_envs, num_joints).
    """
    i, j = wp.tid()
    if source_joint_positions:
        target_joint_positions[i, joint_indices[j]] = source_joint_positions[i, j]
    if source_joint_velocities:
        target_joint_velocities[i, joint_indices[j]] = source_joint_velocities[i, j]
    if source_joint_efforts:
        target_joint_efforts[i, joint_indices[j]] = source_joint_efforts[i, j]


@wp.kernel
def update_actuator_state_model(
    source_computed_effort: wp.array2d(dtype=wp.float32),
    source_applied_effort: wp.array2d(dtype=wp.float32),
    source_gear_ratio: wp.array2d(dtype=wp.float32),
    source_vel_limits: wp.array2d(dtype=wp.float32),
    joint_indices: wp.array(dtype=wp.int32),
    target_computed_effort: wp.array2d(dtype=wp.float32),
    target_applied_effort: wp.array2d(dtype=wp.float32),
    target_gear_ratio: wp.array2d(dtype=wp.float32),
    target_soft_joint_vel_limits: wp.array2d(dtype=wp.float32),
):
    """Update actuator state model parameters from source arrays using joint indices.

    This kernel copies actuator state model parameters (computed effort, applied effort,
    gear ratio, and velocity limits) from source arrays to target arrays, remapping
    joint indices using the provided joint_indices array.

    Args:
        source_computed_effort: Input array of source computed effort values. Shape is
            (num_envs, num_selected_joints).
        source_applied_effort: Input array of source applied effort values. Shape is
            (num_envs, num_selected_joints).
        source_gear_ratio: Input array of source gear ratio values. Shape is
            (num_envs, num_selected_joints). Can be None if not provided.
        source_vel_limits: Input array of source velocity limit values. Shape is
            (num_envs, num_selected_joints).
        joint_indices: Input array of joint indices for remapping. Shape is
            (num_selected_joints,). Specifies which joints in the target arrays to update.
        target_computed_effort: Output array where computed effort values are written.
            Shape is (num_envs, num_joints).
        target_applied_effort: Output array where applied effort values are written.
            Shape is (num_envs, num_joints).
        target_gear_ratio: Output array where gear ratio values are written. Shape is
            (num_envs, num_joints).
        target_soft_joint_vel_limits: Output array where soft joint velocity limits are
            written. Shape is (num_envs, num_joints).
    """
    i, j = wp.tid()
    target_computed_effort[i, joint_indices[j]] = source_computed_effort[i, j]
    target_applied_effort[i, joint_indices[j]] = source_applied_effort[i, j]
    target_soft_joint_vel_limits[i, joint_indices[j]] = source_vel_limits[i, j]
    if source_gear_ratio:
        target_gear_ratio[i, joint_indices[j]] = source_gear_ratio[i, j]


@wp.kernel
def extract_friction_properties(
    friction_props: wp.array3d(dtype=wp.float32),
    out_friction: wp.array2d(dtype=wp.float32),
    out_dynamic_friction: wp.array2d(dtype=wp.float32),
    out_viscous_friction: wp.array2d(dtype=wp.float32),
):
    """Extract friction properties from a 3D array into separate 2D arrays.

    This kernel extracts the three friction components (static friction, dynamic friction,
    and viscous friction) from a 3D friction properties array into three separate 2D arrays.

    Args:
        friction_props: Input 3D array containing friction properties. Shape is
            (num_envs, num_joints, 3) where the last dimension contains
            [friction, dynamic_friction, viscous_friction].
        out_friction: Output array where static friction coefficients are written.
            Shape is (num_envs, num_joints).
        out_dynamic_friction: Output array where dynamic friction coefficients are written.
            Shape is (num_envs, num_joints).
        out_viscous_friction: Output array where viscous friction coefficients are written.
            Shape is (num_envs, num_joints).
    """
    i, j = wp.tid()
    out_friction[i, j] = friction_props[i, j, 0]
    out_dynamic_friction[i, j] = friction_props[i, j, 1]
    out_viscous_friction[i, j] = friction_props[i, j, 2]


@wp.kernel
def concat_joint_pos_limits_lower_and_upper(
    joint_pos_limits_lower: wp.array2d(dtype=wp.float32),
    joint_pos_limits_upper: wp.array2d(dtype=wp.float32),
    joint_pos_limits: wp.array2d(dtype=wp.vec2f),
):
    """Concatenate joint position limits lower and upper in a single array.

    Args:
        joint_pos_limits_lower: Input array of joint position limits lower. Shape is (num_envs, num_joints).
        joint_pos_limits_upper: Input array of joint position limits upper. Shape is (num_envs, num_joints).
        joint_pos_limits: Output array where joint position limits are written. Shape is (num_envs, num_joints, 2).
    """
    i, j = wp.tid()
    joint_pos_limits[i, j] = wp.vec2f(joint_pos_limits_lower[i, j], joint_pos_limits_upper[i, j])
