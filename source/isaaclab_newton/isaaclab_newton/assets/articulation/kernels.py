# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import warp as wp

from isaaclab.assets.articulation.ordering_kernels import resolve_backend_index
from isaaclab.utils.warp.index_kernel import IndexKernelDispatcher

if TYPE_CHECKING:
    import torch

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
def update_wrench_array_with_force_and_torque_ordered(
    forces: wp.array2d(dtype=wp.vec3f),
    torques: wp.array2d(dtype=wp.vec3f),
    user_to_backend: wp.array(dtype=wp.int32),
    wrench: wp.array2d(dtype=wp.spatial_vectorf),
    env_mask: wp.array(dtype=wp.bool),
    body_mask: wp.array(dtype=wp.bool),
) -> None:
    """Write public-order force and torque into a backend-order wrench array.

    Launched only under non-identity body ordering; the caller uses the
    reorder-free :func:`~isaaclab_newton.assets.kernels.update_wrench_array_with_force_and_torque`
    when no body ordering is active, so this kernel always applies the body map.

    Args:
        forces: Body-frame forces [N], shaped [num_envs, num_bodies], in public
            body order.
        torques: Body-frame torques [N*m], shaped [num_envs, num_bodies], in
            public body order.
        user_to_backend: Read-only map shaped [num_bodies] from each public body
            index to its backend body index.
        wrench: Backend-order wrench destination [N, N*m], shaped
            [num_envs, num_bodies].
        env_mask: Environment-selection mask shaped [num_envs].
        body_mask: Public-body selection mask shaped [num_bodies].
    """
    env_id, user_body_id = wp.tid()
    if env_mask[env_id] and body_mask[user_body_id]:
        backend_body_id = user_to_backend[user_body_id]
        wrench[env_id, backend_body_id] = wp.spatial_vector(
            forces[env_id, user_body_id], torques[env_id, user_body_id], wp.float32
        )


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
def get_body_com_acc_from_body_com_vel_ordered(
    body_com_vel_backend: wp.array2d(dtype=wp.spatial_vectorf),
    prev_body_com_vel_backend: wp.array2d(dtype=wp.spatial_vectorf),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    dt: wp.float32,
    body_com_acc_user: wp.array2d(dtype=wp.spatial_vectorf),
) -> None:
    """Derive public-order body COM acceleration from backend-order velocity.

    Args:
        body_com_vel_backend: Current body center-of-mass velocity in backend
            order [m/s, rad/s], shape (num_envs, num_bodies).
        prev_body_com_vel_backend: Previous body center-of-mass velocity in
            backend order [m/s, rad/s], shape (num_envs, num_bodies).
            Updated in place with the current velocity.
        user_to_backend: Public-to-backend body permutation, shape
            (num_bodies,).
        has_ordering: Whether the public-to-backend map is nonidentity.
        dt: Finite-difference time step [s].
        body_com_acc_user: Body center-of-mass acceleration in public order
            [m/s^2, rad/s^2], shape (num_envs, num_bodies).
    """
    env_id, user_body_id = wp.tid()
    backend_body_id = resolve_backend_index(user_body_id, user_to_backend, has_ordering)
    body_com_vel = body_com_vel_backend[env_id, backend_body_id]
    body_com_acc_user[env_id, user_body_id] = (body_com_vel - prev_body_com_vel_backend[env_id, backend_body_id]) / dt
    prev_body_com_vel_backend[env_id, backend_body_id] = body_com_vel


@wp.kernel
def write_joint_limit_data_to_buffer_index(
    in_data: wp.array2d(dtype=wp.vec2f),
    soft_limit_factor: wp.float32,
    env_ids: wp.array(dtype=Any),
    joint_ids: wp.array(dtype=Any),
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
    env_id = wp.int32(env_ids[i])
    joint_id = wp.int32(joint_ids[j])
    joint_pos_limits_lower[env_id, joint_id] = in_data[i, j][0]
    joint_pos_limits_upper[env_id, joint_id] = in_data[i, j][1]
    if (default_joint_pos[env_id, joint_id] < joint_pos_limits_lower[env_id, joint_id]) or default_joint_pos[
        env_id, joint_id
    ] > joint_pos_limits_upper[env_id, joint_id]:
        wp.atomic_add(clamped_defaults, 0, 1)
        default_joint_pos[env_id, joint_id] = wp.clamp(
            default_joint_pos[env_id, joint_id],
            joint_pos_limits_lower[env_id, joint_id],
            joint_pos_limits_upper[env_id, joint_id],
        )
    soft_joint_pos_limits[env_id, joint_id] = compute_soft_joint_pos_limits_func(
        wp.vec2f(joint_pos_limits_lower[env_id, joint_id], joint_pos_limits_upper[env_id, joint_id]),
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
def write_joint_limit_data_to_user_and_backend_index(
    in_data: wp.array2d(dtype=wp.vec2f),
    soft_limit_factor: wp.float32,
    env_ids: wp.array(dtype=Any),
    user_ids: wp.array(dtype=Any),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    user_joint_pos_limits_lower: wp.array2d(dtype=wp.float32),
    user_joint_pos_limits_upper: wp.array2d(dtype=wp.float32),
    user_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    backend_joint_pos_limits_lower: wp.array2d(dtype=wp.float32),
    backend_joint_pos_limits_upper: wp.array2d(dtype=wp.float32),
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    default_joint_pos: wp.array2d(dtype=wp.float32),
    clamped_defaults: wp.array(dtype=wp.int32),
):
    """Write indexed user-order joint limits into public and backend-order buffers."""
    i, j = wp.tid()
    env_id = wp.int32(env_ids[i])
    user_id = wp.int32(user_ids[j])
    backend_id = resolve_backend_index(user_id, user_to_backend, has_ordering)
    limits = in_data[i, j]
    lower = limits[0]
    upper = limits[1]

    user_joint_pos_limits_lower[env_id, user_id] = lower
    user_joint_pos_limits_upper[env_id, user_id] = upper
    user_joint_pos_limits[env_id, user_id] = limits
    backend_joint_pos_limits_lower[env_id, backend_id] = lower
    backend_joint_pos_limits_upper[env_id, backend_id] = upper

    if (default_joint_pos[env_id, user_id] < lower) or default_joint_pos[env_id, user_id] > upper:
        wp.atomic_add(clamped_defaults, 0, 1)
        default_joint_pos[env_id, user_id] = wp.clamp(default_joint_pos[env_id, user_id], lower, upper)

    soft_joint_pos_limits[env_id, user_id] = compute_soft_joint_pos_limits_func(limits, soft_limit_factor)


@wp.kernel
def write_joint_limit_data_to_user_and_backend_mask(
    in_data: wp.array2d(dtype=wp.vec2f),
    soft_limit_factor: wp.float32,
    env_mask: wp.array(dtype=wp.bool),
    user_mask: wp.array(dtype=wp.bool),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    user_joint_pos_limits_lower: wp.array2d(dtype=wp.float32),
    user_joint_pos_limits_upper: wp.array2d(dtype=wp.float32),
    user_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    backend_joint_pos_limits_lower: wp.array2d(dtype=wp.float32),
    backend_joint_pos_limits_upper: wp.array2d(dtype=wp.float32),
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    default_joint_pos: wp.array2d(dtype=wp.float32),
    clamped_defaults: wp.array(dtype=wp.int32),
):
    """Write masked user-order joint limits into public and backend-order buffers."""
    env_id, user_id = wp.tid()
    if not env_mask[env_id] or not user_mask[user_id]:
        return

    backend_id = resolve_backend_index(user_id, user_to_backend, has_ordering)
    limits = in_data[env_id, user_id]
    lower = limits[0]
    upper = limits[1]

    user_joint_pos_limits_lower[env_id, user_id] = lower
    user_joint_pos_limits_upper[env_id, user_id] = upper
    user_joint_pos_limits[env_id, user_id] = limits
    backend_joint_pos_limits_lower[env_id, backend_id] = lower
    backend_joint_pos_limits_upper[env_id, backend_id] = upper

    if (default_joint_pos[env_id, user_id] < lower) or default_joint_pos[env_id, user_id] > upper:
        wp.atomic_add(clamped_defaults, 0, 1)
        default_joint_pos[env_id, user_id] = wp.clamp(default_joint_pos[env_id, user_id], lower, upper)

    soft_joint_pos_limits[env_id, user_id] = compute_soft_joint_pos_limits_func(limits, soft_limit_factor)


@wp.kernel
def write_joint_friction_data_to_buffer(
    in_friction: wp.array2d(dtype=wp.float32),
    in_dynamic_friction: wp.array2d(dtype=wp.float32),
    in_viscous_friction: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=Any),
    joint_ids: wp.array(dtype=Any),
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
    env_id = wp.int32(env_ids[i])
    joint_id = wp.int32(joint_ids[j])
    # First update the output buffers
    if from_mask:
        out_friction[env_id, joint_id] = in_friction[env_id, joint_id]
        if in_dynamic_friction:
            out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[env_id, joint_id]
        if in_viscous_friction:
            out_viscous_friction[env_id, joint_id] = in_viscous_friction[env_id, joint_id]
    else:
        out_friction[env_id, joint_id] = in_friction[i, j]
        if in_dynamic_friction:
            out_dynamic_friction[env_id, joint_id] = in_dynamic_friction[i, j]
        if in_viscous_friction:
            out_viscous_friction[env_id, joint_id] = in_viscous_friction[i, j]
    # Then update the friction properties
    friction_props[env_id, joint_id, 0] = out_friction[env_id, joint_id]
    if in_dynamic_friction:
        friction_props[env_id, joint_id, 1] = out_dynamic_friction[env_id, joint_id]
    if in_viscous_friction:
        friction_props[env_id, joint_id, 2] = out_viscous_friction[env_id, joint_id]


@wp.kernel
def write_joint_friction_param_to_buffer(
    in_data: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=Any),
    joint_ids: wp.array(dtype=Any),
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
    env_id = wp.int32(env_ids[i])
    joint_id = wp.int32(joint_ids[j])
    if from_mask:
        out_data[env_id, joint_id] = in_data[env_id, joint_id]
        out_buffer[env_id, joint_id, buffer_index] = in_data[env_id, joint_id]
    else:
        out_data[env_id, joint_id] = in_data[i, j]
        out_buffer[env_id, joint_id, buffer_index] = in_data[i, j]


@wp.kernel
def float_data_to_buffer_with_indices(
    in_data: wp.float32,
    env_ids: wp.array(dtype=Any),
    joint_ids: wp.array(dtype=Any),
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
    env_id = wp.int32(env_ids[i])
    joint_id = wp.int32(joint_ids[j])
    out_data[env_id, joint_id] = in_data


_WRITE_JOINT_LIMIT_DATA_TO_BUFFER_INDEX_DISPATCHER = IndexKernelDispatcher(
    write_joint_limit_data_to_buffer_index, ("env_ids", "joint_ids")
)
_WRITE_JOINT_LIMIT_DATA_TO_USER_AND_BACKEND_INDEX_DISPATCHER = IndexKernelDispatcher(
    write_joint_limit_data_to_user_and_backend_index, ("env_ids", "user_ids")
)
_WRITE_JOINT_FRICTION_DATA_TO_BUFFER_DISPATCHER = IndexKernelDispatcher(
    write_joint_friction_data_to_buffer, ("env_ids", "joint_ids")
)
_WRITE_JOINT_FRICTION_PARAM_TO_BUFFER_DISPATCHER = IndexKernelDispatcher(
    write_joint_friction_param_to_buffer, ("env_ids", "joint_ids")
)
_FLOAT_DATA_TO_BUFFER_WITH_INDICES_DISPATCHER = IndexKernelDispatcher(
    float_data_to_buffer_with_indices, ("env_ids", "joint_ids")
)


def write_joint_limit_data_to_buffer_index_kernel(
    env_ids: wp.array | torch.Tensor, joint_ids: wp.array | torch.Tensor
) -> wp.Kernel:
    """Select the joint-limit buffer writer matching the selector dtypes."""
    return _WRITE_JOINT_LIMIT_DATA_TO_BUFFER_INDEX_DISPATCHER.select(env_ids, joint_ids)


def write_joint_limit_data_to_user_and_backend_index_kernel(
    env_ids: wp.array | torch.Tensor, user_ids: wp.array | torch.Tensor
) -> wp.Kernel:
    """Select the ordered joint-limit writer matching the selector dtypes."""
    return _WRITE_JOINT_LIMIT_DATA_TO_USER_AND_BACKEND_INDEX_DISPATCHER.select(env_ids, user_ids)


def write_joint_friction_data_to_buffer_kernel(
    env_ids: wp.array | torch.Tensor, joint_ids: wp.array | torch.Tensor
) -> wp.Kernel:
    """Select the joint-friction writer matching the selector dtypes."""
    return _WRITE_JOINT_FRICTION_DATA_TO_BUFFER_DISPATCHER.select(env_ids, joint_ids)


def write_joint_friction_param_to_buffer_kernel(
    env_ids: wp.array | torch.Tensor, joint_ids: wp.array | torch.Tensor
) -> wp.Kernel:
    """Select the joint-friction parameter writer matching the selector dtypes."""
    return _WRITE_JOINT_FRICTION_PARAM_TO_BUFFER_DISPATCHER.select(env_ids, joint_ids)


def float_data_to_buffer_with_indices_kernel(
    env_ids: wp.array | torch.Tensor, joint_ids: wp.array | torch.Tensor
) -> wp.Kernel:
    """Select the scalar buffer writer matching the selector dtypes."""
    return _FLOAT_DATA_TO_BUFFER_WITH_INDICES_DISPATCHER.select(env_ids, joint_ids)


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


@wp.kernel
def gather_jacobian_rows(
    src: wp.array4d(dtype=wp.float32),
    art_ids: wp.array(dtype=wp.int32),
    jacobian_body_user_to_backend: wp.array(dtype=wp.int32),
    joint_user_to_backend: wp.array(dtype=wp.int32),
    num_base_dofs: wp.int32,
    has_joint_ordering: bool,
    dst: wp.array4d(dtype=wp.float32),
):
    """Copy per-view articulation jacobian rows from a model-sized buffer into a view-sized buffer.

    Newton's ``eval_jacobian`` writes every articulation in the model (across all
    :class:`~newton.selection.ArticulationView` instances) into a single 4-D output
    shaped ``(model.articulation_count, max_links, 6, max_dofs)``. An
    ``ArticulationView`` owns only the subset indexed by ``articulation_ids``. This
    kernel gathers those rows into a contiguous view-sized destination so the
    caller-facing
    :attr:`~isaaclab.assets.BaseArticulationData.body_com_jacobian_w` contract
    ``(num_instances, num_jacobi_bodies, 6, num_joints + num_base_dofs)`` is
    preserved.

    Newton's output keeps every link row, including the fixed-base root row at
    index 0. The body-axis mapping is owned entirely by
    ``jacobian_body_user_to_backend``: it holds, for each public Jacobian body
    row, the backend row to read from ``src`` in Newton's full-row layout, and
    for fixed-base articulations it already excludes the root row. The kernel
    therefore indexes ``src`` by that map directly, with no runtime link offset
    and no body-ordering branch. The DoF axis is preserved in full including the
    6 leading free-root joint columns for floating-base, matching the
    cross-library industry convention used by PhysX, Pinocchio, Drake, MuJoCo,
    RBDL, OCS2, and iDynTree.

    The gather is in-place on a pre-allocated ``dst`` buffer, so the kernel launch
    is safe under CUDA graph capture.

    Args:
        src: Input jacobian buffer reshaped to 4-D. Shape is
            (model.articulation_count, max_links, 6, max_dofs).
        art_ids: Model-level articulation indices owned by this view. Shape is
            (num_instances,).
        jacobian_body_user_to_backend: Map from public Jacobian body row to
            backend Jacobian body row in Newton's full-row layout. Built for
            every view (also under identity ordering); for fixed-base
            articulations it excludes the fixed root row.
        joint_user_to_backend: Map from public actuated-joint index to backend
            actuated-joint index.
        num_base_dofs: Number of leading floating-base DoF columns.
        has_joint_ordering: Whether ``joint_user_to_backend`` is non-identity.
        dst: Output jacobian buffer for this view. Shape is
            (num_instances, num_jacobi_bodies, 6, num_joints + num_base_dofs),
            where ``num_jacobi_bodies`` is this asset's ``num_bodies`` minus the
            fixed-base root row (per-asset count, not the model-wide
            ``max_links``).
    """
    i, user_link, s, user_dof = wp.tid()
    backend_link = jacobian_body_user_to_backend[user_link]

    backend_dof = user_dof
    if has_joint_ordering and user_dof >= num_base_dofs:
        backend_dof = num_base_dofs + joint_user_to_backend[user_dof - num_base_dofs]

    dst[i, user_link, s, user_dof] = src[art_ids[i], backend_link, s, backend_dof]


@wp.kernel
def gather_mass_matrix_rows(
    src: wp.array3d(dtype=wp.float32),
    art_ids: wp.array(dtype=wp.int32),
    joint_user_to_backend: wp.array(dtype=wp.int32),
    num_base_dofs: wp.int32,
    has_joint_ordering: bool,
    dst: wp.array3d(dtype=wp.float32),
):
    """Copy per-view articulation mass-matrix rows from a model-sized buffer into a view-sized buffer.

    3-D analogue of :func:`gather_jacobian_rows` for the joint-space mass
    matrix written by :func:`newton.sim.articulation.eval_mass_matrix`. The
    DoF axis is preserved in full (including the leading 6 free-root rows/cols
    for floating-base articulations), matching the cross-library industry
    convention used by PhysX, Pinocchio, Drake, MuJoCo, RBDL, OCS2, and iDynTree.

    Args:
        src: Input mass-matrix buffer. Shape is
            (model.articulation_count, max_dofs, max_dofs).
        art_ids: Model-level articulation indices owned by this view. Shape is
            (num_instances,).
        joint_user_to_backend: Map from public actuated-joint index to backend
            actuated-joint index.
        num_base_dofs: Number of leading floating-base DoF rows/columns.
        has_joint_ordering: Whether ``joint_user_to_backend`` is non-identity.
        dst: Output mass-matrix buffer for this view. Shape is
            (num_instances, num_joints + num_base_dofs,
            num_joints + num_base_dofs).
    """
    i, user_row, user_col = wp.tid()
    backend_row = user_row
    backend_col = user_col
    if has_joint_ordering:
        if user_row >= num_base_dofs:
            backend_row = num_base_dofs + joint_user_to_backend[user_row - num_base_dofs]
        if user_col >= num_base_dofs:
            backend_col = num_base_dofs + joint_user_to_backend[user_col - num_base_dofs]

    dst[i, user_row, user_col] = src[art_ids[i], backend_row, backend_col]


@wp.kernel
def gather_dof_force_rows(
    src: wp.array(dtype=wp.float32),
    art_ids: wp.array(dtype=wp.int32),
    articulation_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_user_to_backend: wp.array(dtype=wp.int32),
    num_base_dofs: wp.int32,
    has_joint_ordering: bool,
    dst: wp.array2d(dtype=wp.float32),
):
    """Copy per-view articulation DoF forces from a flat model-sized buffer into a view-sized buffer.

    Flat-layout analogue of :func:`gather_mass_matrix_rows` for per-DoF outputs
    written in :attr:`newton.Model.joint_qd` layout (e.g. the gravity
    compensation force from ``newton.eval_inverse_dynamics_passive``). The DoF axis is
    preserved in full (including the leading 6 free-root entries for
    floating-base articulations), matching the cross-library industry
    convention used by PhysX, Pinocchio, Drake, MuJoCo, RBDL, OCS2, and
    iDynTree.

    The gather is in-place on a pre-allocated ``dst`` buffer, so the kernel
    launch is safe under CUDA graph capture.

    Args:
        src: Input flat DoF buffer. Shape is (model.joint_dof_count,).
        art_ids: Model-level articulation indices owned by this view. Shape is
            (num_instances,).
        articulation_start: First joint index of each model articulation
            (:attr:`newton.Model.articulation_start`). Shape is
            (model.articulation_count + 1,).
        joint_qd_start: First velocity coordinate of each model joint
            (:attr:`newton.Model.joint_qd_start`). Shape is
            (model.joint_count + 1,). Composed per thread as
            ``joint_qd_start[articulation_start[art_ids[i]]]`` to locate the
            articulation's segment in ``src``.
        joint_user_to_backend: Map from public actuated-joint index to backend
            actuated-joint index.
        num_base_dofs: Number of leading floating-base DoF entries.
        has_joint_ordering: Whether ``joint_user_to_backend`` is non-identity.
        dst: Output DoF-force buffer for this view. Shape is
            (num_instances, num_joints + num_base_dofs).
    """
    i, user_dof = wp.tid()
    backend_dof = user_dof
    if has_joint_ordering and user_dof >= num_base_dofs:
        backend_dof = num_base_dofs + joint_user_to_backend[user_dof - num_base_dofs]

    dof_start = joint_qd_start[articulation_start[art_ids[i]]]
    dst[i, user_dof] = src[dof_start + backend_dof]


@wp.kernel
def shift_jacobian_com_to_origin(
    body_link_pose: wp.array2d(dtype=wp.transformf),
    body_com_pos_b: wp.array2d(dtype=wp.vec3f),
    link_offset: wp.int32,
    src: wp.array4d(dtype=wp.float32),
    dst: wp.array4d(dtype=wp.float32),
):
    """Shift the linear-velocity rows of the Jacobian from COM to link origin.

    Newton's ``eval_jacobian`` returns ``J · q_dot = [v_com_world, omega_world]``
    per link — the linear rows are the velocity of the link's center of mass,
    expressed in world frame. The
    :attr:`~isaaclab.assets.BaseArticulationData.body_link_jacobian_w` contract
    requires the linear rows to be the velocity at the link **origin**
    (USD prim transform) so that ``J · q_dot[body_idx]`` matches
    :attr:`~isaaclab.assets.BaseArticulationData.body_link_lin_vel_w` /
    :attr:`~isaaclab.assets.BaseArticulationData.body_link_ang_vel_w`.

    The shift is the same one applied per-body by
    :func:`get_link_vel_from_root_com_vel_func`, but layered onto every
    Jacobian column: each column represents the spatial velocity contribution
    of one DoF, and shifting a spatial velocity from COM to link origin uses
    the same ``v_origin = v_com - omega x (R · body_com_pos_b)`` identity.

    Notes on layout:
        * Jacobian rows ``[0:3]`` are linear velocity, ``[3:6]`` are angular.
        * ``body_link_pose`` and ``body_com_pos_b`` are indexed by the
          articulation's full body count, so ``link_offset`` must be applied
          to map a row in the (already-gathered) ``src`` to its body index in
          the asset data. ``link_offset = 1`` for fixed-base (Newton's row 0
          fixed-root row was dropped during the prior gather);
          ``link_offset = 0`` for floating-base.

    Args:
        body_link_pose: Per-body link pose in world frame. Shape is
            (num_instances, num_bodies).
        body_com_pos_b: Per-body center-of-mass offset expressed in the body's
            link frame. Shape is (num_instances, num_bodies).
        link_offset: Offset added to the jacobian-row body index to reach the
            full body index. ``1`` for fixed-base, ``0`` for floating-base.
        src: COM-referenced Jacobian (read-only). Shape is
            (num_instances, num_jacobi_bodies, 6, num_joints + num_base_dofs).
        dst: Output buffer for the link-origin Jacobian. Same shape as
            ``src``. Linear rows ``[0:3]`` are written with the shifted
            velocity; angular rows ``[3:6]`` are copied unchanged (angular
            velocity is reference-point invariant).
    """
    n, b, dof = wp.tid()
    full_body_idx = b + link_offset

    R = wp.transform_get_rotation(body_link_pose[n, full_body_idx])
    c_world = wp.quat_rotate(R, body_com_pos_b[n, full_body_idx])

    v_com = wp.vec3(src[n, b, 0, dof], src[n, b, 1, dof], src[n, b, 2, dof])
    omega = wp.vec3(src[n, b, 3, dof], src[n, b, 4, dof], src[n, b, 5, dof])

    v_origin = v_com - wp.cross(omega, c_world)

    dst[n, b, 0, dof] = v_origin[0]
    dst[n, b, 1, dof] = v_origin[1]
    dst[n, b, 2, dof] = v_origin[2]
    dst[n, b, 3, dof] = omega[0]
    dst[n, b, 4, dof] = omega[1]
    dst[n, b, 5, dof] = omega[2]


"""
Deprecated kernels.

These kernels are retained for backward compatibility and will be removed in a future release. Prefer the
public-order asset write APIs (:meth:`~isaaclab.assets.Articulation.write_joint_position_to_sim_index` and its
siblings), which apply the ordering conversion internally, or the public-order reorder kernels from
``isaaclab.assets.articulation.ordering_kernels``. Because Warp kernels cannot emit a runtime deprecation warning
without breaking :func:`warp.launch`, the deprecation is documented here and in the changelog rather than raised at
call time.
"""


@wp.kernel
def write_joint_vel_data_index(
    in_data: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=Any),
    joint_ids: wp.array(dtype=Any),
    joint_vel: wp.array2d(dtype=wp.float32),
    prev_joint_vel: wp.array2d(dtype=wp.float32),
    joint_acc: wp.array2d(dtype=wp.float32),
):
    """Write joint velocity data to the output buffers.

    Deprecated. Prefer :meth:`~isaaclab.assets.Articulation.write_joint_velocity_to_sim_index` (or the
    ``ordering_kernels`` reorder family), which apply the public-to-backend ordering conversion internally.

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
    env_id = wp.int32(env_ids[i])
    joint_id = wp.int32(joint_ids[j])
    joint_vel[env_id, joint_id] = in_data[i, j]
    prev_joint_vel[env_id, joint_id] = in_data[i, j]
    joint_acc[env_id, joint_id] = 0.0


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

    Deprecated. Prefer :meth:`~isaaclab.assets.Articulation.write_joint_velocity_to_sim_mask` (or the
    ``ordering_kernels`` reorder family), which apply the public-to-backend ordering conversion internally.

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
def write_joint_state_data_index(
    pos_data: wp.array2d(dtype=wp.float32),
    vel_data: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=Any),
    joint_ids: wp.array(dtype=Any),
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    prev_joint_vel: wp.array2d(dtype=wp.float32),
    joint_acc: wp.array2d(dtype=wp.float32),
):
    """Write joint position and velocity data in a single kernel launch.

    Deprecated. Prefer :meth:`~isaaclab.assets.Articulation.write_joint_state_to_sim` and its indexed siblings
    (or the ``ordering_kernels`` reorder family), which apply the public-to-backend ordering conversion internally.

    Args:
        pos_data: Input joint positions. Shape is (num_selected_envs, num_selected_joints).
        vel_data: Input joint velocities. Shape is (num_selected_envs, num_selected_joints).
        env_ids: Environment indices. Shape is (num_selected_envs,).
        joint_ids: Joint indices. Shape is (num_selected_joints,).
        joint_pos: Output joint positions. Shape is (num_envs, num_joints).
        joint_vel: Output joint velocities. Shape is (num_envs, num_joints).
        prev_joint_vel: Output previous joint velocities. Shape is (num_envs, num_joints).
        joint_acc: Output joint accelerations (reset to 0). Shape is (num_envs, num_joints).
    """
    i, j = wp.tid()
    env_id = wp.int32(env_ids[i])
    joint_id = wp.int32(joint_ids[j])
    joint_pos[env_id, joint_id] = pos_data[i, j]
    joint_vel[env_id, joint_id] = vel_data[i, j]
    prev_joint_vel[env_id, joint_id] = vel_data[i, j]
    joint_acc[env_id, joint_id] = 0.0


_WRITE_JOINT_VEL_DATA_INDEX_DISPATCHER = IndexKernelDispatcher(write_joint_vel_data_index, ("env_ids", "joint_ids"))
_WRITE_JOINT_STATE_DATA_INDEX_DISPATCHER = IndexKernelDispatcher(write_joint_state_data_index, ("env_ids", "joint_ids"))


def write_joint_vel_data_index_kernel(
    env_ids: wp.array | torch.Tensor, joint_ids: wp.array | torch.Tensor
) -> wp.Kernel:
    """Select the deprecated joint-velocity writer matching the selector dtypes."""
    return _WRITE_JOINT_VEL_DATA_INDEX_DISPATCHER.select(env_ids, joint_ids)


def write_joint_state_data_index_kernel(
    env_ids: wp.array | torch.Tensor, joint_ids: wp.array | torch.Tensor
) -> wp.Kernel:
    """Select the deprecated joint-state writer matching the selector dtypes."""
    return _WRITE_JOINT_STATE_DATA_INDEX_DISPATCHER.select(env_ids, joint_ids)


@wp.kernel
def write_joint_state_data_mask(
    pos_data: wp.array2d(dtype=wp.float32),
    vel_data: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    joint_mask: wp.array(dtype=wp.bool),
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    prev_joint_vel: wp.array2d(dtype=wp.float32),
    joint_acc: wp.array2d(dtype=wp.float32),
):
    """Write joint position and velocity data in a single kernel launch using masks.

    Deprecated. Prefer :meth:`~isaaclab.assets.Articulation.write_joint_state_to_sim` and its masked siblings
    (or the ``ordering_kernels`` reorder family), which apply the public-to-backend ordering conversion internally.

    Args:
        pos_data: Input joint positions. Shape is (num_envs, num_joints).
        vel_data: Input joint velocities. Shape is (num_envs, num_joints).
        env_mask: Environment mask. Shape is (num_envs,).
        joint_mask: Joint mask. Shape is (num_joints,).
        joint_pos: Output joint positions. Shape is (num_envs, num_joints).
        joint_vel: Output joint velocities. Shape is (num_envs, num_joints).
        prev_joint_vel: Output previous joint velocities. Shape is (num_envs, num_joints).
        joint_acc: Output joint accelerations (reset to 0). Shape is (num_envs, num_joints).
    """
    i, j = wp.tid()
    if env_mask[i] and joint_mask[j]:
        joint_pos[i, j] = pos_data[i, j]
        joint_vel[i, j] = vel_data[i, j]
        prev_joint_vel[i, j] = vel_data[i, j]
        joint_acc[i, j] = 0.0
