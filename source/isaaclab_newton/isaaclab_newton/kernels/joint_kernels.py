# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp

"""
Helper kernels for updating joint data.
"""


@wp.kernel
def update_joint_array(
    new_data: wp.array2d(dtype=wp.float32),
    joint_data: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=bool),
    joint_mask: wp.array(dtype=bool),
):
    """
    Update the joint data for the given environment and joint indices from the newton data.

    .. note:: The :arg:`env_mask` length must be equal to the number of instances in the newton data.
    The :arg:`joint_mask` length must be equal to the number of joints in the newton data. The :arg:`new_data` shape
    must match the :arg:`joint_data` shape.

    Args:
        new_data: The new data to update the joint data with. Shape is (num_instances, num_joints).
        joint_data: The joint data to update. Shape is (num_instances, num_joints). (modified)
        env_mask: The environment mask to update the joint data for. Shape is (num_instances,).
        joint_mask: The joint mask to update the joint data for. Shape is (num_joints,).
    """
    env_index, joint_index = wp.tid()
    if env_mask[env_index] and joint_mask[joint_index]:
        joint_data[env_index, joint_index] = new_data[env_index, joint_index]


@wp.kernel
def update_joint_array_int(
    new_data: wp.array2d(dtype=wp.int32),
    joint_data: wp.array2d(dtype=wp.int32),
    env_mask: wp.array(dtype=bool),
    joint_mask: wp.array(dtype=bool),
):
    """
    Update the joint data for the given environment and joint indices from the newton data.

    .. note:: The :arg:`env_mask` length must be equal to the number of instances in the newton data.
    The :arg:`joint_mask` length must be equal to the number of joints in the newton data. The :arg:`new_data` shape
    must match the :arg:`joint_data` shape.

    Args:
        new_data: The new data to update the joint data with. Shape is (num_instances, num_joints).
        joint_data: The joint data to update. Shape is (num_instances, num_joints). (modified)
        env_mask: The environment mask to update the joint data for. Shape is (num_instances,).
        joint_mask: The joint mask to update the joint data for. Shape is (num_joints,).
    """
    env_index, joint_index = wp.tid()
    if env_mask[env_index] and joint_mask[joint_index]:
        joint_data[env_index, joint_index] = new_data[env_index, joint_index]


@wp.kernel
def update_joint_array_with_value_array(
    value: wp.array(dtype=wp.float32),
    joint_data: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=bool),
    joint_mask: wp.array(dtype=bool),
):
    """Update the joint data for the given environment and joint indices with a value array.

    .. note:: The :arg:`env_mask` length must be equal to the number of instances in the newton data.
    The :arg:`joint_mask` length must be equal to the number of joints in the newton data. The :arg:`value` shape
    must be (num_joints,).

    Args:
        value: The value array to update the joint data with. Shape is (num_joints,).
        joint_data: The joint data to update. Shape is (num_instances, num_joints). (modified)
        env_mask: The environment mask to update the joint data for. Shape is (num_instances,).
        joint_mask: The joint mask to update the joint data for. Shape is (num_joints,).
    """
    env_index, joint_index = wp.tid()
    if env_mask[env_index] and joint_mask[joint_index]:
        joint_data[env_index, joint_index] = value[joint_index]


@wp.kernel
def update_joint_array_with_value(
    value: wp.float32,
    joint_data: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=bool),
    joint_mask: wp.array(dtype=bool),
):
    """Update the joint data for the given environment and joint indices with a value.

    .. note:: The :arg:`env_mask` length must be equal to the number of instances in the newton data.
    The :arg:`joint_mask` length must be equal to the number of joints in the newton data. The :arg:`joint_data` shape
    must be (num_instances, num_joints).

    Args:
        value: The value to update the joint data with.
        joint_data: The joint data to update. Shape is (num_instances, num_joints). (modified)
        env_mask: The environment mask to update the joint data for. Shape is (num_instances,).
        joint_mask: The joint mask to update the joint data for. Shape is (num_joints,).
    """
    env_index, joint_index = wp.tid()
    if env_mask[env_index] and joint_mask[joint_index]:
        joint_data[env_index, joint_index] = value


@wp.kernel
def update_joint_array_with_value_int(
    value: wp.int32,
    joint_data: wp.array2d(dtype=wp.int32),
    env_mask: wp.array(dtype=bool),
    joint_mask: wp.array(dtype=bool),
):
    """Update the joint data for the given environment and joint indices with a value.

    .. note:: The :arg:`env_mask` length must be equal to the number of instances in the newton data.
    The :arg:`joint_mask` length must be equal to the number of joints in the newton data. The :arg:`joint_data` shape
    must be (num_instances, num_joints).

    Args:
        value: The value to update the joint data with.
        joint_data: The joint data to update. Shape is (num_instances, num_joints). (modified)
        env_mask: The environment mask to update the joint data for. Shape is (num_instances,).
        joint_mask: The joint mask to update the joint data for. Shape is (num_joints,).
    """
    env_index, joint_index = wp.tid()
    if env_mask[env_index] and joint_mask[joint_index]:
        joint_data[env_index, joint_index] = value


"""
Kernels to update joint limits.
"""


@wp.func
def get_soft_joint_limits(lower_limit: float, upper_limit: float, soft_factor: float) -> wp.vec2f:
    """Get the soft joint limits for the given lower and upper limits and soft factor.

    Args:
        lower_limit: The lower limit of the joint.
        upper_limit: The upper limit of the joint.
        soft_factor: The soft factor to use for the joint limits.

    Returns:
        The soft joint limits. Shape is (2,).
    """
    mean = (lower_limit + upper_limit) / 2.0
    range = upper_limit - lower_limit
    lower_limit = mean - 0.5 * range * soft_factor
    upper_limit = mean + 0.5 * range * soft_factor
    return wp.vec2f(lower_limit, upper_limit)


@wp.kernel
def update_joint_limits(
    new_limits_lower: wp.array2d(dtype=wp.float32),
    new_limits_upper: wp.array2d(dtype=wp.float32),
    soft_factor: float,
    lower_limits: wp.array2d(dtype=wp.float32),
    upper_limits: wp.array2d(dtype=wp.float32),
    soft_joint_limits: wp.array2d(dtype=wp.vec2f),
    env_mask: wp.array(dtype=bool),
    joint_mask: wp.array(dtype=bool),
):
    """Update the joint limits for the given environment and joint indices.

    .. note:: The :arg:`env_mask` length must be equal to the number of instances in the newton data.
    The :arg:`joint_mask` length must be equal to the number of joints in the newton data.

    Args:
        new_limits_lower: The new lower limits to update the joint limits with. Shape is (num_instances, num_joints).
        new_limits_upper: The new upper limits to update the joint limits with. Shape is (num_instances, num_joints).
        soft_factor: The soft factor to use for the soft joint limits.
        lower_limits: The lower limits to update the joint limits with. Shape is (num_instances, num_joints). (modified)
        upper_limits: The upper limits to update the joint limits with. Shape is (num_instances, num_joints). (modified)
        soft_joint_limits: The soft joint limits to update. Shape is (num_instances, num_joints). (modified)
        env_mask: The environment mask to update the joint limits for. Shape is (num_instances,).
        joint_mask: The joint mask to update the joint limits for. Shape is (num_joints,).
    """
    env_index, joint_index = wp.tid()
    if env_mask[env_index] and joint_mask[joint_index]:
        lower_limits[env_index, joint_index] = new_limits_lower[env_index, joint_index]
        upper_limits[env_index, joint_index] = new_limits_upper[env_index, joint_index]

        soft_joint_limits[env_index, joint_index] = get_soft_joint_limits(
            lower_limits[env_index, joint_index], upper_limits[env_index, joint_index], soft_factor
        )


@wp.kernel
def update_joint_limits_with_value(
    new_limits: float,
    soft_factor: float,
    lower_limits: wp.array2d(dtype=wp.float32),
    upper_limits: wp.array2d(dtype=wp.float32),
    soft_joint_limits: wp.array2d(dtype=wp.vec2f),
    env_mask: wp.array(dtype=bool),
    joint_mask: wp.array(dtype=bool),
):
    """Update the joint limits for the given environment and joint indices with a value.

    .. note:: The :arg:`env_mask` length must be equal to the number of instances in the newton data.
    The :arg:`joint_mask` length must be equal to the number of joints in the newton data.

    Args:
        new_limits: The new limits to update the joint limits with.
        soft_factor: The soft factor to use for the soft joint limits.
        lower_limits: The lower limits to update the joint limits with. Shape is (num_instances, num_joints). (modified)
        upper_limits: The upper limits to update the joint limits with. Shape is (num_instances, num_joints). (modified)
        soft_joint_limits: The soft joint limits to update. Shape is (num_instances, num_joints). (modified)
        env_mask: The environment mask to update the joint limits for. Shape is (num_instances,).
        joint_mask: The joint mask to update the joint limits for. Shape is (num_joints,).
    """
    env_index, joint_index = wp.tid()
    if env_mask[env_index] and joint_mask[joint_index]:
        lower_limits[env_index, joint_index] = new_limits
        upper_limits[env_index, joint_index] = new_limits

        soft_joint_limits[env_index, joint_index] = get_soft_joint_limits(
            lower_limits[env_index, joint_index], upper_limits[env_index, joint_index], soft_factor
        )


@wp.kernel
def update_joint_limits_value_vec2f(
    new_limits: wp.vec2f,
    soft_factor: float,
    lower_limits: wp.array2d(dtype=wp.float32),
    upper_limits: wp.array2d(dtype=wp.float32),
    soft_joint_limits: wp.array2d(dtype=wp.vec2f),
    env_mask: wp.array(dtype=bool),
    joint_mask: wp.array(dtype=bool),
):
    """Update the joint limits for the given environment and joint indices with a value.

    Args:
        new_limits: The new limits to update the joint limits with.
        soft_factor: The soft factor to use for the soft joint limits.
        lower_limits: The lower limits to update the joint limits with. Shape is (num_instances, num_joints). (modified)
        upper_limits: The upper limits to update the joint limits with. Shape is (num_instances, num_joints). (modified)
        soft_joint_limits: The soft joint limits to update. Shape is (num_instances, num_joints). (modified)
        env_mask: The environment mask to update the joint limits for. Shape is (num_instances,).
        joint_mask: The joint mask to update the joint limits for. Shape is (num_joints,).
    """
    env_index, joint_index = wp.tid()
    if env_mask[env_index] and joint_mask[joint_index]:
        lower_limits[env_index, joint_index] = new_limits[0]
        upper_limits[env_index, joint_index] = new_limits[1]

        soft_joint_limits[env_index, joint_index] = get_soft_joint_limits(
            lower_limits[env_index, joint_index], upper_limits[env_index, joint_index], soft_factor
        )


"""
Kernels to update joint position from joint limits.
"""


@wp.kernel
def update_joint_pos_with_limits(
    joint_pos_limits_lower: wp.array2d(dtype=wp.float32),
    joint_pos_limits_upper: wp.array2d(dtype=wp.float32),
    joint_pos: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=bool),
    joint_mask: wp.array(dtype=bool),
):
    """Update the joint position for the given environment and joint indices with the limits.

    .. note:: The :arg:`env_mask` length must be equal to the number of instances in the newton data.
    The :arg:`joint_mask` length must be equal to the number of joints in the newton data.

    Args:
        joint_pos_limits_lower: The lower limits to update the joint position with. Shape is (num_instances, num_joints).
        joint_pos_limits_upper: The upper limits to update the joint position with. Shape is (num_instances, num_joints).
        joint_pos: The joint position to update. Shape is (num_instances, num_joints). (modified)
        env_mask: The environment mask to update the joint position for. Shape is (num_instances,).
        joint_mask: The joint mask to update the joint position for. Shape is (num_joints,).
    """
    env_index, joint_index = wp.tid()
    if env_mask[env_index] and joint_mask[joint_index]:
        joint_pos[env_index, joint_index] = wp.clamp(
            joint_pos[env_index, joint_index],
            joint_pos_limits_lower[env_index, joint_index],
            joint_pos_limits_upper[env_index, joint_index],
        )


@wp.kernel
def update_joint_pos_with_limits_value(
    joint_pos_limits: float,
    joint_pos: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=bool),
    joint_mask: wp.array(dtype=bool),
):
    """Update the joint position for the given environment and joint indices with the limits.

    .. note:: The :arg:`env_mask` length must be equal to the number of instances in the newton data.
    The :arg:`joint_mask` length must be equal to the number of joints in the newton data.

    Args:
        joint_pos_limits: The joint position limits to update.
        joint_pos: The joint position to update. Shape is (num_instances, num_joints). (modified)
        env_mask: The environment mask to update the joint position for. Shape is (num_instances,).
        joint_mask: The joint mask to update the joint position for. Shape is (num_joints,).
    """
    env_index, joint_index = wp.tid()
    if env_mask[env_index] and joint_mask[joint_index]:
        joint_pos[env_index, joint_index] = wp.clamp(
            joint_pos[env_index, joint_index], joint_pos_limits, joint_pos_limits
        )


@wp.kernel
def update_joint_pos_with_limits_value_vec2f(
    joint_pos_limits: wp.vec2f,
    joint_pos: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=bool),
    joint_mask: wp.array(dtype=bool),
):
    """Update the joint position for the given environment and joint indices with the limits.

    .. note:: The :arg:`env_mask` length must be equal to the number of instances in the newton data.
    The :arg:`joint_mask` length must be equal to the number of joints in the newton data.

    Args:
        joint_pos_limits: The joint position limits to update. Shape is (2,)
        joint_pos: The joint position to update. Shape is (num_instances, num_joints). (modified)
        env_mask: The environment mask to update the joint position for. Shape is (num_instances,).
        joint_mask: The joint mask to update the joint position for. Shape is (num_joints,).
    """
    env_index, joint_index = wp.tid()
    if env_mask[env_index] and joint_mask[joint_index]:
        joint_pos[env_index, joint_index] = wp.clamp(
            joint_pos[env_index, joint_index], joint_pos_limits[0], joint_pos_limits[1]
        )


"""
Helper kernel to reconstruct limits
"""


@wp.kernel
def make_joint_pos_limits_from_lower_and_upper_limits(
    lower_limits: wp.array2d(dtype=wp.float32),
    upper_limits: wp.array2d(dtype=wp.float32),
    joint_pos_limits: wp.array2d(dtype=wp.vec2f),
):
    """Make the joint position limits from the lower and upper limits.

    Args:
        lower_limits: The lower limits to make the joint position limits from. Shape is (num_instances, num_joints).
        upper_limits: The upper limits to make the joint position limits from. Shape is (num_instances, num_joints).
        joint_pos_limits: The joint position limits to make. Shape is (num_instances, num_joints, 2). (destination)
    """
    env_index, joint_index = wp.tid()
    joint_pos_limits[env_index, joint_index] = wp.vec2f(
        lower_limits[env_index, joint_index], upper_limits[env_index, joint_index]
    )


"""
Helper kernel to update soft joint position limits.
"""


@wp.kernel
def update_soft_joint_pos_limits(
    joint_pos_limits_lower: wp.array2d(dtype=wp.float32),
    joint_pos_limits_upper: wp.array2d(dtype=wp.float32),
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    soft_factor: float,
):
    """Update the soft joint position limits for the given environment and joint indices.

    Args:
        joint_pos_limits_lower: The lower limits to update the soft joint position limits with. Shape is (num_instances, num_joints).
        joint_pos_limits_upper: The upper limits to update the soft joint position limits with. Shape is (num_instances, num_joints).
        soft_joint_pos_limits: The soft joint position limits to update. Shape is (num_instances, num_joints). (modified)
        soft_factor: The soft factor to use for the soft joint position limits.
    """
    env_index, joint_index = wp.tid()
    soft_joint_pos_limits[env_index, joint_index] = get_soft_joint_limits(
        joint_pos_limits_lower[env_index, joint_index], joint_pos_limits_upper[env_index, joint_index], soft_factor
    )


@wp.kernel
def update_default_joint_pos(
    joint_pos_limits_lower: wp.array2d(dtype=wp.float32),
    joint_pos_limits_upper: wp.array2d(dtype=wp.float32),
    joint_pos: wp.array2d(dtype=wp.float32),
):
    """Update the default joint position for the given environment and joint indices."""
    env_index, joint_index = wp.tid()
    joint_pos[env_index, joint_index] = wp.clamp(
        joint_pos[env_index, joint_index],
        joint_pos_limits_lower[env_index, joint_index],
        joint_pos_limits_upper[env_index, joint_index],
    )


"""
Kernels to derive joint acceleration from velocity.
"""


@wp.kernel
def derive_joint_acceleration_from_velocity(
    joint_velocity: wp.array2d(dtype=wp.float32),
    previous_joint_velocity: wp.array2d(dtype=wp.float32),
    dt: float,
    joint_acceleration: wp.array2d(dtype=wp.float32),
):
    """
    Derive the joint acceleration from the velocity.

    Args:
        joint_velocity: The joint velocity. Shape is (num_instances, num_joints).
        previous_joint_velocity: The previous joint velocity. Shape is (num_instances, num_joints).
        dt: The time step.
        joint_acceleration: The joint acceleration. Shape is (num_instances, num_joints). (modified)
    """
    env_index, joint_index = wp.tid()
    # compute acceleration
    joint_acceleration[env_index, joint_index] = (
        joint_velocity[env_index, joint_index] - previous_joint_velocity[env_index, joint_index]
    ) / dt

    # update previous velocity
    previous_joint_velocity[env_index, joint_index] = joint_velocity[env_index, joint_index]


@wp.kernel
def clip_joint_array_with_limits_masked(
    lower_limits: wp.array(dtype=wp.float32),
    upper_limits: wp.array(dtype=wp.float32),
    joint_array: wp.array(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    joint_mask: wp.array(dtype=wp.bool),
):
    joint_index = wp.tid()
    if env_mask[joint_index] and joint_mask[joint_index]:
        joint_array[joint_index] = wp.clamp(
            joint_array[joint_index], lower_limits[joint_index], upper_limits[joint_index]
        )


@wp.kernel
def clip_joint_array_with_limits(
    lower_limits: wp.array(dtype=wp.float32),
    upper_limits: wp.array(dtype=wp.float32),
    joint_array: wp.array(dtype=wp.float32),
):
    index = wp.tid()
    joint_array[index] = wp.clamp(joint_array[index], lower_limits[index], upper_limits[index])
