# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp

"""
Helper kernels for updating joint data.
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
