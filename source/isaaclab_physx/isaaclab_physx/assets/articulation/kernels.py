from typing import Any
import warp as wp

from ..kernels import *

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
        joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor
    )

"""
Articulation-specific warp kernels.
"""

@wp.kernel
def get_joint_acc_from_joint_vel(
    joint_vel: wp.array2d(dtype=wp.float32),
    prev_joint_vel: wp.array2d(dtype=wp.float32),
    joint_acc: wp.array2d(dtype=wp.float32),
    dt: wp.float32,
):
    """Compute the joint acceleration from the joint velocity.
    
    Args:
        joint_vel: The joint velocity.
        prev_joint_vel: The previous joint velocity.
        joint_acc: The joint acceleration.
        dt: The time step.
    """
    i, j = wp.tid()
    joint_acc[i, j] = (joint_vel[i, j] - prev_joint_vel[i, j]) / dt
    prev_joint_vel[i, j] = joint_vel[i, j]

@wp.kernel
def write_joint_vel_data(
    in_data: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    prev_joint_vel: wp.array2d(dtype=wp.float32),
    joint_acc: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=wp.int32),
    joint_ids: wp.array(dtype=wp.int32),
    from_mask: bool,
):
    """Write the joint velocity data to the buffer.

    This kernel also updates the previous joint velocity and sets the joint acceleration to 0.0.
    
    Args:
        in_data: The joint velocity data.
        joint_vel: The joint velocity.
        prev_joint_vel: The previous joint velocity.
        joint_acc: The joint acceleration.
        env_ids: The environment indices.
        joint_ids: The joint indices.
        from_mask: Whether to use a mask.
    """
    i,j = wp.tid()
    if from_mask:
        joint_vel[env_ids[i], joint_ids[j]] = in_data[env_ids[i], joint_ids[j]]
        prev_joint_vel[env_ids[i], joint_ids[j]] = in_data[env_ids[i], joint_ids[j]]
    else:
        joint_vel[env_ids[i], joint_ids[j]] = in_data[i, j]
        prev_joint_vel[env_ids[i], joint_ids[j]] = in_data[i, j]
    joint_acc[env_ids[i], joint_ids[j]] = 0.0


@wp.kernel
def write_joint_limit_data_to_buffer(
    in_data: wp.array2d(dtype=wp.vec2f),
    soft_limit_factor: wp.float32,
    joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    default_joint_pos: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=wp.int32),
    joint_ids: wp.array(dtype=wp.int32),
    from_mask: bool,
    clamped_defaults: bool,
):
    """Write the joint limit data to the buffer.
    
    Args:
        in_data: The joint limit data.
        soft_limit_factor: The soft limit factor.
        joint_pos_limits: The joint position limits.
        soft_joint_pos_limits: The soft joint position limits.
        default_joint_pos: The default joint position.
        env_ids: The environment indices.
        joint_ids: The joint indices.
        from_mask: Whether to use a mask.
        clamped_defaults: Whether the default joint positions are clamped.
    """
    i, j = wp.tid()
    if from_mask:
        joint_pos_limits[env_ids[i], joint_ids[j]] = in_data[env_ids[i], joint_ids[j]]
    else:
        joint_pos_limits[env_ids[i], joint_ids[j]] = in_data[i, j]
    if (default_joint_pos[env_ids[i], joint_ids[j]] < joint_pos_limits[env_ids[i], joint_ids[j]][0]) or default_joint_pos[env_ids[i], joint_ids[j]] > joint_pos_limits[env_ids[i], joint_ids[j]][1]:
        clamped_defaults = True
        default_joint_pos[env_ids[i], joint_ids[j]] = wp.clamp(default_joint_pos[env_ids[i], joint_ids[j]], joint_pos_limits[env_ids[i], joint_ids[j]][0], joint_pos_limits[env_ids[i], joint_ids[j]][1])
    soft_joint_pos_limits[env_ids[i], joint_ids[j]] = compute_soft_joint_pos_limits_func(joint_pos_limits[env_ids[i], joint_ids[j]], soft_limit_factor)

@wp.kernel
def write_joint_friction_data_to_buffer(
    in_friction: wp.array2d(dtype=wp.float32),
    in_dynamic_friction: wp.array2d(dtype=wp.float32),
    in_viscous_friction: wp.array2d(dtype=wp.float32),
    out_friction: wp.array2d(dtype=wp.float32),
    out_dynamic_friction: wp.array2d(dtype=wp.float32),
    out_viscous_friction: wp.array2d(dtype=wp.float32),
    friction_props: wp.array3d(dtype=wp.float32),
    env_ids: wp.array(dtype=wp.int32),
    joint_ids: wp.array(dtype=wp.int32),
    from_mask: bool,
):
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
    out_data: wp.array2d(dtype=wp.float32),
    out_buffer: wp.array3d(dtype=wp.float32),
    env_ids: wp.array(dtype=wp.int32),
    joint_ids: wp.array(dtype=wp.int32),
    buffer_index: wp.int32,
    from_mask: bool,
):
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
    out_data: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=wp.int32),
    joint_ids: wp.array(dtype=wp.int32),
):
    i,j = wp.tid()
    out_data[env_ids[i], joint_ids[j]] = in_data

@wp.kernel
def update_soft_joint_pos_limits(
    joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    soft_limit_factor: wp.float32,
):
    i, j = wp.tid()
    soft_joint_pos_limits[i, j] = compute_soft_joint_pos_limits_func(joint_pos_limits[i, j], soft_limit_factor)

@wp.kernel
def update_default_joint_values(
    target: wp.array2d(dtype=wp.float32),
    source: wp.array(dtype=wp.float32),
    ids: wp.array(dtype=wp.int32),
):
    i, j = wp.tid()
    target[i, ids[j]] = source[j]

@wp.kernel
def update_targets(
    source_joint_positions: wp.array2d(dtype=wp.float32),
    source_joint_velocities: wp.array2d(dtype=wp.float32),
    source_joint_efforts: wp.array2d(dtype=wp.float32),
    target_joint_positions: wp.array2d(dtype=wp.float32),
    target_joint_velocities: wp.array2d(dtype=wp.float32),
    target_joint_efforts: wp.array2d(dtype=wp.float32),
    joint_indices: wp.array(dtype=wp.int32),
):
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
    target_computed_effort: wp.array2d(dtype=wp.float32),
    target_applied_effort: wp.array2d(dtype=wp.float32),
    target_gear_ratio: wp.array2d(dtype=wp.float32),
    target_soft_joint_vel_limits: wp.array2d(dtype=wp.float32),
    joint_indices: wp.array(dtype=wp.int32),
):
    i, j = wp.tid()
    target_computed_effort[i, joint_indices[j]] = source_computed_effort[i, j]
    target_applied_effort[i, joint_indices[j]] = source_applied_effort[i, j]
    target_soft_joint_vel_limits[i, joint_indices[j]] = source_vel_limits[i, j]
    if source_gear_ratio:
        target_gear_ratio[i, joint_indices[j]] = source_gear_ratio[i, j]
