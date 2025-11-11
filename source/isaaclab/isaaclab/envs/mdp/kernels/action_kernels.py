import warp as wp
from typing import Any
from isaaclab.utils.warp.update_kernels import array_switch

@wp.kernel
def where_array2D_binary(
    condition: wp.array2d(dtype=wp.bool),
    true_value: wp.array2d(dtype=wp.float32),
    false_value: wp.array2d(dtype=wp.float32),
    output: wp.array2d(dtype=wp.float32),
):
    """
    Assigns the true value to the output array if the condition is true, otherwise assigns the false value.

    Args:
        condition: The condition. Shape is (N, M).
        true_value: The true value. Shape is (N, M).
        false_value: The false value. Shape is (N, M).
        output: The output array. Shape is (N, M).
    """
    index_1, index_2 = wp.tid()
    output[index_1, index_2] = true_value[index_1, index_2] if condition[index_1, index_2] else false_value[index_1, index_2]


@wp.kernel
def where_array2D_float(
    value: wp.array2d(dtype=wp.float32),
    threshold: wp.float32,
    true_value: wp.array2d(dtype=wp.float32),
    false_value: wp.array2d(dtype=wp.float32),
    output: wp.array2d(dtype=wp.float32),
):
    """
    Assigns the true value to the output array if the value is smaller than the threshold, otherwise assigns the false value.

    Args:
        value: The value. Shape is (N, M).
        threshold: The threshold.
        true_value: The true value. Shape is (N, M).
        false_value: The false value. Shape is (N, M).
        output: The output array. Shape is (N, M).
    """
    index_1, index_2 = wp.tid()
    output[index_1, index_2] = true_value[index_1, index_2] if value[index_1, index_2] < threshold else false_value[index_1, index_2]

@wp.kernel
def clip_array2D(
    value: wp.array2d(dtype=wp.float32),
    clip: wp.array(dtype=wp.vec2f),
):
    """
    Clips the values in the array to the clip.

    Args:
        value: The value. Shape is (N, M).
        clip: The clip. Shape is (N, M).
    """
    index_1, index_2 = wp.tid()
    value[index_1, index_2] = wp.clamp(value[index_1, index_2], clip[index_2, 0], clip[index_2, 1])

@wp.func
def unscale_transform(value: wp.float32, lower_limit: wp.float32, upper_limit: wp.float32) -> wp.float32:
    value_ = wp.clamp(value, -1.0, 1.0)
    offset = (lower_limit + upper_limit) * 0.5
    return value_ * (upper_limit - lower_limit) * 0.5 + offset

@wp.kernel
def process_joint_position_to_limits_action(
    raw_actions: wp.array2d(dtype=wp.float32),
    scale: wp.array(dtype=wp.float32),
    processed_actions: wp.array2d(dtype=wp.float32),
    clip: wp.array(dtype=wp.vec2f),
    rescale_to_limits: bool,
    joint_pos_limits_lower: wp.array2d(dtype=wp.float32),
    joint_pos_limits_upper: wp.array2d(dtype=wp.float32),
    joint_ids: wp.array(dtype=wp.int32),
):
    index_1, index_2 = wp.tid()
    processed_actions[index_1, index_2] = raw_actions[index_1, index_2] * scale[index_2]
    if clip:
        processed_actions[index_1, index_2] = wp.clamp(processed_actions[index_1, index_2], clip[index_2, 0], clip[index_2, 1])
    if rescale_to_limits:
        processed_actions[index_1, index_2] = unscale_transform(processed_actions[index_1, index_2], joint_pos_limits_lower[index_1, joint_ids[index_2]], joint_pos_limits_upper[index_1, joint_ids[index_2]])


@wp.kernel
def process_ema_joint_position_to_limits_action(
    source: wp.array2d(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
    clip: wp.array(dtype=wp.vec2f),
    destination: wp.array2d(dtype=wp.float32),
    destination_prev: wp.array2d(dtype=wp.float32),
):
    index_1, index_2 = wp.tid()
    destination[index_1, index_2] = alpha[index_2] * source[index_1, index_2] + (1.0 - alpha[index_2]) * destination_prev[index_1, index_2]
    destination[index_1, index_2] = wp.clamp(destination[index_1, index_2], clip[index_2, 0], clip[index_2, 1])
    destination_prev[index_1, index_2] = destination[index_1, index_2]

@wp.kernel
def process_joint_action(
    raw_actions: wp.array2d(dtype=wp.float32),
    scale: wp.array(dtype=wp.float32),
    offset: wp.array(dtype=wp.float32),
    clip: wp.array(dtype=wp.vec2f),
    processed_actions: wp.array2d(dtype=wp.float32),
):
    index_1, index_2 = wp.tid()
    processed_actions[index_1, index_2] = raw_actions[index_1, index_2] * scale[index_2] + offset[index_2]
    if clip:
        processed_actions[index_1, index_2] = wp.clamp(processed_actions[index_1, index_2], clip[index_2, 0], clip[index_2, 1])

@wp.kernel
def apply_relative_joint_position_action(
    processed_actions: wp.array2d(dtype=wp.float32),
    current_actions: wp.array2d(dtype=wp.float32),
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_ids: wp.array(dtype=wp.int32),
):
    index_1, index_2 = wp.tid()
    current_actions[index_1, index_2] = processed_actions[index_1, index_2] + joint_pos[index_1, joint_ids[index_2]]


@wp.kernel
def process_non_holonomic_action(
    raw_actions: wp.array2d(dtype=wp.float32),
    processed_actions: wp.array2d(dtype=wp.float32),
    scale: wp.array(dtype=wp.float32),
    offset: wp.array(dtype=wp.float32),
    clip: wp.array(dtype=wp.vec2f),
):
    index_1, index_2 = wp.tid()
    processed_actions[index_1, index_2] = raw_actions[index_1, index_2] * scale[index_2] + offset[index_2]
    if clip:
        processed_actions[index_1, index_2] = wp.clamp(processed_actions[index_1, index_2], clip[index_2, 0], clip[index_2, 1])

@wp.func
def get_yaw_from_quat(quat: wp.quatf) -> wp.float32:
    return wp.atan2(2.0 * (quat[1] * quat[2] + quat[0] * quat[3]), 1.0 - 2.0 * (quat[2] * quat[2] + quat[3] * quat[3]))

@wp.kernel
def apply_non_holonomic_action(
    pose_w: wp.array2d(dtype=wp.transformf),
    yaw_w: wp.array(dtype=wp.float32),
    processed_actions: wp.array2d(dtype=wp.float32),
    joint_vel_command: wp.array2d(dtype=wp.float32),
    body_idx: wp.int32,
):
    index_1 = wp.tid()
    yaw_w[index_1] = get_yaw_from_quat(wp.transform_get_rotation(pose_w[index_1, body_idx]))
    joint_vel_command[index_1, 0] = processed_actions[index_1, 0] * wp.cos(yaw_w[index_1])
    joint_vel_command[index_1, 1] = processed_actions[index_1, 0] * wp.sin(yaw_w[index_1])
    joint_vel_command[index_1, 2] = processed_actions[index_1, 2]