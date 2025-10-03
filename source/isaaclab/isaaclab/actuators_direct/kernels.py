import warp as wp

@wp.kernel
def compute_pd_actuator(
    joint_targets: wp.array2d(dtype=wp.float32),
    added_effort: wp.array2d(dtype=wp.float32),
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    stiffness: wp.array2d(dtype=wp.float32),
    damping: wp.array2d(dtype=wp.float32),
    control_mode: wp.array2d(dtype=wp.int32),
    computed_effort: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    joint_mask: wp.array(dtype=wp.bool),
) -> None:
    env_index, joint_index = wp.tid()
    if env_mask[env_index] and joint_mask[joint_index]:
        # No control
        if control_mode[env_index, joint_index] == 0:
            computed_effort[env_index, joint_index] = stiffness[env_index, joint_index] * (0 - joint_pos[env_index, joint_index]) + damping[env_index, joint_index] * (0 - joint_vel[env_index, joint_index]) + added_effort[env_index, joint_index]
        # Position control
        elif control_mode[env_index, joint_index] == 1:
            computed_effort[env_index, joint_index] = stiffness[env_index, joint_index] * (joint_targets[env_index, joint_index] - joint_pos[env_index, joint_index]) + damping[env_index, joint_index] * (0 - joint_vel[env_index, joint_index]) + added_effort[env_index, joint_index]
        # Velocity control
        elif control_mode[env_index, joint_index] == 2:
            computed_effort[env_index, joint_index] = stiffness[env_index, joint_index] * (0 - joint_pos[env_index, joint_index]) + damping[env_index, joint_index] * (joint_targets[env_index, joint_index] - joint_vel[env_index, joint_index]) + added_effort[env_index, joint_index]

@wp.kernel
def clip_efforts_with_limits(
    limits: wp.array2d(dtype=wp.float32),
    joint_array: wp.array2d(dtype=wp.float32),
    clipped_joint_array: wp.array2d(dtype=wp.float32),
    env_mask: wp.array2d(dtype=wp.bool),
    joint_mask: wp.array2d(dtype=wp.bool),
):
    env_index, joint_index = wp.tid()
    if env_mask[env_index] and joint_mask[joint_index]:
        clipped_joint_array[env_index, joint_index] = wp.clamp(joint_array[env_index, joint_index], -limits[env_index, joint_index], limits[env_index, joint_index])

@wp.func
def clip_effort_dc_motor(
    saturation_effort: float,
    vel_limit: float,
    effort_limit: float,
    joint_vel: float,
    effort: float,
    
):
    max_effort = saturation_effort * (1.0 - joint_vel)/vel_limit
    min_effort = saturation_effort * (-1.0 - joint_vel)/vel_limit
    max_effort = wp.clamp(max_effort, 0, effort_limit)
    min_effort = wp.clamp(min_effort, -effort_limit, 0)
    return wp.clamp(effort, min_effort, max_effort)

@wp.kernel
def clip_efforts_dc_motor(
    saturation_effort: wp.array2d(dtype=wp.float32),
    effort_limit: wp.array2d(dtype=wp.float32),
    vel_limit: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    joint_array: wp.array2d(dtype=wp.float32),
    clipped_joint_array: wp.array2d(dtype=wp.float32),
    env_mask: wp.array2d(dtype=wp.bool),
    joint_mask: wp.array2d(dtype=wp.bool),
):
    env_index, joint_index = wp.tid()
    if env_mask[env_index] and joint_mask[joint_index]:
        clipped_joint_array[env_index, joint_index] = clip_effort_dc_motor(saturation_effort[env_index, joint_index], vel_limit[env_index, joint_index], effort_limit[env_index, joint_index], joint_vel[env_index, joint_index], joint_array[env_index, joint_index])