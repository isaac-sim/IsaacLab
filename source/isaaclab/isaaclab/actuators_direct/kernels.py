import warp as wp

@wp.kernel
def compute_implicit_actuator(
    stiffness: wp.array2d(dtype=wp.float32),
    damping: wp.array2d(dtype=wp.float32),
    effort: wp.array2d(dtype=wp.float32),
    effort_limit: wp.array2d(dtype=wp.float32),
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    control_mode: wp.array2d(dtype=wp.int32),
    joint_targets: wp.array2d(dtype=wp.float32),
    computed_effort: wp.array2d(dtype=wp.float32),
    applied_effort: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    joint_mask: wp.array(dtype=wp.bool),
) -> None:
    bdx, jdx = wp.tid()
    if env_mask[bdx] and joint_mask[jdx]:
        # No control
        if control_mode[bdx, jdx] == 0:
            computed_effort[bdx, jdx] = 0
        # Position control
        elif control_mode[bdx, jdx] == 1:
            computed_effort[bdx, jdx] = stiffness[bdx, jdx] * (joint_targets[bdx, jdx] - joint_pos[bdx, jdx]) + damping[bdx, jdx] * (0 - joint_vel[bdx, jdx]) + effort[bdx, jdx]
        # Velocity control
        elif control_mode[bdx, jdx] == 2:
            computed_effort[bdx, jdx] = stiffness[bdx, jdx] * (0 - joint_pos[bdx, jdx]) + damping[bdx, jdx] * (joint_targets[bdx, jdx] - joint_vel[bdx, jdx]) + effort[bdx, jdx]
        # Clip the computed effort
        applied_effort[bdx, jdx] = wp.clamp(computed_effort[bdx, jdx], -effort_limit[bdx, jdx], effort_limit[bdx, jdx])