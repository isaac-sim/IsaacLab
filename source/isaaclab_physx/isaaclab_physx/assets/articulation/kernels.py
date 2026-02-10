from typing import Any
import warp as wp

vec13f = wp.types.vector(length=13, dtype=wp.float32)

@wp.func
def get_link_vel_from_root_com_vel_func(
    com_vel: wp.spatial_vectorf,
    link_pose: wp.transformf,
    body_com_pose: wp.transformf,
):
    projected_vel = wp.cross(wp.spatial_bottom(com_vel), wp.quat_rotate(wp.transform_get_rotation(link_pose), -wp.transform_get_translation(body_com_pose)))
    return wp.spatial_vector(wp.spatial_top(com_vel) + projected_vel, wp.spatial_bottom(com_vel))

@wp.func
def get_com_pose_from_link_pose_func(
    link_pose: wp.transformf,
    body_com_pose: wp.transformf,
):
    return link_pose * body_com_pose

@wp.func
def concat_pose_and_vel_to_state_func(
    pose: wp.transformf,
    vel: wp.spatial_vectorf,
) -> vec13f:
    # Pose: [pos, quat]
    return vec13f(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6], vel[0], vel[1], vel[2], vel[3], vel[4], vel[5])

@wp.func
def compute_heading_w_func(
    forward_vec: wp.vec3f,
    quat: wp.quatf,
):
    forward_w =  wp.quat_rotate(quat, forward_vec)
    return wp.atan2(forward_w[1], forward_w[0])

@wp.func
def set_state_transforms_func(
    state: vec13f,
    transform: wp.transformf,
):
    state[0] = transform[0]
    state[1] = transform[1]
    state[2] = transform[2]
    state[3] = transform[3]
    state[4] = transform[4]
    state[5] = transform[5]
    state[6] = transform[6]

@wp.func
def set_state_velocities_func(
    state: vec13f,
    velocity: wp.spatial_vectorf,
):
    state[7] = velocity[0]
    state[8] = velocity[1]
    state[9] = velocity[2]
    state[10] = velocity[3]
    state[11] = velocity[4]
    state[12] = velocity[5]

@wp.func
def get_link_velocity_in_com_frame_func(
    link_velocity_w: wp.spatial_vectorf,
    link_pose_w: wp.transformf,
    body_com_pose_b: wp.transformf,
):
    return wp.spatial_vector(
        wp.spatial_top(link_velocity_w) + wp.cross(wp.spatial_bottom(link_velocity_w),
        wp.quat_rotate(wp.transform_get_rotation(link_pose_w), wp.transform_get_translation(body_com_pose_b))),
        wp.spatial_bottom(link_velocity_w)
    )

@wp.func
def get_com_pose_in_link_frame_func(
    com_pose_w: wp.transformf,
    com_pose_b: wp.transformf,
):
    T2 = wp.transform(
        wp.quat_rotate(wp.quat_inverse(wp.transform_get_rotation(com_pose_b)),
        - wp.transform_get_translation(com_pose_b)), wp.quat_inverse(wp.transform_get_rotation(com_pose_b))
    )
    link_pose_w = com_pose_w * T2
    return link_pose_w

@wp.func
def compute_soft_joint_pos_limits_func(
    joint_pos_limits: wp.vec2f,
    soft_limit_factor: wp.float32,
):
    joint_pos_mean = (joint_pos_limits[0] + joint_pos_limits[1]) / 2.0
    joint_pos_range = joint_pos_limits[1] - joint_pos_limits[0]
    return wp.vec2f(
        joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor,
        joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor
    )

@wp.kernel
def get_root_link_vel_from_root_com_vel(
    com_vel: wp.array(dtype=wp.spatial_vectorf),
    link_pose: wp.array(dtype=wp.transformf),
    body_com_pose_b: wp.array2d(dtype=wp.transformf),
    link_vel: wp.array(dtype=wp.spatial_vectorf),
):
    i = wp.tid()
    link_vel[i] = get_link_vel_from_root_com_vel_func(com_vel[i], link_pose[i], body_com_pose_b[0, i])


@wp.kernel
def get_root_com_pose_from_root_link_pose(
    link_pose: wp.array(dtype=wp.transformf),
    body_com_pose_b: wp.array2d(dtype=wp.transformf),
    com_pose_w: wp.array(dtype=wp.transformf),
):
    i = wp.tid()
    com_pose_w[i] = get_com_pose_from_link_pose_func(link_pose[i], body_com_pose_b[0, i])


@wp.kernel
def concat_root_pose_and_vel_to_state(
    pose: wp.array(dtype=wp.transformf),
    vel: wp.array(dtype=wp.spatial_vectorf),
    state: wp.array(dtype=vec13f),
):
    i = wp.tid()
    state[i] = concat_pose_and_vel_to_state_func(pose[i], vel[i])

@wp.kernel
def get_body_link_vel_from_body_com_vel(
    body_com_vel: wp.array2d(dtype=wp.spatial_vectorf),
    body_link_pose: wp.array2d(dtype=wp.transformf),
    body_com_pose: wp.array2d(dtype=wp.transformf),
    body_link_vel: wp.array2d(dtype=wp.spatial_vectorf),
):
    i, j = wp.tid()
    body_link_vel[i, j] = get_link_vel_from_root_com_vel_func(body_com_vel[i, j], body_link_pose[i, j], body_com_pose[i, j])


@wp.kernel
def get_body_com_pose_from_body_link_pose(
    body_link_pose: wp.array2d(dtype=wp.transformf),
    body_com_pose_b: wp.array2d(dtype=wp.transformf),
    body_com_pose_w: wp.array2d(dtype=wp.transformf),
):
    i, j = wp.tid()
    body_com_pose_w[i, j] = get_com_pose_from_link_pose_func(body_link_pose[i, j], body_com_pose_b[i, j])

@wp.kernel
def concat_body_pose_and_vel_to_state(
    pose: wp.array2d(dtype=wp.transformf),
    vel: wp.array2d(dtype=wp.spatial_vectorf),
    state: wp.array2d(dtype=vec13f),
):
    i, j = wp.tid()
    state[i, j] = concat_pose_and_vel_to_state_func(pose[i, j], vel[i, j])

@wp.kernel
def get_joint_acc_from_joint_vel(
    joint_vel: wp.array2d(dtype=wp.float32),
    prev_joint_vel: wp.array2d(dtype=wp.float32),
    joint_acc: wp.array2d(dtype=wp.float32),
    dt: wp.float32,
):
    i, j = wp.tid()
    joint_acc[i, j] = (joint_vel[i, j] - prev_joint_vel[i, j]) / dt
    prev_joint_vel[i, j] = joint_vel[i, j]

@wp.kernel
def quat_apply_inverse_1D_kernel(
    gravity: wp.array(dtype=wp.vec3f),
    quat: wp.array(dtype=wp.quatf),
    projected_gravity: wp.array(dtype=wp.vec3f),
):
    i = wp.tid()
    projected_gravity[i] = wp.quat_rotate_inv(quat[i], gravity[i])

@wp.kernel
def root_heading_w(
    forward_vec: wp.array(dtype=wp.vec3f),
    quat: wp.array(dtype=wp.quatf),
    heading_w: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    heading_w[i] = compute_heading_w_func(forward_vec[i], quat[i])

@wp.kernel
def set_root_link_pose_to_sim(
    data: wp.array(dtype=wp.transformf),
    env_ids: wp.array(dtype=wp.int32),
    root_link_pose_w: wp.array(dtype=wp.transformf),
    root_link_state_w: wp.array(dtype=vec13f),
    root_state_w: wp.array(dtype=vec13f),
    from_mask: bool,
):
    # If from mask, then we get complete data. Otherwise, we get partial data.
    i = wp.tid()
    if from_mask:
        root_link_pose_w[env_ids[i]] = data[env_ids[i]]
        if root_link_state_w:
            set_state_transforms_func(root_link_state_w[env_ids[i]], data[env_ids[i]])
        if root_state_w:
            set_state_transforms_func(root_state_w[env_ids[i]], data[env_ids[i]])
    else:
        root_link_pose_w[env_ids[i]] = data[i]
        if root_link_state_w:
            set_state_transforms_func(root_link_state_w[env_ids[i]], data[i])
        if root_state_w:
            set_state_transforms_func(root_state_w[env_ids[i]], data[i])

@wp.kernel
def set_root_com_pose_to_sim(
    data: wp.array(dtype=wp.transformf),
    body_com_pose_b: wp.array2d(dtype=wp.transformf),
    env_ids: wp.array(dtype=wp.int32),
    root_com_pose_w: wp.array(dtype=wp.transformf),
    root_link_pose_w: wp.array(dtype=wp.transformf),
    root_com_state_w: wp.array(dtype=vec13f),
    root_link_state_w: wp.array(dtype=vec13f),
    root_state_w: wp.array(dtype=vec13f),
    from_mask: bool,
):
    i = wp.tid()
    # If from mask, then we get complete data. Otherwise, we get partial data.
    if from_mask:
        root_com_pose_w[env_ids[i]] = data[env_ids[i]]
        if root_com_state_w:
            set_state_transforms_func(root_com_state_w[env_ids[i]], data[env_ids[i]])
    else:
        root_com_pose_w[env_ids[i]] = data[i]
        if root_com_state_w:
            set_state_transforms_func(root_com_state_w[env_ids[i]], data[i])
    # Get the com pose in the link frame
    root_link_pose_w[env_ids[i]] = get_com_pose_in_link_frame_func(
        root_com_pose_w[env_ids[i]], body_com_pose_b[env_ids[i], 0]
    )
    if root_link_state_w:
        set_state_transforms_func(root_link_state_w[env_ids[i]], root_link_pose_w[env_ids[i]])
    if root_state_w:
        set_state_transforms_func(root_state_w[env_ids[i]], root_link_pose_w[env_ids[i]])

@wp.kernel
def set_root_com_velocity_to_sim(
    data: wp.array(dtype=wp.spatial_vectorf),
    env_ids: wp.array(dtype=wp.int32),
    root_com_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    body_acc_w: wp.array2d(dtype=wp.spatial_vectorf),
    root_state_w: wp.array(dtype=vec13f),
    root_com_state_w: wp.array(dtype=vec13f),
    num_bodies: wp.int32,
    from_mask: bool,
):
    i = wp.tid()
    # If from mask, then we get complete data. Otherwise, we get partial data.
    if from_mask:
        root_com_velocity_w[env_ids[i]] = data[env_ids[i]]
        if root_state_w:
            set_state_velocities_func(root_state_w[env_ids[i]], data[env_ids[i]])
        if root_com_state_w:
            set_state_velocities_func(root_com_state_w[env_ids[i]], data[env_ids[i]])
    else:
        root_com_velocity_w[env_ids[i]] = data[i]
        if root_state_w:
            set_state_velocities_func(root_state_w[env_ids[i]], data[i])
        if root_com_state_w:
            set_state_velocities_func(root_com_state_w[env_ids[i]], data[i])
    # Make the acceleration zero to prevent reporting old values
    for j in range(num_bodies):
        body_acc_w[env_ids[i], j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

@wp.kernel
def set_root_link_velocity_to_sim(
    data: wp.array(dtype=wp.spatial_vectorf),
    body_com_pose_b: wp.array2d(dtype=wp.transformf),
    link_pose_w: wp.array(dtype=wp.transformf),
    env_ids: wp.array(dtype=wp.int32),
    root_link_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    root_com_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    body_acc_w: wp.array2d(dtype=wp.spatial_vectorf),
    root_link_state_w: wp.array(dtype=vec13f),
    root_state_w: wp.array(dtype=vec13f),
    root_com_state_w: wp.array(dtype=vec13f),
    num_bodies: wp.int32,
    from_mask: bool,
):
    # If from mask, then we get complete data. Otherwise, we get partial data.
    i = wp.tid()
    if from_mask:
        root_link_velocity_w[env_ids[i]] = data[env_ids[i]]
        if root_link_state_w:
            set_state_velocities_func(root_link_state_w[env_ids[i]], data[env_ids[i]])
    else:
        root_link_velocity_w[env_ids[i]] = data[i]
        if root_link_state_w:
            set_state_velocities_func(root_link_state_w[env_ids[i]], data[i])
    # Get the link velocity in the com frame
    root_com_velocity_w[env_ids[i]] = get_link_velocity_in_com_frame_func(
        root_link_velocity_w[env_ids[i]], link_pose_w[env_ids[i]], body_com_pose_b[env_ids[i], 0]
    )
    if root_com_state_w:
        set_state_velocities_func(root_com_state_w[env_ids[i]], root_com_velocity_w[env_ids[i]])
    if root_state_w:
        set_state_velocities_func(root_state_w[env_ids[i]], root_com_velocity_w[env_ids[i]])
    # Make the acceleration zero to prevent reporting old values
    for j in range(num_bodies):
        body_acc_w[env_ids[i], j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


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
def write_2d_data_to_buffer_with_indices(
    in_data: wp.array2d(dtype=wp.float32),
    out_data: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=wp.int32),
    joint_ids: wp.array(dtype=wp.int32),
    from_mask: bool,
):
    i,j = wp.tid()
    if from_mask:
        out_data[env_ids[i], joint_ids[j]] = in_data[env_ids[i], joint_ids[j]]
    else:
        out_data[env_ids[i], joint_ids[j]] = in_data[i, j]

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
def write_body_inertia_to_buffer(
    in_data: wp.array3d(dtype=wp.float32),
    out_data: wp.array3d(dtype=wp.float32),
    env_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
    from_mask: bool,
):
    i, j = wp.tid()
    if from_mask:
        for k in range(9):
            out_data[env_ids[i], body_ids[j], k] = in_data[env_ids[i], body_ids[j], k]
    else:
        for k in range(9):
            out_data[env_ids[i], body_ids[j], k] = in_data[i, j, k]


@wp.kernel
def write_body_com_pose_to_buffer(
    in_data: wp.array2d(dtype=wp.transformf),
    out_data: wp.array2d(dtype=wp.transformf),
    env_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
    from_mask: bool,
):
    i, j = wp.tid()
    if from_mask:
        out_data[env_ids[i], body_ids[j]] = in_data[env_ids[i], body_ids[j]]
    else:
        out_data[env_ids[i], body_ids[j]] = in_data[i, j]


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

# TODO:
# Make it like so: + Do we want to leverage warp outputs.
# source_1 
# target_1
# source_2
# target_2
# ...
# source_n
# target_n
# And cap n at 10. Will ignore Nones.
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