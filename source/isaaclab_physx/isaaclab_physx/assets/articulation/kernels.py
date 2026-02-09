from triton.knobs import Env
import warp as wp

vec13f = wp.types.vector(length=13, dtype=wp.float32)

@wp.func
def get_link_vel_from_root_com_vel_func(
    com_vel: wp.spatial_vectorf,
    link_pose: wp.transformf,
    body_com_pose: wp.transformf,
):
    projected_vel = wp.cross(wp.spatial_bottom(com_vel), wp.quat_rotate(wp.transform_get_rotation(link_pose), -wp.transform_get_translation(body_com_pose)))
    return wp.spatial_vectorf(wp.spatial_top(com_vel) + projected_vel, wp.spatial_bottom(com_vel))

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
    state: vec13f,
):
    # Pose: [pos, quat]
    state[0] = pose[0]
    state[1] = pose[1]
    state[2] = pose[2]
    state[3] = pose[3]
    state[4] = pose[4]
    state[5] = pose[5]
    state[6] = pose[6]
    # Velocity: [lin_vel, ang_vel]
    state[7] = vel[0]
    state[8] = vel[1]
    state[9] = vel[2]
    state[10] = vel[3]
    state[11] = vel[4]
    state[12] = vel[5]

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

@wp.kernel
def get_root_link_vel_from_root_com_vel(
    com_vel: wp.array(dtype=wp.spatial_vectorf),
    link_pose: wp.array(dtype=wp.transformf),
    body_com_pose: wp.array2d(dtype=wp.transformf),
    link_vel: wp.array(dtype=wp.spatial_vectorf),
):
    i = wp.tid()
    link_vel[i] = get_link_vel_from_root_com_vel_func(com_vel[i], link_pose[i], body_com_pose[0, i])


@wp.kernel
def get_root_com_pose_from_root_link_pose(
    link_pose: wp.array(dtype=wp.transformf),
    body_com_pose: wp.array2d(dtype=wp.transformf),
    com_pose: wp.array(dtype=wp.transformf),
):
    i = wp.tid()
    com_pose[i] = get_com_pose_from_link_pose_func(link_pose[i], body_com_pose[0, i])


@wp.kernel
def concat_root_pose_and_vel_to_state(
    pose: wp.array(dtype=wp.transformf),
    vel: wp.array(dtype=wp.spatial_vectorf),
    state: wp.array(dtype=vec13f),
):
    i = wp.tid()
    concat_pose_and_vel_to_state_func(pose[i], vel[i], state[i])

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
    body_com_pose: wp.array2d(dtype=wp.transformf),
    body_com_pose: wp.array2d(dtype=wp.transformf),
):
    i, j = wp.tid()
    body_com_pose[i, j] = get_com_pose_from_link_pose_func(body_link_pose[i, j], body_com_pose[i, j])

@wp.kernel
def concat_body_pose_and_vel_to_state(
    pose: wp.array2d(dtype=wp.transformf),
    vel: wp.array2d(dtype=wp.spatial_vectorf),
    state: wp.array2d(dtype=vec13f),
):
    i, j = wp.tid()
    concat_pose_and_vel_to_state_func(pose[i, j], vel[i, j], state[i, j])

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
    root_link_state_w: Any, # wp.array(dtype=vec13f) or None
    root_state_w: Any, # wp.array(dtype=vec13f) or None
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
    root_com_state_w: Any, # wp.array(dtype=vec13f) or None
    root_link_state_w: Any, # wp.array(dtype=vec13f) or None
    root_state_w: Any, # wp.array(dtype=vec13f) or None
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
    root_state_w: Any, # wp.array(dtype=vec13f) or None
    root_com_state_w: Any, # wp.array(dtype=vec13f) or None
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
    body_acc_w[env_ids[i]] = 0.0

@wp.kernel
def set_root_link_velocity_to_sim(
    data: wp.array(dtype=wp.spatial_vectorf),
    body_com_pose_b: wp.array2d(dtype=wp.transformf),
    link_pose_w: wp.array2d(dtype=wp.transformf),
    env_ids: wp.array(dtype=wp.int32),
    root_link_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    root_com_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    body_acc_w: wp.array2d(dtype=wp.spatial_vectorf),
    root_link_state_w: Any, # wp.array(dtype=vec13f) or None
    root_state_w: Any, # wp.array(dtype=vec13f) or None
    root_com_state_w: Any, # wp.array(dtype=vec13f) or None
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
    body_acc_w[env_ids[i]] = 0.0


@wp.kernel
def write_joint_data_to_buffer(
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