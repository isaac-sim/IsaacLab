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