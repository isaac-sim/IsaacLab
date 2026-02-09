import warp as wp

vec13f = wp.types.vector(length=13, dtype=wp.float32)

@wp.func
def get_root_link_vel_from_root_com_vel_func(
    com_vel: wp.spatial_vectorf,
    link_pose: wp.transformf,
    body_com_pose: wp.transformf,
):
    projected_vel = wp.cross(wp.spatial_bottom(com_vel), wp.quat_rotate(wp.transform_get_rotation(link_pose), -wp.transform_get_translation(body_com_pose)))
    return wp.spatial_vectorf(wp.spatial_top(com_vel) + projected_vel, wp.spatial_bottom(com_vel))

@wp.kernel
def get_root_link_vel_from_root_com_vel(
    com_vel: wp.array(dtype=wp.spatial_vectorf),
    link_pose: wp.array(dtype=wp.transformf),
    body_com_pose: wp.array2d(dtype=wp.transformf),
    link_vel: wp.array(dtype=wp.spatial_vectorf),
):
    i = wp.tid()
    link_vel[i] = get_root_link_vel_from_root_com_vel_func(com_vel[i], link_pose[i], body_com_pose[0, i])


@wp.func
def get_root_com_pose_from_root_link_pose_func(
    link_pose: wp.transformf,
    body_com_pose: wp.transformf,
):
    return link_pose * body_com_pose

@wp.kernel
def get_root_com_pose_from_root_link_pose(
    link_pose: wp.array(dtype=wp.transformf),
    body_com_pose: wp.array2d(dtype=wp.transformf),
    com_pose: wp.array(dtype=wp.transformf),
):
    i = wp.tid()
    com_pose[i] = get_root_com_pose_from_root_link_pose_func(link_pose[i], body_com_pose[0, i])

@wp.kernel
def concat_root_pose_and_vel_to_state_func(
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

@wp.kernel
def concat_root_pose_and_vel_to_state(
    pose: wp.array(dtype=wp.transformf),
    vel: wp.array(dtype=wp.spatial_vectorf),
    state: wp.array(dtype=vec13f),
):
    i = wp.tid()
    concat_root_pose_and_vel_to_state_func(pose[i], vel[i], state[i])