import warp as wp

@wp.kernel
def get_position(pose: wp.array(dtype=wp.transformf), position: wp.array(dtype=wp.vec3f)):
    index = wp.tid()
    position[index] = pose.p[index]


@wp.kernel
def get_quat(pose: wp.array(dtype=wp.transformf), quat: wp.array(dtype=wp.quatf)):
    index = wp.tid()
    quat[index] = pose.q[index]



@wp.func
def build_pose_from_position_and_quat(
    position: wp.vec3f,
    quat: wp.quatf,
) -> wp.transformf:
    return wp.transformf(position, quat)


@wp.kernel
def generate_body_com_pose_b(
    body_com_position_b: wp.array(dtype=wp.vec3f),
    body_com_pose_b: wp.array(dtype=wp.transformf),
):
    index = wp.tid()
    body_com_pose_b[index] = build_pose_from_position_and_quat(body_com_position_b[index], wp.quatf(0.0, 0.0, 0.0, 1.0))


@wp.func
def combine_transforms(p1: wp.vec3f, q1: wp.quatf, p2: wp.vec3f, q2: wp.quatf) -> wp.transformf:
    return wp.transformf(
        p1 + wp.quat_rotate(q1, p2),
        q1 * q2
    )

@wp.kernel
def combine_frame_transforms_partial(
    pose_1: wp.array(dtype=wp.transformf),
    position_2: wp.array(dtype=wp.vec3f),
    resulting_pose: wp.array(dtype=wp.transformf)
):
    index = wp.tid()
    resulting_pose[index] = combine_transforms(
        wp.transform_get_translation(pose_1[index]),
        wp.transform_get_rotation(pose_1[index]),
        position_2[index],
        wp.quatf(0.0, 0.0, 0.0, 1.0)
    )


@wp.kernel
def combine_frame_transforms_partial_batch(
    pose_1: wp.array2d(dtype=wp.transformf),
    position_2: wp.array2d(dtype=wp.vec3f),
    resulting_pose: wp.array2d(dtype=wp.transformf)
):
    env_idx, body_idx = wp.tid()
    resulting_pose[env_idx, body_idx] = combine_transforms(
        wp.transform_get_translation(pose_1[env_idx, body_idx]),
        wp.transform_get_rotation(pose_1[env_idx, body_idx]),
        position_2[env_idx, body_idx],
        wp.quatf(0.0, 0.0, 0.0, 1.0)
    )


@wp.kernel
def combine_frame_transforms(
    pose_1: wp.array(dtype=wp.transformf),
    pose_2: wp.array(dtype=wp.transformf),
    resulting_pose: wp.array(dtype=wp.transformf)
):
    index = wp.tid()
    resulting_pose[index] = combine_transforms(
        wp.transform_get_translation(pose_1[index]),
        wp.transform_get_rotation(pose_1[index]),
        wp.transform_get_translation(pose_2[index]),
        wp.transform_get_rotation(pose_2[index])
    )


@wp.kernel
def combine_frame_transforms_batch(
    pose_1: wp.array2d(dtype=wp.transformf),
    pose_2: wp.array2d(dtype=wp.transformf),
    resulting_pose: wp.array2d(dtype=wp.transformf)
):
    env_idx, body_idx = wp.tid()
    resulting_pose[env_idx, body_idx] = combine_transforms(
        wp.transform_get_translation(pose_1[env_idx, body_idx]),
        wp.transform_get_rotation(pose_1[env_idx, body_idx]),
        wp.transform_get_translation(pose_2[env_idx, body_idx]),
        wp.transform_get_rotation(pose_2[env_idx, body_idx])
    )


@wp.kernel
def derive_body_acceleration_from_velocity(
    velocity: wp.array2d(dtype=wp.spatial_vectorf),
    previous_velocity: wp.array2d(dtype=wp.spatial_vectorf),
    dt: float,
    acceleration: wp.array2d(dtype=wp.spatial_vectorf),
):
    env_idx, body_idx = wp.tid()
    acceleration[env_idx, body_idx] = (velocity[env_idx, body_idx] - previous_velocity[env_idx, body_idx]) / dt


@wp.kernel
def derive_joint_acceleration_from_velocity(
    joint_velocity: wp.array2d(dtype=wp.float32),
    previous_joint_velocity: wp.array2d(dtype=wp.float32),
    dt: float,
    joint_acceleration: wp.array2d(dtype=wp.float32),
):
    index = wp.tid()
    # compute acceleration
    joint_acceleration[index] = (joint_velocity[index] - previous_joint_velocity[index]) / dt

    # update previous velocity
    previous_joint_velocity[index] = joint_velocity[index]


@wp.kernel
def project_vec_from_quat_single(
    vec: wp.vec3f,
    quat: wp.array(dtype=wp.quatf),
    resulting_vec: wp.array(dtype=wp.vec3f)
):
    index = wp.tid()
    resulting_vec[index] = wp.quat_rotate(quat[index], vec)


@wp.func
def project_velocity_to_frame(
    velocity: wp.spatial_vectorf,
    pose: wp.transformf,
) -> wp.spatial_vectorf:
    return wp.spatial_vectorf(
        wp.quat_rotate_inv(pose.q, wp.spatial_top(velocity)),
        wp.quat_rotate_inv(pose.q, wp.spatial_bottom(velocity))
    )

@wp.kernel
def project_velocities_to_frame(
    velocity: wp.array(dtype=wp.spatial_vectorf),
    pose: wp.array(dtype=wp.transformf),
    resulting_velocity: wp.array(dtype=wp.spatial_vectorf)
):
    index = wp.tid()
    resulting_velocity[index] = project_velocity_to_frame(velocity[index], pose[index])


@wp.func
def heading_vec_b(quat: wp.quatf, vec: wp.vec3f) -> float:
    quat_rot: wp.quatf = wp.quat_rotate(quat, vec)
    return wp.atan2(quat_rot[0], quat_rot[3])


@wp.kernel
def compute_heading(
    forward_vec_b: wp.vec3f,
    quat_w: wp.array(dtype=wp.quatf),
    heading: wp.array(dtype=wp.float32)
):
    index = wp.tid()
    heading[index] = heading_vec_b(quat_w[index], forward_vec_b)


@wp.kernel
def update_transforms_array(
    pose: wp.array(dtype=wp.transformf),
    indices: wp.array(dtype=wp.int32),
    new_pose: wp.array(dtype=wp.transformf),
):
    index = wp.tid()
    pose[indices[index]] = new_pose[index]

@wp.kernel
def update_transforms_array_with_value(
    value: wp.transformf,
    pose: wp.array(dtype=wp.transformf),
    indices: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    pose[indices[index]] = value

@wp.kernel
def update_spatial_vector_array(
    velocity: wp.array(dtype=wp.spatial_vectorf),
    indices: wp.array(dtype=wp.int32),
    new_velocity: wp.array(dtype=wp.spatial_vectorf),
):
    index = wp.tid()
    velocity[indices[index]] = new_velocity[index]

@wp.kernel
def update_spatial_vector_array_with_value(
    value: wp.spatial_vectorf,
    velocity: wp.array(dtype=wp.spatial_vectorf),
    indices: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    velocity[indices[index]] = value

@wp.kernel
def transform_CoM_pose_to_link_frame(
    com_pose_w: wp.array(dtype=wp.transformf),
    com_pose_link_frame: wp.array(dtype=wp.transformf),
    link_pose_w: wp.array(dtype=wp.transformf),
    indices: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    link_pose_w[indices[index]] = combine_transforms(
        wp.transform_get_translation(com_pose_w[index]),
        wp.transform_get_rotation(com_pose_w[index]),
        wp.transform_get_translation(com_pose_link_frame[indices[index]]),
        wp.quatf(0.0, 0.0, 0.0, 1.0)
    )

@wp.kernel
def update_wrench_array(
    new_value: wp.array2d(dtype=wp.spatial_vectorf),
    wrench: wp.array2d(dtype=wp.spatial_vectorf),
    env_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
):
    env_index, body_index = wp.tid()
    wrench[env_ids[env_index], body_ids[body_index]] = new_value[env_index, body_index]

@wp.kernel
def update_wrench_array_with_value(
    value: wp.spatial_vectorf,
    wrench: wp.array2d(dtype=wp.spatial_vectorf),
    env_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
):
    env_index, body_index = wp.tid()
    wrench[env_ids[env_index], body_ids[body_index]] = value

@wp.func
def update_wrench_with_force(
    force: wp.vec3f,
) -> wp.spatial_vectorf:
    return wp.spatial_vectorf(wp.vec3f(0.0, 0.0, 0.0), force)

@wp.func
def update_wrench_with_torque(
    torque: wp.vec3f,
) -> wp.spatial_vectorf:
    return wp.spatial_vectorf(torque, wp.vec3f(0.0, 0.0, 0.0))

@wp.kernel
def update_wrench_array_with_force(
    forces: wp.array2d(dtype=wp.vec3f),
    wrench: wp.array2d(dtype=wp.spatial_vectorf),
    env_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
):
    env_index, body_index = wp.tid()
    wrench[env_ids[env_index], body_ids[body_index]] = update_wrench_with_force(forces[env_index, body_index])

@wp.kernel
def update_wrench_array_with_torque(
    torques: wp.array2d(dtype=wp.vec3f),
    wrench: wp.array2d(dtype=wp.spatial_vectorf),
    env_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
):
    env_index, body_index = wp.tid()
    wrench[env_ids[env_index], body_ids[body_index]] = update_wrench_with_torque(torques[env_index, body_index])

@wp.func
def split_state_to_pose(
    state: wp.array(dtype=wp.float32),
) -> wp.transformf:
    return wp.transformf(
        wp.vec3f(state[0], state[1], state[2]),
        wp.quatf(state[3], state[4], state[5], state[6])
    )

@wp.func
def split_state_to_velocity(
    state: wp.array(dtype=wp.float32),
) -> wp.spatial_vectorf:
    return wp.spatial_vectorf(
        wp.vec3f(state[7], state[8], state[9]),
        wp.vec3f(state[10], state[11], state[12])
    )

@wp.kernel
def split_root_state(
    root_state: wp.array2d(dtype=wp.float32),
    root_pose:wp.array2d(dtype=wp.transformf),
    root_velocity:wp.array2d(dtype=wp.spatial_vectorf),
    env_indices: wp.array(dtype=wp.int32),
):
    env_index = env_indices[wp.tid()]
    root_pose[env_indices[env_index]] = split_state_to_pose(root_state[env_index])
    root_velocity[env_indices[env_index]] = split_state_to_velocity(root_state[env_index])

