import warp as wp


@wp.kernel
def extract_z_from_pose(pose: wp.array(dtype=wp.transformf), z: wp.array(dtype=wp.float32)):
    env_index = wp.tid()
    z[env_index] = pose[env_index][2]


@wp.kernel
def make_quat_unique_1D(quat: wp.array(dtype=wp.quatf), unique_quat: wp.array(dtype=wp.quatf)):
    env_index = wp.tid()
    if quat[env_index][3] < 0.0:
        unique_quat[env_index] = -quat[env_index]


@wp.kernel
def make_quat_unique_2D(quat: wp.array2d(dtype=wp.quatf), unique_quat: wp.array2d(dtype=wp.quatf)):
    env_index, body_index = wp.tid()
    if quat[env_index, body_index][3] < 0.0:
        unique_quat[env_index, body_index] = -quat[env_index, body_index]


@wp.kernel
def get_body_world_pose_flattened(
    body_pose_w: wp.array2d(dtype=wp.transformf),
    env_origins: wp.array(dtype=wp.vec3f),
    body_pose: wp.array2d(dtype=wp.transformf),
    body_indices: wp.array(dtype=wp.int32),
):
    env_tid, body_tid = wp.tid()
    wp.transform_set_translation(body_pose[env_tid, body_tid], wp.transform_get_translation(body_pose_w[env_tid, body_indices[body_tid]]) - env_origins[env_tid])


@wp.kernel
def project_gravity_to_body(
    body_pose_w: wp.array2d(dtype=wp.transformf),
    gravity_dir: wp.vec3f,
    projected_gravity_b: wp.array2d(dtype=wp.vec3f),
    body_indices: wp.array(dtype=wp.int32),
):
    env_tid, body_tid = wp.tid()
    projected_gravity_b[env_tid, body_tid] = wp.quat_rotate_inv(wp.transform_get_rotation(body_pose_w[env_tid, body_indices[body_tid]]), gravity_dir)


@wp.kernel
def get_joint_data_by_indices(
    joint_data: wp.array2d(dtype=wp.float32),
    joint_indices: wp.array(dtype=wp.int32),
    joint_data_out: wp.array2d(dtype=wp.float32),
):
    env_tid, joint_tid = wp.tid()
    joint_data_out[env_tid, joint_tid] = joint_data[env_tid, joint_indices[joint_tid]]


@wp.kernel
def get_joint_data_rel_by_indices(
    joint_data: wp.array2d(dtype=wp.float32),
    default_joint_data: wp.array2d(dtype=wp.float32),
    joint_indices: wp.array(dtype=wp.int32),
    joint_data_rel_out: wp.array2d(dtype=wp.float32),
):
    env_tid, joint_tid = wp.tid()
    joint_data_rel_out[env_tid, joint_tid] = joint_data[env_tid, joint_indices[joint_tid]] - default_joint_data[env_tid, joint_indices[joint_tid]]