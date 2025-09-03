import warp as wp

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

@wp.kernel
def generate_mask_from_ids(
    mask: wp.array(dtype=wp.bool),
    ids: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    mask[ids[index]] = True

@wp.kernel
def populate_empty_array(
    input_array: wp.array(dtype=wp.float32),
    output_array: wp.array2d(dtype=wp.float32),
    indices: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    output_array[indices[index]] = input_array[index]


@wp.kernel
def clip_joint_array_with_limits_masked(
    lower_limits: wp.array(dtype=wp.float32),
    upper_limits: wp.array(dtype=wp.float32),
    joint_array: wp.array(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    joint_mask: wp.array(dtype=wp.bool),
):
    body_index, joint_index = wp.tid()
    if env_mask[body_index] and joint_mask[joint_index]:
        joint_array[body_index, joint_index] = wp.clamp(joint_array[body_index, joint_index], lower_limits[body_index, joint_index], upper_limits[body_index, joint_index])

@wp.kernel
def clip_joint_array_with_limits(
    lower_limits: wp.array(dtype=wp.float32),
    upper_limits: wp.array(dtype=wp.float32),
    joint_array: wp.array(dtype=wp.float32),
):
    index = wp.tid()
    joint_array[index] = wp.clamp(joint_array[index], lower_limits[index], upper_limits[index])