import warp as wp

@wp.kernel
def update_wrench_array(
    new_value: wp.array2d(dtype=wp.spatial_vectorf),
    wrench: wp.array2d(dtype=wp.spatial_vectorf),
    env_ids: wp.array(dtype=wp.bool),
    body_ids: wp.array(dtype=wp.bool),
):
    env_index, body_index = wp.tid()
    if env_ids[env_index] and body_ids[body_index]:
        wrench[env_index, body_index] = new_value[env_index, body_index]

@wp.kernel
def update_wrench_array_with_value(
    value: wp.spatial_vectorf,
    wrench: wp.array2d(dtype=wp.spatial_vectorf),
    env_ids: wp.array(dtype=wp.bool),
    body_ids: wp.array(dtype=wp.bool),
):
    env_index, body_index = wp.tid()
    if env_ids[env_index] and body_ids[body_index]:
        wrench[env_index, body_index] = value

@wp.func
def update_wrench_with_force(
    force: wp.vec3f,
) -> wp.spatial_vectorf:
    return wp.spatial_vectorf(0.0, 0.0, 0.0, force[0], force[1], force[2])

@wp.func
def update_wrench_with_torque(
    torque: wp.vec3f,
) -> wp.spatial_vectorf:
    return wp.spatial_vectorf(torque[0], torque[1], torque[2], 0.0, 0.0, 0.0)

@wp.kernel
def update_wrench_array_with_force(
    forces: wp.array2d(dtype=wp.vec3f),
    wrench: wp.array2d(dtype=wp.spatial_vectorf),
    env_ids: wp.array(dtype=wp.bool),
    body_ids: wp.array(dtype=wp.bool),
):
    env_index, body_index = wp.tid()
    if env_ids[env_index] and body_ids[body_index]:
        wrench[env_index, body_index] = update_wrench_with_force(forces[env_index, body_index])

@wp.kernel
def update_wrench_array_with_torque(
    torques: wp.array2d(dtype=wp.vec3f),
    wrench: wp.array2d(dtype=wp.spatial_vectorf),
    env_ids: wp.array(dtype=wp.bool),
    body_ids: wp.array(dtype=wp.bool),
):
    env_index, body_index = wp.tid()
    if env_ids[env_index] and body_ids[body_index]:
        wrench[env_index, body_index] = update_wrench_with_torque(torques[env_index, body_index])

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
    output_array: wp.array(dtype=wp.float32),
    indices: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    output_array[indices[index]] = input_array[index]