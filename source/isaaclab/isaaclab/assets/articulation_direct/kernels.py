from sympy import Q
import warp as wp

@wp.func
def cast_pose_to_pos(pose: wp.array(dtype=wp.vec3f)) -> wp.vec3f:
    return wp.vec3f(pose[0], pose[1], pose[2])

@wp.func
def cast_pose_to_quat(pose: wp.array(dtype=wp.float32)) -> wp.quatf:
    return wp.quatf(pose[3], pose[4], pose[5], pose[6])

@wp.func
def cast_vel_to_linear(vel: wp.array(dtype=wp.spatial_vectorf)) -> wp.vec3f:
    return wp.vec3f(vel[3], vel[4], vel[5])

@wp.func
def cast_vel_to_angular(vel: wp.array(dtype=wp.spatial_vectorf)) -> wp.vec3f:
    return wp.vec3f(vel[0], vel[1], vel[2])

@wp.kernel
def split_root_pose(pose: wp.array2d(dtype=wp.float32), position: wp.array(dtype=wp.vec3f), quat: wp.array(dtype=wp.quatf)):
    index = wp.tid()
    position[index] = cast_pose_to_pos(pose[index])
    quat[index] = cast_pose_to_quat(pose[index])

@wp.kernel
def split_root_vel(combined_velocity: wp.array(dtype=wp.spatial_vectorf), linear_velocity: wp.array(dtype=wp.vec3f), angular_velocity: wp.array(dtype=wp.vec3f)):
    index = wp.tid()
    linear_velocity[index] = cast_vel_to_linear(combined_velocity[index])
    angular_velocity[index] = cast_vel_to_angular(combined_velocity[index])

@wp.kernel
def project_linear_velocity_to_link_frame(
    linear_velocity: wp.array(dtype=wp.vec3f),
    position: wp.array(dtype=wp.vec3f),
    quat: wp.array(dtype=wp.quatf),
    transformed_velocity: wp.array(dtype=wp.vec3f)
):
    index = wp.tid()
    transformed_velocity[index] = wp.cross(linear_velocity[index], wp.quat_rotate(quat[index], -position[index]))

@wp.kernel
def split_links_pose(pose: wp.array3d(dtype=wp.float32), position: wp.array2d(dtype=wp.vec3f), quat: wp.array2d(dtype=wp.quatf)):
    env_idx, body_idx = wp.tid()
    position[env_idx, body_idx] = cast_pose_to_pos(pose[env_idx, body_idx])
    quat[env_idx, body_idx] = cast_pose_to_quat(pose[env_idx, body_idx])

@wp.kernel
def split_links_vel(vel: wp.array3d(dtype=wp.spatial_vectorf), linear_velocity: wp.array2d(dtype=wp.vec3f), angular_velocity: wp.array2d(dtype=wp.vec3f)):
    env_idx, body_idx = wp.tid()
    linear_velocity[env_idx, body_idx] = cast_vel_to_linear(vel[env_idx, body_idx])
    angular_velocity[env_idx, body_idx] = cast_vel_to_angular(vel[env_idx, body_idx])


@wp.kernel
def project_linear_velocity_to_links_frame(
    linear_velocity: wp.array2d(dtype=wp.vec3f),
    position: wp.array2d(dtype=wp.vec3f),
    quat: wp.array2d(dtype=wp.quatf),
    transformed_velocity: wp.array2d(dtype=wp.vec3f)
):
    env_idx, body_idx = wp.tid()
    transformed_velocity[env_idx, body_idx] = wp.cross(linear_velocity[env_idx, body_idx], wp.quat_rotate(quat[env_idx, body_idx], -position[env_idx, body_idx]))

@wp.kernel
def combine_frame_transforms(
    position_1: wp.array(dtype=wp.vec3f),
    quat_1: wp.array(dtype=wp.quatf),
    position_2: wp.array(dtype=wp.vec3f),
    quat_2: wp.array(dtype=wp.quatf),
    resulting_position: wp.array(dtype=wp.vec3f),
    resulting_quat: wp.array(dtype=wp.quatf)
):
    index = wp.tid()
    resulting_position[index] = position_1[index] + wp.quat_rotate(quat_1[index], position_2[index])
    if quat_2.shape[0] > 0:
        resulting_quat[index] = quat_1[index] * quat_2[index]
    else:
        resulting_quat[index] = quat_1[index]

@wp.kernel
def combine_frame_transforms_batch(
    position_1: wp.array2d(dtype=wp.vec3f),
    quat_1: wp.array2d(dtype=wp.quatf),
    position_2: wp.array2d(dtype=wp.vec3f),
    quat_2: wp.array2d(dtype=wp.quatf),
    resulting_position: wp.array2d(dtype=wp.vec3f),
    resulting_quat: wp.array2d(dtype=wp.quatf)
):
    env_idx, body_idx = wp.tid()
    resulting_position[env_idx, body_idx] = position_1[env_idx, body_idx] + wp.quat_rotate(quat_1[env_idx, body_idx], position_2[env_idx, body_idx])
    if quat_2.shape[0] > 0:
        resulting_quat[env_idx, body_idx] = quat_1[env_idx, body_idx] * quat_2[env_idx, body_idx]
    else:
        resulting_quat[env_idx, body_idx] = quat_1[env_idx, body_idx]

@wp.kernel
def derive_body_acceleration_from_velocity(
    linear_velocity: wp.array2d(dtype=wp.vec3f),
    angular_velocity: wp.array2d(dtype=wp.vec3f),
    previous_linear_velocity: wp.array2d(dtype=wp.vec3f),
    previous_angular_velocity: wp.array2d(dtype=wp.vec3f),
    dt: float,
    linear_acceleration: wp.array2d(dtype=wp.vec3f),
    angular_acceleration: wp.array2d(dtype=wp.vec3f),
):
    env_idx, body_idx = wp.tid()
    # compute acceleration
    linear_acceleration[env_idx, body_idx] = (linear_velocity[env_idx, body_idx] - previous_linear_velocity[env_idx, body_idx]) / dt
    angular_acceleration[env_idx, body_idx] = (angular_velocity[env_idx, body_idx] - previous_angular_velocity[env_idx, body_idx]) / dt

    # update previous velocities
    previous_linear_velocity[env_idx, body_idx] = linear_velocity[env_idx, body_idx]
    previous_angular_velocity[env_idx, body_idx] = angular_velocity[env_idx, body_idx]

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
def update_root_com_pose(
    position: wp.array(dtype=wp.vec3f),
    quat: wp.array(dtype=wp.quatf),
    position_b: wp.array(dtype=wp.vec3f),
    quat_b: wp.array(dtype=wp.quatf),
):
    index = wp.tid()
    position_b[index] = position[index]
    quat_b[index] = quat[index]


@wp.kernel
def project_vec_from_quat(
    vec: wp.array(dtype=wp.vec3f),
    quat: wp.array(dtype=wp.quatf),
    resulting_vec: wp.array(dtype=wp.vec3f)
):
    index = wp.tid()
    resulting_vec[index] = wp.quat_rotate(quat[index], vec[index])

@wp.kernel
def project_vec_from_quat_inverse(
    vec: wp.array(dtype=wp.vec3f),
    quat: wp.array(dtype=wp.quatf),
    resulting_vec: wp.array(dtype=wp.vec3f)
):
    index = wp.tid()
    resulting_vec[index] = wp.quat_rotate_inv(quat[index], vec[index])