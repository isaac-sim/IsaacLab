from sympy import Q
import warp as wp


@wp.kernel
def get_position(pose: wp.array(dtype=wp.transformf), position: wp.array(dtype=wp.vec3f)):
    index = wp.tid()
    position[index] = pose.p[index]


@wp.kernel
def get_quat(pose: wp.array(dtype=wp.transformf), quat: wp.array(dtype=wp.quatf)):
    index = wp.tid()
    quat[index] = pose.q[index]


@wp.kernel
def get_linear_velocity(velocity: wp.array(dtype=wp.spatial_vectorf), linear_velocity: wp.array(dtype=wp.vec3f)):
    index = wp.tid()
    linear_velocity[index] = wp.spatial_bottom(velocity[index])


@wp.kernel
def get_angular_velocity(velocity: wp.array(dtype=wp.spatial_vectorf), angular_velocity: wp.array(dtype=wp.vec3f)):
    index = wp.tid()
    angular_velocity[index] = wp.spatial_bottom(velocity[index])


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
def velocity_projector(
    velocity: wp.spatial_vectorf,
    pose: wp.transformf) -> wp.spatial_vectorf:
    return wp.spatial_vectorf(wp.spatial_top(velocity), wp.cross(wp.spatial_bottom(velocity), wp.quat_rotate(pose.q, -pose.p)))


@wp.kernel
def project_linear_velocity_to_link_frame(
    velocity: wp.array(dtype=wp.spatial_vectorf),
    pose: wp.array(dtype=wp.transformf),
    transformed_velocity: wp.array(dtype=wp.spatial_vectorf)
):
    index = wp.tid()
    transformed_velocity[index] = velocity_projector(velocity[index], pose[index])


@wp.kernel
def project_linear_velocity_to_links_frame(
    velocity: wp.array2d(dtype=wp.spatial_vectorf),
    pose: wp.array2d(dtype=wp.transformf),
    transformed_velocity: wp.array2d(dtype=wp.spatial_vectorf)
):
    env_idx, body_idx = wp.tid()
    transformed_velocity[env_idx, body_idx] = velocity_projector(velocity[env_idx, body_idx], pose[env_idx, body_idx])


@wp.kernel
def combine_frame_transforms(
    pose_1: wp.array(dtype=wp.transformf),
    position_2: wp.array(dtype=wp.vec3f),
    resulting_pose: wp.array(dtype=wp.transformf)
):
    index = wp.tid()
    resulting_pose.p[index] = pose_1.p[index] + wp.quat_rotate(pose_1.q[index], position_2[index])
    resulting_pose.q[index] = pose_1.q[index]


@wp.kernel
def combine_frame_transforms_batch(
    pose_1: wp.array2d(dtype=wp.transformf),
    position_2: wp.array2d(dtype=wp.vec3f),
    resulting_pose: wp.array2d(dtype=wp.transformf)
):
    env_idx, body_idx = wp.tid()
    resulting_pose.p[env_idx, body_idx] = pose_1.p[env_idx, body_idx] + wp.quat_rotate(pose_1.q[env_idx, body_idx], position_2[env_idx, body_idx])
    resulting_pose.q[env_idx, body_idx] = pose_1.q[env_idx, body_idx]


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
def update_root_link_pose(
    new_position: wp.array(dtype=wp.vec3f),
    new_quat: wp.array(dtype=wp.quatf),
    position: wp.array(dtype=wp.vec3f),
    quat: wp.array(dtype=wp.quatf),
    indices: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    position[indices[index]] = new_position[index]
    quat[indices[index]] = new_quat[index]

