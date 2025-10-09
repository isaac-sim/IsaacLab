import warp as wp

"""
Casters from spatial velocity to linear and angular velocity
"""

@wp.kernel
def get_linear_velocity(velocity: wp.array(dtype=wp.spatial_vectorf), linear_velocity: wp.array(dtype=wp.vec3f)):
    """
    Get the linear velocity from a spatial velocity.

    Velocities are given in the following format: (wx, wy, wz, vx, vy, vz).

    .. caution:: Velocities are given with angular velocity first and linear velocity second.

    Args:
        velocity: The spatial velocity. Shape is (6,).
        linear_velocity: The linear velocity. Shape is (3,). (modified)
    """
    index = wp.tid()
    linear_velocity[index] = wp.spatial_bottom(velocity[index])


@wp.kernel
def get_angular_velocity(velocity: wp.array(dtype=wp.spatial_vectorf), angular_velocity: wp.array(dtype=wp.vec3f)):
    """
    Get the angular velocity from a spatial velocity.

    Velocities are given in the following format: (wx, wy, wz, vx, vy, vz).

    .. caution:: Velocities are given with angular velocity first and linear velocity second.

    Args:
        velocity: The spatial velocity. Shape is (6,).
        angular_velocity: The angular velocity. Shape is (3,). (modified)
    """
    index = wp.tid()
    angular_velocity[index] = wp.spatial_bottom(velocity[index])

"""
Projectors from com frame to link frame and vice versa
"""

@wp.func
def velocity_projector(
    com_velocity: wp.spatial_vectorf,
    link_pose: wp.transformf,
    com_position: wp.vec3f,
) -> wp.spatial_vectorf:
    """
    Project a velocity from the com frame to the link frame.

    Velocities are given in the following format: (wx, wy, wz, vx, vy, vz).

    .. caution:: Velocities are given with angular velocity first and linear velocity second.

    .. note:: Only :arg:`com_position` is needed as in Newton, the CoM orientation is always aligned with the
    link frame.

    Args:
        com_velocity: The velocity in the com frame. Shape is (6,).
        link_pose: The link pose in the world frame. Shape is (7,).
        com_position: The position of the com in the link frame. Shape is (3,).

    Returns:
        wp.spatial_vectorf: The projected velocity in the link frame. Shape is (6,).
    """
    u = wp.spatial_top(com_velocity)
    w = wp.spatial_bottom(com_velocity) + wp.cross(
            wp.spatial_bottom(com_velocity),
            wp.quat_rotate(wp.transform_get_rotation(link_pose), -com_position),
        )
    return wp.spatial_vectorf(u[0], u[1], u[2], w[0], w[1], w[2])


@wp.func
def velocity_projector_inv(
    com_velocity: wp.spatial_vectorf,
    link_pose: wp.transformf,
    com_position: wp.vec3f,
) -> wp.spatial_vectorf:
    """
    Project a velocity from the link frame to the com frame.

    Velocities are given in the following format: (wx, wy, wz, vx, vy, vz).

    .. caution:: Velocities are given with angular velocity first and linear velocity second.

    .. note:: Only :arg:`com_position` is needed as in Newton, the CoM orientation is always aligned with the
    link frame.

    Args:
        com_velocity: The velocity in the link frame. Shape is (6,).
        link_pose: The link pose in the world frame. Shape is (7,).
        com_position: The position of the com in the link frame. Shape is (3,).

    Returns:
        wp.spatial_vectorf: The projected velocity in the com frame. Shape is (6,).
    """
    u = wp.spatial_top(com_velocity)
    w = wp.spatial_bottom(com_velocity) + wp.cross(
            wp.spatial_bottom(com_velocity),
            wp.quat_rotate(wp.transform_get_rotation(link_pose), com_position),
        )
    return wp.spatial_vectorf(u[0], u[1], u[2], w[0], w[1], w[2])

"""
Kernels to project velocities to and from the com frame
"""

@wp.kernel
def project_com_velocity_to_link_frame(
    com_velocity: wp.array(dtype=wp.spatial_vectorf),
    link_pose: wp.array(dtype=wp.transformf),
    com_position: wp.array(dtype=wp.vec3f),
    link_velocity: wp.array(dtype=wp.spatial_vectorf)
):
    """
    Project a velocity from the com frame to the link frame.

    Velocities are given in the following format: (wx, wy, wz, vx, vy, vz).

    .. caution:: Velocities are given with angular velocity first and linear velocity second.

    .. note:: Only :arg:`com_position` is needed as in Newton, the CoM orientation is always aligned with the
    link frame.

    Args:
        com_velocity: The com velocity in the world frame. Shape is (num_links, 6).
        link_pose: The link pose in the world frame. Shape is (num_links, 7).
        com_position: The com position in link frame. Shape is (num_links, 3).
        link_velocity: The link velocity. Shape is (num_links, 6). (modified)
    """
    index = wp.tid()
    link_velocity[index] = velocity_projector(com_velocity[index], link_pose[index], com_position[index])

@wp.kernel
def project_com_velocity_to_link_frame_masked(
    com_velocity: wp.array(dtype=wp.spatial_vectorf),
    link_pose: wp.array(dtype=wp.transformf),
    com_position: wp.array(dtype=wp.vec3f),
    link_velocity: wp.array(dtype=wp.spatial_vectorf),
    mask: wp.array(dtype=wp.bool)
):
    """
    Project a velocity from the com frame to the link frame.

    Velocities are given in the following format: (wx, wy, wz, vx, vy, vz).

    .. caution:: Velocities are given with angular velocity first and linear velocity second.

    .. note:: Only :arg:`com_position` is needed as in Newton, the CoM orientation is always aligned with the
    link frame.

    Args:
        com_velocity: The com velocity in the world frame. Shape is (num_links, 6).
        link_pose: The link pose in the world frame. Shape is (num_links, 7).
        com_position: The com position in link frame. Shape is (num_links, 3).
        link_velocity: The link velocity in the world frame. Shape is (num_links, 6). (modified)
        mask: The mask of the links to project the velocity to. Shape is (num_links,).
    """
    index = wp.tid()
    if mask[index]:
        link_velocity[index] = velocity_projector(
            com_velocity[index],
            link_pose[index],
            com_position[index],
        )

@wp.kernel
def project_com_velocity_to_link_frame_batch(
    com_velocity: wp.array2d(dtype=wp.spatial_vectorf),
    link_pose: wp.array2d(dtype=wp.transformf),
    com_position: wp.array2d(dtype=wp.vec3f),
    link_velocity: wp.array2d(dtype=wp.spatial_vectorf)
):
    """
    Project a velocity from the com frame to the link frame.

    Velocities are given in the following format: (wx, wy, wz, vx, vy, vz).

    .. caution:: Velocities are given with angular velocity first and linear velocity second.

    .. note:: Only :attr:`com_position` is needed as in Newton, the CoM orientation is always aligned with the
    link frame.

    Args:
        com_velocity: The com velocity in the world frame. Shape is (num_links, 6).
        link_pose: The link pose in the world frame. Shape is (num_links, 7).
        com_position: The com position in link frame. Shape is (num_links, 3).
        link_velocity: The link velocity in the world frame. Shape is (num_links, 6). (modified)
    """
    env_idx, body_idx = wp.tid()
    link_velocity[env_idx, body_idx] = velocity_projector(
        com_velocity[env_idx, body_idx],
        link_pose[env_idx, body_idx],
        com_position[env_idx, body_idx],
    )

@wp.kernel
def project_com_velocity_to_link_frame_batch_masked(
    com_velocity: wp.array2d(dtype=wp.spatial_vectorf),
    link_pose: wp.array2d(dtype=wp.transformf),
    com_position: wp.array2d(dtype=wp.vec3f),
    link_velocity: wp.array2d(dtype=wp.spatial_vectorf),
    env_mask: wp.array(dtype=wp.bool),
    body_mask: wp.array(dtype=wp.bool),
):
    """
    Project a velocity from the com frame to the link frame.

    Velocities are given in the following format: (wx, wy, wz, vx, vy, vz).

    .. caution:: Velocities are given with angular velocity first and linear velocity second.

    .. note:: Only :arg:`com_position` is needed as in Newton, the CoM orientation is always aligned with the
    link frame.

    Args:
        com_velocity: The com velocity in the world frame. Shape is (num_links, 6).
        link_pose: The link pose in the world frame. Shape is (num_links, 7).
        com_position: The com position in link frame. Shape is (num_links, 3).
        link_velocity: The link velocity in the world frame. Shape is (num_links, 6). (modified)
        env_mask: The mask of the environments to project the velocity to. Shape is (num_links,).
        body_mask: The mask of the bodies to project the velocity to. Shape is (num_links,).
    """
    env_idx, body_idx = wp.tid()
    if env_mask[env_idx] and body_mask[body_idx]:
        link_velocity[env_idx, body_idx] = velocity_projector(
            com_velocity[env_idx, body_idx],
            link_pose[env_idx, body_idx],
            com_position[env_idx, body_idx],
        )


@wp.kernel
def project_link_velocity_to_com_frame(
    link_velocity: wp.array(dtype=wp.spatial_vectorf),
    link_pose: wp.array(dtype=wp.transformf),
    com_position: wp.array(dtype=wp.vec3f),
    com_velocity: wp.array(dtype=wp.spatial_vectorf)
):
    """
    Project a velocity from the link frame to the com frame.

    Velocities are given in the following format: (wx, wy, wz, vx, vy, vz).

    .. caution:: Velocities are given with angular velocity first and linear velocity second.

    .. note:: Only :arg:`com_position` is needed as in Newton, the CoM orientation is always aligned with the
    link frame.

    Args:
        link_velocity: The link velocity in the world frame. Shape is (num_links, 6).
        link_pose: The link pose in the world frame. Shape is (num_links, 7).
        com_position: The com position in link frame. Shape is (num_links, 3).
        com_velocity: The com velocity in the world frame. Shape is (num_links, 6). (modified)
    """
    index = wp.tid()
    com_velocity[index] = velocity_projector_inv(link_velocity[index], link_pose[index], com_position[index])


@wp.kernel
def project_link_velocity_to_com_frame_masked(
    link_velocity: wp.array(dtype=wp.spatial_vectorf),
    link_pose: wp.array(dtype=wp.transformf),
    com_position: wp.array(dtype=wp.vec3f),
    com_velocity: wp.array(dtype=wp.spatial_vectorf),
    mask: wp.array(dtype=wp.bool)
):
    """
    Project a velocity from the link frame to the com frame.

    Velocities are given in the following format: (wx, wy, wz, vx, vy, vz).

    .. caution:: Velocities are given with angular velocity first and linear velocity second.

    .. note:: Only :arg:`com_position` is needed as in Newton, the CoM orientation is always aligned with the
    link frame.

    Args:
        link_velocity: The link velocity in the world frame. Shape is (num_links, 6).
        link_pose: The link pose in the world frame. Shape is (num_links, 7).
        com_position: The com position in link frame. Shape is (num_links, 3).
        com_velocity: The com velocity in the world frame. Shape is (num_links, 6). (modified)
        mask: The mask of the links to project the velocity to. Shape is (num_links,).
    """
    index = wp.tid()
    if mask[index]:
        com_velocity[index] = velocity_projector_inv(
            link_velocity[index],
            link_pose[index],
            com_position[index],
        )


@wp.kernel
def project_link_velocity_to_com_frame_batch(
    link_velocity: wp.array2d(dtype=wp.spatial_vectorf),
    link_pose: wp.array2d(dtype=wp.transformf),
    com_position: wp.array2d(dtype=wp.vec3f),
    com_velocity: wp.array2d(dtype=wp.spatial_vectorf)
):
    """
    Project a velocity from the link frame to the com frame.

    Velocities are given in the following format: (wx, wy, wz, vx, vy, vz).

    .. caution:: Velocities are given with angular velocity first and linear velocity second.

    .. note:: Only :arg:`com_position` is needed as in Newton, the CoM orientation is always aligned with the
    link frame.

    Args:
        link_velocity (wp.array2d(dtype=wp.spatial_vectorf)): The link velocity in the world frame.
        link_pose (wp.array2d(dtype=wp.transformf)): The link pose in the world frame.
        com_position (wp.array2d(dtype=wp.vec3f)): The com position in link frame.
        com_velocity (wp.array2d(dtype=wp.spatial_vectorf)): The com velocity in the world frame. (destination)
    """
    env_idx, body_idx = wp.tid()
    com_velocity[env_idx, body_idx] = velocity_projector_inv(
        link_velocity[env_idx, body_idx],
        link_pose[env_idx, body_idx],
        com_position[env_idx, body_idx]
    )

@wp.kernel
def project_link_velocity_to_com_frame_batch_masked(
    link_velocity: wp.array2d(dtype=wp.spatial_vectorf),
    link_pose: wp.array2d(dtype=wp.transformf),
    com_position: wp.array2d(dtype=wp.vec3f),
    com_velocity: wp.array2d(dtype=wp.spatial_vectorf),
    env_mask: wp.array(dtype=wp.bool),
    body_mask: wp.array(dtype=wp.bool),
):
    """
    Project a velocity from the link frame to the com frame.

    Velocities are given in the following format: (wx, wy, wz, vx, vy, vz).

    .. caution:: Velocities are given with angular velocity first and linear velocity second.

    .. note:: Only :arg:`com_position` is needed as in Newton, the CoM orientation is always aligned with the
    link frame.

    Args:
        link_velocity: The link velocity in the world frame. Shape is (num_links, 6).
        link_pose: The link pose in the world frame. Shape is (num_links, 7).
        com_position: The com position in link frame. Shape is (num_links, 3).
        com_velocity: The com velocity in the world frame. Shape is (num_links, 6). (modified)
        env_mask: The mask of the environments to project the velocity to. Shape is (num_links,).
        body_mask: The mask of the bodies to project the velocity to. Shape is (num_links,).
    """
    env_idx, body_idx = wp.tid()
    if env_mask[env_idx] and body_mask[body_idx]:
        com_velocity[env_idx, body_idx] = velocity_projector_inv(
            link_velocity[env_idx, body_idx],
            link_pose[env_idx, body_idx],
            com_position[env_idx, body_idx],
        )

"""
Kernels to update velocity arrays
"""

@wp.kernel
def update_velocity_array(
    new_velocity: wp.array(dtype=wp.spatial_vectorf),
    velocity: wp.array(dtype=wp.spatial_vectorf),
    mask: wp.array(dtype=wp.bool),
):
    """
    Update a velocity array with a new velocity.

    Velocities are given in the following format: (wx, wy, wz, vx, vy, vz).

    .. caution:: Velocities are given with angular velocity first and linear velocity second.

    Args:
        new_velocity: The new velocity. Shape is (num_links, 6).
        velocity: The velocity array. Shape is (num_links, 6). (modified)
        mask: The mask of the velocities to update. Shape is (num_links,).
    """
    index = wp.tid()
    if mask[index]:
        velocity[index] = new_velocity[index]


@wp.kernel
def update_velocity_array_batch(
    new_velocity: wp.array2d(dtype=wp.spatial_vectorf),
    velocity: wp.array2d(dtype=wp.spatial_vectorf),
    env_mask: wp.array(dtype=wp.bool),
    body_mask: wp.array(dtype=wp.bool),
):
    """
    Update a velocity array with a new velocity.

    Velocities are given in the following format: (wx, wy, wz, vx, vy, vz).

    .. caution:: Velocities are given with angular velocity first and linear velocity second.

    Args:
        new_velocity: The new velocity. Shape is (num_links, 6).
        velocity: The velocity array. Shape is (num_links, 6). (modified)
        env_mask: The mask of the environments to update. Shape is (num_links,).
        body_mask: The mask of the bodies to update. Shape is (num_links,).
    """
    env_idx, body_idx = wp.tid()
    if env_mask[env_idx] and body_mask[body_idx]:
        velocity[env_idx, body_idx] = new_velocity[env_idx, body_idx]

"""
Kernels to derive body acceleration from velocity.
"""

@wp.kernel
def derive_body_acceleration_from_velocity(
    velocity: wp.array2d(dtype=wp.spatial_vectorf),
    previous_velocity: wp.array2d(dtype=wp.spatial_vectorf),
    dt: float,
    acceleration: wp.array2d(dtype=wp.spatial_vectorf),
):
    """
    Derive the body acceleration from the velocity.

    Args:
        velocity: The velocity. Shape is (num_instances, 6).
        previous_velocity: The previous velocity. Shape is (num_instances, 6).
        dt: The time step.
        acceleration: The acceleration. Shape is (num_instances, 6). (modified)
    """
    env_idx, body_idx = wp.tid()
    acceleration[env_idx, body_idx] = (velocity[env_idx, body_idx] - previous_velocity[env_idx, body_idx]) / dt