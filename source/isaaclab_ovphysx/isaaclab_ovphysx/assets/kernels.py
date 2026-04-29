# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels shared by ovphysx-backed Articulation and RigidObject assets."""

import warp as wp

# 13-element state vector: pos(3) + quat(4) + lin_vel(3) + ang_vel(3).
# Layout matches the PhysX/Newton shared convention used by the deprecated
# state-concat properties (default_root_state, root_state_w, etc.).
vec13f = wp.types.vector(length=13, dtype=wp.float32)


@wp.func
def _concat_pose_and_vel_to_state_func(
    pose: wp.transformf,
    vel: wp.spatial_vectorf,
) -> vec13f:
    """Concatenate a pose and velocity into a 13-element state vector.

    The state vector layout is ``[pos(3), quat(4), ang_vel(3), lin_vel(3)]``,
    matching the PhysX and Newton backend convention.  Warp spatial vectors
    store angular velocity in components ``[0:3]`` (``spatial_top``) and
    linear velocity in components ``[3:6]`` (``spatial_bottom``).

    Args:
        pose: Root pose as a ``wp.transformf`` — components ``[0:3]`` are
            position ``[m]`` and ``[3:7]`` are the quaternion ``(x, y, z, w)`` ``[-]``.
        vel: Root spatial velocity — components ``[0:3]`` are angular velocity
            ``[rad/s]`` and ``[3:6]`` are linear velocity ``[m/s]``.

    Returns:
        13-element state vector ``(px, py, pz, qx, qy, qz, qw, wx, wy, wz, vx, vy, vz)``.
    """
    return vec13f(
        pose[0],
        pose[1],
        pose[2],
        pose[3],
        pose[4],
        pose[5],
        pose[6],
        vel[0],
        vel[1],
        vel[2],
        vel[3],
        vel[4],
        vel[5],
    )


@wp.kernel
def concat_root_pose_and_vel_to_state(
    pose: wp.array(dtype=wp.transformf),
    vel: wp.array(dtype=wp.spatial_vectorf),
    state: wp.array(dtype=vec13f),
):
    """Concatenate root pose and velocity into a 13-element state vector.

    Combines a 7-element pose (pos + quat) and a 6-element spatial velocity
    (linear + angular) into a single ``vec13f`` state vector per instance by
    calling :func:`_concat_pose_and_vel_to_state_func`.

    Args:
        pose: Root poses in world frame ``[m, -]``. Shape is ``(num_envs,)``.
        vel: Root spatial velocities ``[m/s, rad/s]``. Shape is ``(num_envs,)``.
        state: Output concatenated state vectors ``(px, py, pz, qx, qy, qz, qw,
            wx, wy, wz, vx, vy, vz)`` ``[m, -, rad/s, m/s]``. Shape is
            ``(num_envs,)``.
    """
    i = wp.tid()
    state[i] = _concat_pose_and_vel_to_state_func(pose[i], vel[i])


@wp.kernel
def _body_wrench_to_world(
    force_b: wp.array(dtype=wp.vec3f, ndim=2),
    torque_b: wp.array(dtype=wp.vec3f, ndim=2),
    poses: wp.array(dtype=wp.transformf, ndim=2),
    wrench_out: wp.array(dtype=wp.float32, ndim=3),
):
    """Rotate body-frame force/torque to world frame and pack into a flat output array.

    For each instance ``i`` and body ``j``, the body-frame force and torque are
    rotated into the world frame using the quaternion extracted from ``poses[i, j]``.
    The world-frame link position is also extracted and packed alongside the
    rotated wrench.

    Output layout per ``(i, j)`` slice (9 floats total):

    * ``[0:3]`` — world-frame force ``[N]``  ``(fx, fy, fz)``
    * ``[3:6]`` — world-frame torque ``[N·m]``  ``(tx, ty, tz)``
    * ``[6:9]`` — world-frame link position ``[m]``  ``(px, py, pz)``

    Args:
        force_b: Body-frame applied forces ``[N]``. Shape is ``(N, L)``.
        torque_b: Body-frame applied torques ``[N·m]``. Shape is ``(N, L)``.
        poses: Link poses in world frame. Shape is ``(N, L)``.
        wrench_out: Output packed wrench array ``[N, N·m, m]``. Shape is ``(N, L, 9)``.
    """
    i, j = wp.tid()
    q = wp.transform_get_rotation(poses[i, j])
    f_w = wp.quat_rotate(q, force_b[i, j])
    t_w = wp.quat_rotate(q, torque_b[i, j])
    wrench_out[i, j, 0] = f_w[0]
    wrench_out[i, j, 1] = f_w[1]
    wrench_out[i, j, 2] = f_w[2]
    wrench_out[i, j, 3] = t_w[0]
    wrench_out[i, j, 4] = t_w[1]
    wrench_out[i, j, 5] = t_w[2]
    p_w = wp.transform_get_translation(poses[i, j])
    wrench_out[i, j, 6] = p_w[0]
    wrench_out[i, j, 7] = p_w[1]
    wrench_out[i, j, 8] = p_w[2]


@wp.kernel
def _scatter_rows_partial(
    dst: wp.array2d(dtype=wp.float32),
    src: wp.array2d(dtype=wp.float32),
    ids: wp.array(dtype=wp.int32),
):
    """Scatter a partial row-indexed source array into a larger destination array.

    For each thread ``(i, j)`` the kernel writes ``dst[ids[i], j] = src[i, j]``.
    This is a GPU-side indexed scatter that allows writing ``K`` selected rows
    from a ``(K, C)`` source into the corresponding rows of a ``(N, C)``
    destination, where ``K ≤ N``.

    Args:
        dst: Destination array of shape ``(N, C)`` to scatter values into.
        src: Source array of shape ``(K, C)`` containing the values to scatter.
        ids: Row indices into ``dst`` for each row of ``src``. Shape is ``(K,)``.
    """
    i, j = wp.tid()
    dst[ids[i], j] = src[i, j]


@wp.kernel
def _copy_first_body(
    body_vel: wp.array(dtype=wp.spatial_vectorf, ndim=2),
    root_vel: wp.array(dtype=wp.spatial_vectorf),
):
    """Copy the first body's spatial velocity to the root velocity buffer.

    For single rigid-body assets, index 0 is always the root body.  This
    kernel extracts that slice without allocating an intermediate buffer.

    Args:
        body_vel: Body spatial velocities ``[m/s, rad/s]``. Shape is
            ``(num_envs, num_bodies)``.
        root_vel: Output root spatial velocities ``[m/s, rad/s]``. Shape is
            ``(num_envs,)``.
    """
    i = wp.tid()
    root_vel[i] = body_vel[i, 0]


@wp.kernel
def _compose_root_com_pose(
    link_pose: wp.array(dtype=wp.transformf),
    com_pose_b: wp.array(dtype=wp.transformf, ndim=2),
    com_pose_w: wp.array(dtype=wp.transformf),
):
    """Compose root link pose with the body-frame COM offset to get the world-frame COM pose.

    Implements the forward transform:

        ``com_pose_w = link_pose_w * com_pose_b[0]``

    where ``*`` denotes ``wp.transform_multiply``.  Only the first body
    (index ``0``) is used; for rigid objects there is always exactly one body.

    Args:
        link_pose: Root link poses in world frame ``[m, -]``. Shape is
            ``(num_envs,)``.
        com_pose_b: Body-frame COM offsets ``[m, -]`` from the
            ``RIGID_BODY_COM_POSE`` binding. Shape is ``(num_envs, num_bodies)``.
        com_pose_w: Output world-frame root COM poses ``[m, -]``. Shape is
            ``(num_envs,)``.
    """
    i = wp.tid()
    com_pose_w[i] = wp.transform_multiply(link_pose[i], com_pose_b[i, 0])


@wp.kernel
def _compose_root_link_pose_from_com(
    com_pose_w: wp.array(dtype=wp.transformf),
    com_pose_b: wp.array(dtype=wp.transformf, ndim=2),
    link_pose_w: wp.array(dtype=wp.transformf),
):
    """Recover root link pose from world-frame COM pose and body-frame COM offset.

    This is the inverse of :func:`_compose_root_com_pose`.  The forward relation is:

        ``com_pose_w = link_pose_w * com_pose_b``

    Rearranging gives:

        ``link_pose_w = com_pose_w * inverse(com_pose_b)``

    Args:
        com_pose_w: World-frame COM poses ``[m, -]`` (user-provided input).
            Shape is ``(num_envs,)``.
        com_pose_b: Body-frame COM offsets ``[m, -]`` read from the
            ``RIGID_BODY_COM_POSE`` binding. Shape is ``(num_envs, num_bodies)``.
        link_pose_w: Output root link poses in world frame ``[m, -]``. Shape is
            ``(num_envs,)``.
    """
    i = wp.tid()
    link_pose_w[i] = wp.transform_multiply(com_pose_w[i], wp.transform_inverse(com_pose_b[i, 0]))


@wp.kernel
def _projected_gravity(
    gravity_vec_w: wp.array(dtype=wp.vec3f),
    root_pose: wp.array(dtype=wp.transformf),
    out: wp.array(dtype=wp.vec3f),
):
    """Project the world-frame gravity direction into the root body frame.

    Applies the inverse of the root orientation quaternion to the world-frame
    gravity vector, yielding the gravity direction expressed in the body frame.
    The magnitude is preserved (unit vector in, unit vector out if input is a
    unit vector).

    Args:
        gravity_vec_w: Gravity direction per instance in world frame ``[-]``
            (typically the normalised ``(0, 0, -1)`` gravitational acceleration
            direction). Shape is ``(num_envs,)``.
        root_pose: Root link poses in world frame ``[m, -]``. Only the
            rotation component is used. Shape is ``(num_envs,)``.
        out: Output gravity direction in body frame ``[-]``. Shape is
            ``(num_envs,)``.
    """
    i = wp.tid()
    q = wp.transform_get_rotation(root_pose[i])
    out[i] = wp.quat_rotate_inv(q, gravity_vec_w[i])


@wp.kernel
def _compute_heading(
    forward_vec_b: wp.array(dtype=wp.vec3f),
    root_pose: wp.array(dtype=wp.transformf),
    out: wp.array(dtype=wp.float32),
):
    """Compute the yaw heading angle by rotating a body-frame forward vector to world frame.

    Rotates ``forward_vec_b`` by the root orientation quaternion and then computes the
    heading as ``atan2(forward_w.y, forward_w.x)`` ``[rad]``, i.e. the signed angle
    from the world X-axis to the projected forward direction in the XY plane.

    Args:
        forward_vec_b: Forward direction in body frame per instance ``[-]``.
            Shape is ``(num_envs,)``.
        root_pose: Root link poses in world frame ``[m, -]``. Only the rotation
            component is used. Shape is ``(num_envs,)``.
        out: Output heading angles ``[rad]`` in ``[-π, π]``. Shape is
            ``(num_envs,)``.
    """
    i = wp.tid()
    q = wp.transform_get_rotation(root_pose[i])
    forward = wp.quat_rotate(q, forward_vec_b[i])
    out[i] = wp.atan2(forward[1], forward[0])


@wp.kernel
def _world_vel_to_body_lin(
    root_pose: wp.array(dtype=wp.transformf),
    vel_w: wp.array(dtype=wp.spatial_vectorf),
    out: wp.array(dtype=wp.vec3f),
):
    """Rotate the world-frame linear velocity component into the root body frame.

    Extracts the linear velocity from the top three components of the spatial
    velocity vector (``wp.spatial_top``) and rotates it by the inverse of the
    root orientation quaternion.

    Args:
        root_pose: Root link poses in world frame ``[m, -]``. Only the rotation
            component is used. Shape is ``(num_envs,)``.
        vel_w: Root spatial velocities in world frame ``[m/s, rad/s]``.
            Shape is ``(num_envs,)``.
        out: Output linear velocity in body frame ``[m/s]``. Shape is
            ``(num_envs,)``.
    """
    i = wp.tid()
    q = wp.transform_get_rotation(root_pose[i])
    lin = wp.spatial_top(vel_w[i])
    out[i] = wp.quat_rotate_inv(q, lin)


@wp.kernel
def _world_vel_to_body_ang(
    root_pose: wp.array(dtype=wp.transformf),
    vel_w: wp.array(dtype=wp.spatial_vectorf),
    out: wp.array(dtype=wp.vec3f),
):
    """Rotate the world-frame angular velocity component into the root body frame.

    Extracts the angular velocity from the bottom three components of the spatial
    velocity vector (``wp.spatial_bottom``) and rotates it by the inverse of the
    root orientation quaternion.

    Args:
        root_pose: Root link poses in world frame ``[m, -]``. Only the rotation
            component is used. Shape is ``(num_envs,)``.
        vel_w: Root spatial velocities in world frame ``[m/s, rad/s]``.
            Shape is ``(num_envs,)``.
        out: Output angular velocity in body frame ``[rad/s]``. Shape is
            ``(num_envs,)``.
    """
    i = wp.tid()
    q = wp.transform_get_rotation(root_pose[i])
    ang = wp.spatial_bottom(vel_w[i])
    out[i] = wp.quat_rotate_inv(q, ang)


@wp.kernel
def get_root_link_vel_from_root_com_vel(
    com_vel: wp.array(dtype=wp.spatial_vectorf),
    link_pose: wp.array(dtype=wp.transformf),
    body_com_pose_b: wp.array(dtype=wp.transformf, ndim=2),
    link_vel: wp.array(dtype=wp.spatial_vectorf),
):
    """Compute root link velocity from root COM velocity via a lever-arm transform.

    Transforms the COM spatial velocity into the link-frame spatial velocity by
    applying the rigid-body lever-arm correction.  Angular velocity is invariant
    under a change of reference point; linear velocity picks up the cross-product
    contribution from the COM offset:

        ``v_link = v_com + ω × lever``

    where ``lever = rot(link_rot, -com_offset_b)`` is the COM-to-link-origin
    vector expressed in the world frame, and ``ω = angular velocity``.

    Args:
        com_vel: Root COM spatial velocities ``[m/s, rad/s]`` in world frame.
            Components ``[0:3]`` are linear ``[m/s]``, ``[3:6]`` are angular
            ``[rad/s]``. Shape is ``(num_instances,)``.
        link_pose: Root link poses in world frame ``[m, -]``. Shape is
            ``(num_instances,)``.
        body_com_pose_b: Body-frame COM offsets ``[m, -]``. Shape is
            ``(num_instances, num_bodies)``. Only body index ``0`` is used.
        link_vel: Output root link spatial velocities ``[m/s, rad/s]`` in world
            frame. Shape is ``(num_instances,)``.
    """
    i = wp.tid()
    ang = wp.spatial_bottom(com_vel[i])
    lever = wp.quat_rotate(
        wp.transform_get_rotation(link_pose[i]), -wp.transform_get_translation(body_com_pose_b[i, 0])
    )
    link_vel[i] = wp.spatial_vector(wp.spatial_top(com_vel[i]) + wp.cross(ang, lever), ang)


@wp.kernel
def derive_body_acceleration_from_body_com_velocities(
    body_com_vel: wp.array(dtype=wp.spatial_vectorf),
    dt: wp.float32,
    prev_body_com_vel: wp.array(dtype=wp.spatial_vectorf),
    body_acc: wp.array(dtype=wp.spatial_vectorf),
):
    """Derive body acceleration from body COM velocities using finite differencing.

    Mirrors :func:`isaaclab_newton.assets.kernels.derive_body_acceleration_from_body_com_velocities`
    for a 1-D (root-level) array layout used by single rigid-body assets.

    Args:
        body_com_vel: Current body COM spatial velocities [m/s, rad/s].
            Shape is (num_instances,), dtype ``wp.spatial_vectorf``.
        dt: Simulation time step [s].
        prev_body_com_vel: Previous-step body COM spatial velocities [m/s, rad/s].
            Updated in-place after the acceleration is written.
            Shape is (num_instances,), dtype ``wp.spatial_vectorf``.
        body_acc: Output body spatial accelerations [m/s², rad/s²].
            Shape is (num_instances,), dtype ``wp.spatial_vectorf``.
    """
    i = wp.tid()
    body_acc[i] = (body_com_vel[i] - prev_body_com_vel[i]) / dt
    prev_body_com_vel[i] = body_com_vel[i]
