# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp

"""
Articulation-specific warp functions.
"""


@wp.func
def compute_soft_joint_pos_limits_func(
    joint_pos_limits: wp.vec2f,
    soft_limit_factor: wp.float32,
):
    """Compute the soft joint position limits.

    Args:
        joint_pos_limits: Hard joint position limits as ``(lower, upper)`` [m or rad,
            depending on joint type].
        soft_limit_factor: Scale factor in [0, 1] shrinking the soft range around
            the midpoint of the hard range; ``1.0`` makes the soft limits equal the
            hard limits, smaller values create a tighter window.

    Returns:
        The soft joint position limits as ``(lower, upper)``.
    """
    joint_pos_mean = (joint_pos_limits[0] + joint_pos_limits[1]) / 2.0
    joint_pos_range = joint_pos_limits[1] - joint_pos_limits[0]
    return wp.vec2f(
        joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor,
        joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor,
    )


"""
Articulation-specific warp kernels.
"""


@wp.kernel
def _fd_joint_acc(
    cur_vel: wp.array2d(dtype=wp.float32),
    prev_vel: wp.array2d(dtype=wp.float32),
    inv_dt: float,
    out: wp.array2d(dtype=wp.float32),
):
    """Compute the joint acceleration via finite differencing and update the previous velocity.

    Diverges from PhysX's :func:`get_joint_acc_from_joint_vel` in taking the inverse
    time step rather than ``dt`` itself; the multiply-by-reciprocal avoids per-element
    division inside the kernel.

    Args:
        cur_vel: Current joint velocities [m/s or rad/s, depending on joint type].
            Shape is (num_envs, num_joints).
        prev_vel: Previous joint velocities (updated in-place). Same shape and units
            as :paramref:`cur_vel`.
        inv_dt: Inverse time step ``1 / dt`` [1/s].
        out: Output joint accelerations [m/s^2 or rad/s^2, depending on joint type].
            Shape is (num_envs, num_joints).
    """
    i, j = wp.tid()
    out[i, j] = (cur_vel[i, j] - prev_vel[i, j]) * inv_dt
    prev_vel[i, j] = cur_vel[i, j]


@wp.kernel
def _compose_body_com_poses(
    link_pose: wp.array(dtype=wp.transformf, ndim=2),
    com_pose_b: wp.array(dtype=wp.transformf, ndim=2),
    com_pose_w: wp.array(dtype=wp.transformf, ndim=2),
):
    """Compose body link poses with body-frame CoM offsets to get world-frame CoM poses.

    Args:
        link_pose: Body link poses in world frame [m, m, m, qx, qy, qz, qw].
            Shape is (num_envs, num_bodies).
        com_pose_b: Body-frame CoM offsets [m, m, m, qx, qy, qz, qw].
            Shape is (num_envs, num_bodies).
        com_pose_w: Output world-frame body CoM poses [m, m, m, qx, qy, qz, qw].
            Shape is (num_envs, num_bodies).
    """
    i, j = wp.tid()
    com_pose_w[i, j] = wp.transform_multiply(link_pose[i, j], com_pose_b[i, j])


@wp.kernel
def update_soft_joint_pos_limits(
    joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    soft_limit_factor: wp.float32,
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
):
    """Update soft joint position limits from hard limits and a soft limit factor.

    Soft limits provide a safety margin before reaching the hard joint position
    limits. See :func:`compute_soft_joint_pos_limits_func` for the per-joint
    formula.

    Args:
        joint_pos_limits: Hard joint position limits as vec2f ``(lower, upper)``
            [m or rad, depending on joint type]. Shape is (num_envs, num_joints).
        soft_limit_factor: Scale factor in [0, 1]. ``1.0`` makes the soft limits
            equal the hard limits; smaller values create a tighter window.
        soft_joint_pos_limits: Output array. Shape is (num_envs, num_joints).
    """
    i, j = wp.tid()
    soft_joint_pos_limits[i, j] = compute_soft_joint_pos_limits_func(joint_pos_limits[i, j], soft_limit_factor)


@wp.kernel
def clamp_default_joint_pos_and_update_soft_limits_index(
    joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    env_ids: wp.array(dtype=wp.int32),
    joint_ids: wp.array(dtype=wp.int32),
    soft_limit_factor: wp.float32,
    default_joint_pos: wp.array2d(dtype=wp.float32),
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    clamped_count: wp.array(dtype=wp.int32),
):
    """Clamp default joint positions to new limits and refresh soft limits over (env_ids x joint_ids).

    Mirrors PhysX's :func:`isaaclab_physx.assets.articulation.kernels.write_joint_limit_data_to_buffer`
    side-effects, minus the limit-write itself (the existing
    :func:`shared_kernels.write_joint_position_limit_to_buffer_index` launch handles that).

    For each ``(i, j)`` thread the kernel:

    * Clamps :paramref:`default_joint_pos` ``[env_ids[i], joint_ids[j]]`` if it falls outside
      the new limits, atomically incrementing :paramref:`clamped_count`.
    * Recomputes :paramref:`soft_joint_pos_limits` ``[env_ids[i], joint_ids[j]]`` from the new
      hard limits and :paramref:`soft_limit_factor`.

    Args:
        joint_pos_limits: Hard joint position limits as vec2f ``(lower, upper)``
            [m or rad, depending on joint type]. Shape is (num_envs, num_joints).
        env_ids: Environment indices to update. Shape is (num_selected_envs,).
        joint_ids: Joint indices to update. Shape is (num_selected_joints,).
        soft_limit_factor: Scale factor in [0, 1] for the soft limit window.
        default_joint_pos: In/out default joint positions [m or rad, depending on joint type].
            Shape is (num_envs, num_joints).
        soft_joint_pos_limits: Out soft joint position limits as vec2f ``(lower, upper)``
            [m or rad, depending on joint type]. Shape is (num_envs, num_joints).
        clamped_count: One-element output counter incremented atomically each time a
            default joint position was clamped. Shape is (1,).
    """
    i, j = wp.tid()
    e = env_ids[i]
    k = joint_ids[j]
    lo = joint_pos_limits[e, k][0]
    hi = joint_pos_limits[e, k][1]
    if (default_joint_pos[e, k] < lo) or (default_joint_pos[e, k] > hi):
        wp.atomic_add(clamped_count, 0, 1)
        default_joint_pos[e, k] = wp.clamp(default_joint_pos[e, k], lo, hi)
    soft_joint_pos_limits[e, k] = compute_soft_joint_pos_limits_func(joint_pos_limits[e, k], soft_limit_factor)


@wp.kernel
def clamp_default_joint_pos_and_update_soft_limits_mask(
    joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    env_mask: wp.array(dtype=wp.bool),
    joint_mask: wp.array(dtype=wp.bool),
    soft_limit_factor: wp.float32,
    default_joint_pos: wp.array2d(dtype=wp.float32),
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    clamped_count: wp.array(dtype=wp.int32),
):
    """Mask variant of :func:`clamp_default_joint_pos_and_update_soft_limits_index`.

    Iterates the full ``(num_envs, num_joints)`` grid and applies the clamp /
    soft-limit refresh only where both :paramref:`env_mask` and :paramref:`joint_mask`
    are ``True``.

    Args:
        joint_pos_limits: Hard joint position limits as vec2f ``(lower, upper)``
            [m or rad, depending on joint type]. Shape is (num_envs, num_joints).
        env_mask: Boolean mask over environments. Shape is (num_envs,).
        joint_mask: Boolean mask over joints. Shape is (num_joints,).
        soft_limit_factor: Scale factor in [0, 1] for the soft limit window.
        default_joint_pos: In/out default joint positions [m or rad, depending on joint type].
            Shape is (num_envs, num_joints).
        soft_joint_pos_limits: Out soft joint position limits as vec2f ``(lower, upper)``
            [m or rad, depending on joint type]. Shape is (num_envs, num_joints).
        clamped_count: One-element output counter incremented atomically each time a
            default joint position was clamped. Shape is (1,).
    """
    i, j = wp.tid()
    if not env_mask[i] or not joint_mask[j]:
        return
    lo = joint_pos_limits[i, j][0]
    hi = joint_pos_limits[i, j][1]
    if (default_joint_pos[i, j] < lo) or (default_joint_pos[i, j] > hi):
        wp.atomic_add(clamped_count, 0, 1)
        default_joint_pos[i, j] = wp.clamp(default_joint_pos[i, j], lo, hi)
    soft_joint_pos_limits[i, j] = compute_soft_joint_pos_limits_func(joint_pos_limits[i, j], soft_limit_factor)


@wp.kernel
def write_joint_friction_data_to_buffer_index(
    in_static: wp.array2d(dtype=wp.float32),
    in_dynamic: wp.array2d(dtype=wp.float32),
    in_viscous: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=wp.int32),
    joint_ids: wp.array(dtype=wp.int32),
    out_buffer: wp.array3d(dtype=wp.float32),
):
    """Conditionally update the static / dynamic / viscous slots of the friction buffer.

    Mirrors :func:`isaaclab_physx.assets.articulation.kernels.write_joint_friction_data_to_buffer`:
    each of the three input arrays is optional (``None`` translates to a null pointer
    which evaluates ``False`` inside the kernel), so callers can update any subset
    of the friction components without disturbing the others.

    Args:
        in_static: Static friction coefficients, or ``None`` to leave that component
            unchanged. Shape is (num_selected_envs, num_selected_joints).
        in_dynamic: Dynamic friction coefficients, or ``None``. Same shape as
            :paramref:`in_static`.
        in_viscous: Viscous friction coefficients [N·s/m or N·m·s/rad, depending on
            joint type], or ``None``. Same shape as :paramref:`in_static`.
        env_ids: Environment indices to write. Shape is (num_selected_envs,).
        joint_ids: Joint indices to write. Shape is (num_selected_joints,).
        out_buffer: Combined friction buffer. Shape is (num_envs, num_joints, 3) with
            slots [0] static, [1] dynamic, [2] viscous.
    """
    i, j = wp.tid()
    if in_static:
        out_buffer[env_ids[i], joint_ids[j], 0] = in_static[i, j]
    if in_dynamic:
        out_buffer[env_ids[i], joint_ids[j], 1] = in_dynamic[i, j]
    if in_viscous:
        out_buffer[env_ids[i], joint_ids[j], 2] = in_viscous[i, j]


@wp.kernel
def write_joint_friction_data_to_buffer_mask(
    in_static: wp.array2d(dtype=wp.float32),
    in_dynamic: wp.array2d(dtype=wp.float32),
    in_viscous: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    joint_mask: wp.array(dtype=wp.bool),
    out_buffer: wp.array3d(dtype=wp.float32),
):
    """Mask variant of :func:`write_joint_friction_data_to_buffer_index`.

    Args:
        in_static: Static friction coefficients, or ``None`` to leave that component
            unchanged. Shape is (num_envs, num_joints).
        in_dynamic: Dynamic friction coefficients, or ``None``. Same shape as
            :paramref:`in_static`.
        in_viscous: Viscous friction coefficients [N·s/m or N·m·s/rad, depending on
            joint type], or ``None``. Same shape as :paramref:`in_static`.
        env_mask: Boolean mask over environments. Shape is (num_envs,).
        joint_mask: Boolean mask over joints. Shape is (num_joints,).
        out_buffer: Combined friction buffer. Shape is (num_envs, num_joints, 3) with
            slots [0] static, [1] dynamic, [2] viscous.
    """
    i, j = wp.tid()
    if not env_mask[i] or not joint_mask[j]:
        return
    if in_static:
        out_buffer[i, j, 0] = in_static[i, j]
    if in_dynamic:
        out_buffer[i, j, 1] = in_dynamic[i, j]
    if in_viscous:
        out_buffer[i, j, 2] = in_viscous[i, j]
