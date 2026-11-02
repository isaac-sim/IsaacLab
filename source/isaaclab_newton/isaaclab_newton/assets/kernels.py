# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import warp as wp

from isaaclab.utils.warp.index_kernel import IndexKernelDispatcher
from isaaclab.utils.warp.math import velocity_at_offset

if TYPE_CHECKING:
    import torch


"""
Shared @wp.func helpers.
"""


@wp.func
def pack_body_wrench_to_world(
    force_b: wp.vec3f,
    torque_b: wp.vec3f,
    body_rot_w: wp.quatf,
) -> wp.spatial_vectorf:
    """Rotate a body-frame COM wrench into world frame and pack it."""
    return wp.spatial_vector(
        wp.quat_rotate(body_rot_w, force_b),
        wp.quat_rotate(body_rot_w, torque_b),
        wp.float32,
    )


@wp.func
def get_link_vel_from_root_com_vel_func(
    com_vel: wp.spatial_vectorf,
    link_pose: wp.transformf,
    body_com_pos_b: wp.vec3f,
):
    """Compute link velocity from center-of-mass velocity.

    Transforms a COM spatial velocity into a link-frame velocity by projecting
    the angular velocity contribution from the COM offset relative to the link frame.

    Args:
        com_vel: COM spatial velocity (linear, angular).
        link_pose: Link pose in world frame.
        body_com_pos_b: COM position in body (link) frame.

    Returns:
        Link spatial velocity (linear, angular).
    """
    offset = wp.quat_rotate(wp.transform_get_rotation(link_pose), -body_com_pos_b)
    return wp.spatial_vector(velocity_at_offset(com_vel, offset), wp.spatial_bottom(com_vel))


@wp.func
def get_com_pose_from_link_pose_func(
    link_pose: wp.transformf,
    body_com_pos_b: wp.vec3f,
):
    """Compute COM pose in world frame from link pose and body-frame COM offset.

    Args:
        link_pose: Link pose in world frame.
        body_com_pos_b: COM position in body (link) frame.

    Returns:
        COM pose in world frame.
    """
    return link_pose * wp.transformf(body_com_pos_b, wp.quatf(0.0, 0.0, 0.0, 1.0))


@wp.func
def get_link_velocity_in_com_frame_func(
    link_velocity_w: wp.spatial_vectorf,
    link_pose_w: wp.transformf,
    body_com_pos_b: wp.vec3f,
):
    """Compute COM velocity from link velocity by accounting for the COM offset.

    Transforms a link-frame spatial velocity into a COM-frame velocity by adding
    the cross-product contribution of the COM offset rotated into the world frame.

    Args:
        link_velocity_w: Link spatial velocity in world frame (linear, angular).
        link_pose_w: Link pose in world frame.
        body_com_pos_b: COM position in body (link) frame.

    Returns:
        COM spatial velocity in world frame (linear, angular).
    """
    offset = wp.quat_rotate(wp.transform_get_rotation(link_pose_w), body_com_pos_b)
    return wp.spatial_vector(velocity_at_offset(link_velocity_w, offset), wp.spatial_bottom(link_velocity_w))


@wp.func
def get_com_pose_in_link_frame_func(
    com_pose_w: wp.transformf,
    com_pos_b: wp.vec3f,
):
    """Compute link pose in world frame from COM pose by inverting the body-frame COM offset.

    This is the inverse of ``get_com_pose_from_link_pose_func``. Given the COM pose in
    world frame and the COM position offset in body frame, it recovers the link pose in
    world frame. Newton COM always has identity orientation, so only position is needed.

    Args:
        com_pose_w: COM pose in world frame.
        com_pos_b: COM position in body (link) frame.

    Returns:
        Link pose in world frame.
    """
    com_rot_w = wp.transform_get_rotation(com_pose_w)
    link_pos_w = wp.transform_get_translation(com_pose_w) - wp.quat_rotate(com_rot_w, com_pos_b)
    return wp.transform(link_pos_w, com_rot_w)


"""
Root-level @wp.kernel (1D — used by RigidObject + Articulation).
"""


@wp.kernel
def get_root_link_vel_from_root_com_vel(
    com_vel: wp.array(dtype=wp.spatial_vectorf),
    link_pose: wp.array(dtype=wp.transformf),
    body_com_pos_b: wp.array2d(dtype=wp.vec3f),
    link_vel: wp.array(dtype=wp.spatial_vectorf),
):
    """Compute root link velocity from root center-of-mass velocity.

    This kernel transforms the root COM velocity into link-frame velocity by projecting
    the angular velocity contribution from the COM offset.

    Args:
        com_vel: Input array of root COM spatial velocities. Shape is (num_envs,).
        link_pose: Input array of root link poses in world frame. Shape is (num_envs,).
        body_com_pos_b: Input array of body COM positions in body frame. Shape is (num_envs, num_bodies).
            Only the first body (index 0) is used for the root.
        link_vel: Output array where root link velocities are written. Shape is (num_envs,).
    """
    i = wp.tid()
    link_vel[i] = get_link_vel_from_root_com_vel_func(com_vel[i], link_pose[i], body_com_pos_b[i, 0])


@wp.kernel
def get_root_com_pose_from_root_link_pose(
    link_pose: wp.array(dtype=wp.transformf),
    body_com_pos_b: wp.array2d(dtype=wp.vec3f),
    com_pose_w: wp.array(dtype=wp.transformf),
):
    """Compute root COM pose from root link pose.

    This kernel transforms the root link pose to the root COM pose using the body COM offset.

    Args:
        link_pose: Input array of root link poses in world frame. Shape is (num_envs,).
        body_com_pos_b: Input array of body COM positions in body frame. Shape is (num_envs, num_bodies).
            Only the first body (index 0) is used for the root.
        com_pose_w: Output array where root COM poses are written. Shape is (num_envs,).
    """
    i = wp.tid()
    com_pose_w[i] = get_com_pose_from_link_pose_func(link_pose[i], body_com_pos_b[i, 0])


"""
Body-level @wp.kernel (2D — used by Articulation + RigidObjectCollection).
"""


@wp.kernel
def get_body_link_vel_from_body_com_vel(
    body_com_vel: wp.array2d(dtype=wp.spatial_vectorf),
    body_link_pose: wp.array2d(dtype=wp.transformf),
    body_com_pos_b: wp.array2d(dtype=wp.vec3f),
    body_link_vel: wp.array2d(dtype=wp.spatial_vectorf),
):
    """Compute body link velocities from body COM velocities for all bodies.

    This kernel transforms COM velocities into link-frame velocities by projecting
    the angular velocity contribution from the COM offset, for each body in each environment.

    Args:
        body_com_vel: Input array of body COM spatial velocities. Shape is (num_envs, num_bodies).
        body_link_pose: Input array of body link poses in world frame. Shape is (num_envs, num_bodies).
        body_com_pos_b: Input array of body COM positions in body frame. Shape is (num_envs, num_bodies).
        body_link_vel: Output array where body link velocities are written. Shape is (num_envs, num_bodies).
    """
    i, j = wp.tid()
    body_link_vel[i, j] = get_link_vel_from_root_com_vel_func(
        body_com_vel[i, j], body_link_pose[i, j], body_com_pos_b[i, j]
    )


@wp.kernel
def get_body_com_pose_from_body_link_pose(
    body_link_pose: wp.array2d(dtype=wp.transformf),
    body_com_pos_b: wp.array2d(dtype=wp.vec3f),
    body_com_pose_w: wp.array2d(dtype=wp.transformf),
):
    """Compute body COM poses from body link poses for all bodies.

    This kernel transforms link poses to COM poses using the body COM offset in the body frame.

    Args:
        body_link_pose: Input array of body link poses in world frame. Shape is (num_envs, num_bodies).
        body_com_pos_b: Input array of body COM positions in body frame. Shape is (num_envs, num_bodies).
        body_com_pose_w: Output array where body COM poses in world frame are written.
            Shape is (num_envs, num_bodies).
    """
    i, j = wp.tid()
    body_com_pose_w[i, j] = get_com_pose_from_link_pose_func(body_link_pose[i, j], body_com_pos_b[i, j])


"""
Derived property kernels.
"""


@wp.kernel
def projected_gravity_b_kernel(
    gravity_w: wp.array(dtype=wp.vec3f),
    quat: wp.array(dtype=wp.quatf),
    projected_gravity_b: wp.array(dtype=wp.vec3f),
):
    """Project per-env world-frame gravity into the base frame as a unit vector.

    ``gravity_w`` carries magnitude (m/s^2) and is normalized internally, so the
    result stays direction-only under per-env gravity randomization.
    ``wp.normalize`` maps a zero vector to zero, so disabled gravity is safe.

    Args:
        gravity_w: World-frame gravity vector per env. Shape is (num_envs,).
        quat: Per-env body quaternion. Shape is (num_envs,).
        projected_gravity_b: Output unit-vector projection. Shape is (num_envs,).
    """
    i = wp.tid()
    projected_gravity_b[i] = wp.quat_rotate_inv(quat[i], wp.normalize(gravity_w[i]))


@wp.kernel
def projected_gravity_b_2D_kernel(
    gravity_w: wp.array(dtype=wp.vec3f),
    quat: wp.array2d(dtype=wp.quatf),
    projected_gravity_b: wp.array2d(dtype=wp.vec3f),
):
    """Project per-env world-frame gravity into per-body base frame as a unit vector.

    Broadcasts the per-env ``gravity_w[i]`` across all bodies of env ``i``.

    Args:
        gravity_w: World-frame gravity vector per env. Shape is (num_envs,).
        quat: Per-body quaternion. Shape is (num_envs, num_bodies).
        projected_gravity_b: Output unit-vector projection. Shape is (num_envs, num_bodies).
    """
    i, j = wp.tid()
    projected_gravity_b[i, j] = wp.quat_rotate_inv(quat[i, j], wp.normalize(gravity_w[i]))


"""
Root-level write kernels (1D — used by RigidObject + Articulation).
"""


@wp.kernel
def set_root_link_pose_to_sim_index(
    data: wp.array(dtype=wp.transformf),
    env_ids: wp.array(dtype=Any),
    root_link_pose_w: wp.array(dtype=wp.transformf),
):
    """Write root link pose data to simulation buffers.

    This kernel writes root link poses from the input array to the output buffer.

    Args:
        data: Input array of root link poses. Shape is (num_selected_envs,).
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        root_link_pose_w: Output array where root link poses are written. Shape is (num_envs,).
    """
    i = wp.tid()
    env_id = wp.int32(env_ids[i])
    root_link_pose_w[env_id] = data[i]


@wp.kernel
def set_root_link_pose_to_sim_mask(
    data: wp.array(dtype=wp.transformf),
    env_mask: wp.array(dtype=wp.bool),
    root_link_pose_w: wp.array(dtype=wp.transformf),
):
    """Write root link pose data to simulation buffers.

    This kernel writes root link poses from the input array to the output buffer.

    Args:
        data: Input array of root link poses. Shape is (num_instances,).
        env_mask: Input array of environment mask. Shape is (num_instances,).
        root_link_pose_w: Output array where root link poses are written. Shape is (num_envs,).
    """
    i = wp.tid()
    if env_mask[i]:
        root_link_pose_w[i] = data[i]


@wp.kernel
def set_root_com_pose_to_sim_index(
    data: wp.array(dtype=wp.transformf),
    body_com_pos_b: wp.array2d(dtype=wp.vec3f),
    env_ids: wp.array(dtype=Any),
    root_com_pose_w: wp.array(dtype=wp.transformf),
    root_link_pose_w: wp.array(dtype=wp.transformf),
):
    """Write root COM pose data to simulation buffers.

    This kernel writes root COM poses from the input array to the output buffers
    and computes the corresponding link pose from the COM pose.

    Args:
        data: Input array of root COM poses. Shape is (num_selected_envs,).
        body_com_pos_b: Input array of body COM positions in body frame. Shape is
            (num_envs, num_bodies). Only the first body (index 0) is used for the root.
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        root_com_pose_w: Output array where root COM poses are written. Shape is (num_envs,).
        root_link_pose_w: Output array where root link poses (derived from COM) are written.
            Shape is (num_envs,).
    """
    i = wp.tid()
    env_id = wp.int32(env_ids[i])
    root_com_pose_w[env_id] = data[i]
    # Get the com pose in the link frame
    root_link_pose_w[env_id] = get_com_pose_in_link_frame_func(root_com_pose_w[env_id], body_com_pos_b[env_id, 0])


@wp.kernel
def set_root_com_pose_to_sim_mask(
    data: wp.array(dtype=wp.transformf),
    body_com_pos_b: wp.array2d(dtype=wp.vec3f),
    env_mask: wp.array(dtype=wp.bool),
    root_com_pose_w: wp.array(dtype=wp.transformf),
    root_link_pose_w: wp.array(dtype=wp.transformf),
):
    """Write root COM pose data to simulation buffers.

    This kernel writes root COM poses from the input array to the output buffers
    and computes the corresponding link pose from the COM pose.

    Args:
        data: Input array of root COM poses. Shape is (num_instances,).
        body_com_pos_b: Input array of body COM positions in body frame. Shape is
            (num_envs, num_bodies). Only the first body (index 0) is used for the root.
        env_mask: Input array of environment mask. Shape is (num_instances,).
        root_com_pose_w: Output array where root COM poses are written. Shape is (num_envs,).
        root_link_pose_w: Output array where root link poses (derived from COM) are written.
            Shape is (num_envs,).
    """
    i = wp.tid()
    if env_mask[i]:
        root_com_pose_w[i] = data[i]
        # Get the com pose in the link frame
        root_link_pose_w[i] = get_com_pose_in_link_frame_func(root_com_pose_w[i], body_com_pos_b[i, 0])


@wp.kernel
def set_root_com_velocity_to_sim_index(
    data: wp.array(dtype=wp.spatial_vectorf),
    env_ids: wp.array(dtype=Any),
    num_bodies: wp.int32,
    root_com_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    body_acc_w: wp.array2d(dtype=wp.spatial_vectorf),
):
    """Write root COM velocity data to simulation buffers.

    This kernel writes root COM velocities from the input array to the output buffers
    and zeros out the body acceleration buffer to prevent reporting stale values.

    Args:
        data: Input array of root COM spatial velocities. Shape is (num_selected_envs,).
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        num_bodies: Input scalar number of bodies per environment.
        root_com_velocity_w: Output array where root COM velocities are written. Shape is (num_envs,).
        body_acc_w: Output array where body accelerations are zeroed. Shape is
            (num_envs, num_bodies).
    """
    i = wp.tid()
    env_id = wp.int32(env_ids[i])
    root_com_velocity_w[env_id] = data[i]
    # Make the acceleration zero to prevent reporting old values
    for j in range(num_bodies):
        body_acc_w[env_id, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def set_root_com_velocity_to_sim_mask(
    data: wp.array(dtype=wp.spatial_vectorf),
    env_mask: wp.array(dtype=wp.bool),
    num_bodies: wp.int32,
    root_com_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    body_acc_w: wp.array2d(dtype=wp.spatial_vectorf),
):
    """Write root COM velocity data to simulation buffers.

    This kernel writes root COM velocities from the input array to the output buffers
    and zeros out the body acceleration buffer to prevent reporting stale values.

    Args:
        data: Input array of root COM spatial velocities. Shape is (num_instances,).
        env_mask: Input array of environment mask. Shape is (num_instances,).
        num_bodies: Input scalar number of bodies per environment.
        root_com_velocity_w: Output array where root COM velocities are written. Shape is (num_envs,).
        body_acc_w: Output array where body accelerations are zeroed. Shape is
            (num_envs, num_bodies).
    """
    i = wp.tid()
    if env_mask[i]:
        root_com_velocity_w[i] = data[i]
        # Make the acceleration zero to prevent reporting old values
        for j in range(num_bodies):
            body_acc_w[i, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def set_root_link_velocity_to_sim_index(
    data: wp.array(dtype=wp.spatial_vectorf),
    body_com_pos_b: wp.array2d(dtype=wp.vec3f),
    link_pose_w: wp.array(dtype=wp.transformf),
    env_ids: wp.array(dtype=Any),
    num_bodies: wp.int32,
    root_link_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    root_com_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    body_acc_w: wp.array2d(dtype=wp.spatial_vectorf),
):
    """Write root link velocity data to simulation buffers.

    This kernel writes root link velocities from the input array to the output buffers,
    computes the corresponding COM velocity from the link velocity, and zeros out
    the body acceleration buffer.

    Args:
        data: Input array of root link spatial velocities. Shape is (num_selected_envs,).
        body_com_pos_b: Input array of body COM positions in body frame. Shape is
            (num_envs, num_bodies). Only the first body (index 0) is used for the root.
        link_pose_w: Input array of root link poses in world frame. Shape is (num_envs,).
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        num_bodies: Input scalar number of bodies per environment.
        root_link_velocity_w: Output array where root link velocities are written.
            Shape is (num_envs,).
        root_com_velocity_w: Output array where root COM velocities (derived from link)
            are written. Shape is (num_envs,).
        body_acc_w: Output array where body accelerations are zeroed.
            Shape is (num_envs, num_bodies).
    """
    i = wp.tid()
    env_id = wp.int32(env_ids[i])
    root_link_velocity_w[env_id] = data[i]
    # Get the link velocity in the com frame
    root_com_velocity_w[env_id] = get_link_velocity_in_com_frame_func(
        root_link_velocity_w[env_id], link_pose_w[env_id], body_com_pos_b[env_id, 0]
    )
    # Make the acceleration zero to prevent reporting old values
    for j in range(num_bodies):
        body_acc_w[env_id, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def set_root_link_velocity_to_sim_mask(
    data: wp.array(dtype=wp.spatial_vectorf),
    body_com_pos_b: wp.array2d(dtype=wp.vec3f),
    link_pose_w: wp.array(dtype=wp.transformf),
    env_mask: wp.array(dtype=wp.bool),
    num_bodies: wp.int32,
    root_link_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    root_com_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    body_acc_w: wp.array2d(dtype=wp.spatial_vectorf),
):
    """Write root link velocity data to simulation buffers.

    This kernel writes root link velocities from the input array to the output buffers,
    computes the corresponding COM velocity from the link velocity, and zeros out
    the body acceleration buffer.

    Args:
        data: Input array of root link spatial velocities. Shape is (num_instances,).
        body_com_pos_b: Input array of body COM positions in body frame. Shape is
            (num_envs, num_bodies). Only the first body (index 0) is used for the root.
        link_pose_w: Input array of root link poses in world frame. Shape is (num_envs,).
        env_mask: Input array of environment mask. Shape is (num_instances,).
        num_bodies: Input scalar number of bodies per environment.
        root_link_velocity_w: Output array where root link velocities are written.
            Shape is (num_envs,).
        root_com_velocity_w: Output array where root COM velocities (derived from link)
            are written. Shape is (num_envs,).
        body_acc_w: Output array where body accelerations are zeroed.
            Shape is (num_envs, num_bodies).
    """
    i = wp.tid()
    if env_mask[i]:
        root_link_velocity_w[i] = data[i]
        # Get the link velocity in the com frame
        root_com_velocity_w[i] = get_link_velocity_in_com_frame_func(
            root_link_velocity_w[i], link_pose_w[i], body_com_pos_b[i, 0]
        )
        # Make the acceleration zero to prevent reporting old values
        for j in range(num_bodies):
            body_acc_w[i, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


_SET_ROOT_LINK_POSE_TO_SIM_INDEX_DISPATCHER = IndexKernelDispatcher(set_root_link_pose_to_sim_index, ("env_ids",))
_SET_ROOT_COM_POSE_TO_SIM_INDEX_DISPATCHER = IndexKernelDispatcher(set_root_com_pose_to_sim_index, ("env_ids",))
_SET_ROOT_COM_VELOCITY_TO_SIM_INDEX_DISPATCHER = IndexKernelDispatcher(set_root_com_velocity_to_sim_index, ("env_ids",))
_SET_ROOT_LINK_VELOCITY_TO_SIM_INDEX_DISPATCHER = IndexKernelDispatcher(
    set_root_link_velocity_to_sim_index, ("env_ids",)
)


def set_root_link_pose_to_sim_index_kernel(env_ids: wp.array | torch.Tensor) -> wp.Kernel:
    """Select the root-link pose writer matching the environment selector dtype."""
    return _SET_ROOT_LINK_POSE_TO_SIM_INDEX_DISPATCHER.select(env_ids)


def set_root_com_pose_to_sim_index_kernel(env_ids: wp.array | torch.Tensor) -> wp.Kernel:
    """Select the root-COM pose writer matching the environment selector dtype."""
    return _SET_ROOT_COM_POSE_TO_SIM_INDEX_DISPATCHER.select(env_ids)


def set_root_com_velocity_to_sim_index_kernel(env_ids: wp.array | torch.Tensor) -> wp.Kernel:
    """Select the root-COM velocity writer matching the environment selector dtype."""
    return _SET_ROOT_COM_VELOCITY_TO_SIM_INDEX_DISPATCHER.select(env_ids)


def set_root_link_velocity_to_sim_index_kernel(env_ids: wp.array | torch.Tensor) -> wp.Kernel:
    """Select the root-link velocity writer matching the environment selector dtype."""
    return _SET_ROOT_LINK_VELOCITY_TO_SIM_INDEX_DISPATCHER.select(env_ids)


"""
Body-level write kernels (2D — used by RigidObjectCollection).
"""


@wp.kernel
def set_body_link_pose_to_sim(
    data: wp.array2d(dtype=wp.transformf),
    env_ids: wp.array(dtype=Any),
    body_ids: wp.array(dtype=Any),
    from_mask: bool,
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
):
    """Write body link pose data to simulation buffers.

    This kernel writes body link poses from the input array to the output buffers
    and optionally updates the corresponding state vectors, for each body in each environment.

    Args:
        data: Input array of body link poses. Shape is (num_envs, num_bodies) or
            (num_selected_envs, num_selected_bodies) depending on from_mask.
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        body_ids: Input array of body indices to write to. Shape is (num_selected_bodies,).
        from_mask: Input flag indicating whether to use masked indexing.
        body_link_pose_w: Output array where body link poses are written.
            Shape is (num_envs, num_bodies).
    """
    i, j = wp.tid()
    env_id = wp.int32(env_ids[i])
    body_id = wp.int32(body_ids[j])
    if from_mask:
        body_link_pose_w[env_id, body_id] = data[env_id, body_id]
    else:
        body_link_pose_w[env_id, body_id] = data[i, j]


@wp.kernel
def set_body_com_pose_to_sim(
    data: wp.array2d(dtype=wp.transformf),
    body_com_pos_b: wp.array2d(dtype=wp.vec3f),
    env_ids: wp.array(dtype=Any),
    body_ids: wp.array(dtype=Any),
    from_mask: bool,
    body_com_pose_w: wp.array2d(dtype=wp.transformf),
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
):
    """Write body COM pose data to simulation buffers.

    This kernel writes body COM poses from the input array to the output buffers,
    computes the corresponding link poses from the COM poses, and optionally updates
    the corresponding state vectors, for each body in each environment.

    Args:
        data: Input array of body COM poses. Shape is (num_envs, num_bodies) or
            (num_selected_envs, num_selected_bodies) depending on from_mask.
        body_com_pos_b: Input array of body COM positions in body frame. Shape is
            (num_envs, num_bodies).
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        body_ids: Input array of body indices to write to. Shape is (num_selected_bodies,).
        from_mask: Input flag indicating whether to use masked indexing.
        body_com_pose_w: Output array where body COM poses are written.
            Shape is (num_envs, num_bodies).
        body_link_pose_w: Output array where body link poses (derived from COM) are written.
            Shape is (num_envs, num_bodies).
    """
    i, j = wp.tid()
    env_id = wp.int32(env_ids[i])
    body_id = wp.int32(body_ids[j])
    if from_mask:
        body_com_pose_w[env_id, body_id] = data[env_id, body_id]
    else:
        body_com_pose_w[env_id, body_id] = data[i, j]
    # Get the link pose from com pose
    body_link_pose_w[env_id, body_id] = get_com_pose_in_link_frame_func(
        body_com_pose_w[env_id, body_id], body_com_pos_b[env_id, body_id]
    )


@wp.kernel
def set_body_com_velocity_to_sim(
    data: wp.array2d(dtype=wp.spatial_vectorf),
    env_ids: wp.array(dtype=Any),
    body_ids: wp.array(dtype=Any),
    from_mask: bool,
    body_com_velocity_w: wp.array2d(dtype=wp.spatial_vectorf),
    body_acc_w: wp.array2d(dtype=wp.spatial_vectorf),
):
    """Write body COM velocity data to simulation buffers.

    This kernel writes body COM velocities from the input array to the output buffers,
    optionally updates the corresponding state vectors, and zeros out the body
    acceleration buffer, for each body in each environment.

    Args:
        data: Input array of body COM spatial velocities. Shape is (num_envs, num_bodies) or
            (num_selected_envs, num_selected_bodies) depending on from_mask.
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        body_ids: Input array of body indices to write to. Shape is (num_selected_bodies,).
        from_mask: Input flag indicating whether to use masked indexing.
        body_com_velocity_w: Output array where body COM velocities are written.
            Shape is (num_envs, num_bodies).
        body_acc_w: Output array where body accelerations are zeroed.
            Shape is (num_envs, num_bodies).
    """
    i, j = wp.tid()
    env_id = wp.int32(env_ids[i])
    body_id = wp.int32(body_ids[j])
    if from_mask:
        body_com_velocity_w[env_id, body_id] = data[env_id, body_id]
    else:
        body_com_velocity_w[env_id, body_id] = data[i, j]
    # Make the acceleration zero to prevent reporting old values
    body_acc_w[env_id, body_id] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def set_body_link_velocity_to_sim(
    data: wp.array2d(dtype=wp.spatial_vectorf),
    body_com_pos_b: wp.array2d(dtype=wp.vec3f),
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    env_ids: wp.array(dtype=Any),
    body_ids: wp.array(dtype=Any),
    from_mask: bool,
    body_link_velocity_w: wp.array2d(dtype=wp.spatial_vectorf),
    body_com_velocity_w: wp.array2d(dtype=wp.spatial_vectorf),
    body_acc_w: wp.array2d(dtype=wp.spatial_vectorf),
):
    """Write body link velocity data to simulation buffers.

    This kernel writes body link velocities from the input array to the output buffers,
    computes the corresponding COM velocities from the link velocities, optionally updates
    the corresponding state vectors, and zeros out the body acceleration buffer.

    Args:
        data: Input array of body link spatial velocities. Shape is (num_envs, num_bodies)
            or (num_selected_envs, num_selected_bodies) depending on from_mask.
        body_com_pos_b: Input array of body COM positions in body frame. Shape is
            (num_envs, num_bodies).
        body_link_pose_w: Input array of body link poses in world frame. Shape is
            (num_envs, num_bodies).
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        body_ids: Input array of body indices to write to. Shape is (num_selected_bodies,).
        from_mask: Input flag indicating whether to use masked indexing.
        body_link_velocity_w: Output array where body link velocities are written.
            Shape is (num_envs, num_bodies).
        body_com_velocity_w: Output array where body COM velocities (derived from link)
            are written. Shape is (num_envs, num_bodies).
        body_acc_w: Output array where body accelerations are zeroed.
            Shape is (num_envs, num_bodies).
    """
    i, j = wp.tid()
    env_id = wp.int32(env_ids[i])
    body_id = wp.int32(body_ids[j])
    if from_mask:
        body_link_velocity_w[env_id, body_id] = data[env_id, body_id]
    else:
        body_link_velocity_w[env_id, body_id] = data[i, j]
    # Get the link velocity in the com frame
    body_com_velocity_w[env_id, body_id] = get_link_velocity_in_com_frame_func(
        body_link_velocity_w[env_id, body_id],
        body_link_pose_w[env_id, body_id],
        body_com_pos_b[env_id, body_id],
    )
    # Make the acceleration zero to prevent reporting old values
    body_acc_w[env_id, body_id] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


_SET_BODY_LINK_POSE_TO_SIM_DISPATCHER = IndexKernelDispatcher(set_body_link_pose_to_sim, ("env_ids", "body_ids"))
_SET_BODY_COM_POSE_TO_SIM_DISPATCHER = IndexKernelDispatcher(set_body_com_pose_to_sim, ("env_ids", "body_ids"))
_SET_BODY_COM_VELOCITY_TO_SIM_DISPATCHER = IndexKernelDispatcher(set_body_com_velocity_to_sim, ("env_ids", "body_ids"))
_SET_BODY_LINK_VELOCITY_TO_SIM_DISPATCHER = IndexKernelDispatcher(
    set_body_link_velocity_to_sim, ("env_ids", "body_ids")
)


def set_body_link_pose_to_sim_kernel(env_ids: wp.array | torch.Tensor, body_ids: wp.array | torch.Tensor) -> wp.Kernel:
    """Select the body-link pose writer for the selector dtypes."""
    return _SET_BODY_LINK_POSE_TO_SIM_DISPATCHER.select(env_ids, body_ids)


def set_body_com_pose_to_sim_kernel(env_ids: wp.array | torch.Tensor, body_ids: wp.array | torch.Tensor) -> wp.Kernel:
    """Select the body-COM pose writer for the selector dtypes."""
    return _SET_BODY_COM_POSE_TO_SIM_DISPATCHER.select(env_ids, body_ids)


def set_body_com_velocity_to_sim_kernel(
    env_ids: wp.array | torch.Tensor, body_ids: wp.array | torch.Tensor
) -> wp.Kernel:
    """Select the body-COM velocity writer for the selector dtypes."""
    return _SET_BODY_COM_VELOCITY_TO_SIM_DISPATCHER.select(env_ids, body_ids)


def set_body_link_velocity_to_sim_kernel(
    env_ids: wp.array | torch.Tensor, body_ids: wp.array | torch.Tensor
) -> wp.Kernel:
    """Select the body-link velocity writer for the selector dtypes."""
    return _SET_BODY_LINK_VELOCITY_TO_SIM_DISPATCHER.select(env_ids, body_ids)


"""
Generic buffer-writing kernels (used by Articulation + RigidObject + RigidObjectCollection).
"""


@wp.func
def _update_body_inertial_properties_inverse(
    env_id: int,
    body_id: int,
    body_mass: wp.array2d(dtype=wp.float32),
    body_inertia: wp.array3d(dtype=wp.float32),
    update_inv_mass: bool,
    body_inv_mass: wp.array2d(dtype=wp.float32),
    body_inv_inertia: wp.array2d(dtype=wp.mat33f),
):
    """Update inverse inertial properties for one resolved backend body."""
    mass = body_mass[env_id, body_id]
    if mass > 0.0:
        if update_inv_mass:
            body_inv_mass[env_id, body_id] = 1.0 / mass
        inertia = wp.mat33f(
            body_inertia[env_id, body_id, 0],
            body_inertia[env_id, body_id, 1],
            body_inertia[env_id, body_id, 2],
            body_inertia[env_id, body_id, 3],
            body_inertia[env_id, body_id, 4],
            body_inertia[env_id, body_id, 5],
            body_inertia[env_id, body_id, 6],
            body_inertia[env_id, body_id, 7],
            body_inertia[env_id, body_id, 8],
        )
        body_inv_inertia[env_id, body_id] = wp.inverse(inertia)
    else:
        if update_inv_mass:
            body_inv_mass[env_id, body_id] = 0.0
        body_inv_inertia[env_id, body_id] = wp.mat33f(0.0)


@wp.kernel
def write_body_mass_and_inverse_index(
    masses: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=Any),
    body_ids: wp.array(dtype=Any),
    body_user_to_backend: wp.array(dtype=wp.int32),
    has_body_ordering: bool,
    body_inertia_backend: wp.array3d(dtype=wp.float32),
    body_mass_user: wp.array2d(dtype=wp.float32),
    body_mass_backend: wp.array2d(dtype=wp.float32),
    body_inv_mass: wp.array2d(dtype=wp.float32),
    body_inv_inertia: wp.array2d(dtype=wp.mat33f),
):
    """Write selected body masses and update Newton inverse inertial properties.

    Args:
        masses: Selected body masses. Shape is (num_selected_envs, num_selected_bodies).
        env_ids: Selected environment indices. Shape is (num_selected_envs,).
        body_ids: Selected public body indices. Shape is (num_selected_bodies,).
        body_user_to_backend: Public-to-backend body index map. Shape is (num_bodies,).
        has_body_ordering: Whether to apply the public-to-backend body index map.
        body_inertia_backend: Backend-order body inertias. Shape is (num_envs, num_bodies, 9).
        body_mass_user: Public-order body masses. Shape is (num_envs, num_bodies).
        body_mass_backend: Backend-order body masses. Shape is (num_envs, num_bodies).
        body_inv_mass: Backend-order inverse body masses. Shape is (num_envs, num_bodies).
        body_inv_inertia: Backend-order inverse body inertias. Shape is (num_envs, num_bodies).
    """
    i, j = wp.tid()
    env_id = wp.int32(env_ids[i])
    user_body_id = wp.int32(body_ids[j])
    backend_body_id = user_body_id
    if has_body_ordering:
        backend_body_id = body_user_to_backend[user_body_id]
    mass = masses[i, j]
    body_mass_user[env_id, user_body_id] = mass
    if has_body_ordering:
        body_mass_backend[env_id, backend_body_id] = mass
    _update_body_inertial_properties_inverse(
        env_id, backend_body_id, body_mass_backend, body_inertia_backend, True, body_inv_mass, body_inv_inertia
    )


@wp.kernel
def write_body_mass_and_inverse_mask(
    masses: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    body_mask: wp.array(dtype=wp.bool),
    body_user_to_backend: wp.array(dtype=wp.int32),
    has_body_ordering: bool,
    body_inertia_backend: wp.array3d(dtype=wp.float32),
    body_mass_user: wp.array2d(dtype=wp.float32),
    body_mass_backend: wp.array2d(dtype=wp.float32),
    body_inv_mass: wp.array2d(dtype=wp.float32),
    body_inv_inertia: wp.array2d(dtype=wp.mat33f),
):
    """Write masked body masses and update Newton inverse inertial properties.

    Args:
        masses: Full public-order body masses. Shape is (num_envs, num_bodies).
        env_mask: Selected environment mask. Shape is (num_envs,).
        body_mask: Selected public body mask. Shape is (num_bodies,).
        body_user_to_backend: Public-to-backend body index map. Shape is (num_bodies,).
        has_body_ordering: Whether to apply the public-to-backend body index map.
        body_inertia_backend: Backend-order body inertias. Shape is (num_envs, num_bodies, 9).
        body_mass_user: Public-order body masses. Shape is (num_envs, num_bodies).
        body_mass_backend: Backend-order body masses. Shape is (num_envs, num_bodies).
        body_inv_mass: Backend-order inverse body masses. Shape is (num_envs, num_bodies).
        body_inv_inertia: Backend-order inverse body inertias. Shape is (num_envs, num_bodies).
    """
    env_id, user_body_id = wp.tid()
    if not env_mask[env_id] or not body_mask[user_body_id]:
        return
    backend_body_id = user_body_id
    if has_body_ordering:
        backend_body_id = body_user_to_backend[user_body_id]
    mass = masses[env_id, user_body_id]
    body_mass_user[env_id, user_body_id] = mass
    if has_body_ordering:
        body_mass_backend[env_id, backend_body_id] = mass
    _update_body_inertial_properties_inverse(
        env_id, backend_body_id, body_mass_backend, body_inertia_backend, True, body_inv_mass, body_inv_inertia
    )


@wp.kernel
def write_body_inertia_and_inverse_index(
    inertias: wp.array3d(dtype=wp.float32),
    env_ids: wp.array(dtype=Any),
    body_ids: wp.array(dtype=Any),
    body_user_to_backend: wp.array(dtype=wp.int32),
    has_body_ordering: bool,
    body_mass_backend: wp.array2d(dtype=wp.float32),
    body_inertia_user: wp.array3d(dtype=wp.float32),
    body_inertia_backend: wp.array3d(dtype=wp.float32),
    body_inv_mass: wp.array2d(dtype=wp.float32),
    body_inv_inertia: wp.array2d(dtype=wp.mat33f),
):
    """Write selected body inertias and update Newton inverse inertia."""
    i, j = wp.tid()
    env_id = wp.int32(env_ids[i])
    user_body_id = wp.int32(body_ids[j])
    backend_body_id = user_body_id
    if has_body_ordering:
        backend_body_id = body_user_to_backend[user_body_id]
    for k in range(9):
        value = inertias[i, j, k]
        body_inertia_user[env_id, user_body_id, k] = value
        if has_body_ordering:
            body_inertia_backend[env_id, backend_body_id, k] = value
    _update_body_inertial_properties_inverse(
        env_id, backend_body_id, body_mass_backend, body_inertia_backend, False, body_inv_mass, body_inv_inertia
    )


@wp.kernel
def write_body_inertia_and_inverse_mask(
    inertias: wp.array3d(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    body_mask: wp.array(dtype=wp.bool),
    body_user_to_backend: wp.array(dtype=wp.int32),
    has_body_ordering: bool,
    body_mass_backend: wp.array2d(dtype=wp.float32),
    body_inertia_user: wp.array3d(dtype=wp.float32),
    body_inertia_backend: wp.array3d(dtype=wp.float32),
    body_inv_mass: wp.array2d(dtype=wp.float32),
    body_inv_inertia: wp.array2d(dtype=wp.mat33f),
):
    """Write masked body inertias and update Newton inverse inertia."""
    env_id, user_body_id = wp.tid()
    if not env_mask[env_id] or not body_mask[user_body_id]:
        return
    backend_body_id = user_body_id
    if has_body_ordering:
        backend_body_id = body_user_to_backend[user_body_id]
    for k in range(9):
        value = inertias[env_id, user_body_id, k]
        body_inertia_user[env_id, user_body_id, k] = value
        if has_body_ordering:
            body_inertia_backend[env_id, backend_body_id, k] = value
    _update_body_inertial_properties_inverse(
        env_id, backend_body_id, body_mass_backend, body_inertia_backend, False, body_inv_mass, body_inv_inertia
    )


_WRITE_BODY_MASS_AND_INVERSE_INDEX_DISPATCHER = IndexKernelDispatcher(
    write_body_mass_and_inverse_index, ("env_ids", "body_ids")
)
_WRITE_BODY_INERTIA_AND_INVERSE_INDEX_DISPATCHER = IndexKernelDispatcher(
    write_body_inertia_and_inverse_index, ("env_ids", "body_ids")
)


def write_body_mass_and_inverse_index_kernel(
    env_ids: wp.array | torch.Tensor, body_ids: wp.array | torch.Tensor
) -> wp.Kernel:
    """Select the fused body-mass writer for the selector dtypes."""
    return _WRITE_BODY_MASS_AND_INVERSE_INDEX_DISPATCHER.select(env_ids, body_ids)


def write_body_inertia_and_inverse_index_kernel(
    env_ids: wp.array | torch.Tensor, body_ids: wp.array | torch.Tensor
) -> wp.Kernel:
    """Select the fused body-inertia writer for the selector dtypes."""
    return _WRITE_BODY_INERTIA_AND_INVERSE_INDEX_DISPATCHER.select(env_ids, body_ids)


@wp.kernel
def make_dummy_body_com_pose_b(
    body_com_pos_b: wp.array2d(dtype=wp.vec3f),
    body_com_pose_b: wp.array2d(dtype=wp.transformf),
):
    """Make a body COM pose from position by appending an identity quaternion.

    Needed by the ``body_com_pose_b`` property to match the base API that returns
    ``wp.transformf`` (pos + quat).

    Args:
        body_com_pos_b: Input array of body COM positions in body frame. Shape is (num_envs, num_bodies).
        body_com_pose_b: Output array where body COM poses are written. Shape is (num_envs, num_bodies).
    """
    i, j = wp.tid()
    body_com_pose_b[i, j] = wp.transformf(body_com_pos_b[i, j], wp.quatf(0.0, 0.0, 0.0, 1.0))


@wp.kernel
def derive_body_acceleration_from_body_com_velocities(
    body_com_vel: wp.array2d(dtype=wp.spatial_vectorf),
    dt: wp.float32,
    prev_body_com_vel: wp.array2d(dtype=wp.spatial_vectorf),
    body_acc: wp.array2d(dtype=wp.spatial_vectorf),
):
    """Derive body acceleration from body COM velocities.

    This kernel derives body acceleration from body COM velocities using finite differencing.

    Args:
        body_com_vel: Input array of body COM velocities. Shape is (num_envs, num_bodies).
        dt: Input time step (scalar) used for finite differencing.
        prev_body_com_vel: Input/output array of previous body COM velocities. Shape is (num_envs, num_bodies).
        body_acc: Output array where body accelerations are written. Shape is (num_envs, num_bodies).
    """
    i, j = wp.tid()
    # Compute the acceleration
    body_acc[i, j] = (body_com_vel[i, j] - prev_body_com_vel[i, j]) / dt
    # Update the previous body COM velocity
    prev_body_com_vel[i, j] = body_com_vel[i, j]


@wp.kernel
def update_wrench_array_with_force_and_torque(
    forces: wp.array2d(dtype=wp.vec3f),
    torques: wp.array2d(dtype=wp.vec3f),
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
    wrench: wp.array2d(dtype=wp.spatial_vectorf),
    env_ids: wp.array(dtype=wp.bool),
    body_ids: wp.array(dtype=wp.bool),
):
    """Write body-frame COM wrenches into a world-frame destination array.

    Args:
        forces: Body-frame forces at each body center of mass [N], shape
            (num_envs, num_bodies).
        torques: Body-frame torques at each body center of mass [N·m], shape
            (num_envs, num_bodies).
        body_link_pose_w: Public-order body link poses in the world frame,
            shape (num_envs, num_bodies). Only each pose quaternion is used;
            the link translation does not shift the COM-referenced wrench.
        wrench: World-frame force and torque at each body center of mass [N,
            N·m], shape (num_envs, num_bodies).
        env_ids: Environment-selection mask, shape (num_envs,).
        body_ids: Body-selection mask, shape (num_bodies,).
    """
    env_index, body_index = wp.tid()
    if env_ids[env_index] and body_ids[body_index]:
        body_rot_w = wp.transform_get_rotation(body_link_pose_w[env_index, body_index])
        wrench[env_index, body_index] = pack_body_wrench_to_world(
            forces[env_index, body_index],
            torques[env_index, body_index],
            body_rot_w,
        )


def com_positions(coms: float | torch.Tensor | wp.array) -> float | torch.Tensor | wp.array:
    """Return center of mass positions, dropping the orientation when poses are given.

    Newton models a body's center of mass as a position, while the asset API also accepts poses
    (position and quaternion (x, y, z, w)) like the other backends. The orientation is ignored.

    Args:
        coms: Center of mass positions [m] with a trailing dimension of 3 (or dtype ``wp.vec3f``), or poses with a
            trailing dimension of 7 (or dtype ``wp.transformf``).

    Returns:
        The center of mass positions [m]; inputs that are already positions are returned unchanged.
    """
    if isinstance(coms, wp.array) and coms.dtype == wp.transformf:
        coms = wp.to_torch(coms)
    if isinstance(coms, torch.Tensor) and coms.shape[-1] == 7:
        return coms[..., :3].contiguous()
    return coms
