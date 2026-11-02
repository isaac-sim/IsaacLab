# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import torch
import warp as wp

from isaaclab.utils.warp.index_kernel import IndexKernelDispatcher
from isaaclab.utils.warp.math import velocity_at_offset

"""
Shared @wp.func helpers.
"""


@wp.func
def get_link_vel_from_root_com_vel_func(
    com_vel: wp.spatial_vectorf,
    link_pose: wp.transformf,
    body_com_pose: wp.transformf,
):
    """Compute link velocity from center-of-mass velocity.

    Transforms a COM spatial velocity into a link-frame velocity by projecting
    the angular velocity contribution from the COM offset relative to the link frame.

    Args:
        com_vel: COM spatial velocity (linear, angular).
        link_pose: Link pose in world frame.
        body_com_pose: COM pose in body (link) frame.

    Returns:
        Link spatial velocity (linear, angular).
    """
    offset = wp.quat_rotate(wp.transform_get_rotation(link_pose), -wp.transform_get_translation(body_com_pose))
    return wp.spatial_vector(velocity_at_offset(com_vel, offset), wp.spatial_bottom(com_vel))


@wp.func
def get_com_pose_from_link_pose_func(
    link_pose: wp.transformf,
    body_com_pose: wp.transformf,
):
    """Compute COM pose in world frame from link pose and body-frame COM offset.

    Args:
        link_pose: Link pose in world frame.
        body_com_pose: COM pose in body (link) frame.

    Returns:
        COM pose in world frame.
    """
    return link_pose * body_com_pose


@wp.func
def get_link_velocity_in_com_frame_func(
    link_velocity_w: wp.spatial_vectorf,
    link_pose_w: wp.transformf,
    body_com_pose_b: wp.transformf,
):
    """Compute COM velocity from link velocity by accounting for the COM offset.

    Transforms a link-frame spatial velocity into a COM-frame velocity by adding
    the cross-product contribution of the COM offset rotated into the world frame.

    Args:
        link_velocity_w: Link spatial velocity in world frame (linear, angular).
        link_pose_w: Link pose in world frame.
        body_com_pose_b: COM pose in body (link) frame.

    Returns:
        COM spatial velocity in world frame (linear, angular).
    """
    offset = wp.quat_rotate(wp.transform_get_rotation(link_pose_w), wp.transform_get_translation(body_com_pose_b))
    return wp.spatial_vector(velocity_at_offset(link_velocity_w, offset), wp.spatial_bottom(link_velocity_w))


@wp.func
def get_com_pose_in_link_frame_func(
    com_pose_w: wp.transformf,
    com_pose_b: wp.transformf,
):
    """Compute link pose in world frame from COM pose by inverting the body-frame COM offset.

    This is the inverse of ``get_com_pose_from_link_pose_func``. Given the COM pose in
    world frame and the COM offset in body frame, it recovers the link pose in world frame.

    Args:
        com_pose_w: COM pose in world frame.
        com_pose_b: COM pose in body (link) frame.

    Returns:
        Link pose in world frame.
    """
    T2 = wp.transform(
        wp.quat_rotate(
            wp.quat_inverse(wp.transform_get_rotation(com_pose_b)), -wp.transform_get_translation(com_pose_b)
        ),
        wp.quat_inverse(wp.transform_get_rotation(com_pose_b)),
    )
    link_pose_w = com_pose_w * T2
    return link_pose_w


"""
Root-level @wp.kernel (1D — used by RigidObject + Articulation).
"""


@wp.kernel
def get_root_link_vel_from_root_com_vel(
    com_vel: wp.array(dtype=wp.spatial_vectorf),
    link_pose: wp.array(dtype=wp.transformf),
    body_com_pose_b: wp.array2d(dtype=wp.transformf),
    link_vel: wp.array(dtype=wp.spatial_vectorf),
):
    """Compute root link velocity from root center-of-mass velocity.

    This kernel transforms the root COM velocity into link-frame velocity by projecting
    the angular velocity contribution from the COM offset.

    Args:
        com_vel: Input array of root COM spatial velocities. Shape is (num_envs,).
        link_pose: Input array of root link poses in world frame. Shape is (num_envs,).
        body_com_pose_b: Input array of body COM poses in body frame. Shape is (num_envs, num_bodies).
            Only the first body (index 0) is used for the root.
        link_vel: Output array where root link velocities are written. Shape is (num_envs,).
    """
    i = wp.tid()
    link_vel[i] = get_link_vel_from_root_com_vel_func(com_vel[i], link_pose[i], body_com_pose_b[i, 0])


@wp.kernel
def get_root_com_pose_from_root_link_pose(
    link_pose: wp.array(dtype=wp.transformf),
    body_com_pose_b: wp.array2d(dtype=wp.transformf),
    com_pose_w: wp.array(dtype=wp.transformf),
):
    """Compute root COM pose from root link pose.

    This kernel transforms the root link pose to the root COM pose using the body COM offset.

    Args:
        link_pose: Input array of root link poses in world frame. Shape is (num_envs,).
        body_com_pose_b: Input array of body COM poses in body frame. Shape is (num_envs, num_bodies).
            Only the first body (index 0) is used for the root.
        com_pose_w: Output array where root COM poses are written. Shape is (num_envs,).
    """
    i = wp.tid()
    com_pose_w[i] = get_com_pose_from_link_pose_func(link_pose[i], body_com_pose_b[i, 0])


"""
Body-level @wp.kernel (2D — used by Articulation + RigidObjectCollection).
"""


@wp.kernel
def get_body_link_vel_from_body_com_vel(
    body_com_vel: wp.array2d(dtype=wp.spatial_vectorf),
    body_link_pose: wp.array2d(dtype=wp.transformf),
    body_com_pose: wp.array2d(dtype=wp.transformf),
    body_link_vel: wp.array2d(dtype=wp.spatial_vectorf),
):
    """Compute body link velocities from body COM velocities for all bodies.

    This kernel transforms COM velocities into link-frame velocities by projecting
    the angular velocity contribution from the COM offset, for each body in each environment.

    Args:
        body_com_vel: Input array of body COM spatial velocities. Shape is (num_envs, num_bodies).
        body_link_pose: Input array of body link poses in world frame. Shape is (num_envs, num_bodies).
        body_com_pose: Input array of body COM poses in body frame. Shape is (num_envs, num_bodies).
        body_link_vel: Output array where body link velocities are written. Shape is (num_envs, num_bodies).
    """
    i, j = wp.tid()
    body_link_vel[i, j] = get_link_vel_from_root_com_vel_func(
        body_com_vel[i, j], body_link_pose[i, j], body_com_pose[i, j]
    )


@wp.kernel
def get_body_com_pose_from_body_link_pose(
    body_link_pose: wp.array2d(dtype=wp.transformf),
    body_com_pose_b: wp.array2d(dtype=wp.transformf),
    body_com_pose_w: wp.array2d(dtype=wp.transformf),
):
    """Compute body COM poses from body link poses for all bodies.

    This kernel transforms link poses to COM poses using the body COM offset in the body frame.

    Args:
        body_link_pose: Input array of body link poses in world frame. Shape is (num_envs, num_bodies).
        body_com_pose_b: Input array of body COM poses in body frame. Shape is (num_envs, num_bodies).
        body_com_pose_w: Output array where body COM poses in world frame are written.
            Shape is (num_envs, num_bodies).
    """
    i, j = wp.tid()
    body_com_pose_w[i, j] = get_com_pose_from_link_pose_func(body_link_pose[i, j], body_com_pose_b[i, j])


"""
Derived property kernels.
"""


"""
Root-level write kernels (1D — used by RigidObject + Articulation).
"""


@wp.kernel
def set_root_link_pose_to_sim(
    data: wp.array(dtype=wp.transformf),
    env_ids: wp.array(dtype=Any),
    from_mask: bool,
    root_link_pose_w: wp.array(dtype=wp.transformf),
    sim_env_ids: wp.array(dtype=wp.int32),
):
    """Write root link pose data to simulation buffers.

    Args:
        data: Input array of root link poses. Shape is (num_envs,) or (num_selected_envs,)
            depending on from_mask.
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        from_mask: Input flag indicating whether to use masked indexing. If True, env_ids
            are used to index into data. If False, data is indexed sequentially.
        root_link_pose_w: Output array where root link poses are written. Shape is (num_envs,).
        sim_env_ids: Output int32 environment indices for PhysX.
    """
    # If from mask, then we get complete data. Otherwise, we get partial data.
    i = wp.tid()
    env_id = wp.int32(env_ids[i])
    sim_env_ids[i] = env_id
    if from_mask:
        root_link_pose_w[env_id] = data[env_id]
    else:
        root_link_pose_w[env_id] = data[i]


@wp.kernel
def set_root_com_pose_to_sim(
    data: wp.array(dtype=wp.transformf),
    body_com_pose_b: wp.array2d(dtype=wp.transformf),
    env_ids: wp.array(dtype=Any),
    from_mask: bool,
    root_com_pose_w: wp.array(dtype=wp.transformf),
    root_link_pose_w: wp.array(dtype=wp.transformf),
    sim_env_ids: wp.array(dtype=wp.int32),
):
    """Write root COM pose data to simulation buffers.

    This kernel writes root COM poses from the input array to the output buffers
    and computes the corresponding link pose from the COM pose.

    Args:
        data: Input array of root COM poses. Shape is (num_envs,) or (num_selected_envs,)
            depending on from_mask.
        body_com_pose_b: Input array of body COM poses in body frame. Shape is
            (num_envs, num_bodies). Only the first body (index 0) is used for the root.
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        from_mask: Input flag indicating whether to use masked indexing. If True, env_ids
            are used to index into data. If False, data is indexed sequentially.
        root_com_pose_w: Output array where root COM poses are written. Shape is (num_envs,).
        root_link_pose_w: Output array where root link poses (derived from COM) are written.
            Shape is (num_envs,).
        sim_env_ids: Output int32 environment indices for PhysX.
    """
    i = wp.tid()
    env_id = wp.int32(env_ids[i])
    sim_env_ids[i] = env_id
    # If from mask, then we get complete data. Otherwise, we get partial data.
    if from_mask:
        root_com_pose_w[env_id] = data[env_id]
    else:
        root_com_pose_w[env_id] = data[i]
    # Get the com pose in the link frame
    root_link_pose_w[env_id] = get_com_pose_in_link_frame_func(root_com_pose_w[env_id], body_com_pose_b[env_id, 0])


@wp.kernel
def set_root_com_velocity_to_sim(
    data: wp.array(dtype=wp.spatial_vectorf),
    env_ids: wp.array(dtype=Any),
    num_bodies: wp.int32,
    from_mask: bool,
    root_com_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    body_acc_w: wp.array2d(dtype=wp.spatial_vectorf),
    sim_env_ids: wp.array(dtype=wp.int32),
):
    """Write root COM velocity data to simulation buffers.

    This kernel writes root COM velocities from the input array to the output buffers
    and zeros out the body acceleration buffer to prevent reporting stale values.

    Args:
        data: Input array of root COM spatial velocities. Shape is (num_envs,) or
            (num_selected_envs,) depending on from_mask.
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        num_bodies: Input scalar number of bodies per environment.
        from_mask: Input flag indicating whether to use masked indexing. If True, env_ids
            are used to index into data. If False, data is indexed sequentially.
        root_com_velocity_w: Output array where root COM velocities are written. Shape is (num_envs,).
        body_acc_w: Output array where body accelerations are zeroed. Shape is
            (num_envs, num_bodies).
        sim_env_ids: Output int32 environment indices for PhysX.
    """
    i = wp.tid()
    env_id = wp.int32(env_ids[i])
    sim_env_ids[i] = env_id
    # If from mask, then we get complete data. Otherwise, we get partial data.
    if from_mask:
        root_com_velocity_w[env_id] = data[env_id]
    else:
        root_com_velocity_w[env_id] = data[i]
    # Make the acceleration zero to prevent reporting old values
    for j in range(num_bodies):
        body_acc_w[env_id, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def set_root_link_velocity_to_sim(
    data: wp.array(dtype=wp.spatial_vectorf),
    body_com_pose_b: wp.array2d(dtype=wp.transformf),
    link_pose_w: wp.array(dtype=wp.transformf),
    env_ids: wp.array(dtype=Any),
    num_bodies: wp.int32,
    from_mask: bool,
    root_link_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    root_com_velocity_w: wp.array(dtype=wp.spatial_vectorf),
    body_acc_w: wp.array2d(dtype=wp.spatial_vectorf),
    sim_env_ids: wp.array(dtype=wp.int32),
):
    """Write root link velocity data to simulation buffers.

    This kernel writes root link velocities from the input array to the output buffers,
    computes the corresponding COM velocity from the link velocity, and zeros out the
    body acceleration buffer.

    Args:
        data: Input array of root link spatial velocities. Shape is (num_envs,) or
            (num_selected_envs,) depending on from_mask.
        body_com_pose_b: Input array of body COM poses in body frame. Shape is
            (num_envs, num_bodies). Only the first body (index 0) is used for the root.
        link_pose_w: Input array of root link poses in world frame. Shape is (num_envs,).
        env_ids: Input array of environment indices to write to. Shape is (num_selected_envs,).
        num_bodies: Input scalar number of bodies per environment.
        from_mask: Input flag indicating whether to use masked indexing. If True, env_ids
            are used to index into data. If False, data is indexed sequentially.
        root_link_velocity_w: Output array where root link velocities are written.
            Shape is (num_envs,).
        root_com_velocity_w: Output array where root COM velocities (derived from link)
            are written. Shape is (num_envs,).
        body_acc_w: Output array where body accelerations are zeroed.
            Shape is (num_envs, num_bodies).
        sim_env_ids: Output int32 environment indices for PhysX.
    """
    # If from mask, then we get complete data. Otherwise, we get partial data.
    i = wp.tid()
    env_id = wp.int32(env_ids[i])
    sim_env_ids[i] = env_id
    if from_mask:
        root_link_velocity_w[env_id] = data[env_id]
    else:
        root_link_velocity_w[env_id] = data[i]
    # Get the link velocity in the com frame
    root_com_velocity_w[env_id] = get_link_velocity_in_com_frame_func(
        root_link_velocity_w[env_id], link_pose_w[env_id], body_com_pose_b[env_id, 0]
    )
    # Make the acceleration zero to prevent reporting old values
    for j in range(num_bodies):
        body_acc_w[env_id, j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


_SET_ROOT_LINK_POSE_TO_SIM_DISPATCHER = IndexKernelDispatcher(set_root_link_pose_to_sim, ("env_ids",))
_SET_ROOT_COM_POSE_TO_SIM_DISPATCHER = IndexKernelDispatcher(set_root_com_pose_to_sim, ("env_ids",))
_SET_ROOT_COM_VELOCITY_TO_SIM_DISPATCHER = IndexKernelDispatcher(set_root_com_velocity_to_sim, ("env_ids",))
_SET_ROOT_LINK_VELOCITY_TO_SIM_DISPATCHER = IndexKernelDispatcher(set_root_link_velocity_to_sim, ("env_ids",))


def set_root_link_pose_to_sim_kernel(env_ids: wp.array | torch.Tensor) -> wp.Kernel:
    """Select the root-link pose writer for the selector dtype."""
    return _SET_ROOT_LINK_POSE_TO_SIM_DISPATCHER.select(env_ids)


def set_root_com_pose_to_sim_kernel(env_ids: wp.array | torch.Tensor) -> wp.Kernel:
    """Select the root-COM pose writer for the selector dtype."""
    return _SET_ROOT_COM_POSE_TO_SIM_DISPATCHER.select(env_ids)


def set_root_com_velocity_to_sim_kernel(env_ids: wp.array | torch.Tensor) -> wp.Kernel:
    """Select the root-COM velocity writer for the selector dtype."""
    return _SET_ROOT_COM_VELOCITY_TO_SIM_DISPATCHER.select(env_ids)


def set_root_link_velocity_to_sim_kernel(env_ids: wp.array | torch.Tensor) -> wp.Kernel:
    """Select the root-link velocity writer for the selector dtype."""
    return _SET_ROOT_LINK_VELOCITY_TO_SIM_DISPATCHER.select(env_ids)


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
    body_com_pose_b: wp.array2d(dtype=wp.transformf),
    env_ids: wp.array(dtype=Any),
    body_ids: wp.array(dtype=Any),
    from_mask: bool,
    body_com_pose_w: wp.array2d(dtype=wp.transformf),
    body_link_pose_w: wp.array2d(dtype=wp.transformf),
):
    """Write body COM pose data to simulation buffers.

    This kernel writes body COM poses from the input array to the output buffers
    and computes the corresponding link poses from the COM poses, for each body
    in each environment.

    Args:
        data: Input array of body COM poses. Shape is (num_envs, num_bodies) or
            (num_selected_envs, num_selected_bodies) depending on from_mask.
        body_com_pose_b: Input array of body COM poses in body frame. Shape is
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
        body_com_pose_w[env_id, body_id], body_com_pose_b[env_id, body_id]
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

    This kernel writes body COM velocities from the input array to the output buffers
    and zeros out the body acceleration buffer, for each body in each environment.

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
    body_com_pose_b: wp.array2d(dtype=wp.transformf),
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
    computes the corresponding COM velocities from the link velocities, and zeros out
    the body acceleration buffer.

    Args:
        data: Input array of body link spatial velocities. Shape is (num_envs, num_bodies)
            or (num_selected_envs, num_selected_bodies) depending on from_mask.
        body_com_pose_b: Input array of body COM poses in body frame. Shape is
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
        body_com_pose_b[env_id, body_id],
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
