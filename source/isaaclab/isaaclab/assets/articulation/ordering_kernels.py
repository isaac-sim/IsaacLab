# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Warp kernels for articulation ordering conversions.

Visibility contract:
    The ``gather_2d``/``gather_3d`` kernels and their directional
    ``reorder_*`` aliases listed in ``__all__`` are public: they gather a
    single item axis between public and backend order and can legitimately be
    launched on raw solver-view arrays for advanced interop. The
    ``write_2d``/``write_3d``/``write_float`` Python launch wrappers and the
    fused joint-target, body-wrench, joint-state, Jacobian, mass-matrix, and
    generalized-vector kernels are an internal contract between isaaclab core
    and the backend packages: not user API, may change without deprecation.
    Underscore-prefixed kernels and helpers are module-private; backends go
    through the wrappers.

Axis conventions and ownership:
    Axis 0 is always the environment axis. An articulation item axis is
    described as public or backend order at each argument. Direct joint and body
    permutations are validated, read-only maps owned by ArticulationNameMap.
    Derived Jacobian maps and identity/no-order ``_ALL_*_INDICES`` buffers are
    articulation- or data-owned. These kernels treat every map as read-only.
    Component, spatial, and floating-base DoF axes are preserved unless a kernel
    explicitly states otherwise.

    Index-based writers require unique environment and public-item selectors;
    duplicate selectors issue concurrent writes with undefined winners.
    Mask-based writers do not have that precondition. When has_ordering is
    false, fused writers update only the public buffer and permit that public
    buffer to alias the backend output.
"""

from __future__ import annotations

from typing import Any

import warp as wp

__all__ = [
    "gather_2d",
    "gather_3d",
    "reorder_2d_backend_to_user",
    "reorder_2d_user_to_backend",
    "reorder_3d_backend_to_user",
    "reorder_3d_user_to_backend",
    "resolve_backend_index",
]


@wp.func
def resolve_backend_index(
    user_id: wp.int32,
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
):
    """Map a public (user-order) item index to its backend-order index.

    Ordered write kernels that iterate the public joint or body axis and
    scatter into backend-order buffers use this to centralize the
    user-to-backend direction together with the identity fallback used when no
    nonidentity ordering is active, so the routing logic is defined once
    instead of re-implemented inline in each kernel.

    Args:
        user_id: Item index on the public (user-order) axis.
        user_to_backend: Public-to-backend permutation for the axis. Ignored
            when ``has_ordering`` is ``False`` (an identity map may be passed).
        has_ordering: Whether a nonidentity ordering is active.

    Returns:
        The backend-order index for ``user_id``.
    """
    if has_ordering:
        return user_to_backend[user_id]
    return user_id


@wp.kernel
def gather_2d(
    src: wp.array2d(dtype=Any),
    dst_to_src: wp.array(dtype=wp.int32),
    dst: wp.array2d(dtype=Any),
) -> None:
    """Gather a 2-D item axis into destination order.

    Args:
        src: Source array shaped [num_envs, num_items]. Values retain the
            caller-defined units.
        dst_to_src: Read-only map shaped [num_items] from each destination item
            index to its source item index.
        dst: Destination array shaped [num_envs, num_items], with the same
            units as src. It must not alias src for a nonidentity permutation.
    """
    env_id, dst_id = wp.tid()
    dst[env_id, dst_id] = src[env_id, dst_to_src[dst_id]]


@wp.kernel
def reorder_joint_targets_user_to_backend(
    user_effort: wp.array2d(dtype=wp.float32),
    user_pos_target: wp.array2d(dtype=wp.float32),
    user_vel_target: wp.array2d(dtype=wp.float32),
    backend_to_user: wp.array(dtype=wp.int32),
    write_effort: bool,
    write_pos_target: bool,
    write_vel_target: bool,
    write_joint_act: bool,
    backend_effort: wp.array2d(dtype=wp.float32),
    backend_pos_target: wp.array2d(dtype=wp.float32),
    backend_vel_target: wp.array2d(dtype=wp.float32),
    backend_joint_act: wp.array2d(dtype=wp.float32),
) -> None:
    """Gather the per-step joint write-path targets into backend order in one launch.

    This fuses the several :func:`reorder_2d_user_to_backend` launches that a
    single :meth:`write_data_to_sim` would otherwise issue. Each boolean flag is
    fixed at launch (graph-capture safe) and selects whether one backend output
    is written; the effort source is read once and can feed both the actuation
    effort and the direct-drive ``joint_act`` output. A gated-off backend output
    is never indexed and may alias any other argument.

    Args:
        user_effort: Actuation effort or applied torque
            [N or N*m, depending on joint type], shaped [num_envs, num_joints]
            in public joint order.
        user_pos_target: Joint position targets [m or rad, depending on joint
            type], shaped [num_envs, num_joints] in public joint order.
        user_vel_target: Joint velocity targets [m/s or rad/s, depending on
            joint type], shaped [num_envs, num_joints] in public joint order.
        backend_to_user: Read-only map shaped [num_joints] from each backend
            joint index to its public joint index.
        write_effort: Whether to gather ``user_effort`` into ``backend_effort``.
        write_pos_target: Whether to gather ``user_pos_target`` into
            ``backend_pos_target``.
        write_vel_target: Whether to gather ``user_vel_target`` into
            ``backend_vel_target``.
        write_joint_act: Whether to gather ``user_effort`` into
            ``backend_joint_act`` from the same read as ``backend_effort``.
        backend_effort: Effort destination
            [N or N*m, depending on joint type] in backend joint order.
        backend_pos_target: Position-target destination
            [m or rad, depending on joint type] in backend joint order.
        backend_vel_target: Velocity-target destination
            [m/s or rad/s, depending on joint type] in backend joint order.
        backend_joint_act: Direct-drive actuation destination
            [N or N*m, depending on joint type] in backend joint order.
    """
    env_id, backend_id = wp.tid()
    user_id = backend_to_user[backend_id]

    if write_effort or write_joint_act:
        effort = user_effort[env_id, user_id]
        if write_effort:
            backend_effort[env_id, backend_id] = effort
        if write_joint_act:
            backend_joint_act[env_id, backend_id] = effort

    if write_pos_target:
        backend_pos_target[env_id, backend_id] = user_pos_target[env_id, user_id]

    if write_vel_target:
        backend_vel_target[env_id, backend_id] = user_vel_target[env_id, user_id]


@wp.kernel
def reorder_joint_state_backend_to_user(
    backend_pos: wp.array2d(dtype=wp.float32),
    backend_vel: wp.array2d(dtype=wp.float32),
    user_to_backend: wp.array(dtype=wp.int32),
    user_pos: wp.array2d(dtype=wp.float32),
    user_vel: wp.array2d(dtype=wp.float32),
) -> None:
    """Gather backend-order joint position and velocity into public order in one launch.

    This fuses the two :func:`reorder_2d_backend_to_user` launches that the
    post-step publish would otherwise issue for the Tier-1 joint-state shadows.
    Both outputs share the single ``user_to_backend`` gather.

    Args:
        backend_pos: Joint positions [m or rad, depending on joint type], shaped
            [num_envs, num_joints] in backend joint order.
        backend_vel: Joint velocities [m/s or rad/s, depending on joint type],
            shaped [num_envs, num_joints] in backend joint order.
        user_to_backend: Read-only map shaped [num_joints] from each public joint
            index to its backend joint index.
        user_pos: Position destination [m or rad, depending on joint type] in
            public joint order. It must not alias backend_pos for a nonidentity
            permutation.
        user_vel: Velocity destination [m/s or rad/s, depending on joint type] in
            public joint order. It must not alias backend_vel for a nonidentity
            permutation.
    """
    env_id, user_id = wp.tid()
    backend_id = user_to_backend[user_id]
    user_pos[env_id, user_id] = backend_pos[env_id, backend_id]
    user_vel[env_id, user_id] = backend_vel[env_id, backend_id]


@wp.kernel
def reorder_body_wrench_user_to_backend(
    user_force: wp.array2d(dtype=wp.vec3f),
    user_torque: wp.array2d(dtype=wp.vec3f),
    backend_to_user: wp.array(dtype=wp.int32),
    backend_force: wp.array2d(dtype=wp.vec3f),
    backend_torque: wp.array2d(dtype=wp.vec3f),
) -> None:
    """Gather public body wrenches into backend body order.

    Args:
        user_force: Body-frame forces [N], shaped [num_envs, num_bodies], in
            public body order.
        user_torque: Body-frame torques [N*m], shaped [num_envs, num_bodies],
            in public body order.
        backend_to_user: Read-only map shaped [num_bodies] from backend body
            indices to public body indices.
        backend_force: Force destination [N], shaped [num_envs, num_bodies], in
            backend body order.
        backend_torque: Torque destination [N*m], shaped
            [num_envs, num_bodies], in backend body order. Source and
            destination arrays must not alias for a nonidentity permutation.
    """
    env_id, backend_body_id = wp.tid()
    user_body_id = backend_to_user[backend_body_id]
    backend_force[env_id, backend_body_id] = user_force[env_id, user_body_id]
    backend_torque[env_id, backend_body_id] = user_torque[env_id, user_body_id]


@wp.kernel
def reorder_body_state_backend_to_user(
    backend_pose: wp.array2d(dtype=wp.transformf),
    backend_vel: wp.array2d(dtype=wp.spatial_vectorf),
    user_to_backend: wp.array(dtype=wp.int32),
    user_pose: wp.array2d(dtype=wp.transformf),
    user_vel: wp.array2d(dtype=wp.spatial_vectorf),
) -> None:
    """Gather backend-order body pose and COM velocity into public order in one launch.

    This fuses the two :func:`reorder_2d_backend_to_user` launches that the
    post-step publish would otherwise issue for the Tier-1 body-state shadows.
    Both outputs share the single ``user_to_backend`` gather.

    Args:
        backend_pose: Body link poses as [position, quaternion] transforms
            (position [m], orientation a unit quaternion), shaped
            [num_envs, num_bodies] in backend body order.
        backend_vel: Body center-of-mass spatial velocities [m/s, rad/s], shaped
            [num_envs, num_bodies] in backend body order.
        user_to_backend: Read-only map shaped [num_bodies] from each public body
            index to its backend body index.
        user_pose: Pose destination (position [m], orientation a unit quaternion)
            in public body order. It must not alias backend_pose for a
            nonidentity permutation.
        user_vel: Velocity destination [m/s, rad/s] in public body order. It must
            not alias backend_vel for a nonidentity permutation.
    """
    env_id, user_id = wp.tid()
    backend_id = user_to_backend[user_id]
    user_pose[env_id, user_id] = backend_pose[env_id, backend_id]
    user_vel[env_id, user_id] = backend_vel[env_id, backend_id]


@wp.kernel
def gather_3d(
    src: wp.array3d(dtype=Any),
    dst_to_src: wp.array(dtype=wp.int32),
    dst: wp.array3d(dtype=Any),
) -> None:
    """Gather a 3-D item axis into destination order, preserving components.

    Args:
        src: Source shaped [num_envs, num_items, num_components]. Values retain
            the caller-defined units.
        dst_to_src: Read-only map shaped [num_items] from each destination item
            index to its source item index.
        dst: Destination with the same shape and units. The component axis is
            preserved; src and dst must not alias for a nonidentity permutation.
    """
    env_id, dst_id, component_id = wp.tid()
    src_id = dst_to_src[dst_id]
    dst[env_id, dst_id, component_id] = src[env_id, src_id, component_id]


# Directional aliases of the gather kernels. The alias name documents intent at
# a launch site; the map argument follows from it by the dst_to_src contract:
#   *_backend_to_user gathers backend-order data into public order
#       -> pass the ordering's ``user_to_backend`` map (destination is the user axis);
#   *_user_to_backend gathers public-order data into backend order
#       -> pass the ordering's ``backend_to_user`` map (destination is the backend axis).
reorder_2d_backend_to_user = gather_2d
reorder_2d_user_to_backend = gather_2d
reorder_3d_backend_to_user = gather_3d
reorder_3d_user_to_backend = gather_3d


@wp.kernel
def reorder_jacobian_backend_to_user(
    backend_data: wp.array4d(dtype=wp.float32),
    jacobian_body_user_to_backend: wp.array(dtype=wp.int32),
    joint_user_to_backend: wp.array(dtype=wp.int32),
    num_base_dofs: wp.int32,
    has_body_ordering: bool,
    has_joint_ordering: bool,
    user_data: wp.array4d(dtype=wp.float32),
) -> None:
    """Gather Jacobian body rows and actuated-joint columns into public order.

    Args:
        backend_data: Geometric Jacobian shaped
            [num_envs, num_jacobian_bodies, 6, num_dofs] in backend body and
            joint order. Linear rows are [m/s per unit DoF velocity] and angular
            rows are [rad/s per unit DoF velocity].
        jacobian_body_user_to_backend: Read-only map from public Jacobian body
            rows to backend rows. Any fixed-root omission is already encoded.
        joint_user_to_backend: Read-only map from public actuated-joint columns
            to backend actuated-joint columns.
        num_base_dofs: Number of leading floating-base DoFs, either 0 or 6.
        has_body_ordering: Whether to apply the body-row map.
        has_joint_ordering: Whether to apply the joint-column map after the
            leading base DoFs.
        user_data: Destination with the same shape and units in public body and
            joint order. Spatial rows and leading base-DoF columns are
            preserved. It may alias backend_data only when both ordering flags
            are false.
    """
    env_id, user_body_id, spatial_id, user_dof_id = wp.tid()
    backend_body_id = user_body_id
    if has_body_ordering:
        backend_body_id = jacobian_body_user_to_backend[user_body_id]

    backend_dof_id = user_dof_id
    if has_joint_ordering and user_dof_id >= num_base_dofs:
        backend_dof_id = num_base_dofs + joint_user_to_backend[user_dof_id - num_base_dofs]

    user_data[env_id, user_body_id, spatial_id, user_dof_id] = backend_data[
        env_id, backend_body_id, spatial_id, backend_dof_id
    ]


@wp.kernel
def reorder_mass_matrix_backend_to_user(
    backend_data: wp.array3d(dtype=wp.float32),
    joint_user_to_backend: wp.array(dtype=wp.int32),
    num_base_dofs: wp.int32,
    has_joint_ordering: bool,
    user_data: wp.array3d(dtype=wp.float32),
) -> None:
    """Gather both actuated-joint axes of a generalized mass matrix into public order.

    Args:
        backend_data: Generalized mass matrix [kg*m^2 or kg, per DoF type],
            shaped [num_envs, num_dofs, num_dofs], in backend joint order.
        joint_user_to_backend: Read-only map from public actuated-joint indices
            to backend actuated-joint indices.
        num_base_dofs: Number of leading floating-base DoFs, either 0 or 6.
        has_joint_ordering: Whether to apply the map to rows and columns after
            the leading base DoFs.
        user_data: Destination with the same shape and units in public joint
            order. Leading base-DoF rows and columns are preserved. It may alias
            backend_data only when has_joint_ordering is false.
    """
    env_id, user_row_id, user_col_id = wp.tid()
    backend_row_id = user_row_id
    backend_col_id = user_col_id
    if has_joint_ordering:
        if user_row_id >= num_base_dofs:
            backend_row_id = num_base_dofs + joint_user_to_backend[user_row_id - num_base_dofs]
        if user_col_id >= num_base_dofs:
            backend_col_id = num_base_dofs + joint_user_to_backend[user_col_id - num_base_dofs]

    user_data[env_id, user_row_id, user_col_id] = backend_data[env_id, backend_row_id, backend_col_id]


@wp.kernel
def reorder_generalized_vector_backend_to_user(
    backend_data: wp.array2d(dtype=wp.float32),
    joint_user_to_backend: wp.array(dtype=wp.int32),
    num_base_dofs: wp.int32,
    has_joint_ordering: bool,
    user_data: wp.array2d(dtype=wp.float32),
) -> None:
    """Gather an actuated-joint segment of a generalized force vector into public order.

    Args:
        backend_data: Generalized force [N or N*m, depending on DoF type],
            shaped [num_envs, num_dofs], in backend joint order.
        joint_user_to_backend: Read-only map from public actuated-joint indices
            to backend actuated-joint indices.
        num_base_dofs: Number of leading floating-base DoFs, either 0 or 6.
        has_joint_ordering: Whether to apply the map after the leading base DoFs.
        user_data: Destination with the same shape and units in public joint
            order. Leading base-DoF entries are preserved. It may alias
            backend_data only when has_joint_ordering is false.
    """
    env_id, user_dof_id = wp.tid()
    backend_dof_id = user_dof_id
    if has_joint_ordering and user_dof_id >= num_base_dofs:
        backend_dof_id = num_base_dofs + joint_user_to_backend[user_dof_id - num_base_dofs]

    user_data[env_id, user_dof_id] = backend_data[env_id, backend_dof_id]


@wp.kernel
def write_joint_vel_user_to_backend_with_indices(
    in_data: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=wp.int32),
    user_ids: wp.array(dtype=wp.int32),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    full_data: bool,
    user_vel: wp.array2d(dtype=wp.float32),
    user_prev_vel: wp.array2d(dtype=wp.float32),
    user_acc: wp.array2d(dtype=wp.float32),
    backend_vel: wp.array2d(dtype=wp.float32),
) -> None:
    """Write selected public joint velocities and reset their acceleration history.

    The Cartesian-product selectors must each contain unique indices. Repeated
    indices can issue concurrent writes to the same destination cell.

    Args:
        in_data: Joint velocities [m/s or rad/s, depending on joint type].
            With full_data true, shape is [num_envs, num_joints] in public
            order; otherwise it is [len(env_ids), len(user_ids)].
        env_ids: Unique environment indices selected from [0, num_envs).
        user_ids: Unique public joint indices selected from [0, num_joints).
        user_to_backend: Read-only map from public to backend joint indices.
        has_ordering: Whether to scatter velocity into backend_vel. When false,
            backend_vel is not written and may alias user_vel.
        full_data: Whether in_data uses full public indices instead of compact
            selector-local indices.
        user_vel: Public joint velocity destination [m/s or rad/s].
        user_prev_vel: Public previous-velocity destination [m/s or rad/s].
        user_acc: Public acceleration destination [m/s^2 or rad/s^2], set to
            zero for selected cells.
        backend_vel: Backend joint velocity destination [m/s or rad/s].
    """
    i, j = wp.tid()
    env_id = env_ids[i]
    user_id = user_ids[j]

    if full_data:
        value = in_data[env_id, user_id]
    else:
        value = in_data[i, j]

    user_vel[env_id, user_id] = value
    user_prev_vel[env_id, user_id] = value
    user_acc[env_id, user_id] = 0.0
    if has_ordering:
        backend_id = user_to_backend[user_id]
        backend_vel[env_id, backend_id] = value


@wp.kernel
def write_joint_state_user_to_backend_with_indices(
    pos_data: wp.array2d(dtype=wp.float32),
    vel_data: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=wp.int32),
    user_ids: wp.array(dtype=wp.int32),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    full_data: bool,
    user_pos: wp.array2d(dtype=wp.float32),
    user_vel: wp.array2d(dtype=wp.float32),
    user_prev_vel: wp.array2d(dtype=wp.float32),
    user_acc: wp.array2d(dtype=wp.float32),
    backend_pos: wp.array2d(dtype=wp.float32),
    backend_vel: wp.array2d(dtype=wp.float32),
) -> None:
    """Write selected public joint state and reset its acceleration history.

    The Cartesian-product selectors must each contain unique indices. Repeated
    indices can issue concurrent writes to the same destination cell.

    Args:
        pos_data: Joint positions [m or rad, depending on joint type]. With
            full_data true, shape is [num_envs, num_joints] in public order;
            otherwise it is [len(env_ids), len(user_ids)].
        vel_data: Joint velocities [m/s or rad/s, depending on joint type], with
            the same full or compact shape interpretation as pos_data.
        env_ids: Unique environment indices selected from [0, num_envs).
        user_ids: Unique public joint indices selected from [0, num_joints).
        user_to_backend: Read-only map from public to backend joint indices.
        has_ordering: Whether to scatter state into backend_pos and backend_vel.
            When false, backend outputs are not written and may alias the
            corresponding public outputs.
        full_data: Whether input arrays use full public indices instead of
            compact selector-local indices.
        user_pos: Public joint position destination [m or rad].
        user_vel: Public joint velocity destination [m/s or rad/s].
        user_prev_vel: Public previous-velocity destination [m/s or rad/s].
        user_acc: Public acceleration destination [m/s^2 or rad/s^2], set to
            zero for selected cells.
        backend_pos: Backend joint position destination [m or rad].
        backend_vel: Backend joint velocity destination [m/s or rad/s].
    """
    i, j = wp.tid()
    env_id = env_ids[i]
    user_id = user_ids[j]

    if full_data:
        position = pos_data[env_id, user_id]
        velocity = vel_data[env_id, user_id]
    else:
        position = pos_data[i, j]
        velocity = vel_data[i, j]

    user_pos[env_id, user_id] = position
    user_vel[env_id, user_id] = velocity
    user_prev_vel[env_id, user_id] = velocity
    user_acc[env_id, user_id] = 0.0
    if has_ordering:
        backend_id = user_to_backend[user_id]
        backend_pos[env_id, backend_id] = position
        backend_vel[env_id, backend_id] = velocity


@wp.kernel
def write_joint_vel_user_to_backend_with_mask(
    in_data: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    user_mask: wp.array(dtype=wp.bool),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    user_vel: wp.array2d(dtype=wp.float32),
    user_prev_vel: wp.array2d(dtype=wp.float32),
    user_acc: wp.array2d(dtype=wp.float32),
    backend_vel: wp.array2d(dtype=wp.float32),
) -> None:
    """Write masked public joint velocities and reset their acceleration history.

    Args:
        in_data: Public joint velocities [m/s or rad/s, depending on joint
            type], shaped [num_envs, num_joints].
        env_mask: Environment-selection mask shaped [num_envs].
        user_mask: Public-joint selection mask shaped [num_joints].
        user_to_backend: Read-only map from public to backend joint indices.
        has_ordering: Whether to scatter velocity into backend_vel. When false,
            backend_vel is not written and may alias user_vel.
        user_vel: Public joint velocity destination [m/s or rad/s].
        user_prev_vel: Public previous-velocity destination [m/s or rad/s].
        user_acc: Public acceleration destination [m/s^2 or rad/s^2], set to
            zero for selected cells.
        backend_vel: Backend joint velocity destination [m/s or rad/s].
    """
    env_id, user_id = wp.tid()
    if not env_mask[env_id] or not user_mask[user_id]:
        return

    value = in_data[env_id, user_id]
    user_vel[env_id, user_id] = value
    user_prev_vel[env_id, user_id] = value
    user_acc[env_id, user_id] = 0.0
    if has_ordering:
        backend_id = user_to_backend[user_id]
        backend_vel[env_id, backend_id] = value


@wp.kernel
def write_joint_state_user_to_backend_with_mask(
    pos_data: wp.array2d(dtype=wp.float32),
    vel_data: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    user_mask: wp.array(dtype=wp.bool),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    user_pos: wp.array2d(dtype=wp.float32),
    user_vel: wp.array2d(dtype=wp.float32),
    user_prev_vel: wp.array2d(dtype=wp.float32),
    user_acc: wp.array2d(dtype=wp.float32),
    backend_pos: wp.array2d(dtype=wp.float32),
    backend_vel: wp.array2d(dtype=wp.float32),
) -> None:
    """Write masked public joint state and reset its acceleration history.

    Args:
        pos_data: Public joint positions [m or rad, depending on joint type],
            shaped [num_envs, num_joints].
        vel_data: Public joint velocities [m/s or rad/s, depending on joint
            type], shaped [num_envs, num_joints].
        env_mask: Environment-selection mask shaped [num_envs].
        user_mask: Public-joint selection mask shaped [num_joints].
        user_to_backend: Read-only map from public to backend joint indices.
        has_ordering: Whether to scatter state into backend_pos and backend_vel.
            When false, backend outputs are not written and may alias the
            corresponding public outputs.
        user_pos: Public joint position destination [m or rad].
        user_vel: Public joint velocity destination [m/s or rad/s].
        user_prev_vel: Public previous-velocity destination [m/s or rad/s].
        user_acc: Public acceleration destination [m/s^2 or rad/s^2], set to
            zero for selected cells.
        backend_pos: Backend joint position destination [m or rad].
        backend_vel: Backend joint velocity destination [m/s or rad/s].
    """
    env_id, user_id = wp.tid()
    if not env_mask[env_id] or not user_mask[user_id]:
        return

    position = pos_data[env_id, user_id]
    velocity = vel_data[env_id, user_id]
    user_pos[env_id, user_id] = position
    user_vel[env_id, user_id] = velocity
    user_prev_vel[env_id, user_id] = velocity
    user_acc[env_id, user_id] = 0.0
    if has_ordering:
        backend_id = user_to_backend[user_id]
        backend_pos[env_id, backend_id] = position
        backend_vel[env_id, backend_id] = velocity


@wp.kernel
def _write_2d_user_to_backend_with_indices(
    input_data: wp.array2d(dtype=Any),
    env_ids: wp.array(dtype=wp.int32),
    user_ids: wp.array(dtype=wp.int32),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    full_data: bool,
    user_data: wp.array2d(dtype=Any),
    backend_data: wp.array2d(dtype=Any),
) -> None:
    """Write selected structured values to public- and backend-order buffers.

    Args:
        input_data: Values in caller-defined units. With full_data true, shape
            is [num_envs, num_items] in public order; otherwise it is
            [len(env_ids), len(user_ids)].
        env_ids: Unique selected environment indices.
        user_ids: Unique selected public item indices.
        user_to_backend: Read-only public-to-backend item map.
        has_ordering: Whether to scatter values into backend_data. When false,
            backend_data is not written and may alias user_data.
        full_data: Whether input_data uses global public indices.
        user_data: Public-order destination with the same dtype as input_data.
        backend_data: Backend-order destination with the same dtype as input_data.
    """
    local_env_id, local_user_id = wp.tid()
    env_id = env_ids[local_env_id]
    user_id = user_ids[local_user_id]
    if full_data:
        value = input_data[env_id, user_id]
    else:
        value = input_data[local_env_id, local_user_id]

    user_data[env_id, user_id] = value
    if has_ordering:
        backend_data[env_id, user_to_backend[user_id]] = value


@wp.kernel
def _write_2d_user_to_backend_with_mask(
    input_data: wp.array2d(dtype=Any),
    env_mask: wp.array(dtype=wp.bool),
    user_mask: wp.array(dtype=wp.bool),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    user_data: wp.array2d(dtype=Any),
    backend_data: wp.array2d(dtype=Any),
) -> None:
    """Write masked structured values to public- and backend-order buffers.

    Args:
        input_data: Full public-order values shaped [num_envs, num_items] in
            caller-defined units.
        env_mask: Environment-selection mask.
        user_mask: Public-item-selection mask.
        user_to_backend: Read-only public-to-backend item map.
        has_ordering: Whether to scatter values into backend_data. When false,
            backend_data is not written and may alias user_data.
        user_data: Public-order destination with the same dtype as input_data.
        backend_data: Backend-order destination with the same dtype as input_data.
    """
    env_id, user_id = wp.tid()
    if not env_mask[env_id] or not user_mask[user_id]:
        return

    value = input_data[env_id, user_id]
    user_data[env_id, user_id] = value
    if has_ordering:
        backend_data[env_id, user_to_backend[user_id]] = value


@wp.kernel
def _write_3d_user_to_backend_with_indices(
    input_data: wp.array3d(dtype=Any),
    env_ids: wp.array(dtype=wp.int32),
    user_ids: wp.array(dtype=wp.int32),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    full_data: bool,
    user_data: wp.array3d(dtype=Any),
    backend_data: wp.array3d(dtype=Any),
) -> None:
    """Write selected component values to public- and backend-order buffers.

    Args:
        input_data: Values in caller-defined units. With full_data true, shape
            is [num_envs, num_items, num_components] in public order;
            otherwise it is [len(env_ids), len(user_ids), num_components].
        env_ids: Unique selected environment indices.
        user_ids: Unique selected public item indices.
        user_to_backend: Read-only public-to-backend item map.
        has_ordering: Whether to scatter values into backend_data. When false,
            backend_data is not written and may alias user_data.
        full_data: Whether input_data uses global public indices.
        user_data: Public-order destination with the same dtype as input_data.
        backend_data: Backend-order destination with the same dtype as input_data.
    """
    local_env_id, local_user_id, component_id = wp.tid()
    env_id = env_ids[local_env_id]
    user_id = user_ids[local_user_id]
    if full_data:
        value = input_data[env_id, user_id, component_id]
    else:
        value = input_data[local_env_id, local_user_id, component_id]

    user_data[env_id, user_id, component_id] = value
    if has_ordering:
        backend_data[env_id, user_to_backend[user_id], component_id] = value


@wp.kernel
def _write_3d_user_to_backend_with_mask(
    input_data: wp.array3d(dtype=Any),
    env_mask: wp.array(dtype=wp.bool),
    user_mask: wp.array(dtype=wp.bool),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    user_data: wp.array3d(dtype=Any),
    backend_data: wp.array3d(dtype=Any),
) -> None:
    """Write masked component values to public- and backend-order buffers.

    Args:
        input_data: Full public-order values shaped
            [num_envs, num_items, num_components] in caller-defined units.
        env_mask: Environment-selection mask.
        user_mask: Public-item-selection mask.
        user_to_backend: Read-only public-to-backend item map.
        has_ordering: Whether to scatter values into backend_data. When false,
            backend_data is not written and may alias user_data.
        user_data: Public-order destination with the same dtype as input_data.
        backend_data: Backend-order destination with the same dtype as input_data.
    """
    env_id, user_id, component_id = wp.tid()
    if not env_mask[env_id] or not user_mask[user_id]:
        return

    value = input_data[env_id, user_id, component_id]
    user_data[env_id, user_id, component_id] = value
    if has_ordering:
        backend_data[env_id, user_to_backend[user_id], component_id] = value


def _as_warp_array(data, dtype) -> wp.array:
    """Return ``data`` as a Warp array of ``dtype``, adapting torch tensors.

    Warp cannot infer generic (``dtype=Any``) kernel parameters from torch
    tensors, so the launch wrappers below adapt every torch input once, here,
    including trailing-dimension folding for vector and transform dtypes.
    Warp arrays pass through unchanged and must already carry ``dtype``.
    """
    if isinstance(data, wp.array):
        return data
    return wp.from_torch(data, dtype=dtype)


def write_2d_user_to_backend_with_indices(
    input_data,
    env_ids,
    user_ids,
    user_to_backend,
    has_ordering,
    full_data,
    user_data,
    backend_data,
    *,
    dtype,
    device,
) -> None:
    """Write selected structured values to public- and backend-order buffers.

    Launch wrapper over the module-private generic kernel: adapts torch inputs
    to :paramref:`dtype` and derives the launch dim from the selectors, so call
    sites need no dtype-suffixed overload names. Argument contract matches
    :func:`_write_2d_user_to_backend_with_indices`.

    Args:
        input_data: Values in caller-defined units, torch tensor or Warp array.
            With full_data true, shape is [num_envs, num_items] in public order;
            otherwise it is [len(env_ids), len(user_ids)].
        env_ids: Unique selected environment indices, ``wp.int32``.
        user_ids: Unique selected public item indices, ``wp.int32``.
        user_to_backend: Read-only public-to-backend item map, ``wp.int32``.
        has_ordering: Whether to scatter values into backend_data. When false,
            backend_data is not written and may alias user_data.
        full_data: Whether input_data uses global public indices.
        user_data: Public-order destination with element type ``dtype``.
        backend_data: Backend-order destination with element type ``dtype``.
        dtype: Warp element dtype shared by the data arrays (e.g. ``wp.vec3f``,
            ``wp.transformf``).
        device: Warp launch device.
    """
    wp.launch(
        _write_2d_user_to_backend_with_indices,
        dim=(env_ids.shape[0], user_ids.shape[0]),
        inputs=[
            _as_warp_array(input_data, dtype),
            env_ids,
            user_ids,
            user_to_backend,
            has_ordering,
            full_data,
            _as_warp_array(user_data, dtype),
            _as_warp_array(backend_data, dtype),
        ],
        device=device,
    )


def write_2d_user_to_backend_with_mask(
    input_data,
    env_mask,
    user_mask,
    user_to_backend,
    has_ordering,
    user_data,
    backend_data,
    *,
    dtype,
    device,
) -> None:
    """Write masked structured values to public- and backend-order buffers.

    Launch wrapper over :func:`_write_2d_user_to_backend_with_mask`; see
    :func:`write_2d_user_to_backend_with_indices` for the adaptation contract.
    The launch dim is the full [num_envs, num_items] mask domain.
    """
    wp.launch(
        _write_2d_user_to_backend_with_mask,
        dim=(env_mask.shape[0], user_mask.shape[0]),
        inputs=[
            _as_warp_array(input_data, dtype),
            env_mask,
            user_mask,
            user_to_backend,
            has_ordering,
            _as_warp_array(user_data, dtype),
            _as_warp_array(backend_data, dtype),
        ],
        device=device,
    )


def write_3d_user_to_backend_with_indices(
    input_data,
    env_ids,
    user_ids,
    user_to_backend,
    has_ordering,
    full_data,
    user_data,
    backend_data,
    *,
    dtype,
    device,
) -> None:
    """Write selected component values to public- and backend-order buffers.

    Launch wrapper over :func:`_write_3d_user_to_backend_with_indices`; the
    component count of the launch dim comes from the adapted destination.
    """
    user_array = _as_warp_array(user_data, dtype)
    wp.launch(
        _write_3d_user_to_backend_with_indices,
        dim=(env_ids.shape[0], user_ids.shape[0], user_array.shape[2]),
        inputs=[
            _as_warp_array(input_data, dtype),
            env_ids,
            user_ids,
            user_to_backend,
            has_ordering,
            full_data,
            user_array,
            _as_warp_array(backend_data, dtype),
        ],
        device=device,
    )


def write_3d_user_to_backend_with_mask(
    input_data,
    env_mask,
    user_mask,
    user_to_backend,
    has_ordering,
    user_data,
    backend_data,
    *,
    dtype,
    device,
) -> None:
    """Write masked component values to public- and backend-order buffers.

    Launch wrapper over :func:`_write_3d_user_to_backend_with_mask`.
    """
    user_array = _as_warp_array(user_data, dtype)
    wp.launch(
        _write_3d_user_to_backend_with_mask,
        dim=(env_mask.shape[0], user_mask.shape[0], user_array.shape[2]),
        inputs=[
            _as_warp_array(input_data, dtype),
            env_mask,
            user_mask,
            user_to_backend,
            has_ordering,
            user_array,
            _as_warp_array(backend_data, dtype),
        ],
        device=device,
    )


def write_float_user_to_backend_with_indices(
    input_data,
    env_ids,
    user_ids,
    user_to_backend,
    has_ordering,
    full_data,
    user_data,
    backend_data,
    *,
    device,
) -> None:
    """Write a scalar or selected 2-D public values to public and backend buffers.

    ``input_data`` may be a Python scalar, broadcast to the selected cells by
    :func:`_write_scalar_user_to_backend_with_indices` (:paramref:`full_data`
    is irrelevant for a scalar and ignored), or a 2-D float32 torch tensor /
    Warp array, delegated to :func:`write_2d_user_to_backend_with_indices`.
    """
    if isinstance(input_data, (int, float)):
        wp.launch(
            _write_scalar_user_to_backend_with_indices,
            dim=(env_ids.shape[0], user_ids.shape[0]),
            inputs=[
                float(input_data),
                env_ids,
                user_ids,
                user_to_backend,
                has_ordering,
                _as_warp_array(user_data, wp.float32),
                _as_warp_array(backend_data, wp.float32),
            ],
            device=device,
        )
        return
    write_2d_user_to_backend_with_indices(
        input_data,
        env_ids,
        user_ids,
        user_to_backend,
        has_ordering,
        full_data,
        user_data,
        backend_data,
        dtype=wp.float32,
        device=device,
    )


def write_float_user_to_backend_with_mask(
    input_data,
    env_mask,
    user_mask,
    user_to_backend,
    has_ordering,
    user_data,
    backend_data,
    *,
    device,
) -> None:
    """Write a scalar or masked 2-D public values to public and backend buffers.

    ``input_data`` may be a Python scalar, broadcast to the masked cells by
    :func:`_write_scalar_user_to_backend_with_mask`, or a 2-D float32 torch
    tensor / Warp array, delegated to
    :func:`write_2d_user_to_backend_with_mask`.
    """
    if isinstance(input_data, (int, float)):
        wp.launch(
            _write_scalar_user_to_backend_with_mask,
            dim=(env_mask.shape[0], user_mask.shape[0]),
            inputs=[
                float(input_data),
                env_mask,
                user_mask,
                user_to_backend,
                has_ordering,
                _as_warp_array(user_data, wp.float32),
                _as_warp_array(backend_data, wp.float32),
            ],
            device=device,
        )
        return
    write_2d_user_to_backend_with_mask(
        input_data,
        env_mask,
        user_mask,
        user_to_backend,
        has_ordering,
        user_data,
        backend_data,
        dtype=wp.float32,
        device=device,
    )


@wp.kernel
def _write_scalar_user_to_backend_with_indices(
    input_value: wp.float32,
    env_ids: wp.array(dtype=wp.int32),
    user_ids: wp.array(dtype=wp.int32),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    user_data: wp.array2d(dtype=wp.float32),
    backend_data: wp.array2d(dtype=wp.float32),
) -> None:
    """Broadcast one scalar to selected public and backend buffer cells.

    The Cartesian-product selectors must each contain unique indices. Repeated
    indices can issue concurrent writes to the same destination cell.

    Args:
        input_value: Scalar value in caller-defined SI units.
        env_ids: Unique environment indices selected from [0, num_envs).
        user_ids: Unique public item indices selected from [0, num_items).
        user_to_backend: Read-only map from public to backend item indices.
        has_ordering: Whether to scatter the value into backend_data. When
            false, backend_data is not written and may alias user_data.
        user_data: Public-order destination shaped [num_envs, num_items].
        backend_data: Backend-order destination shaped [num_envs, num_items].
    """
    i, j = wp.tid()
    env_id = env_ids[i]
    user_id = user_ids[j]

    user_data[env_id, user_id] = input_value
    if has_ordering:
        backend_id = user_to_backend[user_id]
        backend_data[env_id, backend_id] = input_value


@wp.kernel
def _write_scalar_user_to_backend_with_mask(
    input_value: wp.float32,
    env_mask: wp.array(dtype=wp.bool),
    user_mask: wp.array(dtype=wp.bool),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    user_data: wp.array2d(dtype=wp.float32),
    backend_data: wp.array2d(dtype=wp.float32),
) -> None:
    """Broadcast one scalar to masked public and backend buffer cells.

    Args:
        input_value: Scalar value in caller-defined SI units.
        env_mask: Environment-selection mask shaped [num_envs].
        user_mask: Public-item selection mask shaped [num_items].
        user_to_backend: Read-only map from public to backend item indices.
        has_ordering: Whether to scatter the value into backend_data. When
            false, backend_data is not written and may alias user_data.
        user_data: Public-order destination shaped [num_envs, num_items].
        backend_data: Backend-order destination shaped [num_envs, num_items].
    """
    env_id, user_id = wp.tid()
    if not env_mask[env_id] or not user_mask[user_id]:
        return

    user_data[env_id, user_id] = input_value
    if has_ordering:
        backend_id = user_to_backend[user_id]
        backend_data[env_id, backend_id] = input_value
