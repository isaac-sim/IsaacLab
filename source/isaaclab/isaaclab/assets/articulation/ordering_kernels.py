# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Warp kernels for articulation ordering conversions.

Axis 0 is always the environment axis. An articulation item axis is described
as public or backend order at each argument. Direct joint and body permutations
are validated, read-only maps owned by ArticulationNameMap. Derived Jacobian
maps and identity/no-order ``_ALL_*_INDICES`` buffers are articulation- or
data-owned. These kernels treat every map as read-only. Component, spatial, and
floating-base DoF axes are preserved unless a kernel explicitly states otherwise.

Index-based writers require unique environment and public-item selectors;
duplicate selectors issue concurrent writes with undefined winners. Mask-based
writers do not have that precondition. When has_ordering is false, fused writers
update only the public buffer and permit that public buffer to alias the backend
output.
"""

from __future__ import annotations

from typing import Any

import warp as wp


@wp.kernel
def reorder_2d_backend_to_user(
    backend_data: wp.array2d(dtype=Any),
    user_to_backend: wp.array(dtype=wp.int32),
    user_data: wp.array2d(dtype=Any),
) -> None:
    """Gather a 2-D backend-order item axis into public order.

    Args:
        backend_data: Source array shaped [num_envs, num_items] in backend
            order. Values retain the caller-defined units.
        user_to_backend: Read-only map shaped [num_items] from each public item
            index to its backend item index.
        user_data: Destination array shaped [num_envs, num_items] in public
            order, with the same units as backend_data. It must not alias
            backend_data for a nonidentity permutation.
    """
    env_id, user_id = wp.tid()
    backend_id = user_to_backend[user_id]
    user_data[env_id, user_id] = backend_data[env_id, backend_id]


@wp.kernel
def reorder_2d_user_to_backend(
    user_data: wp.array2d(dtype=Any),
    backend_to_user: wp.array(dtype=wp.int32),
    backend_data: wp.array2d(dtype=Any),
) -> None:
    """Gather a 2-D public-order item axis into backend order.

    Args:
        user_data: Source array shaped [num_envs, num_items] in public order.
            Values retain the caller-defined units.
        backend_to_user: Read-only map shaped [num_items] from each backend item
            index to its public item index.
        backend_data: Destination array shaped [num_envs, num_items] in backend
            order, with the same units as user_data. It must not alias user_data
            for a nonidentity permutation.
    """
    env_id, backend_id = wp.tid()
    user_id = backend_to_user[backend_id]
    backend_data[env_id, backend_id] = user_data[env_id, user_id]


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
def reorder_3d_user_to_backend(
    user_data: wp.array3d(dtype=Any),
    backend_to_user: wp.array(dtype=wp.int32),
    backend_data: wp.array3d(dtype=Any),
) -> None:
    """Gather a 3-D public item axis into backend order.

    Args:
        user_data: Source shaped [num_envs, num_items, num_components] in public
            order. Values retain the caller-defined units.
        backend_to_user: Read-only map shaped [num_items] from backend item
            indices to public item indices.
        backend_data: Destination with the same shape and units in backend
            order. The component axis is preserved, and source and destination
            must not alias for a nonidentity permutation.
    """
    env_id, backend_id, component_id = wp.tid()
    user_id = backend_to_user[backend_id]
    backend_data[env_id, backend_id, component_id] = user_data[env_id, user_id, component_id]


@wp.kernel
def reorder_3d_backend_to_user(
    backend_data: wp.array3d(dtype=Any),
    user_to_backend: wp.array(dtype=wp.int32),
    user_data: wp.array3d(dtype=Any),
) -> None:
    """Gather a 3-D backend item axis into public order.

    Args:
        backend_data: Source shaped [num_envs, num_items, num_components] in
            backend order. Values retain the caller-defined units.
        user_to_backend: Read-only map shaped [num_items] from public item
            indices to backend item indices.
        user_data: Destination with the same shape and units in public order.
            The component axis is preserved, and source and destination must
            not alias for a nonidentity permutation.
    """
    env_id, user_id, component_id = wp.tid()
    backend_id = user_to_backend[user_id]
    user_data[env_id, user_id, component_id] = backend_data[env_id, backend_id, component_id]


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
def write_2d_user_to_backend_with_indices(
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
def write_2d_user_to_backend_with_mask(
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
def write_3d_user_to_backend_with_indices(
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
def write_3d_user_to_backend_with_mask(
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


# Concrete overloads preserve typed torch.Tensor argument adaptation.
_write_2d_user_to_backend_with_indices_vec3 = wp.overload(
    write_2d_user_to_backend_with_indices,
    {
        "input_data": wp.array2d(dtype=wp.vec3f),
        "user_data": wp.array2d(dtype=wp.vec3f),
        "backend_data": wp.array2d(dtype=wp.vec3f),
    },
)
_write_2d_user_to_backend_with_mask_vec3 = wp.overload(
    write_2d_user_to_backend_with_mask,
    {
        "input_data": wp.array2d(dtype=wp.vec3f),
        "user_data": wp.array2d(dtype=wp.vec3f),
        "backend_data": wp.array2d(dtype=wp.vec3f),
    },
)
_write_2d_user_to_backend_with_indices_transform = wp.overload(
    write_2d_user_to_backend_with_indices,
    {
        "input_data": wp.array2d(dtype=wp.transformf),
        "user_data": wp.array2d(dtype=wp.transformf),
        "backend_data": wp.array2d(dtype=wp.transformf),
    },
)
_write_2d_user_to_backend_with_mask_transform = wp.overload(
    write_2d_user_to_backend_with_mask,
    {
        "input_data": wp.array2d(dtype=wp.transformf),
        "user_data": wp.array2d(dtype=wp.transformf),
        "backend_data": wp.array2d(dtype=wp.transformf),
    },
)
_write_3d_user_to_backend_with_indices_float = wp.overload(
    write_3d_user_to_backend_with_indices,
    {
        "input_data": wp.array3d(dtype=wp.float32),
        "user_data": wp.array3d(dtype=wp.float32),
        "backend_data": wp.array3d(dtype=wp.float32),
    },
)
_write_3d_user_to_backend_with_mask_float = wp.overload(
    write_3d_user_to_backend_with_mask,
    {
        "input_data": wp.array3d(dtype=wp.float32),
        "user_data": wp.array3d(dtype=wp.float32),
        "backend_data": wp.array3d(dtype=wp.float32),
    },
)


@wp.func
def _resolve_float_input(
    input_data: wp.float32,
    env_id: int,
    user_id: int,
    local_env_id: int,
    local_user_id: int,
    full_data: bool,
) -> wp.float32:
    """Return a scalar input unchanged."""
    return input_data


@wp.func
def _resolve_float_input(
    input_data: wp.array2d(dtype=wp.float32),
    env_id: int,
    user_id: int,
    local_env_id: int,
    local_user_id: int,
    full_data: bool,
) -> wp.float32:
    """Read a 2-D input using full or selector-local indices."""
    if full_data:
        return input_data[env_id, user_id]
    return input_data[local_env_id, local_user_id]


@wp.kernel
def write_float_user_to_backend_with_indices(
    input_data: Any,
    env_ids: wp.array(dtype=wp.int32),
    user_ids: wp.array(dtype=wp.int32),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    full_data: bool,
    user_data: wp.array2d(dtype=wp.float32),
    backend_data: wp.array2d(dtype=wp.float32),
) -> None:
    """Write a scalar or selected 2-D public values to public and backend buffers.

    The Cartesian-product selectors must each contain unique indices. Repeated
    indices can issue concurrent writes to the same destination cell.

    Args:
        input_data: Scalar or 2-D values in caller-defined SI units. With full_data
            true, 2-D input is shaped [num_envs, num_items] in public order;
            otherwise it is shaped [len(env_ids), len(user_ids)].
        env_ids: Unique environment indices selected from [0, num_envs).
        user_ids: Unique public item indices selected from [0, num_items).
        user_to_backend: Read-only map from public to backend item indices.
        has_ordering: Whether to scatter values into backend_data. When false,
            backend_data is not written and may alias user_data.
        full_data: Whether 2-D input uses full public indices instead of compact
            selector-local indices.
        user_data: Public-order destination shaped [num_envs, num_items].
        backend_data: Backend-order destination shaped [num_envs, num_items].
    """
    i, j = wp.tid()
    env_id = env_ids[i]
    user_id = user_ids[j]

    value = _resolve_float_input(input_data, env_id, user_id, i, j, full_data)

    user_data[env_id, user_id] = value
    if has_ordering:
        backend_id = user_to_backend[user_id]
        backend_data[env_id, backend_id] = value


@wp.kernel
def write_float_user_to_backend_with_mask(
    input_data: Any,
    env_mask: wp.array(dtype=wp.bool),
    user_mask: wp.array(dtype=wp.bool),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: bool,
    user_data: wp.array2d(dtype=wp.float32),
    backend_data: wp.array2d(dtype=wp.float32),
) -> None:
    """Write a scalar or masked 2-D public values to public and backend buffers.

    Args:
        input_data: Scalar or 2-D values in caller-defined SI units. A 2-D input is
            shaped [num_envs, num_items] in public order.
        env_mask: Environment-selection mask shaped [num_envs].
        user_mask: Public-item selection mask shaped [num_items].
        user_to_backend: Read-only map from public to backend item indices.
        has_ordering: Whether to scatter values into backend_data. When false,
            backend_data is not written and may alias user_data.
        user_data: Public-order destination shaped [num_envs, num_items].
        backend_data: Backend-order destination shaped [num_envs, num_items].
    """
    env_id, user_id = wp.tid()
    if not env_mask[env_id] or not user_mask[user_id]:
        return

    value = _resolve_float_input(input_data, env_id, user_id, env_id, user_id, True)
    user_data[env_id, user_id] = value
    if has_ordering:
        backend_id = user_to_backend[user_id]
        backend_data[env_id, backend_id] = value


# Concrete array overloads preserve typed torch.Tensor argument adaptation.
_write_float_user_to_backend_with_indices_array = wp.overload(
    write_float_user_to_backend_with_indices, {"input_data": wp.array2d(dtype=wp.float32)}
)
_write_float_user_to_backend_with_mask_array = wp.overload(
    write_float_user_to_backend_with_mask, {"input_data": wp.array2d(dtype=wp.float32)}
)
