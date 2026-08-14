# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels used by actuator collections."""

from __future__ import annotations

from typing import Any

import torch
import warp as wp

from isaaclab.utils.warp.index_kernel import IndexKernelDispatcher


@wp.kernel(enable_backward=False)
def write_2d_float_with_indices(
    source: wp.array2d(dtype=wp.float32),
    env_ids: wp.array(dtype=Any),
    joint_ids: wp.array(dtype=Any),
    full_data: bool,
    target: wp.array2d(dtype=wp.float32),
):
    """Write 2-D float data into a target buffer using environment and joint indices."""
    env_i, joint_i = wp.tid()
    env_id = env_ids[env_i]
    joint_id = joint_ids[joint_i]
    if full_data:
        target[env_id, joint_id] = source[env_id, joint_id]
    else:
        target[env_id, joint_id] = source[env_i, joint_i]


_WRITE_2D_FLOAT_WITH_INDICES_DISPATCHER = IndexKernelDispatcher(write_2d_float_with_indices, ("env_ids", "joint_ids"))


def write_2d_float_with_indices_kernel(
    env_ids: torch.Tensor | wp.array, joint_ids: torch.Tensor | wp.array
) -> wp.Kernel:
    """Select the indexed float writer for the selector dtypes.

    Args:
        env_ids: Environment indices.
        joint_ids: Joint indices.

    Returns:
        Warp kernel specialized for the two selector dtypes.
    """
    return _WRITE_2D_FLOAT_WITH_INDICES_DISPATCHER.select(env_ids, joint_ids)


@wp.kernel(enable_backward=False)
def write_2d_float_with_mask(
    source: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    joint_mask: wp.array(dtype=wp.bool),
    target: wp.array2d(dtype=wp.float32),
):
    """Write full-sized 2-D float data into a target buffer using masks."""
    env_id, joint_id = wp.tid()
    if env_mask[env_id] and joint_mask[joint_id]:
        target[env_id, joint_id] = source[env_id, joint_id]


@wp.kernel(enable_backward=False)
def scatter_explicit_actuator_outputs(
    source_pos: wp.array2d(dtype=wp.float32),
    source_vel: wp.array2d(dtype=wp.float32),
    source_effort: wp.array2d(dtype=wp.float32),
    source_computed_effort: wp.array2d(dtype=wp.float32),
    source_applied_effort: wp.array2d(dtype=wp.float32),
    source_velocity_limit: wp.array2d(dtype=wp.float32),
    joint_indices: wp.array(dtype=wp.int32),
    target_pos: wp.array2d(dtype=wp.float32),
    target_vel: wp.array2d(dtype=wp.float32),
    target_effort: wp.array2d(dtype=wp.float32),
    target_computed_effort: wp.array2d(dtype=wp.float32),
    target_applied_effort: wp.array2d(dtype=wp.float32),
    target_velocity_limit: wp.array2d(dtype=wp.float32),
):
    """Scatter explicit actuator outputs into full articulation buffers.

    Command outputs are scattered only when present. Computed and applied effort
    and velocity-limit telemetry are always published. Explicit actuator models (for example
    :class:`~isaaclab.actuators.IdealPDActuator`) clear the position and velocity
    commands after computing the effort, so those sources may be null.
    """
    env_id, source_joint_id = wp.tid()
    target_joint_id = joint_indices[source_joint_id]
    if source_pos:
        target_pos[env_id, target_joint_id] = source_pos[env_id, source_joint_id]
    if source_vel:
        target_vel[env_id, target_joint_id] = source_vel[env_id, source_joint_id]
    if source_effort:
        target_effort[env_id, target_joint_id] = source_effort[env_id, source_joint_id]
    target_computed_effort[env_id, target_joint_id] = source_computed_effort[env_id, source_joint_id]
    target_applied_effort[env_id, target_joint_id] = source_applied_effort[env_id, source_joint_id]
    target_velocity_limit[env_id, target_joint_id] = source_velocity_limit[env_id, source_joint_id]


@wp.kernel(enable_backward=False)
def compute_implicit_actuator_batch(
    command_pos: wp.array2d(dtype=wp.float32),
    command_vel: wp.array2d(dtype=wp.float32),
    command_effort: wp.array2d(dtype=wp.float32),
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    joint_stiffness: wp.array2d(dtype=wp.float32),
    joint_damping: wp.array2d(dtype=wp.float32),
    joint_effort_limits: wp.array2d(dtype=wp.float32),
    velocity_limit: wp.array2d(dtype=wp.float32),
    joint_indices: wp.array(dtype=wp.int32),
    batch_computed_effort: wp.array2d(dtype=wp.float32),
    batch_applied_effort: wp.array2d(dtype=wp.float32),
    target_pos: wp.array2d(dtype=wp.float32),
    target_vel: wp.array2d(dtype=wp.float32),
    target_effort: wp.array2d(dtype=wp.float32),
    computed_effort: wp.array2d(dtype=wp.float32),
    applied_effort: wp.array2d(dtype=wp.float32),
    soft_velocity_limit: wp.array2d(dtype=wp.float32),
):
    """Compute and publish one implicit actuator execution batch."""
    env_id, batch_joint_id = wp.tid()
    joint_id = joint_indices[batch_joint_id]

    position_target = command_pos[env_id, joint_id]
    velocity_target = command_vel[env_id, joint_id]
    feedforward = command_effort[env_id, joint_id]
    effort = (
        joint_stiffness[env_id, joint_id] * (position_target - joint_pos[env_id, joint_id])
        + joint_damping[env_id, joint_id] * (velocity_target - joint_vel[env_id, joint_id])
        + feedforward
    )
    limit = joint_effort_limits[env_id, joint_id]
    clamped_effort = wp.clamp(effort, -limit, limit)

    batch_computed_effort[env_id, batch_joint_id] = effort
    batch_applied_effort[env_id, batch_joint_id] = clamped_effort
    target_pos[env_id, joint_id] = position_target
    target_vel[env_id, joint_id] = velocity_target
    target_effort[env_id, joint_id] = feedforward
    computed_effort[env_id, joint_id] = effort
    applied_effort[env_id, joint_id] = clamped_effort
    soft_velocity_limit[env_id, joint_id] = velocity_limit[env_id, batch_joint_id]
