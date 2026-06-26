# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for the OVPhysX frame transformer sensor."""

import warp as wp

# ---- Frame transformer update kernel ----


@wp.kernel
def frame_transformer_update_kernel(
    env_mask: wp.array(dtype=wp.bool),
    raw_transforms: wp.array(dtype=wp.transformf),
    source_raw_indices: wp.array(dtype=wp.int32),
    target_raw_indices: wp.array2d(dtype=wp.int32),
    source_offset_pos: wp.array(dtype=wp.vec3f),
    source_offset_quat: wp.array(dtype=wp.quatf),
    target_offset_pos: wp.array(dtype=wp.vec3f),
    target_offset_quat: wp.array(dtype=wp.quatf),
    source_pos_w: wp.array(dtype=wp.vec3f),
    source_quat_w: wp.array(dtype=wp.quatf),
    target_pos_w: wp.array2d(dtype=wp.vec3f),
    target_quat_w: wp.array2d(dtype=wp.quatf),
    target_pos_source: wp.array2d(dtype=wp.vec3f),
    target_quat_source: wp.array2d(dtype=wp.quatf),
):
    """Update frame transformer sensor data from raw PhysX transforms.

    This kernel processes raw transforms from PhysX and computes:
    1. Source frame pose in world frame (with optional offset)
    2. Target frame poses in world frame (with optional offsets)
    3. Target frame poses relative to source frame

    Args:
        raw_transforms: Raw transforms from PhysX view. Shape is (N*M,) where N is num_envs and M is num_bodies.
        source_raw_indices: Indices into raw_transforms for source frame per environment. Shape is (N,).
        target_raw_indices: Indices into raw_transforms for target frames per (env, frame). Shape is (N, M) where M is
            num_target_frames.
        source_offset_pos: Optional position offset for source frame. Shape is (N, 3).
        source_offset_quat: Optional quaternion offset for source frame. Shape is (N, 4).
        target_offset_pos: Optional position offsets for target frames. Shape is (M, 3).
        target_offset_quat: Optional quaternion offsets for target frames. Shape is (M, 4).
        source_pos_w: Output source position in world frame. Shape is (N, 3).
        source_quat_w: Output source quaternion in world frame. Shape is (N, 4).
        target_pos_w: Output target positions in world frame. Shape is (N, M, 3).
        target_quat_w: Output target quaternions in world frame. Shape is (N, M, 4).
        target_pos_source: Output target positions relative to source frame. Shape is (N, M, 3).
        target_quat_source: Output target quaternions relative to source frame. Shape is (N, M, 4).
    """
    env_id, frame_id = wp.tid()

    if not env_mask[env_id]:
        return

    # Get source frame transform
    source_idx = source_raw_indices[env_id]
    source_tf = raw_transforms[source_idx]

    # Apply source frame offset
    source_offset_tf = wp.transform(source_offset_pos[env_id], source_offset_quat[env_id])
    source_tf_offset = wp.transform_multiply(source_tf, source_offset_tf)
    source_pos_w[env_id] = wp.transform_get_translation(source_tf_offset)
    source_quat_w[env_id] = wp.transform_get_rotation(source_tf_offset)

    # Get target frame transform
    target_idx = target_raw_indices[env_id, frame_id]
    target_tf = raw_transforms[target_idx]

    # Apply target offset if needed
    target_offset_tf = wp.transform(target_offset_pos[frame_id], target_offset_quat[frame_id])
    target_tf_offset = wp.transform_multiply(target_tf, target_offset_tf)
    target_pos_w[env_id, frame_id] = wp.transform_get_translation(target_tf_offset)
    target_quat_w[env_id, frame_id] = wp.transform_get_rotation(target_tf_offset)

    # Compute target frame relative to source frame
    source_tf_inv = wp.transform_inverse(source_tf_offset)
    target_relative_tf = wp.transform_multiply(source_tf_inv, target_tf_offset)
    target_pos_source[env_id, frame_id] = wp.transform_get_translation(target_relative_tf)
    target_quat_source[env_id, frame_id] = wp.transform_get_rotation(target_relative_tf)


# ---- Gather body pose kernel ----


@wp.kernel
def gather_body_pose_kernel(
    env_mask: wp.array(dtype=wp.bool),
    pose_buffer: wp.array(dtype=wp.transformf),
    dst_flat_indices: wp.array(dtype=wp.int32),
    raw_transforms: wp.array(dtype=wp.transformf),
):
    """Copy a single body's per-env pose into the flat raw transforms buffer.

    For each env in the launch, copies ``pose_buffer[env]`` into
    ``raw_transforms[dst_flat_indices[env]]``. Skips envs whose ``env_mask`` is False.

    The pose buffer is a view (``wp.array.view(wp.transformf)``) over a
    ``(num_envs, 7)`` ``float32`` array populated by
    ``binding.read(...)`` for a single ``RIGID_BODY_POSE`` tensor binding,
    so it has shape ``(num_envs,)``. One launch per tracked body fills the
    body's slot column in the flat ``raw_transforms`` buffer.

    Args:
        env_mask: Active environment mask, shape ``(num_envs,)``.
        pose_buffer: Per-env world pose [m, dimensionless], shape ``(num_envs,)``,
            dtype ``wp.transformf`` in ``(px, py, pz, qx, qy, qz, qw)`` format.
        dst_flat_indices: Destination slot in ``raw_transforms`` per env, shape ``(num_envs,)``.
        raw_transforms: Destination flat pose buffer [m, dimensionless], shape
            ``(num_envs * num_unique_bodies,)``, dtype ``wp.transformf``.
    """
    env_id = wp.tid()
    if not env_mask[env_id]:
        return
    raw_transforms[dst_flat_indices[env_id]] = pose_buffer[env_id]
