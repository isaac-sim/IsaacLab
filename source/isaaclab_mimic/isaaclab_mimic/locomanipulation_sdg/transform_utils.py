# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import isaaclab.utils.math as math_utils


def transform_mul(transform_a: torch.Tensor, transform_b: torch.Tensor) -> torch.Tensor:
    """Multiply two translation, quaternion pose representations by converting to matrices first."""
    # Extract position and quaternion components
    pos_a, quat_a = transform_a[..., :3], transform_a[..., 3:]
    pos_b, quat_b = transform_b[..., :3], transform_b[..., 3:]

    # Convert quaternions to rotation matrices
    rot_a = math_utils.matrix_from_quat(quat_a)
    rot_b = math_utils.matrix_from_quat(quat_b)

    # Create pose matrices
    pose_a = math_utils.make_pose(pos_a, rot_a)
    pose_b = math_utils.make_pose(pos_b, rot_b)

    # Multiply pose matrices
    result_pose = torch.matmul(pose_a, pose_b)

    # Extract position and rotation matrix
    result_pos, result_rot = math_utils.unmake_pose(result_pose)

    # Convert rotation matrix back to quaternion
    result_quat = math_utils.quat_from_matrix(result_rot)

    return torch.cat([result_pos, result_quat], dim=-1)


def transform_inv(transform: torch.Tensor) -> torch.Tensor:
    """Invert a translation, quaternion format transformation using math_utils."""
    pos, quat = transform[..., :3], transform[..., 3:]
    quat_inv = math_utils.quat_inv(quat)
    pos_inv = math_utils.quat_apply(quat_inv, -pos)
    return torch.cat([pos_inv, quat_inv], dim=-1)


def transform_relative_pose(world_pose: torch.Tensor, src_frame_pose: torch.Tensor, dst_frame_pose: torch.Tensor):
    """Compute the relative pose with respect to a source frame, and apply this relative pose to a destination frame."""
    device = dst_frame_pose.device
    world_pose = world_pose.to(device)
    src_frame_pose = src_frame_pose.to(device)
    pose = transform_mul(dst_frame_pose, transform_mul(transform_inv(src_frame_pose), world_pose))
    return pose
