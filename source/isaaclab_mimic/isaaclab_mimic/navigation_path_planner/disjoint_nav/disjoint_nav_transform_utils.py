# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import isaaclab.utils.math as math_utils


def transform_to_matrix(transform: torch.Tensor):
    """Convert a translation, quaternion pose representation into a transformation matrix."""
    pose_matrix = math_utils.make_pose(transform[..., :3], math_utils.matrix_from_quat(transform[..., 3:]))
    return pose_matrix


def transform_from_matrix(matrix: torch.Tensor):
    """Convert a transformation matrix to a translation, quaternion pose."""
    pos, rot = math_utils.unmake_pose(matrix)
    quat = math_utils.quat_from_matrix(rot)
    return torch.cat([pos, quat], dim=-1)


def transform_inv(transform: torch.Tensor):
    """Invert a translation, quaternion format transformation."""
    matrix = transform_to_matrix(transform)
    matrix = math_utils.pose_inv(matrix)
    return transform_from_matrix(matrix)


def transform_mul(transform_a, transform_b):
    """Multiply a two translation, quaternion pose representations to apply the transformation."""
    return transform_from_matrix(torch.matmul(transform_to_matrix(transform_a), transform_to_matrix(transform_b)))


def transform_relative_pose(world_pose: torch.Tensor, src_frame_pose: torch.Tensor, dst_frame_pose: torch.Tensor):
    """Compute the relative pose with respect to a source frame, and apply this relative pose to a destination frame."""
    pose = transform_mul(dst_frame_pose, transform_mul(transform_inv(src_frame_pose), world_pose))
    return pose
