# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Geometry shared by cable-routing MDP terms."""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import quat_apply


def cable_relative_joint_gap(segment_poses_w: torch.Tensor, rest_length: float) -> torch.Tensor:
    """Measure separation between consecutive cable-capsule endpoints.

    Unlike adjacent center distance, endpoint separation is invariant to a valid
    hinge rotation between cable segments. It therefore measures constraint
    stretch/compression without penalizing the bending required for routing.

    Args:
        segment_poses_w: Cable capsule poses ``(x, y, z, qx, qy, qz, qw)`` in
            world frame, shape ``(N, S, 7)``.
        rest_length: Capsule centerline length [m].

    Returns:
        Endpoint gap divided by :paramref:`rest_length`, shape ``(N, S - 1)``.

    Raises:
        ValueError: If the input shape or rest length is invalid.
    """
    if segment_poses_w.ndim != 3 or segment_poses_w.shape[-1] != 7:
        raise ValueError(f"segment_poses_w must have shape (N, S, 7); got {tuple(segment_poses_w.shape)}.")
    if segment_poses_w.shape[1] < 2:
        raise ValueError("segment_poses_w must contain at least two cable segments.")
    if not math.isfinite(rest_length) or rest_length <= 0.0:
        raise ValueError(f"rest_length must be finite and positive; got {rest_length}.")

    local_axis = torch.zeros_like(segment_poses_w[..., :3])
    local_axis[..., 2] = 1.0
    half_axis_w = 0.5 * rest_length * quat_apply(segment_poses_w[..., 3:7], local_axis)
    segment_start_w = segment_poses_w[..., :3] - half_axis_w
    segment_end_w = segment_poses_w[..., :3] + half_axis_w
    joint_gap = torch.linalg.vector_norm(segment_end_w[:, :-1] - segment_start_w[:, 1:], dim=-1)
    return joint_gap / rest_length
