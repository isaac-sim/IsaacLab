# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Derived from HybridRobotics/SuperMimic, revision 5efbf114e11f725c31d83e217be987c13c0e4ce2.
# Copyright 2026 the SuperMimic authors. See docs/licenses/dependencies/supermimic-official-license.txt.
# Modified for Isaac Lab package boundaries and simulator interfaces.

"""Shared utility functions used across multiple modules.

Representation-specific helpers live in dedicated modules:
- :mod:`isaaclab_rl.model_based.gencoord` — generalized-coordinate representation
- :mod:`isaaclab_rl.model_based.maxcoord` — maximal-coordinate representation
"""

from __future__ import annotations

import torch

from isaaclab_rl.model_based.quaternion import matrix_from_quat, quat_mul


def matrix_to_quat(rot_mat: torch.Tensor) -> torch.Tensor:
    """Rotation matrix to quaternion — compile-safe (no boolean indexing).

    Same algorithm as pytorch3d/mjlab ``quat_from_matrix`` but uses
    ``torch.clamp`` + ``torch.gather`` instead of boolean advanced indexing,
    so it produces only static-shaped tensors and works inside
    ``torch.compile(mode="reduce-overhead")``.

    Args:
      rot_mat: (..., 3, 3) rotation matrix

    Returns:
      (..., 4) quaternion (w, x, y, z)
    """
    m00, m01, m02 = rot_mat[..., 0, 0], rot_mat[..., 0, 1], rot_mat[..., 0, 2]
    m10, m11, m12 = rot_mat[..., 1, 0], rot_mat[..., 1, 1], rot_mat[..., 1, 2]
    m20, m21, m22 = rot_mat[..., 2, 0], rot_mat[..., 2, 1], rot_mat[..., 2, 2]

    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                dim=-1,
            ),
            min=1e-10,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )  # (..., 4, 4)

    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].clamp(min=0.1))

    best = q_abs.argmax(dim=-1)  # (...)
    best_expanded = best[..., None, None].expand(*best.shape, 1, 4)
    return quat_candidates.gather(-2, best_expanded).squeeze(-2)


def small_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` without TF32 rounding, fusable by ``torch.compile``."""
    return (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(-2)


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix via Gram-Schmidt.

    Args:
      rot6d: (..., 6) first two columns of rotation matrix, flattened

    Returns:
      (..., 3, 3) rotation matrix
    """
    cols = rot6d.unflatten(-1, (3, 2))
    a1 = cols[..., 0]
    a2 = cols[..., 1]

    b1 = torch.nn.functional.normalize(a1, dim=-1)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = torch.nn.functional.normalize(a2 - dot * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)


def rotation_error_magnitude(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """Angular distance between two rotation matrices, in radians.

    The clamp keeps ``cos_angle`` away from the ±1 acos singularity so the
    gradient stays finite for identical or 180°-apart inputs.

    Args:
      mat1: (..., 3, 3) rotation matrix
      mat2: (..., 3, 3) rotation matrix

    Returns:
      (...,) angle in radians
    """
    # trace(R1^T R2) via element-wise product (avoids materializing R1^T R2)
    tr = torch.sum(mat1 * mat2, dim=(-2, -1))
    cos_angle = torch.clamp((tr - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos_angle)


def quat_to_rot6d(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternions to 6D rotation representation (first two columns of rotation matrix)."""
    mat = matrix_from_quat(quat)  # (..., 3, 3)
    return mat[..., :2].flatten(-2, -1)  # (..., 6)


def rodrigues_precompute(
    omega: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose axis-angle into (unit_axis, cos_angle, sin_angle).

    Args:
      omega: (..., 3) axis-angle rotation (axis * angle)

    Returns:
      Tuple of (k, cos_a, sin_a) where k is the unit axis.
    """
    angle = torch.sqrt((omega * omega).sum(-1, keepdim=True) + 1e-8)
    k = omega / angle
    return k, torch.cos(angle), torch.sin(angle)


def rodrigues_apply(
    v: torch.Tensor,
    k: torch.Tensor,
    cos_a: torch.Tensor,
    sin_a: torch.Tensor,
) -> torch.Tensor:
    """Rodrigues rotation with precomputed trig.

    Args:
      v: (..., 3) vectors to rotate
      k: (..., 3) unit rotation axis
      cos_a: (..., 1) cosine of angle
      sin_a: (..., 1) sine of angle

    Returns:
      (..., 3) rotated vectors
    """
    k_cross_v = torch.linalg.cross(k, v, dim=-1)
    k_dot_v = (k * v).sum(-1, keepdim=True)
    return cos_a * v + sin_a * k_cross_v + (1.0 - cos_a) * k_dot_v * k


def rodrigues_rotate(v: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v by axis-angle omega using Rodrigues' formula.

    Args:
      v: (..., 3) vectors to rotate
      omega: (..., 3) axis-angle rotation (axis * angle)

    Returns:
      (..., 3) rotated vectors
    """
    k, cos_a, sin_a = rodrigues_precompute(omega)
    return rodrigues_apply(v, k, cos_a, sin_a)


def quat_integrate(quat: torch.Tensor, ang_vel: torch.Tensor, dt: float) -> torch.Tensor:
    """Integrate quaternion by angular velocity over dt (axis-angle method).

    Args:
      quat: (..., 4) current quaternion (w, x, y, z)
      ang_vel: (..., 3) angular velocity in world frame
      dt: time step

    Returns:
      (..., 4) next quaternion, normalized
    """
    omega = ang_vel * dt
    norm_sq = (omega * omega).sum(dim=-1, keepdim=True)
    norm_safe = torch.sqrt(norm_sq + 1e-8)
    half_angle = norm_safe / 2
    axis = omega / norm_safe
    delta_q = torch.cat([torch.cos(half_angle), torch.sin(half_angle) * axis], dim=-1)
    q_next = quat_mul(delta_q, quat)
    return q_next / torch.norm(q_next, dim=-1, keepdim=True)
