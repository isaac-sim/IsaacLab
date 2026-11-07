# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing utilities for various math operations."""

# needed to import for allowing type-hinting: torch.Tensor | np.ndarray
from __future__ import annotations

import logging
import warnings

import torch
import torch.nn.functional

from isaaclab.utils.math import matrix_from_quat

# import logger
logger = logging.getLogger(__name__)


def aggregate_inertia_about_robot_com(
    body_inertias_local: torch.Tensor,
    body_inv_mass_local: torch.Tensor,
    body_com_pos_b: torch.Tensor,
    body_com_quat_b: torch.Tensor | None,
    body_pos_b: torch.Tensor,
    body_quat_b: torch.Tensor,
    eps=1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aggregate per-link inertias into a single inertia about the robot COM,
    expressed in the base (root link) frame.

    Shapes:
      num_envs=N, num_bodies=B

    Args:
      body_inertias_local (N,B,9|3,3): Link inertias about each link COM, expressed in the
        link frame (as returned by ``data.body_inertia`` on all backends).
      body_inv_mass_local (N,B): Inverse link masses (<=0 treated as padding).
      body_com_pos_b (N,B,3): Link COM position relative to the link frame
        (massLocalPose translation); used as body_pos_b + R_link_base @ body_com_pos_b.
      body_com_quat_b (None): Deprecated and ignored; pass ``None``. ``body_inertias_local`` is already
        expressed in the link frame, so the principal-axes rotation must not be applied.

        .. deprecated:: 6.0.0
           The argument will be removed in a future release.
      body_pos_b (N,B,3): Link origins in base frame.
      body_quat_b (N,B,4 xyzw): Link→base orientation.
      eps (float): Small value to guard division by zero.

    Returns:
      total_mass (N,): Sum of link masses.
      I_total (N,3,3): Inertia about robot COM in base frame (symmetrized).
      com_robot_b (N,3): Robot COM in base frame.

    Method (base frame throughout):
      1) COM of each link: com_link_b = body_pos_b + R_link_base @ body_com_pos_b
      2) Robot COM: mass-weighted average of com_link_b
      3) Rotate each link inertia into the base frame: I_b = R_link_base I_local R_link_base^T
      4) Parallel-axis: I_pa = m (‖r‖² I - r rᵀ), r = com_link_b - com_robot_b
      5) Sum over links and symmetrize
    """
    if body_com_quat_b is not None:
        warnings.warn(
            "The `body_com_quat_b` argument of `aggregate_inertia_about_robot_com` is deprecated and ignored:"
            " `body_inertias_local` is already expressed in the link frame. Pass `None` instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Inertia about the link COM, expressed in the link frame
    num_envs, num_bodies, _ = body_inertias_local.shape
    I_local = body_inertias_local.view(num_envs, num_bodies, 3, 3)

    # Masses
    m = torch.where(body_inv_mass_local > 0, 1.0 / body_inv_mass_local, torch.zeros_like(body_inv_mass_local))
    m_sum = m.sum(dim=1, keepdim=True)
    valid = (m > 0).float().unsqueeze(-1)

    # Link COM positions in base frame
    R_link_base = matrix_from_quat(body_quat_b)
    com_link_b = body_pos_b + (R_link_base @ body_com_pos_b[..., :, None]).squeeze(-1)

    # Robot COM base frame (mass-weighted)
    com_robot_b = (m.unsqueeze(-1) * com_link_b).sum(dim=1) / (m_sum + eps)

    # Rotate inertia from link frame to base frame (it is already in the link frame, so the
    # principal-axes rotation must not be applied on top).
    I_base = R_link_base @ I_local @ R_link_base.transpose(-1, -2)

    # Parallel-axis to robot COM
    r = com_link_b - com_robot_b[:, None, :]
    rrT = r[..., :, None] @ r[..., None, :]
    r2 = (r * r).sum(dim=-1, keepdim=True)
    I3 = torch.eye(3, device=body_pos_b.device).reshape(1, 1, 3, 3).expand(num_envs, num_bodies, 3, 3)
    I_pa = m[..., None, None] * (r2[..., None] * I3 - rrT)

    # Sum over links (ignore zero-mass pads)
    I_total = ((I_base + I_pa) * valid[..., None]).sum(dim=1)
    I_total = 0.5 * (I_total + I_total.transpose(-1, -2))
    total_mass = m.sum(dim=1)

    return total_mass, I_total, com_robot_b
