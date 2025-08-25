# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import torch

import isaaclab.utils.math as math_utils


@torch.jit.script
def torch_rand_float_tensor(lower, upper):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    return (upper - lower) * torch.rand_like(upper) + lower


def aggregate_inertia_about_robot_com(
    body_inertias_local: torch.Tensor,
    body_inv_mass_local: torch.Tensor,
    body_com_pos_b: torch.Tensor,
    body_com_quat_b: torch.Tensor,
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
      body_inertias_local (N,B,9|3,3): Link inertias in the mass/COM frame.
      body_inv_mass_local (N,B): Inverse link masses (<=0 treated as padding).
      body_com_pos_b (N,B,3): Link COM position relative to the link frame
        (massLocalPose translation); used as body_pos_b + R_link_base @ body_com_pos_b.
      body_com_quat_b (N,B,4 wxyz): Mass→link rotation (massLocalPose rotation).
      body_pos_b (N,B,3): Link origins in base frame.
      body_quat_b (N,B,4 wxyz): Link→base orientation.
      eps (float): Small value to guard division by zero.

    Returns:
      total_mass (N,): Sum of link masses.
      I_total (N,3,3): Inertia about robot COM in base frame (symmetrized).
      com_robot_b (N,3): Robot COM in base frame.

    Method (base frame throughout):
      1) COM of each link: com_link_b = body_pos_b + R_link_base @ body_com_pos_b
      2) Robot COM: mass-weighted average of com_link_b
      3) Rotate each link inertia: I_b = (R_link_base @ R_mass_link) I_local (⋯)^T
      4) Parallel-axis: I_pa = m (‖r‖² I - r rᵀ), r = com_link_b - com_robot_b
      5) Sum over links and symmetrize
    """
    # Inertia in mass frame (local to COM)
    num_envs, num_bodies, _ = body_inertias_local.shape
    I_local = body_inertias_local.view(num_envs, num_bodies, 3, 3)

    # Masses
    m = torch.where(body_inv_mass_local > 0, 1.0 / body_inv_mass_local, torch.zeros_like(body_inv_mass_local))
    m_sum = m.sum(dim=1, keepdim=True)
    valid = (m > 0).float().unsqueeze(-1)

    # Rotations: link->base (R_link_base) and mass->link (R_mass_link)
    R_link_base = math_utils.matrix_from_quat(body_quat_b)
    R_mass_link = body_pos_b + (R_link_base @ body_com_pos_b[..., :, None]).squeeze(-1)

    # Robot COM base frame (mass-weighted)
    com_robot_b = (m.unsqueeze(-1) * R_mass_link).sum(dim=1) / (m_sum + eps)

    # Rotate inertia from mass frame to world: R = R_link_base * R_mass
    R_mass = math_utils.matrix_from_quat(body_com_quat_b)
    R = R_link_base @ R_mass
    I_world = R @ I_local @ R.transpose(-1, -2)

    # Parallel-axis to robot COM
    r = R_mass_link - com_robot_b[:, None, :]
    rrT = r[..., :, None] @ r[..., None, :]
    r2 = (r * r).sum(dim=-1, keepdim=True)
    I3 = torch.eye(3, device=body_pos_b.device).reshape(1, 1, 3, 3).expand(num_envs, num_bodies, 3, 3)
    I_pa = m[..., None, None] * (r2[..., None] * I3 - rrT)

    # Sum over links (ignore zero-mass pads)
    I_total = ((I_world + I_pa) * valid[..., None]).sum(dim=1)
    I_total = 0.5 * (I_total + I_total.transpose(-1, -2))
    total_mass = m.sum(dim=1)

    return total_mass, I_total, com_robot_b
