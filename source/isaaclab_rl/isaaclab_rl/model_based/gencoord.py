# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Derived from HybridRobotics/SuperMimic, revision 5efbf114e11f725c31d83e217be987c13c0e4ce2.
# Copyright 2026 the SuperMimic authors. See docs/licenses/dependencies/supermimic-official-license.txt.
# Modified for Isaac Lab package boundaries and simulator interfaces.

"""Generalized-coordinate (gencoord) representation helper functions.

Full state: [root_pos(3), root_quat(4), qj(N), root_lin_vel(3), root_ang_vel(3), vj(N)]
Rep state:  [z(1), g_b(3), v_b(3), w_b(3), qj(N), vj(N)]
"""

import torch

from isaaclab_rl.model_based.math import quat_apply_inverse


def infer_num_joints_from_state_dim(state_dim: int) -> int:
    """Infers number of joints from full-state dimension.

    Full-state layout is:
    [root_pos(3), root_quat(4), qj(N), root_lin_vel(3), root_ang_vel(3), vj(N)]
    """
    if state_dim < 13 or (state_dim - 13) % 2 != 0:
        raise ValueError(f"Invalid full-state dim {state_dim}: expected 13 + 2 * num_joints")
    return (state_dim - 13) // 2


def rep_state_dim_from_num_joints(num_joints: int) -> int:
    return 10 + 2 * num_joints


def full_state_offsets(num_joints: int) -> tuple[int, int, int, int]:
    """Returns start offsets for [qj, root_lin_vel, root_ang_vel, vj] in full state."""
    idx_qj = 7
    idx_lin = idx_qj + num_joints
    idx_ang = idx_lin + 3
    idx_vj = idx_ang + 3
    return idx_qj, idx_lin, idx_ang, idx_vj


def rep_state_offsets(num_joints: int) -> tuple[int, int]:
    """Returns start offsets for [qj, vj] in rep state."""
    idx_qj = 10
    idx_vj = idx_qj + num_joints
    return idx_qj, idx_vj


def decompose_state(
    full_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decomposes full state into canonical components."""
    num_joints = infer_num_joints_from_state_dim(full_state.shape[-1])

    idx_qj, idx_lin, idx_ang, idx_vj = full_state_offsets(num_joints)

    root_pos = full_state[..., 0:3]
    root_quat = full_state[..., 3:7]
    qj = full_state[..., idx_qj:idx_lin]
    root_lin_vel = full_state[..., idx_lin:idx_ang]
    root_ang_vel = full_state[..., idx_ang:idx_vj]
    vj = full_state[..., idx_vj : idx_vj + num_joints]
    return root_pos, root_quat, qj, root_lin_vel, root_ang_vel, vj


def decompose_rep_state(
    rep_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decomposes gencoord rep state into canonical components."""
    # rep state layout: [z(1), g_b(3), v_b(3), w_b(3), qj(N), vj(N)]
    if rep_state.shape[-1] < 10 or (rep_state.shape[-1] - 10) % 2 != 0:
        raise ValueError(f"Invalid rep-state dim {rep_state.shape[-1]}: expected 10 + 2 * num_joints")
    num_joints = (rep_state.shape[-1] - 10) // 2

    idx_qj, idx_vj = rep_state_offsets(num_joints)

    z = rep_state[..., 0:1]
    g_b = rep_state[..., 1:4]
    v_b = rep_state[..., 4:7]
    w_b = rep_state[..., 7:10]
    qj = rep_state[..., idx_qj:idx_vj]
    vj = rep_state[..., idx_vj : idx_vj + num_joints]
    return z, g_b, v_b, w_b, qj, vj


def compute_gencoord_state(full_state: torch.Tensor) -> torch.Tensor:
    """Converts raw full state to gencoord representation.

    Uses root body frame for velocities (no heading alignment).
    This avoids all singularities since we're just rotating to body frame.
    """
    (
        root_pos,
        root_quat,
        qj,
        root_lin_vel,
        root_ang_vel,
        vj,
    ) = decompose_state(full_state)

    z = root_pos[..., 2:3]
    world_down = torch.zeros_like(root_lin_vel)
    world_down[..., 2] = -1.0
    g_b = quat_apply_inverse(root_quat, world_down)
    v_b = quat_apply_inverse(root_quat, root_lin_vel)
    w_b = quat_apply_inverse(root_quat, root_ang_vel)

    return torch.cat([z, g_b, v_b, w_b, qj, vj], dim=-1)
