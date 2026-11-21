# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Derived from HybridRobotics/SuperMimic, revision 5efbf114e11f725c31d83e217be987c13c0e4ce2.
# Copyright 2026 the SuperMimic authors. See docs/licenses/dependencies/supermimic-official-license.txt.
# Modified for Isaac Lab package boundaries and simulator interfaces.

"""Generalized-coordinate and maximal-coordinate state layouts and conversions."""

from __future__ import annotations

import torch

from isaaclab_rl.model_based.kinematics import (
    matrix_from_quat,
    matrix_to_quat,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    quat_to_rot6d,
    rot6d_to_matrix,
    small_matmul,
)


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


def maxcoord_full_state_dim(num_bodies: int) -> int:
    return num_bodies * 13


def maxcoord_rep_state_dim(num_bodies: int) -> int:
    return num_bodies * 15 + 4


def maxcoord_prediction_dim(num_bodies: int) -> int:
    return num_bodies * 6


def _reshape_bodies(flat: torch.Tensor, num_bodies: int, feat: int) -> torch.Tensor:
    """(..., B*feat) -> (..., B, feat)"""
    return flat.unflatten(-1, (num_bodies, feat))


def decompose_maxcoord_state(
    full_state: torch.Tensor, num_bodies: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose flat maxcoord full state into per-body tensors.

    Returns:
      body_pos: (..., B, 3)
      body_quat: (..., B, 4)
      body_lin_vel: (..., B, 3)
      body_ang_vel: (..., B, 3)
    """
    B = num_bodies
    body_pos = _reshape_bodies(full_state[..., : B * 3], B, 3)
    body_quat = _reshape_bodies(full_state[..., B * 3 : B * 7], B, 4)
    body_lin_vel = _reshape_bodies(full_state[..., B * 7 : B * 10], B, 3)
    body_ang_vel = _reshape_bodies(full_state[..., B * 10 : B * 13], B, 3)
    return body_pos, body_quat, body_lin_vel, body_ang_vel


def build_maxcoord_full_state(
    body_pos: torch.Tensor,
    body_quat: torch.Tensor,
    body_lin_vel: torch.Tensor,
    body_ang_vel: torch.Tensor,
) -> torch.Tensor:
    """Build flat maxcoord full state from per-body tensors.

    Args:
      body_pos: (..., B, 3) body positions
      body_quat: (..., B, 4) body quaternions
      body_lin_vel: (..., B, 3) body linear velocities
      body_ang_vel: (..., B, 3) body angular velocities

    Returns:
      (..., B*13) flat maxcoord full state
    """
    return torch.cat(
        [
            body_pos.flatten(-2, -1),
            body_quat.flatten(-2, -1),
            body_lin_vel.flatten(-2, -1),
            body_ang_vel.flatten(-2, -1),
        ],
        dim=-1,
    )


def compute_maxcoord_rep_state(full_state: torch.Tensor, num_bodies: int) -> torch.Tensor:
    """Convert maxcoord full state to root-relative rep state.

    Layout:
      [local_pos(B*3), local_rot_6d(B*6), local_vel(B*3), local_avel(B*3),
       root_height(1), up_dir(3)]

    Args:
      full_state: (..., B*13) flat maxcoord state
      num_bodies: B

    Returns:
      (..., B*15 + 4) rep state
    """
    body_pos, body_quat, body_lin_vel, body_ang_vel = decompose_maxcoord_state(full_state, num_bodies)

    # Root frame
    root_pos = body_pos[..., 0:1, :]  # (..., 1, 3)
    root_quat = body_quat[..., 0:1, :]  # (..., 1, 4)
    root_quat_inv = quat_inv(root_quat.reshape(-1, 4)).reshape(root_quat.shape)

    # Local positions: relative to root, rotated into root frame
    rel_pos_w = body_pos - root_pos  # (..., B, 3)
    local_pos = quat_apply_inverse(
        root_quat.expand_as(body_quat),
        rel_pos_w,
    )  # (..., B, 3)

    # Local rotations as 6D representation (first two columns of rotation matrix)
    rel_quat = quat_mul(
        root_quat_inv.expand_as(body_quat),
        body_quat,
    )  # (..., B, 4)
    local_rot_6d = quat_to_rot6d(rel_quat)  # (..., B, 6)

    # Local velocities: rotated into root frame
    local_vel = quat_apply_inverse(
        root_quat.expand_as(body_quat),
        body_lin_vel,
    )  # (..., B, 3)
    local_avel = quat_apply_inverse(
        root_quat.expand_as(body_quat),
        body_ang_vel,
    )  # (..., B, 3)

    # Root height (Z coordinate in world frame)
    root_height = body_pos[..., 0, 2:3]  # (..., 1)

    # Up direction in root frame
    world_down = torch.zeros_like(root_pos[..., 0, :])  # (..., 3)
    world_down[..., 2] = -1.0
    up_dir = quat_apply_inverse(root_quat[..., 0, :], world_down)  # (..., 3)

    # Flatten per-body features and concatenate
    return torch.cat(
        [
            local_pos.flatten(-2, -1),  # B*3
            local_rot_6d.flatten(-2, -1),  # B*6
            local_vel.flatten(-2, -1),  # B*3
            local_avel.flatten(-2, -1),  # B*3
            root_height,  # 1
            up_dir,  # 3
        ],
        dim=-1,
    )


def decompose_maxcoord_rep_state(
    rep_state: torch.Tensor, num_bodies: int
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Decompose flat maxcoord rep state into per-component tensors.

    Returns:
      local_pos: (..., B, 3)
      local_rot_6d: (..., B, 6)
      local_vel: (..., B, 3)
      local_avel: (..., B, 3)
      root_height: (..., 1)
      up_dir: (..., 3)
    """
    B = num_bodies
    off = 0
    local_pos = _reshape_bodies(rep_state[..., off : off + B * 3], B, 3)
    off += B * 3
    local_rot_6d = _reshape_bodies(rep_state[..., off : off + B * 6], B, 6)
    off += B * 6
    local_vel = _reshape_bodies(rep_state[..., off : off + B * 3], B, 3)
    off += B * 3
    local_avel = _reshape_bodies(rep_state[..., off : off + B * 3], B, 3)
    off += B * 3
    root_height = rep_state[..., off : off + 1]
    off += 1
    up_dir = rep_state[..., off : off + 3]
    return local_pos, local_rot_6d, local_vel, local_avel, root_height, up_dir


def world_frame_bodies(
    aux_state: torch.Tensor,
    rep_state: torch.Tensor,
    num_bodies: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover world-frame body kinematics from (aux_state, rep_state).

    Args:
      aux_state: (..., 7) [root_pos(3), root_quat(4)]
      rep_state: (..., B*15+4) rep state
      num_bodies: B

    Returns:
      body_pos: (..., B, 3)
      body_rot_mat: (..., B, 3, 3) world-frame rotation matrices
      body_lin_vel: (..., B, 3)
      body_ang_vel: (..., B, 3)
    """
    local_pos, local_rot_6d, local_vel, local_avel, _, _ = decompose_maxcoord_rep_state(rep_state, num_bodies)
    root_pos = aux_state[..., :3]
    root_mat = matrix_from_quat(aux_state[..., 3:7]).unsqueeze(-3)

    def rotate(vec: torch.Tensor) -> torch.Tensor:
        return small_matmul(root_mat, vec.unsqueeze(-1))[..., 0]

    body_pos = root_pos.unsqueeze(-2) + rotate(local_pos)
    body_rot_mat = small_matmul(root_mat, rot6d_to_matrix(local_rot_6d))
    body_lin_vel = rotate(local_vel)
    body_ang_vel = rotate(local_avel)
    return body_pos, body_rot_mat, body_lin_vel, body_ang_vel


def reconstruct_full_from_aux_and_rep(
    aux_state: torch.Tensor,
    rep_state: torch.Tensor,
    num_bodies: int,
) -> torch.Tensor:
    """Reconstruct maxcoord full state from aux_state + rep_state."""
    body_pos, body_rot_mat, body_lin_vel, body_ang_vel = world_frame_bodies(aux_state, rep_state, num_bodies)
    body_quat = matrix_to_quat(body_rot_mat)
    return build_maxcoord_full_state(body_pos, body_quat, body_lin_vel, body_ang_vel)
