# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Derived from HybridRobotics/SuperMimic, revision 5efbf114e11f725c31d83e217be987c13c0e4ce2.
# Copyright 2026 the SuperMimic authors. See docs/licenses/dependencies/supermimic-official-license.txt.
# Modified for Isaac Lab package boundaries and simulator interfaces.

"""Maximal-coordinate representation helper functions.

Full state layout:
  [body_pos(B*3), body_quat(B*4), body_lin_vel(B*3), body_ang_vel(B*3)]
  where B = num_bodies.  Total dim = B*13.

Rep state layout (matches control_vae's state2ob):
  [local_pos(B*3), local_rot_6d(B*6), local_vel(B*3), local_avel(B*3),
   root_height(1), up_dir(3)]
  Total dim = B*15 + 4

Aux state layout:
  [root_pos(3), root_quat(4)]
  Total dim = 7

Prediction layout (dyn_head output):
  [vel_delta(B*3), avel_delta(B*3)]  (in root body frame)
  Total dim = B*6
"""

from __future__ import annotations

import math

import mujoco
import torch

from isaaclab_rl.model_based.math import (
    matrix_from_quat,
    matrix_to_quat,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    quat_to_rot6d,
    rot6d_to_matrix,
    small_matmul,
)


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


_TWO_PI = 2 * math.pi
_SIN_CLAMP = 1e-6  # keeps asin' finite where |sin(u2)| reaches 1


def _atan2(y: torch.Tensor, x: torch.Tensor, damp: float) -> torch.Tensor:
    """The exact angle, with a gradient bounded by ``1 / (2 sqrt(damp))``: the
    true gradient ``(x dy - y dx) / (x^2 + y^2)`` is unbounded where both vanish."""
    xd, yd = x.detach(), y.detach()
    return torch.atan2(yd, xd) + (xd * y - yd * x) / (xd * xd + yd * yd + damp)


def _damped(value: torch.Tensor, surrogate: torch.Tensor) -> torch.Tensor:
    """``value`` in the forward pass; ``surrogate`` supplies the gradient."""
    return surrogate + (value - surrogate).detach()


def _reduce(mj_model: mujoco.MjModel, body: int, row_of: dict, col_of: dict):
    """Constants of the chain from tracked ``body`` up to its tracked ancestor,
    padded to three joints: ``(parent_row, A, B, sign, qpos0, lo, hi, cols)``."""
    path = [body]
    while (ancestor := int(mj_model.body_parentid[path[-1]])) not in row_of:
        path.append(ancestor)
    y = torch.eye(3, dtype=torch.float64)
    axes, joints = [], []
    for link in reversed(path):
        y = y @ matrix_from_quat(torch.from_numpy(mj_model.body_quat[link]).double())
        if (joint := int(mj_model.body_jntadr[link])) >= 0:
            axes.append(y @ torch.from_numpy(mj_model.jnt_axis[joint]).double())
            joints.append(joint)
    x = torch.stack([axes[0], axes[1], torch.linalg.cross(axes[0], axes[1])])

    k = len(joints)
    qadr = [int(mj_model.jnt_qposadr[j]) for j in joints]
    sign, qpos0, lo, hi = torch.zeros(4, 3, dtype=torch.float64)
    sign[:k] = 1.0
    if k == 3:
        sign[2] = round(float(axes[2] @ x[2]))
    qpos0[:k] = torch.from_numpy(mj_model.qpos0[qadr])
    lo[:k], hi[:k] = torch.from_numpy(mj_model.jnt_range[joints]).T
    cols = [col_of[adr] for adr in qadr] + [-1] * (3 - k)
    return row_of[ancestor], x.T, (x @ y).T, sign, qpos0, lo, hi, cols


class AnalyticIK:
    """One robot's chains as stacked constants, and the batched solve over them.

    ``body_ids`` are the robot's global body ids in entity order, ``joint_q_adr``
    the qpos addresses in ``qj`` column order, ``tracked_body_rows`` the entity
    slots the maxcoord state carries; row 0 is the root every chain hangs below.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        body_ids: torch.Tensor,
        joint_q_adr: torch.Tensor,
        tracked_body_rows: torch.Tensor,
        device: torch.device | str,
        dtype: torch.dtype,
    ):
        tracked = [int(body_ids[row]) for row in tracked_body_rows]
        row_of = {body: row for row, body in enumerate(tracked)}
        col_of = {int(adr): col for col, adr in enumerate(joint_q_adr)}
        parent, a, b, sign, qpos0, lo, hi, cols = zip(
            *(_reduce(mj_model, body, row_of, col_of) for body in tracked[1:]), strict=True
        )
        self.parent = torch.tensor(parent, device=device)
        self.child = torch.arange(1, len(tracked), device=device)
        self.A, self.B, self.sign, self.qpos0, self.lo, self.hi = (
            torch.stack(t).to(device, dtype) for t in (a, b, sign, qpos0, lo, hi)
        )
        self.three = self.sign[:, 2] != 0
        # qj column of each (chain, slot); -1 on the unused slot of a two-joint chain.
        slot_col = torch.tensor(cols).flatten()
        self.perm = slot_col.argsort()[slot_col.lt(0).sum() :].to(device)

    def __call__(
        self, rots: torch.Tensor, avels: torch.Tensor, damp: float = 1e-2
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Joints from tracked-body rotations ``(..., B, 3, 3)`` and angular
        velocities ``(..., B, 3)`` in a common frame, in ``qj`` column order.

        ``damp`` bounds every gradient and, for the outer pair of a three-joint
        chain within ``cos^2(u2) < damp`` of the gimbal, the rates: the split of
        the observed rate between their coinciding axes is unobservable there.
        Every other value is exact.
        """
        p = small_matmul(rots[..., self.parent, :, :], self.A).transpose(-1, -2)
        e = small_matmul(p, small_matmul(rots[..., self.child, :, :], self.B))
        dw = avels[..., self.child, :] - avels[..., self.parent, :]
        w = small_matmul(p, dw[..., None])[..., 0]

        # Two joints: column y of E is (0, cos u1, sin u1), row x is (cos u2, 0, sin u2).
        u1_xy = _atan2(e[..., 2, 1], e[..., 1, 1], damp)
        u2_xy = _atan2(e[..., 0, 2], e[..., 0, 0], damp)
        # Three joints: E_xz = sin u2; below it column z is cos u2 (-sin u1, cos u1)
        # and beside it row x is cos u2 (-sin u3, cos u3). The same E is also
        # (u1 + pi, pi - u2, u3 + pi): the joint limits pick, and the neutral pose
        # breaks ties.
        u1 = _atan2(-e[..., 1, 2], e[..., 2, 2], damp)
        u2 = torch.asin(e[..., 0, 2].clamp(-1 + _SIN_CLAMP, 1 - _SIN_CLAMP))
        u3 = _atan2(-e[..., 0, 1], e[..., 0, 0], damp)
        u = torch.where(
            self.three[:, None],
            torch.stack([u1, u2, u3], dim=-1),
            torch.stack([u1_xy, u2_xy, torch.zeros_like(u3)], dim=-1),
        )
        twin = torch.stack([u1 + math.pi, math.pi - u2, u3 + math.pi], dim=-1)
        q, q_twin = self._qpos(u), self._qpos(twin)
        use_twin = (self.three & (self._cost(q_twin) < self._cost(q)))[..., None]
        u, q = torch.where(use_twin, twin, u), torch.where(use_twin, q_twin, q)

        # Rates: E z = (sin u2, -sin u1 cos u2, cos u1 cos u2), so projecting w on
        # column z below E_xz gives cos^2(u2) u3', and on R_x(u1) y gives u2'.
        num = e[..., 1, 2] * w[..., 1] + e[..., 2, 2] * w[..., 2]
        cos2 = e[..., 1, 2] ** 2 + e[..., 2, 2] ** 2
        u3d = _damped(num / cos2.clamp(min=damp), num / (cos2 + damp))
        u3d = torch.where(self.three, u3d, 0.0)
        u2d = torch.cos(u[..., 0]) * w[..., 1] + torch.sin(u[..., 0]) * w[..., 2]
        u1d = w[..., 0] - e[..., 0, 2] * u3d
        v = self.sign * torch.stack([u1d, u2d, u3d], dim=-1)
        return q.flatten(-2)[..., self.perm], v.flatten(-2)[..., self.perm]

    def _qpos(self, u: torch.Tensor) -> torch.Tensor:
        """Hinge angles, each on the turn of the circle centered on its joint's range."""
        q = self.sign * u + self.qpos0
        return q - _TWO_PI * torch.round((q - (self.lo + self.hi) / 2) / _TWO_PI)

    def _cost(self, q: torch.Tensor) -> torch.Tensor:
        """Range violation, tie-broken a thousandfold weaker by distance from neutral."""
        violation = (self.lo - q).clamp(min=0) + (q - self.hi).clamp(min=0)
        return (violation + 1e-3 * (q - self.qpos0).abs()).sum(-1)
