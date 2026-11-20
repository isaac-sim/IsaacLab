# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Derived from HybridRobotics/SuperMimic, revision 5efbf114e11f725c31d83e217be987c13c0e4ce2.
# Copyright 2026 the SuperMimic authors. See docs/licenses/dependencies/supermimic-official-license.txt.
# Modified for Isaac Lab package boundaries and simulator interfaces.

"""Analytic inverse kinematics: joint angles and rates from tracked body poses.

Between a tracked parent and child sit two or three hinges. Moving each fixed
link rotation Q left through the hinges (``Q R_a(t) = R_{Qa}(t) Q``) gives
``R_p^T R_c = R_{b1}(t1) R_{b2}(t2) [R_{b3}(t3)] Y``, with ``b_i`` the hinge axes
in the parent frame at the zero pose and ``Y`` the fixed rotations' product. In
the frame ``X`` with columns ``b1, b2, b1 x b2``, with ``A = X`` and ``B = Y^T X``,
every chain is one XYZ Euler product ``E = (R_p A)^T (R_c B) = R_x(u1) R_y(u2) R_z(u3)``
with ``t = sign * u`` (``u3 = 0`` on two joints; ``sign`` flips a third axis
opposite ``b1 x b2``), and the relative angular velocity in that frame is
``(R_p A)^T (w_c - w_p) = u1' x + u2' R_x(u1) y + u3' E z``.
"""

from __future__ import annotations

import math

import mujoco
import torch

from isaaclab_rl.model_based.math import small_matmul
from isaaclab_rl.model_based.quaternion import matrix_from_quat

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
