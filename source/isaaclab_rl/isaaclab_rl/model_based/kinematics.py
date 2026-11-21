# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Portions derived from SuperMimic (MIT), Copyright (c) 2026 ETH Zurich, Chong Zhang.
# See docs/licenses/dependencies/supermimic-license.txt.

# Rotation and analytic IK portions derived from HybridRobotics/SuperMimic
# (Apache-2.0), revision 5efbf114e11f725c31d83e217be987c13c0e4ce2.
# Copyright 2026 the SuperMimic authors; see supermimic-official-license.txt
# in docs/licenses/dependencies/. Modified for native math and articulation interfaces.

"""Differentiable forward/inverse kinematics and WXYZ adapters for native rotation utilities.

Reference states and checkpoints store WXYZ quaternions. Matrix conversion and
inversion retain the reference norm reduction order. Native utilities use XYZW."""

from __future__ import annotations

import math

import numpy as np
import torch
from newton import JointType

from isaaclab.utils import math as native_math
from isaaclab.utils.math import axis_angle_from_quat, quat_from_angle_axis, quat_from_matrix


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


def to_native(quat: torch.Tensor) -> torch.Tensor:
    """Convert reference/checkpoint WXYZ quaternions to native XYZW."""
    return native_math.convert_quat(quat, to="xyzw")


def from_native(quat: torch.Tensor) -> torch.Tensor:
    """Convert native XYZW quaternions to reference/checkpoint WXYZ."""
    return native_math.convert_quat(quat, to="wxyz")


# Matrix conversion retains BSD-3-Clause Isaac Lab/mjlab arithmetic.
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# See the repository LICENSE.
@torch.jit.script
def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """WXYZ matrix conversion with the reference's FP32 norm reduction order.

    Reordering the four squared components before summation changes BPTT optimizer
    updates. Keep this BSD-licensed Isaac Lab formula at the reference boundary;
    multiplication and vector rotation delegate to the native utilities.
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def quat_inv(quat: torch.Tensor) -> torch.Tensor:
    """Inverse with the reference WXYZ norm reduction used by frame transforms."""
    conjugate = from_native(native_math.quat_conjugate(to_native(quat)))
    return conjugate / quat.pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-9)


def quat_mul(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Multiply reference WXYZ quaternions."""
    return from_native(native_math.quat_mul(to_native(first), to_native(second)))


def quat_apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by a reference WXYZ quaternion."""
    return native_math.quat_apply(to_native(quat), vector)


def quat_apply_inverse(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a reference WXYZ quaternion."""
    return native_math.quat_apply_inverse(to_native(quat), vector)


def yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    """Yaw component of a reference WXYZ quaternion."""
    return from_native(native_math.yaw_quat(to_native(quat)))


def subtract_frame_transforms(
    t01: torch.Tensor, q01: torch.Tensor, t02: torch.Tensor, q02: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Relative frame transform with reference WXYZ orientations."""
    inverse = quat_inv(q01)
    return quat_apply(inverse, t02 - t01), quat_mul(inverse, q02)


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


def _reduce(kinematics: ArticulationKinematics, body: int, row_of: dict):
    """Reduce a tracked-body chain to the reference analytic IK constants."""
    path = [body]
    while (ancestor := kinematics.parent[path[-1]]) not in row_of:
        if ancestor < 0:
            raise ValueError("Tracked bodies must descend from the first tracked body.")
        path.append(ancestor)
    y = torch.eye(3, dtype=torch.float64)
    axes, joints = [], []
    for link in reversed(path):
        y = y @ kinematics.off_R[link].cpu().double()
        if (joint := kinematics.joint_idx[link]) >= 0:
            if kinematics.joint_type[link] != JointType.REVOLUTE:
                raise ValueError("Analytic IK requires revolute joints.")
            axes.append(y @ kinematics.axes[link].cpu().double())
            joints.append(joint)
    if len(joints) not in (2, 3):
        raise ValueError("Analytic IK requires two or three joints per tracked-body chain.")
    x = torch.stack([axes[0], axes[1], torch.linalg.cross(axes[0], axes[1])])
    k = len(joints)
    sign, qpos0, lo, hi = torch.zeros(4, 3, dtype=torch.float64)
    sign[:k] = 1.0
    if k == 3:
        sign[2] = round(float(axes[2] @ x[2]))
    lo[:k], hi[:k] = kinematics.q_lo[joints].cpu().double(), kinematics.q_hi[joints].cpu().double()
    cols = joints + [-1] * (3 - k)
    return row_of[ancestor], x.T, (x @ y).T, sign, qpos0, lo, hi, cols


class AnalyticIK:
    """The reference batched analytic solve, bound to the simulated articulation's chains."""

    def __init__(self, kinematics: ArticulationKinematics):
        tracked = kinematics.tracked_idx
        row_of = {body: row for row, body in enumerate(tracked)}
        parent, a, b, sign, qpos0, lo, hi, cols = zip(
            *(_reduce(kinematics, body, row_of) for body in tracked[1:]), strict=True
        )
        device, dtype = kinematics.device, kinematics.axes.dtype
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


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Matrix to rotation vector with a finite zero-angle derivative for model fitting."""
    return axis_angle_from_quat(quat_from_matrix(R))


class ArticulationKinematics:
    """Differentiable FK using the joint frames of the already loaded Newton articulation.

    The USD importer owns topology, anchors, axes, and limits. Constants are read once
    from the first environment; FK evaluates arbitrary states without touching the simulator.
    """

    def __init__(self, model, articulation_id: int, device: str, joint_names: list[str], tracked: list[str]):
        self.device = device
        start = int(model.articulation_start.numpy()[articulation_id])
        end = int(model.articulation_end.numpy()[articulation_id])
        parents, children, kinds = (
            a.numpy()[start:end] for a in (model.joint_parent, model.joint_child, model.joint_type)
        )
        roots = [i for i, kind in enumerate(kinds) if kind == JointType.FREE and parents[i] < 0]
        if len(roots) != 1:
            raise ValueError("Motion tracking requires one floating root in the articulation.")
        root = roots[0]
        order, local = [root], {int(children[root]): 0}
        pending = [i for i in range(end - start) if i != root]
        while pending:
            ready = [i for i in pending if int(parents[i]) in local]
            if not ready:
                raise ValueError("Kinematics requires an articulation tree rooted at the floating body.")
            for i in ready:
                if int(children[i]) in local:
                    raise ValueError("Kinematics does not support closed joint loops.")
                local[int(children[i])] = len(order)
                order.append(i)
                pending.remove(i)
        self.body_names = [model.body_label[int(children[i])].rsplit("/", 1)[-1] for i in order]
        self.tracked_idx = [self.body_names.index(name) for name in tracked]
        self.parent = [-1 if i == root else local[int(parents[i])] for i in order]
        self.joint_type = [int(kinds[i]) for i in order]
        self.joint_idx = []
        dofs = model.joint_qd_start.numpy()[start:end]
        joint_axes = model.joint_axis.numpy()
        lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
        axes = np.zeros((len(order), 3))
        lo, hi = np.full(len(joint_names), -np.inf), np.full(len(joint_names), np.inf)
        used = set()
        for row, i in enumerate(order):
            kind = kinds[i]
            index = -1
            if i != root and kind != JointType.FIXED:
                if kind not in (JointType.REVOLUTE, JointType.PRISMATIC):
                    raise ValueError("Differentiable FK supports fixed, revolute, and prismatic child joints.")
                name = model.joint_label[start + i].rsplit("/", 1)[-1]
                index = joint_names.index(name)
                if index in used:
                    raise ValueError(f"Repeated kinematic joint {name!r}.")
                used.add(index)
                axes[row] = joint_axes[dofs[i]]
                lo[index], hi[index] = lower[dofs[i]], upper[dofs[i]]
            self.joint_idx.append(index)
        if len(used) != len(joint_names):
            raise ValueError(f"Missing kinematic joints: {set(range(len(joint_names))) - used}.")

        def tensor(values):
            return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device)

        xp, xc = (tensor(a.numpy()[start:end][order]) for a in (model.joint_X_p, model.joint_X_c))
        rp, rc = native_math.matrix_from_quat(xp[:, 3:]), native_math.matrix_from_quat(xc[:, 3:])
        self.off_R = rp @ rc.transpose(-1, -2)
        self.anchors = xc[:, :3]
        self.off_p = xp[:, :3] - (self.off_R @ self.anchors[..., None]).squeeze(-1)
        self.axes = (rc @ tensor(axes)[..., None]).squeeze(-1)
        self.q_lo, self.q_hi = tensor(lo), tensor(hi)
        self.num_bodies = len(order)

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        positions, rotations, velocities, angular = [p], [R], [v], [w]
        for i in range(1, self.num_bodies):
            parent, joint = self.parent[i], self.joint_idx[i]
            Rp, pp, vp, wp = rotations[parent], positions[parent], velocities[parent], angular[parent]
            origin = pp + (Rp @ self.off_p[i, :, None]).squeeze(-1)
            base_R = Rp @ self.off_R[i]
            Ri, pi, wi = base_R, origin, wp
            vi = vp + torch.cross(wp, origin - pp, dim=-1)
            if joint >= 0:
                axis = (base_R @ self.axes[i, :, None]).squeeze(-1)
                angle = q[..., joint]
                speed = qd[..., joint : joint + 1]
                if self.joint_type[i] == JointType.REVOLUTE:
                    Ri = base_R @ native_math.matrix_from_quat(quat_from_angle_axis(angle, self.axes[i]))
                    pivot = origin + (base_R @ self.anchors[i, :, None]).squeeze(-1)
                    pi = pivot - (Ri @ self.anchors[i, :, None]).squeeze(-1)
                    wi = wp + axis * speed
                    vi = vp + torch.cross(wp, pivot - pp, dim=-1) + torch.cross(wi, pi - pivot, dim=-1)
                else:
                    pi = origin + axis * angle[..., None]
                    vi = vp + torch.cross(wp, pi - pp, dim=-1) + axis * speed
            positions.append(pi)
            rotations.append(Ri)
            velocities.append(vi)
            angular.append(wi)
        idx = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (
            torch.stack([positions[i] for i in idx], -2),
            torch.stack([rotations[i] for i in idx], -3),
            torch.stack([velocities[i] for i in idx], -2),
            torch.stack([angular[i] for i in idx], -2),
        )


class ForwardKinematics:
    """Adapt native articulation kinematics to the reference algorithm's callable interface."""

    def __init__(self, robot, model, body_names: list[str], device: str):
        articulation_id = int(robot.root_view.articulation_ids.numpy().flat[0])
        self.joint_names = robot.joint_names
        self.body_names = body_names
        self.kinematics = ArticulationKinematics(model, articulation_id, device, self.joint_names, body_names)

    def __call__(self, root_pos, root_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel):
        pos, rot, vel, ang_vel = self.kinematics.forward(
            root_pos,
            native_math.matrix_from_quat(to_native(root_quat)),
            root_lin_vel,
            root_ang_vel,
            joint_pos,
            joint_vel,
        )
        return pos, matrix_to_quat(rot), vel, ang_vel

    def analytic_ik(self) -> AnalyticIK:
        """Build the reference maximal-coordinate joint reconstruction for this articulation."""
        return AnalyticIK(self.kinematics)
