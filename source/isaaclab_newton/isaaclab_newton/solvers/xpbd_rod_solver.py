# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# XPBD algorithm and Warp kernels adapted from:
#   Newton Physics Engine — SolverXPBDRod (PR #1981)
#   SPDX-License-Identifier: Apache-2.0
#   Copyright (c) 2025 The Newton Developers

"""Self-contained XPBD Cosserat rod solver with block-Thomas direct solve.

All Warp kernels are embedded — **no external ``newton`` dependency required**.
The algorithm is a faithful port of Newton's ``SolverXPBDRod`` (block Thomas path):

1. Predict positions & rotations (explicit Euler + gravity + damping).
2. Zero Lagrange-multiplier accumulators and compute compliance.
3. Project constraints via block-Thomas direct solve on the 6×6
   block-tridiagonal JMJT system (stretch + Darboux).
4. Optional floor collision.
5. Integrate positions & rotations.

Usage::

    from isaaclab_newton.solvers import XPBDRodSolver, RodConfig

    cfg = RodConfig()
    solver = XPBDRodSolver(cfg)
    for _ in range(300):
        solver.step(cfg.solver.dt)
    print(solver.positions)
"""

from __future__ import annotations

import math

import numpy as np
import torch
import warp as wp

from .rod_data import RodConfig

__all__ = [
    "XPBDRodSolver",
    "XPBD_DIRECT_SOLVE_BLOCK_THOMAS",
    "XPBD_DIRECT_SOLVE_SPLIT_THOMAS",
    "XPBD_DIRECT_SOLVE_BLOCK_JACOBI",
    "XPBD_DIRECT_SOLVE_BANDED_CHOLESKY",
    "XPBD_DIRECT_SOLVE_BACKENDS",
]

# ---------------------------------------------------------------------------
# Solver backend constants (parity with Newton upstream)
# ---------------------------------------------------------------------------
XPBD_DIRECT_SOLVE_BLOCK_THOMAS = "block_thomas"
XPBD_DIRECT_SOLVE_SPLIT_THOMAS = "split_thomas"
XPBD_DIRECT_SOLVE_BLOCK_JACOBI = "block_jacobi"
XPBD_DIRECT_SOLVE_BANDED_CHOLESKY = "banded_cholesky"
XPBD_DIRECT_SOLVE_BACKENDS = (
    XPBD_DIRECT_SOLVE_BLOCK_THOMAS,
    XPBD_DIRECT_SOLVE_SPLIT_THOMAS,
    XPBD_DIRECT_SOLVE_BLOCK_JACOBI,
    XPBD_DIRECT_SOLVE_BANDED_CHOLESKY,
)
_XPBD_TILE = 64        # tiled Cholesky threshold (n_dofs <= _XPBD_TILE)
_XPBD_BAND_KD = 11     # banded Cholesky half-bandwidth
_XPBD_BAND_LDAB = 34   # banded storage leading dimension (2*KD + KD + 1 = 34)
_XPBD_BLOCK_DIM = 128  # tiled launch block dimension


# ============================================================================
# Warp @wp.func helpers — quaternion operations
# ============================================================================


@wp.func
def _xr_qmul(q1: wp.quat, q2: wp.quat) -> wp.quat:
    return wp.quat(
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
    )


@wp.func
def _xr_qconj(q: wp.quat) -> wp.quat:
    return wp.quat(-q.x, -q.y, -q.z, q.w)


@wp.func
def _xr_qnorm(q: wp.quat) -> wp.quat:
    n = wp.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if n < 1.0e-8:
        return q
    inv = 1.0 / n
    return wp.quat(q.x * inv, q.y * inv, q.z * inv, q.w * inv)


@wp.func
def _xr_qrot(q: wp.quat, v: wp.vec3) -> wp.vec3:
    tx = 2.0 * (q.y * v.z - q.z * v.y)
    ty = 2.0 * (q.z * v.x - q.x * v.z)
    tz = 2.0 * (q.x * v.y - q.y * v.x)
    return wp.vec3(
        v.x + q.w * tx + q.y * tz - q.z * ty,
        v.y + q.w * ty + q.z * tx - q.x * tz,
        v.z + q.w * tz + q.x * ty - q.y * tx,
    )


@wp.func
def _xr_qcorr(q: wp.quat, dtheta: wp.vec3) -> wp.quat:
    nsq = dtheta.x * dtheta.x + dtheta.y * dtheta.y + dtheta.z * dtheta.z
    if nsq < 1.0e-20:
        return q
    cx = 0.5 * (q.w * dtheta.x + q.z * dtheta.y - q.y * dtheta.z)
    cy = 0.5 * (-q.z * dtheta.x + q.w * dtheta.y + q.x * dtheta.z)
    cz = 0.5 * (q.y * dtheta.x - q.x * dtheta.y + q.w * dtheta.z)
    cw = 0.5 * (-q.x * dtheta.x - q.y * dtheta.y - q.z * dtheta.z)
    return _xr_qnorm(wp.quat(q.x + cx, q.y + cy, q.z + cz, q.w + cw))


# ============================================================================
# Warp @wp.func helpers — Jacobian indexing
# ============================================================================


@wp.func
def _xr_jidx(edge: int, row: int, col: int) -> int:
    return edge * 36 + row * 6 + col


# ============================================================================
# Warp @wp.func helpers — 3×3 matrix operations
# ============================================================================


@wp.func
def _xr_m33_add(a: wp.mat33, b: wp.mat33) -> wp.mat33:
    return wp.mat33(
        a[0, 0] + b[0, 0], a[0, 1] + b[0, 1], a[0, 2] + b[0, 2],
        a[1, 0] + b[1, 0], a[1, 1] + b[1, 1], a[1, 2] + b[1, 2],
        a[2, 0] + b[2, 0], a[2, 1] + b[2, 1], a[2, 2] + b[2, 2],
    )


@wp.func
def _xr_m33_sub(a: wp.mat33, b: wp.mat33) -> wp.mat33:
    return wp.mat33(
        a[0, 0] - b[0, 0], a[0, 1] - b[0, 1], a[0, 2] - b[0, 2],
        a[1, 0] - b[1, 0], a[1, 1] - b[1, 1], a[1, 2] - b[1, 2],
        a[2, 0] - b[2, 0], a[2, 1] - b[2, 1], a[2, 2] - b[2, 2],
    )


@wp.func
def _xr_m33_mul(a: wp.mat33, b: wp.mat33) -> wp.mat33:
    return wp.mat33(
        a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0] + a[0, 2] * b[2, 0],
        a[0, 0] * b[0, 1] + a[0, 1] * b[1, 1] + a[0, 2] * b[2, 1],
        a[0, 0] * b[0, 2] + a[0, 1] * b[1, 2] + a[0, 2] * b[2, 2],
        a[1, 0] * b[0, 0] + a[1, 1] * b[1, 0] + a[1, 2] * b[2, 0],
        a[1, 0] * b[0, 1] + a[1, 1] * b[1, 1] + a[1, 2] * b[2, 1],
        a[1, 0] * b[0, 2] + a[1, 1] * b[1, 2] + a[1, 2] * b[2, 2],
        a[2, 0] * b[0, 0] + a[2, 1] * b[1, 0] + a[2, 2] * b[2, 0],
        a[2, 0] * b[0, 1] + a[2, 1] * b[1, 1] + a[2, 2] * b[2, 1],
        a[2, 0] * b[0, 2] + a[2, 1] * b[1, 2] + a[2, 2] * b[2, 2],
    )


@wp.func
def _xr_m33v(a: wp.mat33, v: wp.vec3) -> wp.vec3:
    return wp.vec3(
        a[0, 0] * v[0] + a[0, 1] * v[1] + a[0, 2] * v[2],
        a[1, 0] * v[0] + a[1, 1] * v[1] + a[1, 2] * v[2],
        a[2, 0] * v[0] + a[2, 1] * v[1] + a[2, 2] * v[2],
    )


@wp.func
def _xr_m33_t(a: wp.mat33) -> wp.mat33:
    return wp.mat33(
        a[0, 0], a[1, 0], a[2, 0],
        a[0, 1], a[1, 1], a[2, 1],
        a[0, 2], a[1, 2], a[2, 2],
    )


@wp.func
def _xr_chol(a: wp.mat33) -> wp.mat33:
    eps = 1.0e-9
    l00 = wp.sqrt(wp.max(a[0, 0], eps))
    l10 = a[1, 0] / l00
    l20 = a[2, 0] / l00
    l11 = wp.sqrt(wp.max(a[1, 1] - l10 * l10, eps))
    l21 = (a[2, 1] - l20 * l10) / l11
    l22 = wp.sqrt(wp.max(a[2, 2] - l20 * l20 - l21 * l21, eps))
    return wp.mat33(l00, 0.0, 0.0, l10, l11, 0.0, l20, l21, l22)


@wp.func
def _xr_solvL(L: wp.mat33, b: wp.vec3) -> wp.vec3:
    y0 = b[0] / L[0, 0]
    y1 = (b[1] - L[1, 0] * y0) / L[1, 1]
    y2 = (b[2] - L[2, 0] * y0 - L[2, 1] * y1) / L[2, 2]
    return wp.vec3(y0, y1, y2)


@wp.func
def _xr_solvU(L: wp.mat33, b: wp.vec3) -> wp.vec3:
    x2 = b[2] / L[2, 2]
    x1 = (b[1] - L[2, 1] * x2) / L[1, 1]
    x0 = (b[0] - L[1, 0] * x1 - L[2, 0] * x2) / L[0, 0]
    return wp.vec3(x0, x1, x2)


@wp.func
def _xr_cholsol(L: wp.mat33, b: wp.vec3) -> wp.vec3:
    return _xr_solvU(L, _xr_solvL(L, b))


# ============================================================================
# Warp @wp.func helpers — inverse inertia
# ============================================================================


@wp.func
def _xr_invI_v(inv_inertia: wp.array(dtype=wp.float32), idx: int, v: wp.vec3) -> wp.vec3:
    b = idx * 9
    return wp.vec3(
        inv_inertia[b + 0] * v.x + inv_inertia[b + 1] * v.y + inv_inertia[b + 2] * v.z,
        inv_inertia[b + 3] * v.x + inv_inertia[b + 4] * v.y + inv_inertia[b + 5] * v.z,
        inv_inertia[b + 6] * v.x + inv_inertia[b + 7] * v.y + inv_inertia[b + 8] * v.z,
    )


# ============================================================================
# Warp @wp.func helpers — 6×6 block operations (stored as 4 × mat33)
# ============================================================================


@wp.func
def _xr_bidx(block: int, row: int, col: int) -> int:
    return block * 36 + row * 6 + col


@wp.func
def _xr_load_blk(
    blks: wp.array(dtype=wp.float32), blk: int
) -> tuple[wp.mat33, wp.mat33, wp.mat33, wp.mat33]:
    base = blk * 36
    A = wp.mat33(
        blks[base + 0], blks[base + 1], blks[base + 2],
        blks[base + 6], blks[base + 7], blks[base + 8],
        blks[base + 12], blks[base + 13], blks[base + 14],
    )
    B = wp.mat33(
        blks[base + 3], blks[base + 4], blks[base + 5],
        blks[base + 9], blks[base + 10], blks[base + 11],
        blks[base + 15], blks[base + 16], blks[base + 17],
    )
    C = wp.mat33(
        blks[base + 18], blks[base + 19], blks[base + 20],
        blks[base + 24], blks[base + 25], blks[base + 26],
        blks[base + 30], blks[base + 31], blks[base + 32],
    )
    D = wp.mat33(
        blks[base + 21], blks[base + 22], blks[base + 23],
        blks[base + 27], blks[base + 28], blks[base + 29],
        blks[base + 33], blks[base + 34], blks[base + 35],
    )
    return A, B, C, D


@wp.func
def _xr_store_blk(
    blks: wp.array(dtype=wp.float32), blk: int,
    A: wp.mat33, B: wp.mat33, C: wp.mat33, D: wp.mat33,
):
    base = blk * 36
    blks[base + 0] = A[0, 0]; blks[base + 1] = A[0, 1]; blks[base + 2] = A[0, 2]
    blks[base + 3] = B[0, 0]; blks[base + 4] = B[0, 1]; blks[base + 5] = B[0, 2]
    blks[base + 6] = A[1, 0]; blks[base + 7] = A[1, 1]; blks[base + 8] = A[1, 2]
    blks[base + 9] = B[1, 0]; blks[base + 10] = B[1, 1]; blks[base + 11] = B[1, 2]
    blks[base + 12] = A[2, 0]; blks[base + 13] = A[2, 1]; blks[base + 14] = A[2, 2]
    blks[base + 15] = B[2, 0]; blks[base + 16] = B[2, 1]; blks[base + 17] = B[2, 2]
    blks[base + 18] = C[0, 0]; blks[base + 19] = C[0, 1]; blks[base + 20] = C[0, 2]
    blks[base + 21] = D[0, 0]; blks[base + 22] = D[0, 1]; blks[base + 23] = D[0, 2]
    blks[base + 24] = C[1, 0]; blks[base + 25] = C[1, 1]; blks[base + 26] = C[1, 2]
    blks[base + 27] = D[1, 0]; blks[base + 28] = D[1, 1]; blks[base + 29] = D[1, 2]
    blks[base + 30] = C[2, 0]; blks[base + 31] = C[2, 1]; blks[base + 32] = C[2, 2]
    blks[base + 33] = D[2, 0]; blks[base + 34] = D[2, 1]; blks[base + 35] = D[2, 2]


@wp.func
def _xr_load_v6(vals: wp.array(dtype=wp.float32), blk: int) -> tuple[wp.vec3, wp.vec3]:
    base = blk * 6
    return (
        wp.vec3(vals[base + 0], vals[base + 1], vals[base + 2]),
        wp.vec3(vals[base + 3], vals[base + 4], vals[base + 5]),
    )


@wp.func
def _xr_store_v6(vals: wp.array(dtype=wp.float32), blk: int, v0: wp.vec3, v1: wp.vec3):
    base = blk * 6
    vals[base + 0] = v0[0]; vals[base + 1] = v0[1]; vals[base + 2] = v0[2]
    vals[base + 3] = v1[0]; vals[base + 4] = v1[1]; vals[base + 5] = v1[2]


@wp.func
def _xr_blk_col(
    blks: wp.array(dtype=wp.float32), blk: int, col: int
) -> tuple[wp.vec3, wp.vec3]:
    base = blk * 36
    v0 = wp.vec3(blks[base + 0 * 6 + col], blks[base + 1 * 6 + col], blks[base + 2 * 6 + col])
    v1 = wp.vec3(blks[base + 3 * 6 + col], blks[base + 4 * 6 + col], blks[base + 5 * 6 + col])
    return v0, v1


@wp.func
def _xr_blk_set_col(
    blks: wp.array(dtype=wp.float32), blk: int, col: int, v0: wp.vec3, v1: wp.vec3,
):
    base = blk * 36
    blks[base + 0 * 6 + col] = v0[0]
    blks[base + 1 * 6 + col] = v0[1]
    blks[base + 2 * 6 + col] = v0[2]
    blks[base + 3 * 6 + col] = v1[0]
    blks[base + 4 * 6 + col] = v1[1]
    blks[base + 5 * 6 + col] = v1[2]


@wp.func
def _xr_blk_mul(
    A: wp.mat33, B: wp.mat33, C: wp.mat33, D: wp.mat33,
    E: wp.mat33, F: wp.mat33, G: wp.mat33, H: wp.mat33,
) -> tuple[wp.mat33, wp.mat33, wp.mat33, wp.mat33]:
    return (
        _xr_m33_add(_xr_m33_mul(A, E), _xr_m33_mul(B, G)),
        _xr_m33_add(_xr_m33_mul(A, F), _xr_m33_mul(B, H)),
        _xr_m33_add(_xr_m33_mul(C, E), _xr_m33_mul(D, G)),
        _xr_m33_add(_xr_m33_mul(C, F), _xr_m33_mul(D, H)),
    )


@wp.func
def _xr_blk_sub(
    A: wp.mat33, B: wp.mat33, C: wp.mat33, D: wp.mat33,
    E: wp.mat33, F: wp.mat33, G: wp.mat33, H: wp.mat33,
) -> tuple[wp.mat33, wp.mat33, wp.mat33, wp.mat33]:
    return (_xr_m33_sub(A, E), _xr_m33_sub(B, F), _xr_m33_sub(C, G), _xr_m33_sub(D, H))


@wp.func
def _xr_blk_mv(
    A: wp.mat33, B: wp.mat33, C: wp.mat33, D: wp.mat33,
    v0: wp.vec3, v1: wp.vec3,
) -> tuple[wp.vec3, wp.vec3]:
    top = _xr_m33v(A, v0) + _xr_m33v(B, v1)
    bot = _xr_m33v(C, v0) + _xr_m33v(D, v1)
    return top, bot


@wp.func
def _xr_blk_solve(
    A: wp.mat33, B: wp.mat33, C: wp.mat33, D: wp.mat33,
    b0: wp.vec3, b1: wp.vec3,
) -> tuple[wp.vec3, wp.vec3]:
    L11 = _xr_chol(A)
    c0 = wp.vec3(C[0, 0], C[0, 1], C[0, 2])
    c1 = wp.vec3(C[1, 0], C[1, 1], C[1, 2])
    c2 = wp.vec3(C[2, 0], C[2, 1], C[2, 2])
    y0 = _xr_solvL(L11, c0)
    y1 = _xr_solvL(L11, c1)
    y2 = _xr_solvL(L11, c2)
    L21 = wp.mat33(y0[0], y0[1], y0[2], y1[0], y1[1], y1[2], y2[0], y2[1], y2[2])
    L21t = _xr_m33_t(L21)
    S = _xr_m33_sub(D, _xr_m33_mul(L21, L21t))
    L22 = _xr_chol(S)
    yb0 = _xr_solvL(L11, b0)
    tmp = b1 - _xr_m33v(L21, yb0)
    yb1 = _xr_solvL(L22, tmp)
    x1 = _xr_solvU(L22, yb1)
    x0 = _xr_solvU(L11, yb0 - _xr_m33v(L21t, x1))
    return x0, x1


# ============================================================================
# Warp @wp.func helpers — 3×3 block storage (for split-Thomas backend)
# ============================================================================


@wp.func
def _xr_bidx_3x3(block: int, row: int, col: int) -> int:
    return block * 9 + row * 3 + col


@wp.func
def _xr_load_blk_3x3(blocks: wp.array(dtype=wp.float32), block: int) -> wp.mat33:
    base = block * 9
    return wp.mat33(
        blocks[base + 0], blocks[base + 1], blocks[base + 2],
        blocks[base + 3], blocks[base + 4], blocks[base + 5],
        blocks[base + 6], blocks[base + 7], blocks[base + 8],
    )


@wp.func
def _xr_load_v3b(values: wp.array(dtype=wp.float32), block: int) -> wp.vec3:
    base = block * 3
    return wp.vec3(values[base], values[base + 1], values[base + 2])


@wp.func
def _xr_store_v3b(values: wp.array(dtype=wp.float32), block: int, v: wp.vec3):
    base = block * 3
    values[base] = v[0]
    values[base + 1] = v[1]
    values[base + 2] = v[2]


@wp.func
def _xr_blk_col_3x3(blocks: wp.array(dtype=wp.float32), block: int, col: int) -> wp.vec3:
    base = block * 9
    return wp.vec3(
        blocks[base + 0 * 3 + col],
        blocks[base + 1 * 3 + col],
        blocks[base + 2 * 3 + col],
    )


@wp.func
def _xr_blk_set_col_3x3(blocks: wp.array(dtype=wp.float32), block: int, col: int, v: wp.vec3):
    base = block * 9
    blocks[base + 0 * 3 + col] = v[0]
    blocks[base + 1 * 3 + col] = v[1]
    blocks[base + 2 * 3 + col] = v[2]


# ============================================================================
# Warp @wp.func helpers — Jacobian dot-product for corrections
# ============================================================================


@wp.func
def _xr_jdot(
    J: wp.array(dtype=wp.float32), edge: int, col: int,
    d0: float, d1: float, d2: float, d3: float, d4: float, d5: float,
) -> float:
    return (
        J[_xr_jidx(edge, 0, col)] * d0
        + J[_xr_jidx(edge, 1, col)] * d1
        + J[_xr_jidx(edge, 2, col)] * d2
        + J[_xr_jidx(edge, 3, col)] * d3
        + J[_xr_jidx(edge, 4, col)] * d4
        + J[_xr_jidx(edge, 5, col)] * d5
    )


# ============================================================================
# Warp kernels — integration
# ============================================================================


@wp.kernel
def _xr_predict_pos(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=wp.float32),
    gravity: wp.vec3,
    dt: float,
    damping: float,
    pred: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    w = inv_mass[i]
    if w > 0.0:
        v = vel[i] + (forces[i] * w + gravity) * dt
        v = v * (1.0 - damping)
        vel[i] = v
        pred[i] = pos[i] + v * dt
    else:
        vel[i] = wp.vec3(0.0, 0.0, 0.0)
        pred[i] = pos[i]


@wp.kernel
def _xr_predict_rot(
    ori: wp.array(dtype=wp.quat),
    angvel: wp.array(dtype=wp.vec3),
    torques: wp.array(dtype=wp.vec3),
    qinv: wp.array(dtype=wp.float32),
    dt: float,
    damping: float,
    pred: wp.array(dtype=wp.quat),
):
    i = wp.tid()
    w = qinv[i]
    if w > 0.0:
        hdt = 0.5 * dt
        av = angvel[i] + torques[i] * w * dt
        av = av * (1.0 - damping)
        angvel[i] = av
        q = ori[i]
        oq = wp.quat(av.x, av.y, av.z, 0.0)
        qd = _xr_qmul(oq, q)
        qp = wp.quat(q.x + qd.x * hdt, q.y + qd.y * hdt, q.z + qd.z * hdt, q.w + qd.w * hdt)
        pred[i] = _xr_qnorm(qp)
    else:
        angvel[i] = wp.vec3(0.0, 0.0, 0.0)
        pred[i] = ori[i]


@wp.kernel
def _xr_integrate_pos(
    pos: wp.array(dtype=wp.vec3),
    pred: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=wp.float32),
    dt: float,
):
    i = wp.tid()
    if inv_mass[i] > 0.0:
        vel[i] = (pred[i] - pos[i]) * (1.0 / dt)
        pos[i] = pred[i]


@wp.kernel
def _xr_integrate_rot(
    ori: wp.array(dtype=wp.quat),
    pred: wp.array(dtype=wp.quat),
    prev: wp.array(dtype=wp.quat),
    angvel: wp.array(dtype=wp.vec3),
    qinv: wp.array(dtype=wp.float32),
    dt: float,
):
    i = wp.tid()
    if qinv[i] > 0.0:
        q = ori[i]
        rel = _xr_qmul(pred[i], _xr_qconj(q))
        angvel[i] = wp.vec3(rel.x, rel.y, rel.z) * (2.0 / dt)
        prev[i] = q
        ori[i] = pred[i]


# ============================================================================
# Warp kernels — constraints
# ============================================================================


@wp.kernel
def _xr_prepare_compliance(
    rest_lengths: wp.array(dtype=wp.float32),
    bend_stiff: wp.array(dtype=wp.vec3),
    young_mod: float,
    torsion_mod: float,
    dt: float,
    compliance: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    L = rest_lengths[i]
    dt2 = dt * dt
    eps = 1.0e-10
    bs = bend_stiff[i]
    base = i * 6
    sc = 1.0e-10
    compliance[base + 0] = sc
    compliance[base + 1] = sc
    compliance[base + 2] = sc
    compliance[base + 3] = 1.0 / (young_mod * bs.x * L * dt2 + eps)
    compliance[base + 4] = 1.0 / (young_mod * bs.y * L * dt2 + eps)
    compliance[base + 5] = 1.0 / (torsion_mod * bs.z * L * dt2 + eps)


@wp.kernel
def _xr_update_constraints(
    pos: wp.array(dtype=wp.vec3),
    ori: wp.array(dtype=wp.quat),
    rest_len: wp.array(dtype=wp.float32),
    rest_dar: wp.array(dtype=wp.vec3),
    cvals: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    p0 = pos[i]; p1 = pos[i + 1]
    q0 = ori[i]; q1 = ori[i + 1]
    hL = 0.5 * rest_len[i]
    r0 = _xr_qrot(q0, wp.vec3(0.0, 0.0, hL))
    r1 = _xr_qrot(q1, wp.vec3(0.0, 0.0, -hL))
    se = (p0 + r0) - (p1 + r1)
    qrel = _xr_qmul(_xr_qconj(q0), q1)
    de = wp.vec3(qrel.x, qrel.y, qrel.z) - rest_dar[i]
    base = i * 6
    cvals[base + 0] = se.x; cvals[base + 1] = se.y; cvals[base + 2] = se.z
    cvals[base + 3] = de.x; cvals[base + 4] = de.y; cvals[base + 5] = de.z


@wp.kernel
def _xr_compute_jacobians(
    ori: wp.array(dtype=wp.quat),
    rest_len: wp.array(dtype=wp.float32),
    jpos: wp.array(dtype=wp.float32),
    jrot: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    q0 = ori[i]; q1 = ori[i + 1]
    hL = 0.5 * rest_len[i]
    r0 = _xr_qrot(q0, wp.vec3(0.0, 0.0, hL))
    r1 = _xr_qrot(q1, wp.vec3(0.0, 0.0, -hL))

    for d in range(3):
        jpos[_xr_jidx(i, d, d)] = 1.0
        jpos[_xr_jidx(i, d, d + 3)] = -1.0

    jrot[_xr_jidx(i, 0, 0)] = 0.0;  jrot[_xr_jidx(i, 0, 1)] = r0.z;  jrot[_xr_jidx(i, 0, 2)] = -r0.y
    jrot[_xr_jidx(i, 1, 0)] = -r0.z; jrot[_xr_jidx(i, 1, 1)] = 0.0;   jrot[_xr_jidx(i, 1, 2)] = r0.x
    jrot[_xr_jidx(i, 2, 0)] = r0.y;  jrot[_xr_jidx(i, 2, 1)] = -r0.x; jrot[_xr_jidx(i, 2, 2)] = 0.0

    jrot[_xr_jidx(i, 0, 3)] = 0.0;   jrot[_xr_jidx(i, 0, 4)] = -r1.z; jrot[_xr_jidx(i, 0, 5)] = r1.y
    jrot[_xr_jidx(i, 1, 3)] = r1.z;  jrot[_xr_jidx(i, 1, 4)] = 0.0;   jrot[_xr_jidx(i, 1, 5)] = -r1.x
    jrot[_xr_jidx(i, 2, 3)] = -r1.y; jrot[_xr_jidx(i, 2, 4)] = r1.x;  jrot[_xr_jidx(i, 2, 5)] = 0.0

    x0 = q0.x; y0 = q0.y; z0 = q0.z; w0 = q0.w
    x1 = q1.x; y1 = q1.y; z1 = q1.z; w1 = q1.w

    j0r0 = wp.vec4(-w1, -z1, y1, x1)
    j0r1 = wp.vec4(z1, -w1, -x1, y1)
    j0r2 = wp.vec4(-y1, x1, -w1, z1)
    g0c0 = wp.vec4(0.5 * w0, -0.5 * z0, 0.5 * y0, -0.5 * x0)
    g0c1 = wp.vec4(0.5 * z0, 0.5 * w0, -0.5 * x0, -0.5 * y0)
    g0c2 = wp.vec4(-0.5 * y0, 0.5 * x0, 0.5 * w0, -0.5 * z0)

    j1r0 = wp.vec4(w0, z0, -y0, -x0)
    j1r1 = wp.vec4(-z0, w0, x0, -y0)
    j1r2 = wp.vec4(y0, -x0, w0, -z0)
    g1c0 = wp.vec4(0.5 * w1, -0.5 * z1, 0.5 * y1, -0.5 * x1)
    g1c1 = wp.vec4(0.5 * z1, 0.5 * w1, -0.5 * x1, -0.5 * y1)
    g1c2 = wp.vec4(-0.5 * y1, 0.5 * x1, 0.5 * w1, -0.5 * z1)

    jrot[_xr_jidx(i, 3, 0)] = wp.dot(j0r0, g0c0)
    jrot[_xr_jidx(i, 3, 1)] = wp.dot(j0r0, g0c1)
    jrot[_xr_jidx(i, 3, 2)] = wp.dot(j0r0, g0c2)
    jrot[_xr_jidx(i, 4, 0)] = wp.dot(j0r1, g0c0)
    jrot[_xr_jidx(i, 4, 1)] = wp.dot(j0r1, g0c1)
    jrot[_xr_jidx(i, 4, 2)] = wp.dot(j0r1, g0c2)
    jrot[_xr_jidx(i, 5, 0)] = wp.dot(j0r2, g0c0)
    jrot[_xr_jidx(i, 5, 1)] = wp.dot(j0r2, g0c1)
    jrot[_xr_jidx(i, 5, 2)] = wp.dot(j0r2, g0c2)

    jrot[_xr_jidx(i, 3, 3)] = wp.dot(j1r0, g1c0)
    jrot[_xr_jidx(i, 3, 4)] = wp.dot(j1r0, g1c1)
    jrot[_xr_jidx(i, 3, 5)] = wp.dot(j1r0, g1c2)
    jrot[_xr_jidx(i, 4, 3)] = wp.dot(j1r1, g1c0)
    jrot[_xr_jidx(i, 4, 4)] = wp.dot(j1r1, g1c1)
    jrot[_xr_jidx(i, 4, 5)] = wp.dot(j1r1, g1c2)
    jrot[_xr_jidx(i, 5, 3)] = wp.dot(j1r2, g1c0)
    jrot[_xr_jidx(i, 5, 4)] = wp.dot(j1r2, g1c1)
    jrot[_xr_jidx(i, 5, 5)] = wp.dot(j1r2, g1c2)


@wp.kernel
def _xr_build_rhs(
    cvals: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    lam_sum: wp.array(dtype=wp.float32),
    n_dofs: int,
    rhs: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    if i < n_dofs:
        rhs[i] = -cvals[i] - compliance[i] * lam_sum[i]
    else:
        rhs[i] = 0.0


# ============================================================================
# Warp kernels — inverse inertia & JMJT assembly
# ============================================================================


@wp.kernel
def _xr_compute_inv_inertia(
    ori: wp.array(dtype=wp.quat),
    qinv: wp.array(dtype=wp.float32),
    inv_I_local: wp.vec3,
    inv_I_out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    base = i * 9
    if qinv[i] <= 0.0:
        for j in range(9):
            inv_I_out[base + j] = 0.0
        return

    q = ori[i]
    xx = q.x * q.x; yy = q.y * q.y; zz = q.z * q.z
    xy = q.x * q.y; xz = q.x * q.z; yz = q.y * q.z
    wx = q.w * q.x; wy = q.w * q.y; wz = q.w * q.z

    r00 = 1.0 - 2.0 * (yy + zz); r01 = 2.0 * (xy - wz); r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz); r11 = 1.0 - 2.0 * (xx + zz); r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy); r21 = 2.0 * (yz + wx); r22 = 1.0 - 2.0 * (xx + yy)

    d0 = inv_I_local.x; d1 = inv_I_local.y; d2 = inv_I_local.z
    inv_I_out[base + 0] = r00 * d0 * r00 + r01 * d1 * r01 + r02 * d2 * r02
    inv_I_out[base + 1] = r00 * d0 * r10 + r01 * d1 * r11 + r02 * d2 * r12
    inv_I_out[base + 2] = r00 * d0 * r20 + r01 * d1 * r21 + r02 * d2 * r22
    inv_I_out[base + 3] = r10 * d0 * r00 + r11 * d1 * r01 + r12 * d2 * r02
    inv_I_out[base + 4] = r10 * d0 * r10 + r11 * d1 * r11 + r12 * d2 * r12
    inv_I_out[base + 5] = r10 * d0 * r20 + r11 * d1 * r21 + r12 * d2 * r22
    inv_I_out[base + 6] = r20 * d0 * r00 + r21 * d1 * r01 + r22 * d2 * r02
    inv_I_out[base + 7] = r20 * d0 * r10 + r21 * d1 * r11 + r22 * d2 * r12
    inv_I_out[base + 8] = r20 * d0 * r20 + r21 * d1 * r21 + r22 * d2 * r22


@wp.kernel
def _xr_assemble_jmjt(
    jpos: wp.array(dtype=wp.float32),
    jrot: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    inv_mass: wp.array(dtype=wp.float32),
    inv_inertia: wp.array(dtype=wp.float32),
    n_edges: int,
    diag: wp.array(dtype=wp.float32),
    offdiag: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    if i >= n_edges:
        return
    reg = 1.0e-6
    im0 = inv_mass[i]; im1 = inv_mass[i + 1]

    for row in range(6):
        for col in range(6):
            val = float(0.0)
            for k in range(3):
                val += jpos[_xr_jidx(i, row, k)] * im0 * jpos[_xr_jidx(i, col, k)]
            for k in range(3):
                val += jpos[_xr_jidx(i, row, k + 3)] * im1 * jpos[_xr_jidx(i, col, k + 3)]

            jr0r = wp.vec3(jrot[_xr_jidx(i, row, 0)], jrot[_xr_jidx(i, row, 1)], jrot[_xr_jidx(i, row, 2)])
            jr0c = wp.vec3(jrot[_xr_jidx(i, col, 0)], jrot[_xr_jidx(i, col, 1)], jrot[_xr_jidx(i, col, 2)])
            val += wp.dot(jr0r, _xr_invI_v(inv_inertia, i, jr0c))

            jr1r = wp.vec3(jrot[_xr_jidx(i, row, 3)], jrot[_xr_jidx(i, row, 4)], jrot[_xr_jidx(i, row, 5)])
            jr1c = wp.vec3(jrot[_xr_jidx(i, col, 3)], jrot[_xr_jidx(i, col, 4)], jrot[_xr_jidx(i, col, 5)])
            val += wp.dot(jr1r, _xr_invI_v(inv_inertia, i + 1, jr1c))

            if row == col:
                val += compliance[i * 6 + row] + reg
            diag[_xr_bidx(i, row, col)] = val

    if i == 0:
        for row in range(6):
            for col in range(6):
                offdiag[_xr_bidx(i, row, col)] = 0.0
        return

    prev = i - 1
    for row in range(6):
        for col in range(6):
            val = float(0.0)
            for k in range(3):
                val += jpos[_xr_jidx(prev, row, k + 3)] * im0 * jpos[_xr_jidx(i, col, k)]
            jp_vec = wp.vec3(jrot[_xr_jidx(prev, row, 3)], jrot[_xr_jidx(prev, row, 4)], jrot[_xr_jidx(prev, row, 5)])
            jc_vec = wp.vec3(jrot[_xr_jidx(i, col, 0)], jrot[_xr_jidx(i, col, 1)], jrot[_xr_jidx(i, col, 2)])
            val += wp.dot(jp_vec, _xr_invI_v(inv_inertia, i, jc_vec))
            offdiag[_xr_bidx(i, row, col)] = val


# ============================================================================
# Warp kernels — block Thomas solver
# ============================================================================


@wp.kernel
def _xr_block_thomas(
    diag: wp.array(dtype=wp.float32),
    offdiag: wp.array(dtype=wp.float32),
    rhs: wp.array(dtype=wp.float32),
    n_edges: int,
    c_blks: wp.array(dtype=wp.float32),
    d_prime: wp.array(dtype=wp.float32),
    x: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if tid != 0:
        return
    if n_edges <= 0:
        return

    A0, B0, C0, D0 = _xr_load_blk(diag, 0)
    b0, b1 = _xr_load_v6(rhs, 0)

    if n_edges > 1:
        for col in range(6):
            u0, u1 = _xr_blk_col(offdiag, 1, col)
            s0, s1 = _xr_blk_solve(A0, B0, C0, D0, u0, u1)
            _xr_blk_set_col(c_blks, 0, col, s0, s1)
    else:
        z = wp.vec3(0.0, 0.0, 0.0)
        for col in range(6):
            _xr_blk_set_col(c_blks, 0, col, z, z)

    d0, d1 = _xr_blk_solve(A0, B0, C0, D0, b0, b1)
    _xr_store_v6(d_prime, 0, d0, d1)

    for i in range(1, n_edges):
        DiA, DiB, DiC, DiD = _xr_load_blk(diag, i)
        LtA, LtB, LtC, LtD = _xr_load_blk(offdiag, i)
        CpA, CpB, CpC, CpD = _xr_load_blk(c_blks, i - 1)

        LiA = _xr_m33_t(LtA)
        LiB = _xr_m33_t(LtC)
        LiC = _xr_m33_t(LtB)
        LiD = _xr_m33_t(LtD)

        LCA, LCB, LCC, LCD = _xr_blk_mul(LiA, LiB, LiC, LiD, CpA, CpB, CpC, CpD)
        TiA, TiB, TiC, TiD = _xr_blk_sub(DiA, DiB, DiC, DiD, LCA, LCB, LCC, LCD)

        bi0, bi1 = _xr_load_v6(rhs, i)
        dp0, dp1 = _xr_load_v6(d_prime, i - 1)
        ld0, ld1 = _xr_blk_mv(LiA, LiB, LiC, LiD, dp0, dp1)
        bi0 = bi0 - ld0
        bi1 = bi1 - ld1

        if i < n_edges - 1:
            for col in range(6):
                u0, u1 = _xr_blk_col(offdiag, i + 1, col)
                s0, s1 = _xr_blk_solve(TiA, TiB, TiC, TiD, u0, u1)
                _xr_blk_set_col(c_blks, i, col, s0, s1)
        else:
            z = wp.vec3(0.0, 0.0, 0.0)
            for col in range(6):
                _xr_blk_set_col(c_blks, i, col, z, z)

        di0, di1 = _xr_blk_solve(TiA, TiB, TiC, TiD, bi0, bi1)
        _xr_store_v6(d_prime, i, di0, di1)

    dn0, dn1 = _xr_load_v6(d_prime, n_edges - 1)
    _xr_store_v6(x, n_edges - 1, dn0, dn1)
    for i in range(n_edges - 2, -1, -1):
        CiA, CiB, CiC, CiD = _xr_load_blk(c_blks, i)
        xn0, xn1 = _xr_load_v6(x, i + 1)
        cx0, cx1 = _xr_blk_mv(CiA, CiB, CiC, CiD, xn0, xn1)
        di0, di1 = _xr_load_v6(d_prime, i)
        _xr_store_v6(x, i, di0 - cx0, di1 - cx1)


# ============================================================================
# Warp kernels — additional solver backends (parity with Newton upstream)
# ============================================================================


@wp.kernel
def _xr_zero_2d(arr: wp.array2d(dtype=wp.float32), rows: int, cols: int):
    tid = wp.tid()
    if tid < rows * cols:
        row = tid // cols
        col = tid - row * cols
        arr[row, col] = 0.0


@wp.kernel
def _xr_block_thomas_3x3(
    diag: wp.array(dtype=wp.float32),
    offdiag: wp.array(dtype=wp.float32),
    rhs: wp.array(dtype=wp.float32),
    n_edges: int,
    c_blks: wp.array(dtype=wp.float32),
    d_prime: wp.array(dtype=wp.float32),
    x: wp.array(dtype=wp.float32),
):
    """3×3 block Thomas solve for the split-Thomas backend."""
    tid = wp.tid()
    if tid != 0:
        return
    if n_edges <= 0:
        return

    D0 = _xr_load_blk_3x3(diag, 0)
    b0 = _xr_load_v3b(rhs, 0)
    L0 = _xr_chol(D0)

    if n_edges > 1:
        for col in range(3):
            u = _xr_blk_col_3x3(offdiag, 1, col)
            v = _xr_cholsol(L0, u)
            _xr_blk_set_col_3x3(c_blks, 0, col, v)
    else:
        zero = wp.vec3(0.0, 0.0, 0.0)
        for col in range(3):
            _xr_blk_set_col_3x3(c_blks, 0, col, zero)

    d0 = _xr_cholsol(L0, b0)
    _xr_store_v3b(d_prime, 0, d0)

    for i in range(1, n_edges):
        Di = _xr_load_blk_3x3(diag, i)
        Lti = _xr_load_blk_3x3(offdiag, i)
        Cp = _xr_load_blk_3x3(c_blks, i - 1)

        Li = _xr_m33_t(Lti)
        LC = _xr_m33_mul(Li, Cp)
        Ti = _xr_m33_sub(Di, LC)
        Li_factor = _xr_chol(Ti)

        bi = _xr_load_v3b(rhs, i)
        dp = _xr_load_v3b(d_prime, i - 1)
        ld = _xr_m33v(Li, dp)
        bi = bi - ld

        if i < n_edges - 1:
            for col in range(3):
                u = _xr_blk_col_3x3(offdiag, i + 1, col)
                v = _xr_cholsol(Li_factor, u)
                _xr_blk_set_col_3x3(c_blks, i, col, v)
        else:
            zero = wp.vec3(0.0, 0.0, 0.0)
            for col in range(3):
                _xr_blk_set_col_3x3(c_blks, i, col, zero)

        di = _xr_cholsol(Li_factor, bi)
        _xr_store_v3b(d_prime, i, di)

    xn = _xr_load_v3b(d_prime, n_edges - 1)
    _xr_store_v3b(x, n_edges - 1, xn)

    for i in range(n_edges - 2, -1, -1):
        Ci = _xr_load_blk_3x3(c_blks, i)
        x_next = _xr_load_v3b(x, i + 1)
        cx = _xr_m33v(Ci, x_next)
        di = _xr_load_v3b(d_prime, i)
        _xr_store_v3b(x, i, di - cx)


@wp.kernel
def _xr_solve_blocks_jacobi(
    diag: wp.array(dtype=wp.float32),
    rhs: wp.array(dtype=wp.float32),
    delta_lambda: wp.array(dtype=wp.float32),
    n_edges: int,
):
    """Block-diagonal Jacobi solve: each thread solves one 6×6 diagonal block."""
    edge = wp.tid()
    if edge >= n_edges:
        return
    A, B, C, D = _xr_load_blk(diag, edge)
    b0, b1 = _xr_load_v6(rhs, edge)
    x0, x1 = _xr_blk_solve(A, B, C, D, b0, b1)
    _xr_store_v6(delta_lambda, edge, x0, x1)


@wp.kernel
def _xr_cholesky_solve_tile(
    A: wp.array2d(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
    x: wp.array(dtype=wp.float32),
):
    """Tiled Cholesky solve for small dense systems (n_dofs <= _XPBD_TILE)."""
    a_tile = wp.tile_load(A, shape=(_XPBD_TILE, _XPBD_TILE))
    b_tile = wp.tile_load(b, shape=_XPBD_TILE)
    L = wp.tile_cholesky(a_tile)
    x_tile = wp.tile_cholesky_solve(L, b_tile)
    wp.tile_store(x, x_tile)


@wp.kernel
def _xr_spbsv_u11_1rhs(
    n: int,
    ab: wp.array2d(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
):
    """Banded Cholesky solve (in-place factorization + solve, upper triangular LAPACK format)."""
    tid = wp.tid()
    if tid != 0:
        return

    REL_TOL = float(1.0e-8)
    ABS_FLOOR = float(1.0e-12)
    max_diag = float(0.0)

    for j in range(n):
        sum_val = float(0.0)
        kmax = j if j < _XPBD_BAND_KD else _XPBD_BAND_KD
        for k in range(1, kmax + 1):
            u = ab[_XPBD_BAND_KD - k, j]
            sum_val += u * u

        ajj = ab[_XPBD_BAND_KD, j] - sum_val
        orig_diag = ab[_XPBD_BAND_KD, j]
        if orig_diag > max_diag:
            max_diag = orig_diag

        tol = REL_TOL * max_diag
        if tol < ABS_FLOOR:
            tol = ABS_FLOOR
        if ajj <= tol:
            ajj = tol

        ujj = wp.sqrt(ajj)
        ab[_XPBD_BAND_KD, j] = ujj

        imax = (n - j - 1) if (n - j - 1) < _XPBD_BAND_KD else _XPBD_BAND_KD
        for i in range(1, imax + 1):
            dot = float(0.0)
            k2max = _XPBD_BAND_KD - i
            if k2max > j:
                k2max = j
            if k2max < 0:
                k2max = 0
            for k in range(1, k2max + 1):
                dot += ab[_XPBD_BAND_KD - k, j] * ab[_XPBD_BAND_KD - i - k, j + i]
            aji = ab[_XPBD_BAND_KD - i, j + i] - dot
            ab[_XPBD_BAND_KD - i, j + i] = aji / ujj

    for i in range(n):
        sum_val = float(0.0)
        k0 = 0 if i < _XPBD_BAND_KD else i - _XPBD_BAND_KD
        for k in range(k0, i):
            sum_val += ab[_XPBD_BAND_KD + k - i, i] * b[k]
        b[i] = (b[i] - sum_val) / ab[_XPBD_BAND_KD, i]

    for i in range(n - 1, -1, -1):
        sum_val = float(0.0)
        k1 = i + _XPBD_BAND_KD if i + _XPBD_BAND_KD < n else n - 1
        for k in range(i + 1, k1 + 1):
            sum_val += ab[_XPBD_BAND_KD + i - k, k] * b[k]
        b[i] = (b[i] - sum_val) / ab[_XPBD_BAND_KD, i]


@wp.kernel
def _xr_assemble_jmjt_dense(
    jpos: wp.array(dtype=wp.float32),
    jrot: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    inv_mass: wp.array(dtype=wp.float32),
    inv_inertia: wp.array(dtype=wp.float32),
    n_dofs: int,
    A: wp.array2d(dtype=wp.float32),
):
    """Dense JMJT assembly for the tiled-Cholesky path."""
    i = wp.tid()
    block_start = 6 * i
    inv_m0 = inv_mass[i]
    inv_m1 = inv_mass[i + 1]

    for row in range(6):
        for col in range(6):
            val = float(0.0)
            for k in range(3):
                val += jpos[_xr_jidx(i, row, k)] * inv_m0 * jpos[_xr_jidx(i, col, k)]
            for k in range(3):
                val += jpos[_xr_jidx(i, row, k + 3)] * inv_m1 * jpos[_xr_jidx(i, col, k + 3)]
            jr0r = wp.vec3(jrot[_xr_jidx(i, row, 0)], jrot[_xr_jidx(i, row, 1)], jrot[_xr_jidx(i, row, 2)])
            jr0c = wp.vec3(jrot[_xr_jidx(i, col, 0)], jrot[_xr_jidx(i, col, 1)], jrot[_xr_jidx(i, col, 2)])
            val += wp.dot(jr0r, _xr_invI_v(inv_inertia, i, jr0c))
            jr1r = wp.vec3(jrot[_xr_jidx(i, row, 3)], jrot[_xr_jidx(i, row, 4)], jrot[_xr_jidx(i, row, 5)])
            jr1c = wp.vec3(jrot[_xr_jidx(i, col, 3)], jrot[_xr_jidx(i, col, 4)], jrot[_xr_jidx(i, col, 5)])
            val += wp.dot(jr1r, _xr_invI_v(inv_inertia, i + 1, jr1c))
            if row == col:
                val += compliance[i * 6 + row] + 1.0e-6
            row_idx = block_start + row
            col_idx = block_start + col
            if row_idx < n_dofs and col_idx < n_dofs:
                A[row_idx, col_idx] = val

    if i > 0:
        prev = i - 1
        prev_block = 6 * prev
        for row in range(6):
            for col in range(6):
                val = float(0.0)
                for k in range(3):
                    val += jpos[_xr_jidx(prev, row, k + 3)] * inv_m0 * jpos[_xr_jidx(i, col, k)]
                jp_vec = wp.vec3(
                    jrot[_xr_jidx(prev, row, 3)], jrot[_xr_jidx(prev, row, 4)], jrot[_xr_jidx(prev, row, 5)],
                )
                jc_vec = wp.vec3(
                    jrot[_xr_jidx(i, col, 0)], jrot[_xr_jidx(i, col, 1)], jrot[_xr_jidx(i, col, 2)],
                )
                val += wp.dot(jp_vec, _xr_invI_v(inv_inertia, i, jc_vec))
                row_idx = prev_block + row
                col_idx = block_start + col
                if row_idx < n_dofs and col_idx < n_dofs:
                    A[row_idx, col_idx] = val
                row_idx2 = block_start + col
                col_idx2 = prev_block + row
                if row_idx2 < n_dofs and col_idx2 < n_dofs:
                    A[row_idx2, col_idx2] = val


@wp.kernel
def _xr_pad_diagonal(
    A: wp.array2d(dtype=wp.float32),
    n_dofs: int,
    tile: int,
):
    """Pad unused DOFs on the diagonal with 1s for the tiled Cholesky solver."""
    i = wp.tid()
    if i >= n_dofs and i < tile:
        A[i, i] = 1.0


@wp.kernel
def _xr_assemble_jmjt_banded(
    jpos: wp.array(dtype=wp.float32),
    jrot: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    inv_mass: wp.array(dtype=wp.float32),
    inv_inertia: wp.array(dtype=wp.float32),
    n_dofs: int,
    ab: wp.array2d(dtype=wp.float32),
):
    """Banded JMJT assembly (upper triangular LAPACK banded format)."""
    i = wp.tid()
    block_start = 6 * i
    if block_start >= n_dofs:
        return
    inv_m0 = inv_mass[i]
    inv_m1 = inv_mass[i + 1]

    for row in range(6):
        for col in range(6):
            val = float(0.0)
            for k in range(3):
                val += jpos[_xr_jidx(i, row, k)] * inv_m0 * jpos[_xr_jidx(i, col, k)]
            for k in range(3):
                val += jpos[_xr_jidx(i, row, k + 3)] * inv_m1 * jpos[_xr_jidx(i, col, k + 3)]
            jr0r = wp.vec3(jrot[_xr_jidx(i, row, 0)], jrot[_xr_jidx(i, row, 1)], jrot[_xr_jidx(i, row, 2)])
            jr0c = wp.vec3(jrot[_xr_jidx(i, col, 0)], jrot[_xr_jidx(i, col, 1)], jrot[_xr_jidx(i, col, 2)])
            val += wp.dot(jr0r, _xr_invI_v(inv_inertia, i, jr0c))
            jr1r = wp.vec3(jrot[_xr_jidx(i, row, 3)], jrot[_xr_jidx(i, row, 4)], jrot[_xr_jidx(i, row, 5)])
            jr1c = wp.vec3(jrot[_xr_jidx(i, col, 3)], jrot[_xr_jidx(i, col, 4)], jrot[_xr_jidx(i, col, 5)])
            val += wp.dot(jr1r, _xr_invI_v(inv_inertia, i + 1, jr1c))
            if row == col:
                val += compliance[i * 6 + row] + 1.0e-6
            row_idx = block_start + row
            col_idx = block_start + col
            if row_idx <= col_idx:
                band_row = _XPBD_BAND_KD + row_idx - col_idx
                if band_row >= 0 and band_row <= _XPBD_BAND_KD:
                    ab[band_row, col_idx] = val

    if i > 0:
        prev = i - 1
        prev_block = block_start - 6
        for row in range(6):
            for col in range(6):
                val = float(0.0)
                for k in range(3):
                    val += jpos[_xr_jidx(prev, row, k + 3)] * inv_m0 * jpos[_xr_jidx(i, col, k)]
                jp_vec = wp.vec3(
                    jrot[_xr_jidx(prev, row, 3)], jrot[_xr_jidx(prev, row, 4)], jrot[_xr_jidx(prev, row, 5)],
                )
                jc_vec = wp.vec3(
                    jrot[_xr_jidx(i, col, 0)], jrot[_xr_jidx(i, col, 1)], jrot[_xr_jidx(i, col, 2)],
                )
                val += wp.dot(jp_vec, _xr_invI_v(inv_inertia, i, jc_vec))
                row_idx = prev_block + row
                col_idx = block_start + col
                band_row = _XPBD_BAND_KD + row_idx - col_idx
                if band_row >= 0 and band_row <= _XPBD_BAND_KD:
                    ab[band_row, col_idx] = val


@wp.kernel
def _xr_assemble_stretch_blocks(
    jpos: wp.array(dtype=wp.float32),
    jrot: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    inv_mass: wp.array(dtype=wp.float32),
    inv_inertia: wp.array(dtype=wp.float32),
    n_edges: int,
    diag: wp.array(dtype=wp.float32),
    offdiag: wp.array(dtype=wp.float32),
):
    """Assemble 3×3 block-tridiagonal system for stretch constraints (split-Thomas)."""
    i = wp.tid()
    if i >= n_edges:
        return
    inv_m0 = inv_mass[i]
    inv_m1 = inv_mass[i + 1]

    for row in range(3):
        for col in range(3):
            val = float(0.0)
            for k in range(3):
                val += jpos[_xr_jidx(i, row, k)] * inv_m0 * jpos[_xr_jidx(i, col, k)]
            for k in range(3):
                val += jpos[_xr_jidx(i, row, k + 3)] * inv_m1 * jpos[_xr_jidx(i, col, k + 3)]
            jr0r = wp.vec3(jrot[_xr_jidx(i, row, 0)], jrot[_xr_jidx(i, row, 1)], jrot[_xr_jidx(i, row, 2)])
            jr0c = wp.vec3(jrot[_xr_jidx(i, col, 0)], jrot[_xr_jidx(i, col, 1)], jrot[_xr_jidx(i, col, 2)])
            val += wp.dot(jr0r, _xr_invI_v(inv_inertia, i, jr0c))
            jr1r = wp.vec3(jrot[_xr_jidx(i, row, 3)], jrot[_xr_jidx(i, row, 4)], jrot[_xr_jidx(i, row, 5)])
            jr1c = wp.vec3(jrot[_xr_jidx(i, col, 3)], jrot[_xr_jidx(i, col, 4)], jrot[_xr_jidx(i, col, 5)])
            val += wp.dot(jr1r, _xr_invI_v(inv_inertia, i + 1, jr1c))
            if row == col:
                val += compliance[i * 6 + row] + 1.0e-6
            diag[_xr_bidx_3x3(i, row, col)] = val

    if i == 0:
        for row in range(3):
            for col in range(3):
                offdiag[_xr_bidx_3x3(i, row, col)] = 0.0
        return

    prev = i - 1
    for row in range(3):
        for col in range(3):
            val = float(0.0)
            for k in range(3):
                val += jpos[_xr_jidx(prev, row, k + 3)] * inv_m0 * jpos[_xr_jidx(i, col, k)]
            jp_vec = wp.vec3(
                jrot[_xr_jidx(prev, row, 3)], jrot[_xr_jidx(prev, row, 4)], jrot[_xr_jidx(prev, row, 5)],
            )
            jc_vec = wp.vec3(
                jrot[_xr_jidx(i, col, 0)], jrot[_xr_jidx(i, col, 1)], jrot[_xr_jidx(i, col, 2)],
            )
            val += wp.dot(jp_vec, _xr_invI_v(inv_inertia, i, jc_vec))
            offdiag[_xr_bidx_3x3(i, row, col)] = val


@wp.kernel
def _xr_assemble_darboux_blocks(
    jrot: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    inv_inertia: wp.array(dtype=wp.float32),
    n_edges: int,
    diag: wp.array(dtype=wp.float32),
    offdiag: wp.array(dtype=wp.float32),
):
    """Assemble 3×3 block-tridiagonal system for Darboux (bend/twist) constraints (split-Thomas)."""
    i = wp.tid()
    if i >= n_edges:
        return

    for row in range(3):
        for col in range(3):
            val = float(0.0)
            jr0r = wp.vec3(
                jrot[_xr_jidx(i, row + 3, 0)], jrot[_xr_jidx(i, row + 3, 1)], jrot[_xr_jidx(i, row + 3, 2)],
            )
            jr0c = wp.vec3(
                jrot[_xr_jidx(i, col + 3, 0)], jrot[_xr_jidx(i, col + 3, 1)], jrot[_xr_jidx(i, col + 3, 2)],
            )
            val += wp.dot(jr0r, _xr_invI_v(inv_inertia, i, jr0c))
            jr1r = wp.vec3(
                jrot[_xr_jidx(i, row + 3, 3)], jrot[_xr_jidx(i, row + 3, 4)], jrot[_xr_jidx(i, row + 3, 5)],
            )
            jr1c = wp.vec3(
                jrot[_xr_jidx(i, col + 3, 3)], jrot[_xr_jidx(i, col + 3, 4)], jrot[_xr_jidx(i, col + 3, 5)],
            )
            val += wp.dot(jr1r, _xr_invI_v(inv_inertia, i + 1, jr1c))
            if row == col:
                val += compliance[i * 6 + row + 3] + 1.0e-6
            diag[_xr_bidx_3x3(i, row, col)] = val

    if i == 0:
        for row in range(3):
            for col in range(3):
                offdiag[_xr_bidx_3x3(i, row, col)] = 0.0
        return

    prev = i - 1
    for row in range(3):
        for col in range(3):
            val = float(0.0)
            jp_vec = wp.vec3(
                jrot[_xr_jidx(prev, row + 3, 3)], jrot[_xr_jidx(prev, row + 3, 4)], jrot[_xr_jidx(prev, row + 3, 5)],
            )
            jc_vec = wp.vec3(
                jrot[_xr_jidx(i, col + 3, 0)], jrot[_xr_jidx(i, col + 3, 1)], jrot[_xr_jidx(i, col + 3, 2)],
            )
            val += wp.dot(jp_vec, _xr_invI_v(inv_inertia, i, jc_vec))
            offdiag[_xr_bidx_3x3(i, row, col)] = val


@wp.kernel
def _xr_build_rhs_stretch(
    cvals: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    lam_sum: wp.array(dtype=wp.float32),
    n_edges: int,
    rhs: wp.array(dtype=wp.float32),
):
    """Build RHS vector for stretch constraints (first 3 DOFs per edge)."""
    edge = wp.tid()
    if edge >= n_edges:
        return
    for i in range(3):
        idx = edge * 6 + i
        rhs[edge * 3 + i] = -cvals[idx] - compliance[idx] * lam_sum[idx]


@wp.kernel
def _xr_build_rhs_darboux(
    cvals: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    lam_sum: wp.array(dtype=wp.float32),
    n_edges: int,
    rhs: wp.array(dtype=wp.float32),
):
    """Build RHS vector for Darboux constraints (last 3 DOFs per edge)."""
    edge = wp.tid()
    if edge >= n_edges:
        return
    for i in range(3):
        idx = edge * 6 + i + 3
        rhs[edge * 3 + i] = -cvals[idx] - compliance[idx] * lam_sum[idx]


@wp.kernel
def _xr_merge_delta_lambda(
    stretch_dl: wp.array(dtype=wp.float32),
    darboux_dl: wp.array(dtype=wp.float32),
    combined_dl: wp.array(dtype=wp.float32),
    n_edges: int,
):
    """Merge split Δλ arrays (stretch + darboux) into combined 6-vector format."""
    edge = wp.tid()
    if edge >= n_edges:
        return
    for i in range(3):
        combined_dl[edge * 6 + i] = stretch_dl[edge * 3 + i]
    for i in range(3):
        combined_dl[edge * 6 + i + 3] = darboux_dl[edge * 3 + i]


# ============================================================================
# Warp kernels — utility / collision / corrections
# ============================================================================


@wp.kernel
def _xr_zero_f(arr: wp.array(dtype=wp.float32)):
    arr[wp.tid()] = 0.0


@wp.kernel
def _xr_zero_v3(arr: wp.array(dtype=wp.vec3)):
    arr[wp.tid()] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _xr_floor_collision(
    pos: wp.array(dtype=wp.vec3),
    pred: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    min_z: float,
    restitution: float,
):
    """Floor collision with optional restitution (parity with Newton upstream).

    Args:
        pos: Current particle positions (unused; kept for API compat).
        pred: Predicted particle positions (clamped in-place).
        vel: Current particle velocities (reflected in-place).
        min_z: Z coordinate of the floor plane (mm or m, in solver units).
        restitution: Bounce coefficient. ``0`` = inelastic (vel.z → 0),
            ``1`` = perfectly elastic (vel.z → -vel.z).
    """
    i = wp.tid()
    p = pred[i]
    if p.z < min_z:
        pred[i] = wp.vec3(p.x, p.y, min_z)
        v = vel[i]
        if v.z < 0.0:
            vel[i] = wp.vec3(v.x, v.y, -restitution * v.z)


# ============================================================================
# Warp kernels — GPU-side root control (CUDA-graph capturable)
# ============================================================================


@wp.kernel
def _xr_set_root_orientation_kernel(
    ori: wp.array(dtype=wp.quat),
    pred_ori: wp.array(dtype=wp.quat),
    prev_ori: wp.array(dtype=wp.quat),
    root_idx: int,
    q: wp.quat,
):
    """Force a root particle's orientation to ``q`` from a single-thread launch.

    Avoids the CPU sync that ``apply_proximal_control`` uses; safe to capture
    inside a CUDA graph.
    """
    tid = wp.tid()
    if tid != 0:
        return
    ori[root_idx] = q
    pred_ori[root_idx] = q
    prev_ori[root_idx] = q


@wp.kernel
def _xr_proximal_push_kernel(
    pos: wp.array(dtype=wp.vec3),
    ori: wp.array(dtype=wp.quat),
    push_velocities: wp.array(dtype=wp.float32),
    rotate_velocities: wp.array(dtype=wp.float32),
    root_idx: wp.array(dtype=wp.int32),
    next_idx: wp.array(dtype=wp.int32),
    dt: float,
    n_envs: int,
):
    """GPU proximal push/rotate for one or many environments.

    Applies an axial push and an axial twist to each environment's root
    particle. The axial direction is taken from ``pos[next_idx] - pos[root_idx]``
    (the first segment tangent), so the controller follows the catheter's
    current insertion direction.

    Args:
        pos: Concatenated particle positions for all envs (in-place write at root).
        ori: Concatenated orientations (in-place quaternion update at root).
        push_velocities: Per-env push velocity in solver units / second.
        rotate_velocities: Per-env axial rotation velocity in rad / second.
        root_idx: Per-env global index of the root particle (size n_envs).
        next_idx: Per-env global index of the second particle (size n_envs).
        dt: Control time step.
        n_envs: Number of environments to update.
    """
    e = wp.tid()
    if e >= n_envs:
        return

    r = root_idx[e]
    n = next_idx[e]

    p_root = pos[r]
    p_next = pos[n]
    seg = p_next - p_root
    seg_len = wp.length(seg)
    if seg_len < 1.0e-8:
        tangent = wp.vec3(1.0, 0.0, 0.0)
    else:
        tangent = seg * (1.0 / seg_len)

    push_v = push_velocities[e]
    pos[r] = p_root + tangent * (push_v * dt)

    rot_v = rotate_velocities[e]
    if wp.abs(rot_v) > 1.0e-10:
        half_angle = 0.5 * rot_v * dt
        s = wp.sin(half_angle)
        c = wp.cos(half_angle)
        dq = wp.quat(tangent.x * s, tangent.y * s, tangent.z * s, c)
        q0 = ori[r]
        ori[r] = _xr_qnorm(_xr_qmul(dq, q0))


@wp.kernel
def _xr_compute_corrections(
    inv_mass: wp.array(dtype=wp.float32),
    qinv: wp.array(dtype=wp.float32),
    inv_inertia: wp.array(dtype=wp.float32),
    jpos: wp.array(dtype=wp.float32),
    jrot: wp.array(dtype=wp.float32),
    dlam: wp.array(dtype=wp.float32),
    lam_sum: wp.array(dtype=wp.float32),
    n_edges: int,
    pcorr: wp.array(dtype=wp.vec3),
    rcorr: wp.array(dtype=wp.vec3),
):
    edge = wp.tid()
    if edge >= n_edges:
        return
    base = edge * 6
    d0 = dlam[base + 0]; d1 = dlam[base + 1]; d2 = dlam[base + 2]
    d3 = dlam[base + 3]; d4 = dlam[base + 4]; d5 = dlam[base + 5]
    lam_sum[base + 0] = lam_sum[base + 0] + d0
    lam_sum[base + 1] = lam_sum[base + 1] + d1
    lam_sum[base + 2] = lam_sum[base + 2] + d2
    lam_sum[base + 3] = lam_sum[base + 3] + d3
    lam_sum[base + 4] = lam_sum[base + 4] + d4
    lam_sum[base + 5] = lam_sum[base + 5] + d5

    im0 = inv_mass[edge]; im1 = inv_mass[edge + 1]
    if im0 > 0.0:
        dp = wp.vec3(
            _xr_jdot(jpos, edge, 0, d0, d1, d2, d3, d4, d5) * im0,
            _xr_jdot(jpos, edge, 1, d0, d1, d2, d3, d4, d5) * im0,
            _xr_jdot(jpos, edge, 2, d0, d1, d2, d3, d4, d5) * im0,
        )
        wp.atomic_add(pcorr, edge, dp)
    if im1 > 0.0:
        dp = wp.vec3(
            _xr_jdot(jpos, edge, 3, d0, d1, d2, d3, d4, d5) * im1,
            _xr_jdot(jpos, edge, 4, d0, d1, d2, d3, d4, d5) * im1,
            _xr_jdot(jpos, edge, 5, d0, d1, d2, d3, d4, d5) * im1,
        )
        wp.atomic_add(pcorr, edge + 1, dp)

    if qinv[edge] > 0.0:
        jt0 = wp.vec3(
            _xr_jdot(jrot, edge, 0, d0, d1, d2, d3, d4, d5),
            _xr_jdot(jrot, edge, 1, d0, d1, d2, d3, d4, d5),
            _xr_jdot(jrot, edge, 2, d0, d1, d2, d3, d4, d5),
        )
        wp.atomic_add(rcorr, edge, _xr_invI_v(inv_inertia, edge, jt0))
    if qinv[edge + 1] > 0.0:
        jt1 = wp.vec3(
            _xr_jdot(jrot, edge, 3, d0, d1, d2, d3, d4, d5),
            _xr_jdot(jrot, edge, 4, d0, d1, d2, d3, d4, d5),
            _xr_jdot(jrot, edge, 5, d0, d1, d2, d3, d4, d5),
        )
        wp.atomic_add(rcorr, edge + 1, _xr_invI_v(inv_inertia, edge + 1, jt1))


@wp.kernel
def _xr_apply_corrections(
    pred_pos: wp.array(dtype=wp.vec3),
    pred_ori: wp.array(dtype=wp.quat),
    pcorr: wp.array(dtype=wp.vec3),
    rcorr: wp.array(dtype=wp.vec3),
    n_pts: int,
):
    i = wp.tid()
    if i >= n_pts:
        return
    pred_pos[i] = pred_pos[i] + pcorr[i]
    pred_ori[i] = _xr_qcorr(pred_ori[i], rcorr[i])


# ============================================================================
# Warp kernels — batched (multi-rod / multi-env) variants
# ----------------------------------------------------------------------------
# Each kernel below operates on concatenated state for ``E`` rods. Index
# arithmetic uses two pre-built CPU→GPU lookup arrays:
#   particle_rod_id[i]  → rod that particle i belongs to (size = total particles)
#   edge_rod_id[g_edge] → rod that global edge g_edge belongs to (size = total edges)
#   rod_offsets[r]      → starting particle index of rod r (size = E + 1)
#   edge_offsets[r]     → starting edge index of rod r (size = E + 1)
# ============================================================================


@wp.kernel
def _xr_predict_pos_batched(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=wp.float32),
    gravity: wp.array(dtype=wp.vec3),
    particle_rod_id: wp.array(dtype=wp.int32),
    dt: float,
    damping: float,
    pred: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    w = inv_mass[i]
    if w > 0.0:
        rod = particle_rod_id[i]
        v = vel[i] + (forces[i] * w + gravity[rod]) * dt
        v = v * (1.0 - damping)
        vel[i] = v
        pred[i] = pos[i] + v * dt
    else:
        vel[i] = wp.vec3(0.0, 0.0, 0.0)
        pred[i] = pos[i]


@wp.kernel
def _xr_predict_rot_batched(
    ori: wp.array(dtype=wp.quat),
    angvel: wp.array(dtype=wp.vec3),
    torques: wp.array(dtype=wp.vec3),
    qinv: wp.array(dtype=wp.float32),
    dt: float,
    damping: float,
    pred: wp.array(dtype=wp.quat),
):
    i = wp.tid()
    w = qinv[i]
    if w > 0.0:
        hdt = 0.5 * dt
        av = angvel[i] + torques[i] * w * dt
        av = av * (1.0 - damping)
        angvel[i] = av
        q = ori[i]
        oq = wp.quat(av.x, av.y, av.z, 0.0)
        qd = _xr_qmul(oq, q)
        qp = wp.quat(q.x + qd.x * hdt, q.y + qd.y * hdt, q.z + qd.z * hdt, q.w + qd.w * hdt)
        pred[i] = _xr_qnorm(qp)
    else:
        angvel[i] = wp.vec3(0.0, 0.0, 0.0)
        pred[i] = ori[i]


@wp.kernel
def _xr_integrate_pos_batched(
    pos: wp.array(dtype=wp.vec3),
    pred: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=wp.float32),
    dt: float,
):
    i = wp.tid()
    if inv_mass[i] > 0.0:
        vel[i] = (pred[i] - pos[i]) * (1.0 / dt)
        pos[i] = pred[i]


@wp.kernel
def _xr_integrate_rot_batched(
    ori: wp.array(dtype=wp.quat),
    pred: wp.array(dtype=wp.quat),
    prev: wp.array(dtype=wp.quat),
    angvel: wp.array(dtype=wp.vec3),
    qinv: wp.array(dtype=wp.float32),
    dt: float,
):
    i = wp.tid()
    if qinv[i] > 0.0:
        q = ori[i]
        rel = _xr_qmul(pred[i], _xr_qconj(q))
        angvel[i] = wp.vec3(rel.x, rel.y, rel.z) * (2.0 / dt)
        prev[i] = q
        ori[i] = pred[i]


@wp.kernel
def _xr_prepare_compliance_batched(
    rest_lengths: wp.array(dtype=wp.float32),
    bend_stiff: wp.array(dtype=wp.vec3),
    edge_rod_id: wp.array(dtype=wp.int32),
    young_mod: wp.array(dtype=wp.float32),
    torsion_mod: wp.array(dtype=wp.float32),
    dt: float,
    compliance: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    L = rest_lengths[i]
    dt2 = dt * dt
    eps = 1.0e-10
    bs = bend_stiff[i]
    rod = edge_rod_id[i]
    E = young_mod[rod]
    G = torsion_mod[rod]
    base = i * 6
    sc = 1.0e-10
    compliance[base + 0] = sc
    compliance[base + 1] = sc
    compliance[base + 2] = sc
    compliance[base + 3] = 1.0 / (E * bs.x * L * dt2 + eps)
    compliance[base + 4] = 1.0 / (E * bs.y * L * dt2 + eps)
    compliance[base + 5] = 1.0 / (G * bs.z * L * dt2 + eps)


@wp.kernel
def _xr_update_constraints_batched(
    pos: wp.array(dtype=wp.vec3),
    ori: wp.array(dtype=wp.quat),
    rest_len: wp.array(dtype=wp.float32),
    rest_dar: wp.array(dtype=wp.vec3),
    rod_offsets: wp.array(dtype=wp.int32),
    edge_offsets: wp.array(dtype=wp.int32),
    edge_rod_id: wp.array(dtype=wp.int32),
    cvals: wp.array(dtype=wp.float32),
):
    g_edge = wp.tid()
    rod = edge_rod_id[g_edge]
    local_edge = g_edge - edge_offsets[rod]
    p0_idx = rod_offsets[rod] + local_edge
    p1_idx = p0_idx + 1

    p0 = pos[p0_idx]
    p1 = pos[p1_idx]
    q0 = ori[p0_idx]
    q1 = ori[p1_idx]
    hL = 0.5 * rest_len[g_edge]
    r0 = _xr_qrot(q0, wp.vec3(0.0, 0.0, hL))
    r1 = _xr_qrot(q1, wp.vec3(0.0, 0.0, -hL))
    se = (p0 + r0) - (p1 + r1)
    qrel = _xr_qmul(_xr_qconj(q0), q1)
    de = wp.vec3(qrel.x, qrel.y, qrel.z) - rest_dar[g_edge]
    base = g_edge * 6
    cvals[base + 0] = se.x
    cvals[base + 1] = se.y
    cvals[base + 2] = se.z
    cvals[base + 3] = de.x
    cvals[base + 4] = de.y
    cvals[base + 5] = de.z


@wp.kernel
def _xr_compute_jacobians_batched(
    ori: wp.array(dtype=wp.quat),
    rest_len: wp.array(dtype=wp.float32),
    rod_offsets: wp.array(dtype=wp.int32),
    edge_offsets: wp.array(dtype=wp.int32),
    edge_rod_id: wp.array(dtype=wp.int32),
    jpos: wp.array(dtype=wp.float32),
    jrot: wp.array(dtype=wp.float32),
):
    g_edge = wp.tid()
    rod = edge_rod_id[g_edge]
    local_edge = g_edge - edge_offsets[rod]
    p0_idx = rod_offsets[rod] + local_edge
    p1_idx = p0_idx + 1

    q0 = ori[p0_idx]
    q1 = ori[p1_idx]
    hL = 0.5 * rest_len[g_edge]
    r0 = _xr_qrot(q0, wp.vec3(0.0, 0.0, hL))
    r1 = _xr_qrot(q1, wp.vec3(0.0, 0.0, -hL))

    i = g_edge

    for d in range(3):
        jpos[_xr_jidx(i, d, d)] = 1.0
        jpos[_xr_jidx(i, d, d + 3)] = -1.0

    jrot[_xr_jidx(i, 0, 0)] = 0.0;   jrot[_xr_jidx(i, 0, 1)] = r0.z;  jrot[_xr_jidx(i, 0, 2)] = -r0.y
    jrot[_xr_jidx(i, 1, 0)] = -r0.z; jrot[_xr_jidx(i, 1, 1)] = 0.0;   jrot[_xr_jidx(i, 1, 2)] = r0.x
    jrot[_xr_jidx(i, 2, 0)] = r0.y;  jrot[_xr_jidx(i, 2, 1)] = -r0.x; jrot[_xr_jidx(i, 2, 2)] = 0.0

    jrot[_xr_jidx(i, 0, 3)] = 0.0;   jrot[_xr_jidx(i, 0, 4)] = -r1.z; jrot[_xr_jidx(i, 0, 5)] = r1.y
    jrot[_xr_jidx(i, 1, 3)] = r1.z;  jrot[_xr_jidx(i, 1, 4)] = 0.0;   jrot[_xr_jidx(i, 1, 5)] = -r1.x
    jrot[_xr_jidx(i, 2, 3)] = -r1.y; jrot[_xr_jidx(i, 2, 4)] = r1.x;  jrot[_xr_jidx(i, 2, 5)] = 0.0

    x0 = q0.x; y0 = q0.y; z0 = q0.z; w0 = q0.w
    x1 = q1.x; y1 = q1.y; z1 = q1.z; w1 = q1.w

    j0r0 = wp.vec4(-w1, -z1, y1, x1)
    j0r1 = wp.vec4(z1, -w1, -x1, y1)
    j0r2 = wp.vec4(-y1, x1, -w1, z1)
    g0c0 = wp.vec4(0.5 * w0, -0.5 * z0, 0.5 * y0, -0.5 * x0)
    g0c1 = wp.vec4(0.5 * z0, 0.5 * w0, -0.5 * x0, -0.5 * y0)
    g0c2 = wp.vec4(-0.5 * y0, 0.5 * x0, 0.5 * w0, -0.5 * z0)

    j1r0 = wp.vec4(w0, z0, -y0, -x0)
    j1r1 = wp.vec4(-z0, w0, x0, -y0)
    j1r2 = wp.vec4(y0, -x0, w0, -z0)
    g1c0 = wp.vec4(0.5 * w1, -0.5 * z1, 0.5 * y1, -0.5 * x1)
    g1c1 = wp.vec4(0.5 * z1, 0.5 * w1, -0.5 * x1, -0.5 * y1)
    g1c2 = wp.vec4(-0.5 * y1, 0.5 * x1, 0.5 * w1, -0.5 * z1)

    jrot[_xr_jidx(i, 3, 0)] = wp.dot(j0r0, g0c0)
    jrot[_xr_jidx(i, 3, 1)] = wp.dot(j0r0, g0c1)
    jrot[_xr_jidx(i, 3, 2)] = wp.dot(j0r0, g0c2)
    jrot[_xr_jidx(i, 4, 0)] = wp.dot(j0r1, g0c0)
    jrot[_xr_jidx(i, 4, 1)] = wp.dot(j0r1, g0c1)
    jrot[_xr_jidx(i, 4, 2)] = wp.dot(j0r1, g0c2)
    jrot[_xr_jidx(i, 5, 0)] = wp.dot(j0r2, g0c0)
    jrot[_xr_jidx(i, 5, 1)] = wp.dot(j0r2, g0c1)
    jrot[_xr_jidx(i, 5, 2)] = wp.dot(j0r2, g0c2)

    jrot[_xr_jidx(i, 3, 3)] = wp.dot(j1r0, g1c0)
    jrot[_xr_jidx(i, 3, 4)] = wp.dot(j1r0, g1c1)
    jrot[_xr_jidx(i, 3, 5)] = wp.dot(j1r0, g1c2)
    jrot[_xr_jidx(i, 4, 3)] = wp.dot(j1r1, g1c0)
    jrot[_xr_jidx(i, 4, 4)] = wp.dot(j1r1, g1c1)
    jrot[_xr_jidx(i, 4, 5)] = wp.dot(j1r1, g1c2)
    jrot[_xr_jidx(i, 5, 3)] = wp.dot(j1r2, g1c0)
    jrot[_xr_jidx(i, 5, 4)] = wp.dot(j1r2, g1c1)
    jrot[_xr_jidx(i, 5, 5)] = wp.dot(j1r2, g1c2)


@wp.kernel
def _xr_compute_inv_inertia_batched(
    ori: wp.array(dtype=wp.quat),
    qinv: wp.array(dtype=wp.float32),
    inv_I_local: wp.array(dtype=wp.vec3),
    particle_rod_id: wp.array(dtype=wp.int32),
    inv_I_out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    base = i * 9
    if qinv[i] <= 0.0:
        for j in range(9):
            inv_I_out[base + j] = 0.0
        return

    rod = particle_rod_id[i]
    inv_I_l = inv_I_local[rod]

    q = ori[i]
    xx = q.x * q.x; yy = q.y * q.y; zz = q.z * q.z
    xy = q.x * q.y; xz = q.x * q.z; yz = q.y * q.z
    wx = q.w * q.x; wy = q.w * q.y; wz = q.w * q.z

    r00 = 1.0 - 2.0 * (yy + zz); r01 = 2.0 * (xy - wz); r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz); r11 = 1.0 - 2.0 * (xx + zz); r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy); r21 = 2.0 * (yz + wx); r22 = 1.0 - 2.0 * (xx + yy)

    d0 = inv_I_l.x; d1 = inv_I_l.y; d2 = inv_I_l.z
    inv_I_out[base + 0] = r00 * d0 * r00 + r01 * d1 * r01 + r02 * d2 * r02
    inv_I_out[base + 1] = r00 * d0 * r10 + r01 * d1 * r11 + r02 * d2 * r12
    inv_I_out[base + 2] = r00 * d0 * r20 + r01 * d1 * r21 + r02 * d2 * r22
    inv_I_out[base + 3] = r10 * d0 * r00 + r11 * d1 * r01 + r12 * d2 * r02
    inv_I_out[base + 4] = r10 * d0 * r10 + r11 * d1 * r11 + r12 * d2 * r12
    inv_I_out[base + 5] = r10 * d0 * r20 + r11 * d1 * r21 + r12 * d2 * r22
    inv_I_out[base + 6] = r20 * d0 * r00 + r21 * d1 * r01 + r22 * d2 * r02
    inv_I_out[base + 7] = r20 * d0 * r10 + r21 * d1 * r11 + r22 * d2 * r12
    inv_I_out[base + 8] = r20 * d0 * r20 + r21 * d1 * r21 + r22 * d2 * r22


@wp.kernel
def _xr_assemble_jmjt_batched(
    jpos: wp.array(dtype=wp.float32),
    jrot: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    inv_mass: wp.array(dtype=wp.float32),
    inv_inertia: wp.array(dtype=wp.float32),
    rod_offsets: wp.array(dtype=wp.int32),
    edge_offsets: wp.array(dtype=wp.int32),
    edge_rod_id: wp.array(dtype=wp.int32),
    diag: wp.array(dtype=wp.float32),
    offdiag: wp.array(dtype=wp.float32),
):
    g_edge = wp.tid()
    rod = edge_rod_id[g_edge]
    local_edge = g_edge - edge_offsets[rod]
    rod_start = rod_offsets[rod]
    p0_idx = rod_start + local_edge
    p1_idx = p0_idx + 1

    reg = 1.0e-6
    i = g_edge
    im0 = inv_mass[p0_idx]
    im1 = inv_mass[p1_idx]

    for row in range(6):
        for col in range(6):
            val = float(0.0)
            for k in range(3):
                val += jpos[_xr_jidx(i, row, k)] * im0 * jpos[_xr_jidx(i, col, k)]
            for k in range(3):
                val += jpos[_xr_jidx(i, row, k + 3)] * im1 * jpos[_xr_jidx(i, col, k + 3)]

            jr0r = wp.vec3(jrot[_xr_jidx(i, row, 0)], jrot[_xr_jidx(i, row, 1)], jrot[_xr_jidx(i, row, 2)])
            jr0c = wp.vec3(jrot[_xr_jidx(i, col, 0)], jrot[_xr_jidx(i, col, 1)], jrot[_xr_jidx(i, col, 2)])
            val += wp.dot(jr0r, _xr_invI_v(inv_inertia, p0_idx, jr0c))

            jr1r = wp.vec3(jrot[_xr_jidx(i, row, 3)], jrot[_xr_jidx(i, row, 4)], jrot[_xr_jidx(i, row, 5)])
            jr1c = wp.vec3(jrot[_xr_jidx(i, col, 3)], jrot[_xr_jidx(i, col, 4)], jrot[_xr_jidx(i, col, 5)])
            val += wp.dot(jr1r, _xr_invI_v(inv_inertia, p1_idx, jr1c))

            if row == col:
                val += compliance[i * 6 + row] + reg
            diag[_xr_bidx(i, row, col)] = val

    if local_edge == 0:
        for row in range(6):
            for col in range(6):
                offdiag[_xr_bidx(i, row, col)] = 0.0
        return

    prev = g_edge - 1
    for row in range(6):
        for col in range(6):
            val = float(0.0)
            for k in range(3):
                val += jpos[_xr_jidx(prev, row, k + 3)] * im0 * jpos[_xr_jidx(i, col, k)]
            jp_vec = wp.vec3(jrot[_xr_jidx(prev, row, 3)], jrot[_xr_jidx(prev, row, 4)], jrot[_xr_jidx(prev, row, 5)])
            jc_vec = wp.vec3(jrot[_xr_jidx(i, col, 0)], jrot[_xr_jidx(i, col, 1)], jrot[_xr_jidx(i, col, 2)])
            val += wp.dot(jp_vec, _xr_invI_v(inv_inertia, p0_idx, jc_vec))
            offdiag[_xr_bidx(i, row, col)] = val


@wp.kernel
def _xr_block_thomas_batched(
    diag: wp.array(dtype=wp.float32),
    offdiag: wp.array(dtype=wp.float32),
    rhs: wp.array(dtype=wp.float32),
    edge_offsets: wp.array(dtype=wp.int32),
    n_envs: int,
    c_blks: wp.array(dtype=wp.float32),
    d_prime: wp.array(dtype=wp.float32),
    x: wp.array(dtype=wp.float32),
):
    """Batched block-Thomas solve: one GPU thread per rod, all rods in parallel.

    Each thread (one per rod) walks its own [edge_offsets[r], edge_offsets[r+1])
    range of edges. This is the change that turns N×rod_solve_time into
    O(rod_solve_time) when N rods are solved together.
    """
    rod = wp.tid()
    if rod >= n_envs:
        return

    e_start = edge_offsets[rod]
    e_end = edge_offsets[rod + 1]
    n_edges = e_end - e_start

    if n_edges <= 0:
        return

    # Forward sweep — first edge
    A0, B0, C0, D0 = _xr_load_blk(diag, e_start + 0)
    b0, b1 = _xr_load_v6(rhs, e_start + 0)

    if n_edges > 1:
        for col in range(6):
            u0, u1 = _xr_blk_col(offdiag, e_start + 1, col)
            s0, s1 = _xr_blk_solve(A0, B0, C0, D0, u0, u1)
            _xr_blk_set_col(c_blks, e_start + 0, col, s0, s1)
    else:
        z = wp.vec3(0.0, 0.0, 0.0)
        for col in range(6):
            _xr_blk_set_col(c_blks, e_start + 0, col, z, z)

    d0, d1 = _xr_blk_solve(A0, B0, C0, D0, b0, b1)
    _xr_store_v6(d_prime, e_start + 0, d0, d1)

    # Forward sweep — remaining edges
    for li in range(1, n_edges):
        gi = e_start + li
        DiA, DiB, DiC, DiD = _xr_load_blk(diag, gi)
        LtA, LtB, LtC, LtD = _xr_load_blk(offdiag, gi)
        CpA, CpB, CpC, CpD = _xr_load_blk(c_blks, gi - 1)

        LiA = _xr_m33_t(LtA)
        LiB = _xr_m33_t(LtC)
        LiC = _xr_m33_t(LtB)
        LiD = _xr_m33_t(LtD)

        LCA, LCB, LCC, LCD = _xr_blk_mul(LiA, LiB, LiC, LiD, CpA, CpB, CpC, CpD)
        TiA, TiB, TiC, TiD = _xr_blk_sub(DiA, DiB, DiC, DiD, LCA, LCB, LCC, LCD)

        bi0, bi1 = _xr_load_v6(rhs, gi)
        dp0, dp1 = _xr_load_v6(d_prime, gi - 1)
        ld0, ld1 = _xr_blk_mv(LiA, LiB, LiC, LiD, dp0, dp1)
        bi0 = bi0 - ld0
        bi1 = bi1 - ld1

        if li < n_edges - 1:
            for col in range(6):
                u0, u1 = _xr_blk_col(offdiag, gi + 1, col)
                s0, s1 = _xr_blk_solve(TiA, TiB, TiC, TiD, u0, u1)
                _xr_blk_set_col(c_blks, gi, col, s0, s1)
        else:
            z = wp.vec3(0.0, 0.0, 0.0)
            for col in range(6):
                _xr_blk_set_col(c_blks, gi, col, z, z)

        di0, di1 = _xr_blk_solve(TiA, TiB, TiC, TiD, bi0, bi1)
        _xr_store_v6(d_prime, gi, di0, di1)

    # Back substitution
    last = e_start + n_edges - 1
    dn0, dn1 = _xr_load_v6(d_prime, last)
    _xr_store_v6(x, last, dn0, dn1)
    for li in range(n_edges - 2, -1, -1):
        gi = e_start + li
        CiA, CiB, CiC, CiD = _xr_load_blk(c_blks, gi)
        xn0, xn1 = _xr_load_v6(x, gi + 1)
        cx0, cx1 = _xr_blk_mv(CiA, CiB, CiC, CiD, xn0, xn1)
        di0, di1 = _xr_load_v6(d_prime, gi)
        _xr_store_v6(x, gi, di0 - cx0, di1 - cx1)


@wp.kernel
def _xr_compute_corrections_batched(
    inv_mass: wp.array(dtype=wp.float32),
    qinv: wp.array(dtype=wp.float32),
    inv_inertia: wp.array(dtype=wp.float32),
    jpos: wp.array(dtype=wp.float32),
    jrot: wp.array(dtype=wp.float32),
    dlam: wp.array(dtype=wp.float32),
    lam_sum: wp.array(dtype=wp.float32),
    rod_offsets: wp.array(dtype=wp.int32),
    edge_offsets: wp.array(dtype=wp.int32),
    edge_rod_id: wp.array(dtype=wp.int32),
    pcorr: wp.array(dtype=wp.vec3),
    rcorr: wp.array(dtype=wp.vec3),
    max_delta_out: wp.array(dtype=wp.float32),
    max_correction: wp.array(dtype=wp.float32),
):
    g_edge = wp.tid()
    rod = edge_rod_id[g_edge]
    local_edge = g_edge - edge_offsets[rod]
    p0_idx = rod_offsets[rod] + local_edge
    p1_idx = p0_idx + 1

    base = g_edge * 6
    d0 = dlam[base + 0]; d1 = dlam[base + 1]; d2 = dlam[base + 2]
    d3 = dlam[base + 3]; d4 = dlam[base + 4]; d5 = dlam[base + 5]
    lam_sum[base + 0] = lam_sum[base + 0] + d0
    lam_sum[base + 1] = lam_sum[base + 1] + d1
    lam_sum[base + 2] = lam_sum[base + 2] + d2
    lam_sum[base + 3] = lam_sum[base + 3] + d3
    lam_sum[base + 4] = lam_sum[base + 4] + d4
    lam_sum[base + 5] = lam_sum[base + 5] + d5

    local_max = wp.max(
        wp.max(wp.max(wp.abs(d0), wp.abs(d1)), wp.max(wp.abs(d2), wp.abs(d3))),
        wp.max(wp.abs(d4), wp.abs(d5)),
    )
    wp.atomic_max(max_delta_out, 0, local_max)

    local_max_corr = float(0.0)

    im0 = inv_mass[p0_idx]
    im1 = inv_mass[p1_idx]
    if im0 > 0.0:
        dp = wp.vec3(
            _xr_jdot(jpos, g_edge, 0, d0, d1, d2, d3, d4, d5) * im0,
            _xr_jdot(jpos, g_edge, 1, d0, d1, d2, d3, d4, d5) * im0,
            _xr_jdot(jpos, g_edge, 2, d0, d1, d2, d3, d4, d5) * im0,
        )
        local_max_corr = wp.max(local_max_corr, wp.length(dp))
        wp.atomic_add(pcorr, p0_idx, dp)
    if im1 > 0.0:
        dp = wp.vec3(
            _xr_jdot(jpos, g_edge, 3, d0, d1, d2, d3, d4, d5) * im1,
            _xr_jdot(jpos, g_edge, 4, d0, d1, d2, d3, d4, d5) * im1,
            _xr_jdot(jpos, g_edge, 5, d0, d1, d2, d3, d4, d5) * im1,
        )
        local_max_corr = wp.max(local_max_corr, wp.length(dp))
        wp.atomic_add(pcorr, p1_idx, dp)

    if qinv[p0_idx] > 0.0:
        jt0 = wp.vec3(
            _xr_jdot(jrot, g_edge, 0, d0, d1, d2, d3, d4, d5),
            _xr_jdot(jrot, g_edge, 1, d0, d1, d2, d3, d4, d5),
            _xr_jdot(jrot, g_edge, 2, d0, d1, d2, d3, d4, d5),
        )
        dtheta0 = _xr_invI_v(inv_inertia, p0_idx, jt0)
        local_max_corr = wp.max(local_max_corr, wp.length(dtheta0))
        wp.atomic_add(rcorr, p0_idx, dtheta0)
    if qinv[p1_idx] > 0.0:
        jt1 = wp.vec3(
            _xr_jdot(jrot, g_edge, 3, d0, d1, d2, d3, d4, d5),
            _xr_jdot(jrot, g_edge, 4, d0, d1, d2, d3, d4, d5),
            _xr_jdot(jrot, g_edge, 5, d0, d1, d2, d3, d4, d5),
        )
        dtheta1 = _xr_invI_v(inv_inertia, p1_idx, jt1)
        local_max_corr = wp.max(local_max_corr, wp.length(dtheta1))
        wp.atomic_add(rcorr, p1_idx, dtheta1)

    wp.atomic_max(max_correction, 0, local_max_corr)


# ============================================================================
# GPU workspace
# ============================================================================


class _Workspace:
    """Allocates all Warp arrays needed for a single-rod XPBD step."""

    def __init__(self, num_points: int, num_edges: int, device: str):
        np_ = max(1, num_points)
        ne = max(1, num_edges)
        nd = ne * 6

        self.num_points = num_points
        self.num_edges = num_edges
        self.n_dofs = nd
        d = device

        self.positions = wp.zeros(np_, dtype=wp.vec3, device=d)
        self.predicted_positions = wp.zeros(np_, dtype=wp.vec3, device=d)
        self.velocities = wp.zeros(np_, dtype=wp.vec3, device=d)
        self.forces = wp.zeros(np_, dtype=wp.vec3, device=d)

        self.orientations = wp.zeros(np_, dtype=wp.quat, device=d)
        self.predicted_orientations = wp.zeros(np_, dtype=wp.quat, device=d)
        self.prev_orientations = wp.zeros(np_, dtype=wp.quat, device=d)
        self.angular_velocities = wp.zeros(np_, dtype=wp.vec3, device=d)
        self.torques = wp.zeros(np_, dtype=wp.vec3, device=d)

        self.inv_masses = wp.zeros(np_, dtype=wp.float32, device=d)
        self.quat_inv_masses = wp.zeros(np_, dtype=wp.float32, device=d)

        self.rest_lengths = wp.zeros(ne, dtype=wp.float32, device=d)
        self.rest_darboux = wp.zeros(ne, dtype=wp.vec3, device=d)
        self.bend_stiffness = wp.zeros(ne, dtype=wp.vec3, device=d)

        self.constraint_values = wp.zeros(nd, dtype=wp.float32, device=d)
        self.compliance = wp.zeros(nd, dtype=wp.float32, device=d)
        self.lambda_sum = wp.zeros(nd, dtype=wp.float32, device=d)
        self.jacobian_pos = wp.zeros(ne * 36, dtype=wp.float32, device=d)
        self.jacobian_rot = wp.zeros(ne * 36, dtype=wp.float32, device=d)

        self.rhs = wp.zeros(nd, dtype=wp.float32, device=d)
        self.delta_lambda = wp.zeros(nd, dtype=wp.float32, device=d)
        self.diag_blocks = wp.zeros(ne * 36, dtype=wp.float32, device=d)
        self.offdiag_blocks = wp.zeros(ne * 36, dtype=wp.float32, device=d)
        self.c_blocks = wp.zeros(ne * 36, dtype=wp.float32, device=d)
        self.d_prime = wp.zeros(ne * 6, dtype=wp.float32, device=d)

        self.inv_inertia = wp.zeros(np_ * 9, dtype=wp.float32, device=d)
        self.pos_corrections = wp.zeros(np_, dtype=wp.vec3, device=d)
        self.rot_corrections = wp.zeros(np_, dtype=wp.vec3, device=d)

        self.inv_inertia_local_diag = wp.vec3(1.0, 1.0, 1.0)
        self.young_modulus: float = 1.0e6
        self.torsion_modulus: float = 1.0e6
        self.gravity = wp.vec3(0.0, 0.0, -9.81)

        # Lazily allocated scratch for alternative backends
        self._split_stretch_diag = None  # lazily allocated in _solve_split_thomas
        self._A = None        # dense matrix for tiled Cholesky (TILE × TILE)
        self._rhs_tile = None  # rhs buffer for tiled Cholesky (TILE)
        self._dl_tile = None   # solution buffer for tiled Cholesky (TILE)
        self._ab = None        # banded matrix for banded Cholesky (BAND_LDAB × n_dofs)


class _BatchedWorkspace:
    """Concatenated GPU workspace for ``E`` rods with identical topology.

    All per-particle / per-edge / per-DOF arrays for every environment are
    stored in one contiguous Warp buffer each, so a single kernel launch can
    process the entire batch. This is the data layout that lets the solver
    scale to 512+ envs in one substep — see :func:`_xr_block_thomas_batched`.

    Args:
        num_envs: Number of parallel environments.
        num_points_per_rod: Number of particles in each rod (must be the same
            across envs for now — heterogeneous topology can be supported by
            generalising the offset arrays at a small cost).
        num_edges_per_rod: ``num_points_per_rod - 1``.
        device: Warp device string (``"cuda"`` or ``"cpu"``).
    """

    def __init__(self, num_envs: int, num_points_per_rod: int, num_edges_per_rod: int, device: str):
        ne_total = num_envs * num_edges_per_rod
        np_total = num_envs * num_points_per_rod
        nd_total = ne_total * 6

        self.num_envs = num_envs
        self.num_points_per_rod = num_points_per_rod
        self.num_edges_per_rod = num_edges_per_rod
        self.np_total = np_total
        self.ne_total = ne_total
        self.nd_total = nd_total
        self.device = device

        d = device

        # Per-particle state (concatenated across envs)
        self.positions = wp.zeros(np_total, dtype=wp.vec3, device=d)
        self.predicted_positions = wp.zeros(np_total, dtype=wp.vec3, device=d)
        self.velocities = wp.zeros(np_total, dtype=wp.vec3, device=d)
        self.forces = wp.zeros(np_total, dtype=wp.vec3, device=d)

        self.orientations = wp.zeros(np_total, dtype=wp.quat, device=d)
        self.predicted_orientations = wp.zeros(np_total, dtype=wp.quat, device=d)
        self.prev_orientations = wp.zeros(np_total, dtype=wp.quat, device=d)
        self.angular_velocities = wp.zeros(np_total, dtype=wp.vec3, device=d)
        self.torques = wp.zeros(np_total, dtype=wp.vec3, device=d)

        self.inv_masses = wp.zeros(np_total, dtype=wp.float32, device=d)
        self.quat_inv_masses = wp.zeros(np_total, dtype=wp.float32, device=d)

        # Per-edge arrays
        self.rest_lengths = wp.zeros(ne_total, dtype=wp.float32, device=d)
        self.rest_darboux = wp.zeros(ne_total, dtype=wp.vec3, device=d)
        self.bend_stiffness = wp.zeros(ne_total, dtype=wp.vec3, device=d)

        # Constraint workspace
        self.constraint_values = wp.zeros(nd_total, dtype=wp.float32, device=d)
        self.compliance = wp.zeros(nd_total, dtype=wp.float32, device=d)
        self.lambda_sum = wp.zeros(nd_total, dtype=wp.float32, device=d)
        self.jacobian_pos = wp.zeros(ne_total * 36, dtype=wp.float32, device=d)
        self.jacobian_rot = wp.zeros(ne_total * 36, dtype=wp.float32, device=d)

        self.rhs = wp.zeros(nd_total, dtype=wp.float32, device=d)
        self.delta_lambda = wp.zeros(nd_total, dtype=wp.float32, device=d)
        self.diag_blocks = wp.zeros(ne_total * 36, dtype=wp.float32, device=d)
        self.offdiag_blocks = wp.zeros(ne_total * 36, dtype=wp.float32, device=d)
        self.c_blocks = wp.zeros(ne_total * 36, dtype=wp.float32, device=d)
        self.d_prime = wp.zeros(ne_total * 6, dtype=wp.float32, device=d)

        self.inv_inertia = wp.zeros(np_total * 9, dtype=wp.float32, device=d)
        self.pos_corrections = wp.zeros(np_total, dtype=wp.vec3, device=d)
        self.rot_corrections = wp.zeros(np_total, dtype=wp.vec3, device=d)

        # Per-env material parameters (vec3 inv-inertia, float Young/torsion)
        self.inv_inertia_local_diag = wp.zeros(num_envs, dtype=wp.vec3, device=d)
        self.young_modulus = wp.zeros(num_envs, dtype=wp.float32, device=d)
        self.torsion_modulus = wp.zeros(num_envs, dtype=wp.float32, device=d)
        # Per-env gravity (lets envs sit in different orientations)
        self.gravity = wp.zeros(num_envs, dtype=wp.vec3, device=d)

        # Index arrays — built on CPU once at construction
        rod_offsets_np = np.arange(num_envs + 1, dtype=np.int32) * num_points_per_rod
        edge_offsets_np = np.arange(num_envs + 1, dtype=np.int32) * num_edges_per_rod
        particle_rod_id_np = np.repeat(np.arange(num_envs, dtype=np.int32), num_points_per_rod)
        edge_rod_id_np = np.repeat(np.arange(num_envs, dtype=np.int32), num_edges_per_rod)
        # Per-env root particle indices (for proximal control / orientation set)
        root_idx_np = (np.arange(num_envs, dtype=np.int32) * num_points_per_rod)
        next_idx_np = root_idx_np + 1

        self.rod_offsets = wp.array(rod_offsets_np, dtype=wp.int32, device=d)
        self.edge_offsets = wp.array(edge_offsets_np, dtype=wp.int32, device=d)
        self.particle_rod_id = wp.array(particle_rod_id_np, dtype=wp.int32, device=d)
        self.edge_rod_id = wp.array(edge_rod_id_np, dtype=wp.int32, device=d)
        self.root_idx = wp.array(root_idx_np, dtype=wp.int32, device=d)
        self.next_idx = wp.array(next_idx_np, dtype=wp.int32, device=d)

        # Per-env scratch for control kernels (allocated once, written each call)
        self.push_velocities = wp.zeros(num_envs, dtype=wp.float32, device=d)
        self.rotate_velocities = wp.zeros(num_envs, dtype=wp.float32, device=d)

        # Diagnostics (written via atomic_max, read via numpy())
        self.max_delta_lambda = wp.zeros(1, dtype=wp.float32, device=d)
        self.max_correction = wp.zeros(1, dtype=wp.float32, device=d)


# ============================================================================
# XPBDRodSolver — public API
# ============================================================================


class XPBDRodSolver:
    """Self-contained XPBD Cosserat rod solver (block-Thomas direct solve).

    Drop-in replacement for :class:`NewtonXPBDRodSolver` that does **not**
    require the external ``newton`` package — all Warp kernels are embedded.

    Multi-env support: when ``num_envs > 1`` the solver allocates a
    :class:`_BatchedWorkspace` and runs each substep through batched kernels
    that process all rods in a single launch each. The block-Thomas direct
    solve in particular launches one GPU thread per rod, so the per-step
    cost scales with ``max(num_edges)`` rather than ``num_envs × num_edges``.

    Args:
        config: :class:`RodConfig` describing material, geometry, and solver params.
        num_envs: Number of parallel environments. ``1`` uses the single-rod
            path (backward compatible). ``>1`` uses the batched path.
        floor_z: Z coordinate of the floor plane, or ``None`` to disable.
        floor_restitution: Bounce coefficient on floor contact (0..1). Applied
            in both single- and multi-env paths.
        initial_height: Z offset for the rod root particle.
    """

    def __init__(
        self,
        config: RodConfig,
        *,
        num_envs: int = 1,
        floor_z: float | None = 0.0,
        floor_restitution: float = 0.0,
        initial_height: float = 0.5,
        solver_backend: str = XPBD_DIRECT_SOLVE_BLOCK_THOMAS,
    ):
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if solver_backend not in XPBD_DIRECT_SOLVE_BACKENDS:
            raise ValueError(
                f"solver_backend must be one of {XPBD_DIRECT_SOLVE_BACKENDS!r}, got {solver_backend!r}"
            )

        self._config = config
        self._num_envs = int(num_envs)
        self._floor_z = floor_z
        self._floor_restitution = float(floor_restitution)
        self._solver_backend = solver_backend

        device = config.device
        geo = config.geometry
        mat = config.material
        sol = config.solver

        num_segments = int(geo.num_segments)
        seg_len = float(geo.segment_length)
        num_points = num_segments + 1
        num_edges = num_segments

        self._num_points = num_points
        self._num_edges = num_edges
        self._num_substeps = int(sol.num_substeps)
        self._linear_damping = float(mat.damping)
        self._angular_damping = float(mat.damping)

        # ---- compute material properties ----
        r = float(geo.radius) if not isinstance(geo.radius, list) else float(geo.radius[0])
        particle_mass = float(mat.density) * math.pi * r * r * seg_len
        inv_mass = 1.0 / particle_mass if particle_mass > 0.0 else 0.0

        young_modulus = float(mat.young_modulus)
        if mat.shear_modulus is not None:
            torsion_modulus = float(mat.shear_modulus)
        else:
            torsion_modulus = young_modulus / (2.0 * (1.0 + float(mat.poisson_ratio)))

        bend_coeff = float(mat.bend_stiffness)
        twist_coeff = float(mat.twist_stiffness)

        gx, gy, gz = sol.gravity
        gravity = wp.vec3(float(gx), float(gy), float(gz))

        # ---- build initial positions along X axis ----
        positions_np = np.zeros((num_points, 3), dtype=np.float32)
        for i in range(num_points):
            positions_np[i, 0] = i * seg_len
            positions_np[i, 2] = initial_height

        # ---- initial orientations: rotate local Z → +X ----
        half = math.pi * 0.25
        q_align = np.array([0.0, math.sin(half), 0.0, math.cos(half)], dtype=np.float32)
        orientations_np = np.tile(q_align, (num_points, 1))

        # ---- per-particle arrays ----
        inv_masses_np = np.full(num_points, inv_mass, dtype=np.float32)
        inv_masses_np[0] = 0.0  # lock root position
        quat_inv_masses_np = np.ones(num_points, dtype=np.float32)
        quat_inv_masses_np[0] = 0.0  # lock root rotation

        # ---- per-edge arrays ----
        rest_lengths_np = np.full(num_edges, seg_len, dtype=np.float32)
        rest_darboux_np = np.zeros((num_edges, 3), dtype=np.float32)
        bend_stiff_np = np.zeros((num_edges, 3), dtype=np.float32)
        bend_stiff_np[:, 0] = bend_coeff
        bend_stiff_np[:, 1] = bend_coeff
        bend_stiff_np[:, 2] = twist_coeff

        # ---- populate workspace ----
        # Always allocate the single-rod workspace; it provides the canonical
        # public-API surface and is used directly when num_envs == 1.
        ws = _Workspace(num_points, num_edges, device)
        ws.young_modulus = young_modulus
        ws.torsion_modulus = torsion_modulus
        ws.gravity = gravity

        ws.positions.assign(wp.array(positions_np, dtype=wp.vec3, device=device))
        ws.predicted_positions.assign(wp.array(positions_np, dtype=wp.vec3, device=device))
        ws.inv_masses.assign(wp.array(inv_masses_np, dtype=wp.float32, device=device))
        ws.orientations.assign(wp.array(orientations_np, dtype=wp.quat, device=device))
        ws.predicted_orientations.assign(wp.array(orientations_np, dtype=wp.quat, device=device))
        ws.prev_orientations.assign(wp.array(orientations_np, dtype=wp.quat, device=device))
        ws.quat_inv_masses.assign(wp.array(quat_inv_masses_np, dtype=wp.float32, device=device))
        ws.rest_lengths.assign(wp.array(rest_lengths_np, dtype=wp.float32, device=device))
        ws.rest_darboux.assign(wp.array(rest_darboux_np, dtype=wp.vec3, device=device))
        ws.bend_stiffness.assign(wp.array(bend_stiff_np, dtype=wp.vec3, device=device))

        self._ws = ws
        self._device = device

        # Multi-env: build the batched workspace, replicating per-env state.
        self._bws: _BatchedWorkspace | None = None
        if self._num_envs > 1:
            bws = _BatchedWorkspace(self._num_envs, num_points, num_edges, device)

            # Replicate scalar material params per env
            inv_I_local_np = np.tile(np.array([1.0, 1.0, 1.0], dtype=np.float32), (self._num_envs, 1))
            young_np = np.full(self._num_envs, young_modulus, dtype=np.float32)
            tors_np = np.full(self._num_envs, torsion_modulus, dtype=np.float32)
            gravity_np = np.tile(np.array([gx, gy, gz], dtype=np.float32), (self._num_envs, 1))

            bws.inv_inertia_local_diag.assign(wp.array(inv_I_local_np, dtype=wp.vec3, device=device))
            bws.young_modulus.assign(wp.array(young_np, dtype=wp.float32, device=device))
            bws.torsion_modulus.assign(wp.array(tors_np, dtype=wp.float32, device=device))
            bws.gravity.assign(wp.array(gravity_np, dtype=wp.vec3, device=device))

            # Tile per-rod state across envs
            pos_tiled = np.tile(positions_np, (self._num_envs, 1))
            ori_tiled = np.tile(orientations_np, (self._num_envs, 1))
            inv_mass_tiled = np.tile(inv_masses_np, self._num_envs)
            quat_inv_tiled = np.tile(quat_inv_masses_np, self._num_envs)
            rest_len_tiled = np.tile(rest_lengths_np, self._num_envs)
            rest_dar_tiled = np.tile(rest_darboux_np, (self._num_envs, 1))
            bend_stiff_tiled = np.tile(bend_stiff_np, (self._num_envs, 1))

            bws.positions.assign(wp.array(pos_tiled, dtype=wp.vec3, device=device))
            bws.predicted_positions.assign(wp.array(pos_tiled, dtype=wp.vec3, device=device))
            bws.orientations.assign(wp.array(ori_tiled, dtype=wp.quat, device=device))
            bws.predicted_orientations.assign(wp.array(ori_tiled, dtype=wp.quat, device=device))
            bws.prev_orientations.assign(wp.array(ori_tiled, dtype=wp.quat, device=device))
            bws.inv_masses.assign(wp.array(inv_mass_tiled, dtype=wp.float32, device=device))
            bws.quat_inv_masses.assign(wp.array(quat_inv_tiled, dtype=wp.float32, device=device))
            bws.rest_lengths.assign(wp.array(rest_len_tiled, dtype=wp.float32, device=device))
            bws.rest_darboux.assign(wp.array(rest_dar_tiled, dtype=wp.vec3, device=device))
            bws.bend_stiffness.assign(wp.array(bend_stiff_tiled, dtype=wp.vec3, device=device))

            self._bws = bws

        # CUDA graph (lazily captured on first .step() if device supports it)
        self._cuda_graph = None
        self._cuda_graph_dt = None

    # ------------------------------------------------------------------ step
    def step(self, dt: float) -> None:
        """Advance the simulation by *dt* seconds (subdivided into substeps).

        When ``num_envs > 1`` the batched substep is used (one launch per
        kernel covers all rods). Otherwise the single-rod substep is used.

        On CUDA devices, the substep loop is captured into a CUDA graph the
        first time it is run with a given ``dt``; subsequent calls replay the
        graph for ~zero CPU overhead. The capture is invalidated automatically
        if ``dt`` changes or :meth:`reset_cuda_graph` is called explicitly.
        """
        sub_dt = dt / self._num_substeps

        if self._can_use_cuda_graph(sub_dt):
            if self._cuda_graph is None or self._cuda_graph_dt != sub_dt:
                self._capture_cuda_graph(sub_dt)
            wp.capture_launch(self._cuda_graph)
            return

        substep_fn = self._substep_batched if self._bws is not None else self._substep
        for _ in range(self._num_substeps):
            substep_fn(sub_dt)

    # ----------------------------------------------------------- CUDA graph
    def _can_use_cuda_graph(self, sub_dt: float) -> bool:
        """Return True iff the substep loop is safe to capture with CUDA graphs.

        Capture is disabled when:
          - the device is CPU
          - any external callback might mutate state (e.g. user-supplied
            forces are written between substeps — currently not exposed, so
            always safe by default)

        Subclasses can override this to disable capture when they inject
        per-step bookkeeping that the graph would skip.
        """
        if not str(self._device).startswith("cuda"):
            return False
        return True

    def _capture_cuda_graph(self, sub_dt: float) -> None:
        substep_fn = self._substep_batched if self._bws is not None else self._substep
        # Warm up once outside capture so all kernels are JIT-compiled.
        substep_fn(sub_dt)
        wp.synchronize_device(self._device)

        with wp.ScopedCapture(device=self._device) as capture:
            for _ in range(self._num_substeps):
                substep_fn(sub_dt)
        self._cuda_graph = capture.graph
        self._cuda_graph_dt = sub_dt

    def reset_cuda_graph(self) -> None:
        """Drop any cached CUDA graph (call after externally mutating state size)."""
        self._cuda_graph = None
        self._cuda_graph_dt = None

    def _substep(self, dt: float) -> None:
        ws = self._ws
        dev = self._device
        np_ = ws.num_points
        ne = ws.num_edges
        nd = ws.n_dofs

        # 1. Predict positions & rotations
        wp.launch(_xr_predict_pos, dim=np_, inputs=[
            ws.positions, ws.velocities, ws.forces, ws.inv_masses,
            ws.gravity, dt, self._linear_damping, ws.predicted_positions,
        ], device=dev)
        wp.launch(_xr_predict_rot, dim=np_, inputs=[
            ws.orientations, ws.angular_velocities, ws.torques, ws.quat_inv_masses,
            dt, self._angular_damping, ws.predicted_orientations,
        ], device=dev)

        # 2. Pre-constraint hook (e.g. vessel containment, track guidance — override in subclass)
        self._pre_constraints_hook(ws, dt, dev)

        # 3. Prepare constraints
        wp.launch(_xr_zero_f, dim=nd, inputs=[ws.lambda_sum], device=dev)
        wp.launch(_xr_prepare_compliance, dim=ne, inputs=[
            ws.rest_lengths, ws.bend_stiffness,
            float(ws.young_modulus), float(ws.torsion_modulus), dt, ws.compliance,
        ], device=dev)

        # 4. Project constraints (block-Thomas direct solve)
        self._project_direct(ws, dt, dev)

        # 5. Post-constraint hook (e.g. vessel containment post-solve — override in subclass)
        self._post_constraints_hook(ws, dt, dev)

        # 6. Floor collision
        if self._floor_z is not None:
            wp.launch(_xr_floor_collision, dim=np_, inputs=[
                ws.positions, ws.predicted_positions, ws.velocities,
                float(self._floor_z), float(self._floor_restitution),
            ], device=dev)

        # 5. Integrate
        wp.launch(_xr_integrate_pos, dim=np_, inputs=[
            ws.positions, ws.predicted_positions, ws.velocities, ws.inv_masses, dt,
        ], device=dev)
        wp.launch(_xr_integrate_rot, dim=np_, inputs=[
            ws.orientations, ws.predicted_orientations, ws.prev_orientations,
            ws.angular_velocities, ws.quat_inv_masses, dt,
        ], device=dev)

    def _pre_constraints_hook(self, ws: "_Workspace", dt: float, dev: str) -> None:
        """Extension point called before XPBD constraint prepare + solve.

        Override in a subclass (e.g. :class:`~isaaclab_newton.solvers.XCathRodSolver`)
        to inject custom predicted-position projections — such as vessel containment or
        track-guided insertion — *before* compliance is computed and the block-Thomas
        system is solved.  The base implementation is a no-op.
        """
        del ws, dt, dev

    def _post_constraints_hook(self, ws: "_Workspace", dt: float, dev: str) -> None:
        """Extension point called after XPBD constraint solve, before floor collision.

        Override in a subclass to apply vessel containment or any position correction
        that should observe the constraint-projected state.  The base implementation
        is a no-op.
        """
        del ws, dt, dev

    def _pre_constraints_hook_batched(self, bws: "_BatchedWorkspace", dt: float, dev: str) -> None:
        """Batched counterpart of :meth:`_pre_constraints_hook`.  No-op by default."""
        del bws, dt, dev

    def _post_constraints_hook_batched(self, bws: "_BatchedWorkspace", dt: float, dev: str) -> None:
        """Batched counterpart of :meth:`_post_constraints_hook`.  No-op by default."""
        del bws, dt, dev

    def _project_direct(self, ws: _Workspace, dt: float, dev: str) -> None:
        ne = ws.num_edges
        np_ = ws.num_points

        wp.launch(_xr_update_constraints, dim=ne, inputs=[
            ws.predicted_positions, ws.predicted_orientations,
            ws.rest_lengths, ws.rest_darboux, ws.constraint_values,
        ], device=dev)

        wp.launch(_xr_compute_jacobians, dim=ne, inputs=[
            ws.predicted_orientations, ws.rest_lengths,
            ws.jacobian_pos, ws.jacobian_rot,
        ], device=dev)

        wp.launch(_xr_compute_inv_inertia, dim=np_, inputs=[
            ws.predicted_orientations, ws.quat_inv_masses,
            ws.inv_inertia_local_diag, ws.inv_inertia,
        ], device=dev)

        dl = self._solve_system(ws, dt, dev)

        # Two-phase parallel corrections
        wp.launch(_xr_zero_v3, dim=np_, inputs=[ws.pos_corrections], device=dev)
        wp.launch(_xr_zero_v3, dim=np_, inputs=[ws.rot_corrections], device=dev)

        wp.launch(_xr_compute_corrections, dim=ne, inputs=[
            ws.inv_masses, ws.quat_inv_masses, ws.inv_inertia,
            ws.jacobian_pos, ws.jacobian_rot, dl,
            ws.lambda_sum, int(ne), ws.pos_corrections, ws.rot_corrections,
        ], device=dev)

        wp.launch(_xr_apply_corrections, dim=np_, inputs=[
            ws.predicted_positions, ws.predicted_orientations,
            ws.pos_corrections, ws.rot_corrections, int(np_),
        ], device=dev)

    def _solve_system(self, ws: _Workspace, dt: float, dev: str) -> "wp.array":
        """Assemble and solve the linear system, dispatching to the configured backend."""
        ne = ws.num_edges
        nd = ws.n_dofs

        if self._solver_backend == XPBD_DIRECT_SOLVE_SPLIT_THOMAS:
            return self._solve_split_thomas(ws, ne, dev)

        if self._solver_backend == XPBD_DIRECT_SOLVE_BLOCK_JACOBI:
            wp.launch(_xr_assemble_jmjt, dim=ne, inputs=[
                ws.jacobian_pos, ws.jacobian_rot, ws.compliance,
                ws.inv_masses, ws.inv_inertia, int(ne),
                ws.diag_blocks, ws.offdiag_blocks,
            ], device=dev)
            wp.launch(_xr_build_rhs, dim=nd, inputs=[
                ws.constraint_values, ws.compliance, ws.lambda_sum, int(nd), ws.rhs,
            ], device=dev)
            wp.launch(_xr_solve_blocks_jacobi, dim=ne, inputs=[
                ws.diag_blocks, ws.rhs, ws.delta_lambda, int(ne),
            ], device=dev)
            return ws.delta_lambda

        if self._solver_backend == XPBD_DIRECT_SOLVE_BANDED_CHOLESKY:
            if ws._ab is None:
                ws._ab = wp.zeros((_XPBD_BAND_LDAB, max(1, nd)), dtype=wp.float32, device=dev)
            wp.launch(_xr_zero_2d, dim=_XPBD_BAND_LDAB * max(1, nd), inputs=[
                ws._ab, int(_XPBD_BAND_LDAB), int(max(1, nd)),
            ], device=dev)
            wp.launch(_xr_assemble_jmjt_banded, dim=ne, inputs=[
                ws.jacobian_pos, ws.jacobian_rot, ws.compliance,
                ws.inv_masses, ws.inv_inertia, int(nd), ws._ab,
            ], device=dev)
            wp.launch(_xr_build_rhs, dim=nd, inputs=[
                ws.constraint_values, ws.compliance, ws.lambda_sum, int(nd), ws.rhs,
            ], device=dev)
            wp.launch(_xr_spbsv_u11_1rhs, dim=1, inputs=[int(nd), ws._ab, ws.rhs], device=dev)
            return ws.rhs

        # Default: block_thomas (with tiled Cholesky for small systems)
        if nd <= _XPBD_TILE:
            if ws._A is None:
                ws._A = wp.zeros((_XPBD_TILE, _XPBD_TILE), dtype=wp.float32, device=dev)
                ws._rhs_tile = wp.zeros(_XPBD_TILE, dtype=wp.float32, device=dev)
                ws._dl_tile = wp.zeros(_XPBD_TILE, dtype=wp.float32, device=dev)
            wp.launch(_xr_zero_2d, dim=_XPBD_TILE * _XPBD_TILE, inputs=[
                ws._A, int(_XPBD_TILE), int(_XPBD_TILE),
            ], device=dev)
            wp.launch(_xr_assemble_jmjt_dense, dim=ne, inputs=[
                ws.jacobian_pos, ws.jacobian_rot, ws.compliance,
                ws.inv_masses, ws.inv_inertia, int(nd), ws._A,
            ], device=dev)
            wp.launch(_xr_build_rhs, dim=_XPBD_TILE, inputs=[
                ws.constraint_values, ws.compliance, ws.lambda_sum, int(nd), ws._rhs_tile,
            ], device=dev)
            if nd < _XPBD_TILE:
                wp.launch(_xr_pad_diagonal, dim=_XPBD_TILE, inputs=[
                    ws._A, int(nd), int(_XPBD_TILE),
                ], device=dev)
            wp.launch_tiled(
                _xr_cholesky_solve_tile,
                dim=[1, 1],
                inputs=[ws._A, ws._rhs_tile],
                outputs=[ws._dl_tile],
                block_dim=_XPBD_BLOCK_DIM,
                device=dev,
            )
            return ws._dl_tile

        # Block Thomas for larger systems
        wp.launch(_xr_assemble_jmjt, dim=ne, inputs=[
            ws.jacobian_pos, ws.jacobian_rot, ws.compliance,
            ws.inv_masses, ws.inv_inertia, int(ne),
            ws.diag_blocks, ws.offdiag_blocks,
        ], device=dev)
        wp.launch(_xr_build_rhs, dim=nd, inputs=[
            ws.constraint_values, ws.compliance, ws.lambda_sum, int(nd), ws.rhs,
        ], device=dev)
        wp.launch(_xr_block_thomas, dim=1, inputs=[
            ws.diag_blocks, ws.offdiag_blocks, ws.rhs, int(ne),
            ws.c_blocks, ws.d_prime, ws.delta_lambda,
        ], device=dev)
        return ws.delta_lambda

    def _solve_split_thomas(self, ws: _Workspace, ne: int, dev: str) -> "wp.array":
        """Solve using split 3×3 block Thomas for stretch and Darboux separately."""
        if ws._split_stretch_diag is None:
            ws._split_stretch_diag = wp.zeros(ne * 9, dtype=wp.float32, device=dev)
            ws._split_stretch_offdiag = wp.zeros(ne * 9, dtype=wp.float32, device=dev)
            ws._split_stretch_rhs = wp.zeros(ne * 3, dtype=wp.float32, device=dev)
            ws._split_stretch_c = wp.zeros(ne * 9, dtype=wp.float32, device=dev)
            ws._split_stretch_d_prime = wp.zeros(ne * 3, dtype=wp.float32, device=dev)
            ws._split_stretch_dl = wp.zeros(ne * 3, dtype=wp.float32, device=dev)
            ws._split_darboux_diag = wp.zeros(ne * 9, dtype=wp.float32, device=dev)
            ws._split_darboux_offdiag = wp.zeros(ne * 9, dtype=wp.float32, device=dev)
            ws._split_darboux_rhs = wp.zeros(ne * 3, dtype=wp.float32, device=dev)
            ws._split_darboux_c = wp.zeros(ne * 9, dtype=wp.float32, device=dev)
            ws._split_darboux_d_prime = wp.zeros(ne * 3, dtype=wp.float32, device=dev)
            ws._split_darboux_dl = wp.zeros(ne * 3, dtype=wp.float32, device=dev)

        wp.launch(_xr_assemble_stretch_blocks, dim=ne, inputs=[
            ws.jacobian_pos, ws.jacobian_rot, ws.compliance,
            ws.inv_masses, ws.inv_inertia, int(ne),
            ws._split_stretch_diag, ws._split_stretch_offdiag,
        ], device=dev)
        wp.launch(_xr_assemble_darboux_blocks, dim=ne, inputs=[
            ws.jacobian_rot, ws.compliance, ws.inv_inertia, int(ne),
            ws._split_darboux_diag, ws._split_darboux_offdiag,
        ], device=dev)

        wp.launch(_xr_build_rhs_stretch, dim=ne, inputs=[
            ws.constraint_values, ws.compliance, ws.lambda_sum, int(ne), ws._split_stretch_rhs,
        ], device=dev)
        wp.launch(_xr_build_rhs_darboux, dim=ne, inputs=[
            ws.constraint_values, ws.compliance, ws.lambda_sum, int(ne), ws._split_darboux_rhs,
        ], device=dev)

        wp.launch(_xr_block_thomas_3x3, dim=1, inputs=[
            ws._split_stretch_diag, ws._split_stretch_offdiag, ws._split_stretch_rhs, int(ne),
            ws._split_stretch_c, ws._split_stretch_d_prime, ws._split_stretch_dl,
        ], device=dev)
        wp.launch(_xr_block_thomas_3x3, dim=1, inputs=[
            ws._split_darboux_diag, ws._split_darboux_offdiag, ws._split_darboux_rhs, int(ne),
            ws._split_darboux_c, ws._split_darboux_d_prime, ws._split_darboux_dl,
        ], device=dev)

        wp.launch(_xr_merge_delta_lambda, dim=ne, inputs=[
            ws._split_stretch_dl, ws._split_darboux_dl, ws.delta_lambda, int(ne),
        ], device=dev)
        return ws.delta_lambda

    # ----------------------------------------------------------- batched path
    def _substep_batched(self, dt: float) -> None:
        """Run one XPBD substep across all envs using batched kernels.

        Mirrors :meth:`_substep` but every launch covers ``num_envs`` rods at
        once. The block-Thomas solve in particular runs one GPU thread per
        rod in parallel (see :func:`_xr_block_thomas_batched`).
        """
        bws = self._bws
        assert bws is not None
        dev = self._device
        tp = bws.np_total
        te = bws.ne_total
        td = bws.nd_total

        # 1. Predict positions & rotations (per-env gravity)
        wp.launch(_xr_predict_pos_batched, dim=tp, inputs=[
            bws.positions, bws.velocities, bws.forces, bws.inv_masses,
            bws.gravity, bws.particle_rod_id, dt, self._linear_damping,
            bws.predicted_positions,
        ], device=dev)
        wp.launch(_xr_predict_rot_batched, dim=tp, inputs=[
            bws.orientations, bws.angular_velocities, bws.torques,
            bws.quat_inv_masses, dt, self._angular_damping,
            bws.predicted_orientations,
        ], device=dev)

        # 2. Pre-constraint hook (batched — override in subclass for custom per-env projections)
        self._pre_constraints_hook_batched(bws, dt, dev)

        # 3. Prepare constraints (per-env Young / torsion moduli)
        wp.launch(_xr_zero_f, dim=td, inputs=[bws.lambda_sum], device=dev)
        wp.launch(_xr_prepare_compliance_batched, dim=te, inputs=[
            bws.rest_lengths, bws.bend_stiffness, bws.edge_rod_id,
            bws.young_modulus, bws.torsion_modulus, dt, bws.compliance,
        ], device=dev)

        # 4. Project constraints (block-Thomas direct solve, one thread per rod)
        self._project_direct_batched(bws, dt, dev)

        # 5. Post-constraint hook (batched — override in subclass for custom per-env projections)
        self._post_constraints_hook_batched(bws, dt, dev)

        # 6. Floor collision
        if self._floor_z is not None:
            wp.launch(_xr_floor_collision, dim=tp, inputs=[
                bws.positions, bws.predicted_positions, bws.velocities,
                float(self._floor_z), float(self._floor_restitution),
            ], device=dev)

        # 5. Integrate
        wp.launch(_xr_integrate_pos_batched, dim=tp, inputs=[
            bws.positions, bws.predicted_positions, bws.velocities,
            bws.inv_masses, dt,
        ], device=dev)
        wp.launch(_xr_integrate_rot_batched, dim=tp, inputs=[
            bws.orientations, bws.predicted_orientations, bws.prev_orientations,
            bws.angular_velocities, bws.quat_inv_masses, dt,
        ], device=dev)

    def _project_direct_batched(self, bws: _BatchedWorkspace, dt: float, dev: str) -> None:
        tp = bws.np_total
        te = bws.ne_total
        td = bws.nd_total

        wp.launch(_xr_update_constraints_batched, dim=te, inputs=[
            bws.predicted_positions, bws.predicted_orientations,
            bws.rest_lengths, bws.rest_darboux,
            bws.rod_offsets, bws.edge_offsets, bws.edge_rod_id,
            bws.constraint_values,
        ], device=dev)

        wp.launch(_xr_compute_jacobians_batched, dim=te, inputs=[
            bws.predicted_orientations, bws.rest_lengths,
            bws.rod_offsets, bws.edge_offsets, bws.edge_rod_id,
            bws.jacobian_pos, bws.jacobian_rot,
        ], device=dev)

        wp.launch(_xr_compute_inv_inertia_batched, dim=tp, inputs=[
            bws.predicted_orientations, bws.quat_inv_masses,
            bws.inv_inertia_local_diag, bws.particle_rod_id,
            bws.inv_inertia,
        ], device=dev)

        wp.launch(_xr_assemble_jmjt_batched, dim=te, inputs=[
            bws.jacobian_pos, bws.jacobian_rot, bws.compliance,
            bws.inv_masses, bws.inv_inertia,
            bws.rod_offsets, bws.edge_offsets, bws.edge_rod_id,
            bws.diag_blocks, bws.offdiag_blocks,
        ], device=dev)

        wp.launch(_xr_build_rhs, dim=td, inputs=[
            bws.constraint_values, bws.compliance, bws.lambda_sum, int(td), bws.rhs,
        ], device=dev)

        wp.launch(_xr_block_thomas_batched, dim=bws.num_envs, inputs=[
            bws.diag_blocks, bws.offdiag_blocks, bws.rhs,
            bws.edge_offsets, int(bws.num_envs),
            bws.c_blocks, bws.d_prime, bws.delta_lambda,
        ], device=dev)

        # Two-phase parallel corrections + diagnostic max-|Δλ| and max-correction
        wp.launch(_xr_zero_v3, dim=tp, inputs=[bws.pos_corrections], device=dev)
        wp.launch(_xr_zero_v3, dim=tp, inputs=[bws.rot_corrections], device=dev)
        wp.launch(_xr_zero_f, dim=1, inputs=[bws.max_delta_lambda], device=dev)
        wp.launch(_xr_zero_f, dim=1, inputs=[bws.max_correction], device=dev)

        wp.launch(_xr_compute_corrections_batched, dim=te, inputs=[
            bws.inv_masses, bws.quat_inv_masses, bws.inv_inertia,
            bws.jacobian_pos, bws.jacobian_rot,
            bws.delta_lambda, bws.lambda_sum,
            bws.rod_offsets, bws.edge_offsets, bws.edge_rod_id,
            bws.pos_corrections, bws.rot_corrections, bws.max_delta_lambda,
            bws.max_correction,
        ], device=dev)

        wp.launch(_xr_apply_corrections, dim=tp, inputs=[
            bws.predicted_positions, bws.predicted_orientations,
            bws.pos_corrections, bws.rot_corrections, int(tp),
        ], device=dev)

    # --------------------------------------------------------- proximal control
    def apply_proximal_control(
        self,
        push_velocity: float,
        rotate_velocity: float,
        dt: float,
    ) -> None:
        """Apply kinematic push/rotate control to the root (proximal) particle.

        The root particle has ``inv_mass = 0`` so the XPBD solver will not move
        it — this kinematic update is the sole controller.

        This is the **CPU-bridged** path (uses PyTorch + a ``.norm()`` sync).
        For the multi-env case, prefer :meth:`apply_proximal_control_gpu`,
        which is fully GPU-resident and CUDA-graph capturable.

        Args:
            push_velocity: Insertion speed along the rod axis (m/s).
                Positive = advance (push), negative = retract (pull).
            rotate_velocity: Axial rotation speed (rad/s).
                Positive = counter-clockwise when looking from proximal end.
            dt: Time delta for this control step (seconds).
        """
        if self._bws is not None:
            push = np.full(self._num_envs, float(push_velocity), dtype=np.float32)
            rot = np.full(self._num_envs, float(rotate_velocity), dtype=np.float32)
            self.apply_proximal_control_gpu(push, rot, dt)
            return

        pos_torch = wp.to_torch(self._ws.positions)
        ori_torch = wp.to_torch(self._ws.orientations)

        if self._num_points < 2:
            return

        tangent = pos_torch[1] - pos_torch[0]
        tangent_len = tangent.norm()
        if tangent_len > 1e-8:
            tangent = tangent / tangent_len
        else:
            tangent = torch.tensor([1.0, 0.0, 0.0], device=pos_torch.device)

        pos_torch[0] += tangent * push_velocity * dt

        if abs(rotate_velocity) > 1e-10:
            half_angle = 0.5 * rotate_velocity * dt
            s = math.sin(half_angle)
            c = math.cos(half_angle)
            dq = torch.tensor(
                [tangent[0] * s, tangent[1] * s, tangent[2] * s, c],
                device=ori_torch.device, dtype=ori_torch.dtype,
            )
            q0 = ori_torch[0].clone()
            ori_torch[0] = torch.tensor([
                dq[3] * q0[0] + dq[0] * q0[3] + dq[1] * q0[2] - dq[2] * q0[1],
                dq[3] * q0[1] - dq[0] * q0[2] + dq[1] * q0[3] + dq[2] * q0[0],
                dq[3] * q0[2] + dq[0] * q0[1] - dq[1] * q0[0] + dq[2] * q0[3],
                dq[3] * q0[3] - dq[0] * q0[0] - dq[1] * q0[1] - dq[2] * q0[2],
            ], device=ori_torch.device, dtype=ori_torch.dtype)
            n = ori_torch[0].norm()
            if n > 1e-8:
                ori_torch[0] /= n

    def apply_proximal_control_gpu(
        self,
        push_velocities,
        rotate_velocities,
        dt: float,
    ) -> None:
        """Apply per-env push/rotate control fully on the GPU (no CPU sync).

        This is the recommended controller for multi-env training: it issues a
        single Warp launch over ``num_envs`` threads and is safe to capture
        inside a CUDA graph.

        Args:
            push_velocities: ``(num_envs,)`` array-like of axial push speeds.
                Accepts ``np.ndarray``, ``torch.Tensor``, ``wp.array``, or list.
            rotate_velocities: ``(num_envs,)`` array-like of axial rotation
                speeds (rad/s).
            dt: Time delta for this control step (seconds).
        """
        # Single-rod fast path: just delegate to the proven CPU-bridged method
        # so behaviour is bit-for-bit identical to legacy code.
        if self._bws is None:
            push_v = float(np.asarray(push_velocities).reshape(-1)[0])
            rot_v = float(np.asarray(rotate_velocities).reshape(-1)[0])
            self.apply_proximal_control(push_v, rot_v, dt)
            return

        bws = self._bws
        dev = self._device
        n = bws.num_envs

        # Marshal to wp.array of length num_envs (works for np / torch / list / wp).
        push_arr = self._coerce_per_env_float_array(push_velocities, n, dev, bws.push_velocities)
        rot_arr = self._coerce_per_env_float_array(rotate_velocities, n, dev, bws.rotate_velocities)

        wp.launch(
            _xr_proximal_push_kernel,
            dim=n,
            inputs=[
                bws.positions, bws.orientations,
                push_arr, rot_arr,
                bws.root_idx, bws.next_idx,
                float(dt), int(n),
            ],
            device=dev,
        )

    def set_root_orientation(self, env_idx: int, q) -> None:
        """Force the root particle of env ``env_idx`` to orientation ``q``.

        GPU-side, single-thread launch — safe inside a CUDA graph.

        Args:
            env_idx: Environment index in ``[0, num_envs)``.
            q: New root orientation as a ``(4,)`` xyzw array-like or
                :class:`wp.quat`.
        """
        if isinstance(q, wp.quat):
            q_wp = q
        else:
            q_arr = np.asarray(q, dtype=np.float32).reshape(-1)
            if q_arr.size != 4:
                raise ValueError(f"set_root_orientation expects a 4-vector, got shape {q_arr.shape}")
            q_wp = wp.quat(float(q_arr[0]), float(q_arr[1]), float(q_arr[2]), float(q_arr[3]))

        if self._bws is None:
            if env_idx != 0:
                raise IndexError(f"env_idx {env_idx} out of range for single-env solver")
            wp.launch(
                _xr_set_root_orientation_kernel,
                dim=1,
                inputs=[
                    self._ws.orientations,
                    self._ws.predicted_orientations,
                    self._ws.prev_orientations,
                    int(0), q_wp,
                ],
                device=self._device,
            )
            return

        bws = self._bws
        if not (0 <= env_idx < bws.num_envs):
            raise IndexError(f"env_idx {env_idx} out of range [0, {bws.num_envs})")
        root_idx = int(env_idx * bws.num_points_per_rod)
        wp.launch(
            _xr_set_root_orientation_kernel,
            dim=1,
            inputs=[
                bws.orientations, bws.predicted_orientations, bws.prev_orientations,
                root_idx, q_wp,
            ],
            device=self._device,
        )

    @staticmethod
    def _coerce_per_env_float_array(values, n: int, dev: str, scratch: "wp.array") -> "wp.array":
        """Materialise ``values`` into a ``wp.array`` of float32 length ``n``.

        Reuses the workspace ``scratch`` buffer to avoid an alloc per step.
        """
        if isinstance(values, wp.array):
            if values.size == n and values.dtype == wp.float32:
                return values
            arr_np = values.numpy().astype(np.float32, copy=False).reshape(-1)
        elif isinstance(values, torch.Tensor):
            arr_np = values.detach().to("cpu").contiguous().view(-1).numpy().astype(np.float32, copy=False)
        else:
            arr_np = np.asarray(values, dtype=np.float32).reshape(-1)

        if arr_np.size == 1 and n > 1:
            arr_np = np.full(n, float(arr_np[0]), dtype=np.float32)
        if arr_np.size != n:
            raise ValueError(f"Expected per-env array of length {n}, got {arr_np.size}")

        scratch.assign(wp.array(arr_np, dtype=wp.float32, device=dev))
        return scratch

    # ------------------------------------------------------------ properties
    @property
    def positions(self) -> torch.Tensor:
        """Current particle positions as a PyTorch tensor.

        Shape:
          - single env: ``(num_points, 3)``
          - multi env:  ``(num_envs, num_points, 3)``
        """
        if self._bws is not None:
            t = wp.to_torch(self._bws.positions).clone()
            return t.view(self._num_envs, self._num_points, 3)
        return wp.to_torch(self._ws.positions).clone()

    @property
    def orientations(self) -> torch.Tensor:
        """Current particle orientations xyzw as a PyTorch tensor.

        Shape:
          - single env: ``(num_points, 4)``
          - multi env:  ``(num_envs, num_points, 4)``
        """
        if self._bws is not None:
            t = wp.to_torch(self._bws.orientations).clone()
            return t.view(self._num_envs, self._num_points, 4)
        return wp.to_torch(self._ws.orientations).clone()

    @property
    def velocities(self) -> torch.Tensor:
        """Current particle velocities as a PyTorch tensor.

        Shape:
          - single env: ``(num_points, 3)``
          - multi env:  ``(num_envs, num_points, 3)``
        """
        if self._bws is not None:
            t = wp.to_torch(self._bws.velocities).clone()
            return t.view(self._num_envs, self._num_points, 3)
        return wp.to_torch(self._ws.velocities).clone()

    @property
    def num_points(self) -> int:
        return self._num_points

    @property
    def num_edges(self) -> int:
        return self._num_edges

    @property
    def num_envs(self) -> int:
        """Number of parallel environments (rods) the solver was built for."""
        return self._num_envs

    @property
    def max_delta_lambda(self) -> float:
        """Largest ``|Δλ|`` written by the most recent batched substep.

        Useful for monitoring solver convergence in multi-env training.
        Returns ``0.0`` for the single-env path (diagnostic not collected).
        """
        if self._bws is None:
            return 0.0
        return float(self._bws.max_delta_lambda.numpy()[0])

    @property
    def max_correction(self) -> float:
        """Largest correction magnitude written by the most recent batched substep.

        Tracks the maximum L2 norm of per-particle position/rotation corrections
        across all edges in the batch. Returns ``0.0`` for the single-env path.
        """
        if self._bws is None:
            return 0.0
        return float(self._bws.max_correction.numpy()[0])
