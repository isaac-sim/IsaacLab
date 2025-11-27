# x7_kinematics.py
#
# Kinematics for X7-duo(2) robot described by the given URDF.
# - Left arm  : Joint1 .. Joint7, end-effector = Link7 origin
# - Right arm : Joint8 .. Joint14, end-effector = Link14 origin
#
# Provides:
#   fk_left(q)  -> (pos[3], R[3,3], T[4,4])
#   fk_right(q) -> (pos[3], R[3,3], T[4,4])
#   ik_left(target_pos, target_R, q_init, ...)
#   ik_right(...)
#
# Orientation in IK 用 3D 旋转向量（axis * angle）来表示误差，保证是 6D 控制。

import numpy as np


# ---------- basic rotation / transform helpers ----------

def rot_from_axis_angle(axis, angle):
    """Rodrigues formula, axis is 3D unit vector, angle in rad."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    R = np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C]
    ])
    return R


def rot_x(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([
        [1, 0, 0],
        [0, ca, -sa],
        [0, sa, ca],
    ])


def rot_y(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([
        [ca, 0, sa],
        [0, 1, 0],
        [-sa, 0, ca],
    ])


def rot_z(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([
        [ca, -sa, 0],
        [sa, ca, 0],
        [0, 0, 1],
    ])


def rpy_to_rot(rpy):
    """URDF 风格 RPY: roll (X) -> pitch (Y) -> yaw (Z)."""
    r, p, y = rpy
    return rot_z(y) @ rot_y(p) @ rot_x(r)


def make_transform(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def axis_angle_from_rot(R):
    """R ∈ SO(3) -> 3D axis-angle vector (axis * angle)."""
    R = np.asarray(R, dtype=float)
    trace = np.clip(np.trace(R), -1.0, 3.0)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-8:
        # very small angle -> approximate as zero
        return np.zeros(3)

    wx = R[2, 1] - R[1, 2]
    wy = R[0, 2] - R[2, 0]
    wz = R[1, 0] - R[0, 1]
    v = np.array([wx, wy, wz])
    v = v / (2.0 * np.sin(theta))
    return v * theta


# ---------- X7 joint definitions (from URDF) ----------

# 每个元素: (origin_xyz, origin_rpy, axis)
# 注意：这些是 parent_link -> child_link 的 joint 原点和 axis
LEFT_JOINTS = [
    # Joint1
    (np.array([0.0, 0.0, 0.4345]),
     np.array([0.0, 0.0, -1.5251]),
     np.array([0.0, 0.0, 1.0])),
    # Joint2
    (np.array([0.0, 0.0, 0.0]),
     np.array([1.5708, 0.0, 3.1416]),
     np.array([0.0, 0.0, -1.0])),
    # Joint3
    (np.array([0.0, 0.2525, 0.0]),
     np.array([1.5708, 0.0, 3.1416]),
     np.array([0.0, 0.0, -1.0])),
    # Joint4
    (np.array([0.0, 0.0, 0.0]),
     np.array([1.5708, 0.0, 3.1416]),
     np.array([0.0, 0.0, -1.0])),
    # Joint5
    (np.array([0.0, 0.2525, 0.0]),
     np.array([1.5708, 0.0, 3.1416]),
     np.array([0.0, 0.0, -1.0])),
    # Joint6
    (np.array([0.0, 0.0, 0.0]),
     np.array([1.5708, 0.0, 3.1416]),
     np.array([0.0, 0.0, -1.0])),
    # Joint7
    (np.array([0.0, 0.0, 0.0]),
     np.array([1.5708, 0.0, 3.1416]),
     np.array([0.0, 0.0, 1.0])),
]

RIGHT_JOINTS = [
    # Joint8
    (np.array([0.0, 0.0, -0.4385]),
     np.array([3.1416, 0.0, 1.6376]),
     np.array([0.0, 0.0, 1.0])),
    # Joint9
    (np.array([0.0, 0.0, 0.0]),
     np.array([1.5708, 0.0, 3.1416]),
     np.array([0.0, 0.0, -1.0])),
    # Joint10
    (np.array([0.0, 0.2525, 0.0]),
     np.array([1.5708, 0.0, 3.1416]),
     np.array([0.0, 0.0, -1.0])),
    # Joint11
    (np.array([0.0, 0.0, 0.0]),
     np.array([1.5708, 0.0, 3.1416]),
     np.array([0.0, 0.0, -1.0])),
    # Joint12
    (np.array([0.0, 0.2525, 0.0]),
     np.array([1.5708, 0.0, 3.1416]),
     np.array([0.0, 0.0, -1.0])),
    # Joint13
    (np.array([0.0, 0.0, 0.0]),
     np.array([1.5708, 0.0, 3.1416]),
     np.array([0.0, 0.0, -1.0])),
    # Joint14
    (np.array([0.0, 0.0, 0.0]),
     np.array([1.5708, 0.0, -3.1416]),
     np.array([0.0, 0.0, 1.0])),
]

# 关节上下限（左右臂都一样）
# lower="-3.14" upper="3.14"
JOINT_LIMITS = [(-3.14, 3.14)] * 7


# ---------- FK ----------

def _fk_chain(q, joint_defs):
    """Generic FK: q shape (7,), joint_defs is LEFT_JOINTS or RIGHT_JOINTS.
    Returns:
        pos (3,), R (3,3), T (4,4) in base_link frame.
    """
    assert len(q) == 7
    T = np.eye(4)
    for angle, (xyz, rpy, axis) in zip(q, joint_defs):
        # parent_link -> joint frame
        R0 = rpy_to_rot(rpy)
        T0 = make_transform(R0, xyz)
        # joint rotation about given axis
        Rj = rot_from_axis_angle(axis, angle)
        Tj = make_transform(Rj, np.zeros(3))
        T = T @ T0 @ Tj

    R = T[:3, :3]
    pos = T[:3, 3]
    return pos, R, T


def fk_left(q):
    """Forward kinematics for left arm (Joint1..Joint7 -> Link7 origin)."""
    q = np.asarray(q, dtype=float).reshape(7)
    return _fk_chain(q, LEFT_JOINTS)


def fk_right(q):
    """Forward kinematics for right arm (Joint8..Joint14 -> Link14 origin)."""
    q = np.asarray(q, dtype=float).reshape(7)
    return _fk_chain(q, RIGHT_JOINTS)


# ---------- numeric Jacobian ----------

def _numeric_jacobian(q, joint_defs, eps=1e-5):
    """6x7 Jacobian by finite differences on [pos; orient_vec]."""
    q = np.asarray(q, dtype=float).reshape(7)
    pos0, R0, _ = _fk_chain(q, joint_defs)
    J = np.zeros((6, 7))

    for i in range(7):
        dq = np.zeros(7)
        dq[i] = eps
        pos1, R1, _ = _fk_chain(q + dq, joint_defs)

        dp = (pos1 - pos0) / eps
        # orientation difference from R0 -> R1
        R_rel = R1 @ R0.T
        dw = axis_angle_from_rot(R_rel) / eps

        J[:, i] = np.concatenate([dp, dw])

    return J


# ---------- IK (iterative, 6D) ----------

def _ik_numeric(target_pos, target_R, q_init, joint_defs,
                max_iters=100, tol=1e-4, step=0.5, damping=1e-4):
    """
    target_pos: (3,)
    target_R: (3,3)
    q_init: (7,)
    Returns:
        q_sol (7,), success (bool), iters (int)
    """
    q = np.asarray(q_init, dtype=float).reshape(7)
    target_pos = np.asarray(target_pos, dtype=float).reshape(3)
    target_R = np.asarray(target_R, dtype=float).reshape(3, 3)

    for it in range(max_iters):
        pos, R, _ = _fk_chain(q, joint_defs)

        # position error
        e_pos = target_pos - pos
        # orientation error: R_err = R_target * R_current^T
        R_err = target_R @ R.T
        e_ori = axis_angle_from_rot(R_err)

        e = np.concatenate([e_pos, e_ori])

        if np.linalg.norm(e) < tol:
            # clamp to joint limits before return
            for j in range(7):
                lo, hi = JOINT_LIMITS[j]
                q[j] = np.clip(q[j], lo, hi)
            return q, True, it + 1

        J = _numeric_jacobian(q, joint_defs)

        # Damped least squares: dq = (J^T J + λI)^-1 J^T e
        JT = J.T
        H = JT @ J + damping * np.eye(7)
        g = JT @ e
        dq = np.linalg.solve(H, g)

        q = q + step * dq

        # clamp to limits
        for j in range(7):
            lo, hi = JOINT_LIMITS[j]
            q[j] = np.clip(q[j], lo, hi)

    # not converged
    return q, False, max_iters


def ik_left(target_pos, target_R, q_init,
            max_iters=100, tol=1e-4, step=0.5, damping=1e-4):
    """
    IK for left arm (Joint1..Joint7 -> Link7 origin).

    target_pos: (3,)  目标末端位置（base_link 坐标系）
    target_R:   (3,3) 目标末端姿态旋转矩阵
    q_init:     (7,)  初始关节角
    """
    return _ik_numeric(target_pos, target_R, q_init, LEFT_JOINTS,
                       max_iters=max_iters, tol=tol,
                       step=step, damping=damping)


def ik_right(target_pos, target_R, q_init,
             max_iters=100, tol=1e-4, step=0.5, damping=1e-4):
    """
    IK for right arm (Joint8..Joint14 -> Link14 origin).
    """
    return _ik_numeric(target_pos, target_R, q_init, RIGHT_JOINTS,
                       max_iters=max_iters, tol=tol,
                       step=step, damping=damping)


# ---------- Example usage (for quick test) ----------
if __name__ == "__main__":
    # zero config test
    q0 = np.zeros(7)
    pos_l, R_l, T_l = fk_left(q0)
    pos_r, R_r, T_r = fk_right(q0)
    print("Left FK @ zero:", pos_l)
    print("Right FK @ zero:", pos_r)

    # try IK back to the same pose
    q_sol_l, ok_l, it_l = ik_left(pos_l, R_l, q0)
    print("IK left success:", ok_l, "iters:", it_l, "q:", q_sol_l)

    q_sol_r, ok_r, it_r = ik_right(pos_r, R_r, q0)
    print("IK right success:", ok_r, "iters:", it_r, "q:", q_sol_r)

"""
Lightweight OO wrapper to align with mapping.py usage.
Provides X7Kinematics(arm).fk(q) and .ik(T_target, q_init) using function implementations above.
"""


class X7Kinematics:
    def __init__(self, arm: str = "left"):
        assert arm in ("left", "right"), "arm must be 'left' or 'right'"
        self.arm = arm

    def fk(self, q):
        """Return 4x4 homogeneous transform from base_link to EE origin.
        q: (7,) X7 joint vector (no gripper DOF).
        """
        if self.arm == "left":
            pos, R, T = fk_left(q)
        else:
            pos, R, T = fk_right(q)
        return T

    def ik(self, T_target, q_init,
           max_iters: int = 100, tol: float = 1e-4, step: float = 0.5, damping: float = 1e-4):
        """Solve X7 IK to reach T_target using numeric IK functions above.
        Inputs:
          - T_target: (4,4) target homogeneous transform
          - q_init:   (7,) initial joint guess
        Returns:
          - q_sol: (7,) joint solution (clamped to limits)
          - success: bool
          - iters: int
        """
        T_target = np.asarray(T_target, dtype=float)
        target_pos = T_target[:3, 3]
        target_R = T_target[:3, :3]
        if self.arm == "left":
            q_sol, ok, iters = ik_left(target_pos, target_R, q_init,
                                       max_iters=max_iters, tol=tol, step=step, damping=damping)
        else:
            q_sol, ok, iters = ik_right(target_pos, target_R, q_init,
                                        max_iters=max_iters, tol=tol, step=step, damping=damping)
        return q_sol, ok, iters
