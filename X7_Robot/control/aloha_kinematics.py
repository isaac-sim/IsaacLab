import numpy as np

def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0, 0],
        [0, c,-s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1],
    ], dtype=float)

def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1],
    ], dtype=float)

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c,-s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)

def trans(x, y, z):
    T = np.eye(4, dtype=float)
    T[:3, 3] = [x, y, z]
    return T

def aloha_left_fk(q):
    """
    q: iterable with 6 joint angles [q1..q6] in radians
    return: 4x4 homogeneous transform base -> gripper_link
    """
    q1, q2, q3, q4, q5, q6 = q

    T = np.eye(4, dtype=float)

    # base -> shoulder_link
    T = T @ trans(0.0, 0.0, 0.079)
    T = T @ rot_z(q1)
    # shoulder_link -> upper_arm_link
    T = T @ trans(0.0, 0.0, 0.04805)

    # upper_arm_link -> upper_forearm_link
    T = T @ rot_y(q2)
    T = T @ trans(0.05955, 0.0, 0.3)

    # upper_forearm_link -> lower_forearm_link
    T = T @ rot_y(q3)
    T = T @ trans(0.2, 0.0, 0.0)

    # lower_forearm_link -> wrist_link
    T = T @ rot_x(q4)
    T = T @ trans(0.1, 0.0, 0.0)

    # wrist_link -> gripper_link
    T = T @ rot_y(q5)
    T = T @ trans(0.069744, 0.0, 0.0)

    # gripper_link -> wrist_rotate frame
    T = T @ rot_x(q6)

    return T

def fk_pos_only(q):
    """只要末端位置 (x,y,z)"""
    T = aloha_left_fk(q)
    return T[:3, 3]

def pose_to_vec(T):
    """
    把 4x4 齐次变换转成 6 维向量 [p(3), w(3)]
    w 是姿态误差的小角度近似用的 so(3) 向量
    """
    R = T[:3, :3]
    p = T[:3, 3]

    # 取反对称部分，近似为旋转向量
    w = 0.5 * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], dtype=float)
    return np.concatenate([p, w], axis=0)

def aloha_left_ik(
    T_target,
    q_init=None,
    max_iters=100,
    tol_pos=1e-3,
    tol_rot=1e-3,
    damping=1e-3,
):
    """
    数值 IK：给定目标位姿 T_target (4x4)，求 aloha 左臂关节角 q (6,)

    返回:
        q_sol, success
    """
    if q_init is None:
        q = np.zeros(6, dtype=float)
    else:
        q = np.array(q_init, dtype=float).copy()

    pose_target = pose_to_vec(T_target)
    for it in range(max_iters):
        T_cur = aloha_left_fk(q)
        pose_cur = pose_to_vec(T_cur)

        err = pose_target - pose_cur
        pos_err_norm = np.linalg.norm(err[:3])
        rot_err_norm = np.linalg.norm(err[3:])

        if pos_err_norm < tol_pos and rot_err_norm < tol_rot:
            return q, True

        # 数值 Jacobian: 6x6
        J = np.zeros((6, 6), dtype=float)
        eps = 1e-4
        for i in range(6):
            dq = np.zeros(6, dtype=float)
            dq[i] = eps
            pose_plus  = pose_to_vec(aloha_left_fk(q + dq))
            pose_minus = pose_to_vec(aloha_left_fk(q - dq))
            J[:, i] = (pose_plus - pose_minus) / (2 * eps)

        # Damped least squares: dq = J^T (J J^T + λ^2 I)^(-1) err
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(
            JJt + (damping ** 2) * np.eye(6, dtype=float),
            err
        )

        q += dq

        # 可选：简单限幅（基于 XML 的 range，大致是 [-pi, pi]）
        q = np.clip(q, -np.pi, np.pi)

    return q, False

def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]])

def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]])

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])

def trans(x, y, z):
    T = np.eye(4); T[:3,3] = [x,y,z]; return T

def aloha_right_fk(q):
    q1, q2, q3, q4, q5, q6 = q

    # 右臂 base：与左臂不同
    T = trans(0.469, 0.5, 0) @ rot_z(np.pi)

    # base -> shoulder_link
    T = T @ trans(0, 0, 0.079)
    T = T @ rot_z(q1)

    # shoulder_link -> upper_arm_link
    T = T @ trans(0, 0, 0.04805)
    T = T @ rot_y(q2)

    # upper_arm_link -> upper_forearm_link
    T = T @ trans(0.05955, 0, 0.3)
    T = T @ rot_y(q3)

    # upper_forearm_link -> lower_forearm_link
    T = T @ trans(0.2, 0, 0)
    T = T @ rot_x(q4)

    # lower_forearm_link -> wrist_link
    T = T @ trans(0.1, 0, 0)
    T = T @ rot_y(q5)

    # wrist_link -> gripper_link
    T = T @ trans(0.069744, 0, 0)
    T = T @ rot_x(q6)

    return T

def fk_right_pos(q):
    return aloha_right_fk(q)[:3,3]

def aloha_right_ik(T_target, q_init=None):
    """占位：建议改用下方类封装的 ik 方法以使用右臂 FK。"""
    q, ok = AlohaKinematics(arm="right").ik(T_target, q_init)
    return q, ok


class AlohaKinematics:
    """
    轻量封装：提供 ALOHA 左/右臂的 fk(q6) 与 ik(T, q_init6)。
    - fk(q): 返回 4x4 位姿（基座到末端）
    - ik(T, q_init): 数值 IK（6 轴），返回 (q_sol(6,), success)
    """
    def __init__(self, arm: str = "left"):
        assert arm in ("left", "right"), "arm must be 'left' or 'right'"
        self.arm = arm

    def fk(self, q):
        q = np.asarray(q, dtype=float).reshape(6)
        if self.arm == "left":
            return aloha_left_fk(q)
        else:
            return aloha_right_fk(q)

    def ik(self, T_target, q_init=None,
           max_iters: int = 100, tol_pos: float = 1e-3, tol_rot: float = 1e-3, damping: float = 1e-3):
        """数值 IK（6 轴），目标为 4x4 位姿。
        返回: (q_sol(6,), success)
        """
        # 初始化
        if q_init is None:
            q = np.zeros(6, dtype=float)
        else:
            q = np.array(q_init, dtype=float).reshape(6).copy()

        pose_target = pose_to_vec(T_target)
        for _ in range(max_iters):
            # 当前位姿与误差
            T_cur = self.fk(q)
            pose_cur = pose_to_vec(T_cur)
            err = pose_target - pose_cur
            pos_err_norm = np.linalg.norm(err[:3])
            rot_err_norm = np.linalg.norm(err[3:])
            if pos_err_norm < tol_pos and rot_err_norm < tol_rot:
                return q, True

            # 数值雅可比 6x6（中央差分）
            J = np.zeros((6, 6), dtype=float)
            eps = 1e-4
            for i in range(6):
                dq = np.zeros(6, dtype=float)
                dq[i] = eps
                pose_plus = pose_to_vec(self.fk(q + dq))
                pose_minus = pose_to_vec(self.fk(q - dq))
                J[:, i] = (pose_plus - pose_minus) / (2 * eps)

            # 阻尼最小二乘
            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + (damping ** 2) * np.eye(6), err)
            q += dq
            q = np.clip(q, -np.pi, np.pi)
        return q, False

