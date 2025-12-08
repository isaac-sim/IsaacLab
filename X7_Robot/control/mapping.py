import numpy as np

from control.aloha_kinematics import AlohaKinematics
from control.x7_kinematics import X7Kinematics


class ArmMapper:
    """
    单臂映射：
    - ALOHA delta-q  → X7 绝对关节角（通过 ALOHA FK + X7 IK）
    - X7 关节角     → ALOHA 关节角（通过 X7 FK + ALOHA IK）

    约定（很重要）：
    - ALOHA：每臂 7 维，前 6 维为手臂关节，第 7 维为 gripper；FK/IK 仅使用前 6 维。
    - X7：每臂 7 维，全部为手臂关节（无 gripper 维度）；FK/IK 使用全部 7 维。
    """

    def __init__(self, arm: str = "left"):
        assert arm in ("left", "right")
        self.arm = arm
        self.aloha = AlohaKinematics(arm=arm)
        self.x7 = X7Kinematics(arm=arm)
        # 静态坐标变换（默认为单位）：
        # 将 ALOHA 末端位姿转换到 X7 末端坐标系；以及反向变换
        # 形状均为 (4,4)
        self.T_aloha_to_x7 = np.eye(4, dtype=float)
        self.T_x7_to_aloha = np.eye(4, dtype=float)

    def set_static_transform(self, T_aloha_to_x7: 'np.ndarray | None' = None, T_x7_to_aloha: 'np.ndarray | None' = None):
        """
        设置左右臂的静态坐标系对齐变换。
        - `T_aloha_to_x7`: 将 ALOHA 末端位姿映射到 X7 末端位姿的 4x4 变换矩阵。
        - `T_x7_to_aloha`: 将 X7 末端位姿映射到 ALOHA 末端位姿的 4x4 变换矩阵。
        若只提供其中一个，另一个将尝试用矩阵逆求出（若可逆）。
        """
        if T_aloha_to_x7 is not None:
            T_aloha_to_x7 = np.asarray(T_aloha_to_x7, dtype=float)
            assert T_aloha_to_x7.shape == (4, 4)
            self.T_aloha_to_x7 = T_aloha_to_x7
            # 若未给反向，则用逆
            if T_x7_to_aloha is None:
                # 计算逆
                R = T_aloha_to_x7[:3, :3]
                t = T_aloha_to_x7[:3, 3]
                R_inv = R.T
                t_inv = -R_inv @ t
                T_inv = np.eye(4)
                T_inv[:3, :3] = R_inv
                T_inv[:3, 3] = t_inv
                self.T_x7_to_aloha = T_inv
        if T_x7_to_aloha is not None:
            T_x7_to_aloha = np.asarray(T_x7_to_aloha, dtype=float)
            assert T_x7_to_aloha.shape == (4, 4)
            self.T_x7_to_aloha = T_x7_to_aloha
            # 若未给正向，则用逆
            if T_aloha_to_x7 is None:
                R = T_x7_to_aloha[:3, :3]
                t = T_x7_to_aloha[:3, 3]
                R_inv = R.T
                t_inv = -R_inv @ t
                T_inv = np.eye(4)
                T_inv[:3, :3] = R_inv
                T_inv[:3, 3] = t_inv
                self.T_aloha_to_x7 = T_inv

    # ------------------------------------------------------------------
    # 1. ALOHA 增量 q → X7 绝对关节角
    # ------------------------------------------------------------------
    def aloha_delta_to_x7(
        self,
        q_aloha_curr: np.ndarray,
        delta_q_aloha: np.ndarray,
        q_x7_curr: np.ndarray,
    ):
        """
        输入:
            q_aloha_curr : (7,) 当前 ALOHA 关节角
            delta_q_aloha: (7,) ALOHA 增量动作（RL 输出）
            q_x7_curr    : (7,) 当前 X7 关节角

        输出:
            q_x7_next    : (7,) 映射后的 X7 绝对关节角
            q_aloha_next : (7,) 更新后的 ALOHA 关节角（作为内部“虚拟ALOHA”状态）
        """
        q_aloha_curr = np.asarray(q_aloha_curr, dtype=float)
        delta_q_aloha = np.asarray(delta_q_aloha, dtype=float)
        q_x7_curr = np.asarray(q_x7_curr, dtype=float)

        assert q_aloha_curr.shape == (7,)
        assert delta_q_aloha.shape == (7,)
        assert q_x7_curr.shape == (7,)

        # 1) 先在“虚拟 ALOHA”上累积动作，得到新的 ALOHA 关节
        q_aloha_next = q_aloha_curr + delta_q_aloha

        # 只对前 6 维做 FK / IK（gripper 不参与位姿）
        q_aloha_arm = q_aloha_next[:6]

        # 2) 用 ALOHA FK 计算末端位姿 (4x4)
        T_target_aloha = self.aloha.fk(q_aloha_arm)
        # 坐标系对齐：ALOHA → X7
        T_target = self.T_aloha_to_x7 @ T_target_aloha

        # 3) 用 X7 IK 求解对应的 X7 关节（X7 为 7 维手臂）
        q_x7_arm_init = q_x7_curr[:7]
        q_x7_arm_next = self.x7.ik(T_target, q_init=q_x7_arm_init)
        # 兼容返回 (q, success, iters) 或 (q, success) 的情况
        if isinstance(q_x7_arm_next, (tuple, list)):
            q_x7_arm_next = q_x7_arm_next[0]

        # 如果 IK 失败，你自己在 ik 里处理返回 None 或 raise，
        # 这里简单一点：如果返回 None 就用当前值
        if q_x7_arm_next is None:
            q_x7_arm_next = q_x7_arm_init

        q_x7_arm_next = np.asarray(q_x7_arm_next, dtype=float)
        assert q_x7_arm_next.shape == (7,)

        # 4) X7 为 7 维手臂，无 gripper 维度，直接作为该臂的 7 维输出
        q_x7_next = q_x7_arm_next.copy()

        return q_x7_next, q_aloha_next

    # ------------------------------------------------------------------
    # 2. X7 关节状态 → ALOHA 关节状态
    # ------------------------------------------------------------------
    def x7_state_to_aloha(
        self,
        q_x7_curr: np.ndarray,
        q_aloha_guess: np.ndarray,
    ):
        """
        输入:
            q_x7_curr    : (7,) 当前 X7 关节角
            q_aloha_guess: (7,) ALOHA 初始 guess，用于 IK 初值（通常用上一时刻的 ALOHA 状态）

        输出:
            q_aloha_curr : (7,) 对应的 ALOHA 关节角
        """
        q_x7_curr = np.asarray(q_x7_curr, dtype=float)
        q_aloha_guess = np.asarray(q_aloha_guess, dtype=float)

        assert q_x7_curr.shape == (7,)
        assert q_aloha_guess.shape == (7,)

        # 1) X7 FK 得到当前末端位姿（X7 为 7 维手臂）
        q_x7_arm = q_x7_curr[:7]
        T_curr_x7 = self.x7.fk(q_x7_arm)
        # 坐标系对齐：X7 → ALOHA
        T_curr = self.T_x7_to_aloha @ T_curr_x7

        # 2) 用 ALOHA IK 求解对应的 ALOHA 关节
        q_aloha_arm_init = q_aloha_guess[:6]
        q_aloha_arm = self.aloha.ik(T_curr, q_init=q_aloha_arm_init)
        if isinstance(q_aloha_arm, (tuple, list)):
            q_aloha_arm = q_aloha_arm[0]

        if q_aloha_arm is None:
            q_aloha_arm = q_aloha_arm_init

        q_aloha_arm = np.asarray(q_aloha_arm, dtype=float)
        assert q_aloha_arm.shape == (6,)

        q_aloha_curr = np.zeros(7, dtype=float)
        q_aloha_curr[:6] = q_aloha_arm
        # gripper：X7 无 gripper 维度，这里保留猜测值的第 7 维（或由上层策略决定）
        q_aloha_curr[6] = q_aloha_guess[6]

        return q_aloha_curr


# ======================================================================
# 双臂接口：方便你直接丢 14 维向量进来
#   ALOHA: [L0..L6, R0..R6]
#   X7   : [L0..L6, R0..R6]
# ======================================================================

class BiArmMapper:
    """
    双臂映射器：
    - 假设 ALOHA 与 X7 都是 14 维：[左 7, 右 7]
    - 内部各自用一个 ArmMapper(left) 和 ArmMapper(right)
    """

    def __init__(self):
        self.left = ArmMapper(arm="left")
        self.right = ArmMapper(arm="right")

    def aloha_delta_to_x7(
        self,
        q_aloha_curr: np.ndarray,
        delta_q_aloha: np.ndarray,
        q_x7_curr: np.ndarray,
    ):
        """
        输入:
            q_aloha_curr : (14,)
            delta_q_aloha: (14,)
            q_x7_curr    : (14,)

        输出:
            q_x7_next    : (14,)
            q_aloha_next : (14,)
        """
        q_aloha_curr = np.asarray(q_aloha_curr, dtype=float)
        delta_q_aloha = np.asarray(delta_q_aloha, dtype=float)
        q_x7_curr = np.asarray(q_x7_curr, dtype=float)

        assert q_aloha_curr.shape == (14,)
        assert delta_q_aloha.shape == (14,)
        assert q_x7_curr.shape == (14,)

        # 拆成左右
        q_a_L = q_aloha_curr[:7]
        q_a_R = q_aloha_curr[7:]
        dq_a_L = delta_q_aloha[:7]
        dq_a_R = delta_q_aloha[7:]
        q_x7_L = q_x7_curr[:7]
        q_x7_R = q_x7_curr[7:]

        # 左臂
        q_x7_L_next, q_a_L_next = self.left.aloha_delta_to_x7(
            q_aloha_curr=q_a_L,
            delta_q_aloha=dq_a_L,
            q_x7_curr=q_x7_L,
        )

        # 右臂
        q_x7_R_next, q_a_R_next = self.right.aloha_delta_to_x7(
            q_aloha_curr=q_a_R,
            delta_q_aloha=dq_a_R,
            q_x7_curr=q_x7_R,
        )

        q_x7_next = np.concatenate([q_x7_L_next, q_x7_R_next], axis=0)
        q_aloha_next = np.concatenate([q_a_L_next, q_a_R_next], axis=0)

        return q_x7_next, q_aloha_next

    def x7_state_to_aloha(
        self,
        q_x7_curr: np.ndarray,
        q_aloha_guess: np.ndarray,
    ):
        """
        输入:
            q_x7_curr    : (14,)
            q_aloha_guess: (14,)

        输出:
            q_aloha_curr : (14,)
        """
        q_x7_curr = np.asarray(q_x7_curr, dtype=float)
        q_aloha_guess = np.asarray(q_aloha_guess, dtype=float)

        assert q_x7_curr.shape == (14,)
        assert q_aloha_guess.shape == (14,)

        q_x7_L = q_x7_curr[:7]
        q_x7_R = q_x7_curr[7:]
        q_a_L_guess = q_aloha_guess[:7]
        q_a_R_guess = q_aloha_guess[7:]

        q_a_L_curr = self.left.x7_state_to_aloha(q_x7_L, q_a_L_guess)
        q_a_R_curr = self.right.x7_state_to_aloha(q_x7_R, q_a_R_guess)

        q_aloha_curr = np.concatenate([q_a_L_curr, q_a_R_curr], axis=0)
        return q_aloha_curr
