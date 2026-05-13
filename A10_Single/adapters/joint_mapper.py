# adapters/joint_mapper.py

import numpy as np

class JointMapper12to14:
    """
    自动关节映射器：
    输入：pi0.5（Franka 双臂 12 DOF）
        Franka_L[0..5], Franka_R[0..5]
    输出：你的机器人 14 DOF
        X7 左臂 Joint1~Joint7
        X7 右臂 Joint8~Joint14
    """

    def __init__(self):
        # 默认映射 6→7，最后一个补 0 或固定角度
        self.left_mapping =  [0, 1, 2, 3, 4, 5]   # → Joint1..Joint6
        self.right_mapping = [6, 7, 8, 9,10,11]   # → Joint8..Joint13

        # 第 7 DOF 用什么？
        self.left_extra  = 0.0   # 给 Joint7
        self.right_extra = 0.0   # 给 Joint14

    def map(self, frankas_12dof):
        """
        输入 12D Franka（numpy 或 torch 都行）
        返回 14D X7 关节角
        """

        frankas_12dof = np.asarray(frankas_12dof).reshape(-1)

        # 左臂 X7 (Joint1~7)
        left = [frankas_12dof[i] for i in self.left_mapping]
        left.append(self.left_extra)

        # 右臂 X7 (Joint8~14)
        right = [frankas_12dof[i] for i in self.right_mapping]
        right.append(self.right_extra)

        return np.array(left + right)
