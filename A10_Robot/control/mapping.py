import numpy as np

class ArmMapper:
    def __init__(self, arm: str = "left"):
        assert arm in ("left", "right")
        self.arm = arm

    def aloha_delta_to_x7(self, q_aloha_curr: np.ndarray, delta_q_aloha: np.ndarray, q_x7_curr: np.ndarray):
        q_aloha_curr = np.asarray(q_aloha_curr, dtype=float)
        delta_q_aloha = np.asarray(delta_q_aloha, dtype=float)
        q_x7_curr = np.asarray(q_x7_curr, dtype=float)
        assert q_aloha_curr.shape == (7,) and delta_q_aloha.shape == (7,) and q_x7_curr.shape == (7,)
        # Identity placeholder: treat X7 as same 7-DOF layout per arm
        q_aloha_next = q_aloha_curr + delta_q_aloha
        q_x7_next = q_x7_curr.copy()
        q_x7_next[:] = q_aloha_next[:]
        return q_x7_next, q_aloha_next

    def x7_state_to_aloha(self, q_x7_curr: np.ndarray, q_aloha_guess: np.ndarray):
        q_x7_curr = np.asarray(q_x7_curr, dtype=float)
        q_aloha_guess = np.asarray(q_aloha_guess, dtype=float)
        assert q_x7_curr.shape == (7,) and q_aloha_guess.shape == (7,)
        # Identity placeholder
        q_aloha_curr = q_x7_curr.copy()
        return q_aloha_curr


class BiArmMapper:
    def __init__(self):
        self.left = ArmMapper("left")
        self.right = ArmMapper("right")

    def aloha_delta_to_x7(self, q_aloha_curr: np.ndarray, delta_q_aloha: np.ndarray, q_x7_curr: np.ndarray):
        q_aloha_curr = np.asarray(q_aloha_curr, dtype=float)
        delta_q_aloha = np.asarray(delta_q_aloha, dtype=float)
        q_x7_curr = np.asarray(q_x7_curr, dtype=float)
        assert q_aloha_curr.shape == (14,) and delta_q_aloha.shape == (14,) and q_x7_curr.shape == (14,)
        L_x7, L_aloha = self.left.aloha_delta_to_x7(q_aloha_curr[:7], delta_q_aloha[:7], q_x7_curr[:7])
        R_x7, R_aloha = self.right.aloha_delta_to_x7(q_aloha_curr[7:], delta_q_aloha[7:], q_x7_curr[7:])
        return np.concatenate([L_x7, R_x7]), np.concatenate([L_aloha, R_aloha])

    def x7_state_to_aloha(self, q_x7_curr: np.ndarray, q_aloha_guess: np.ndarray):
        q_x7_curr = np.asarray(q_x7_curr, dtype=float)
        q_aloha_guess = np.asarray(q_aloha_guess, dtype=float)
        assert q_x7_curr.shape == (14,) and q_aloha_guess.shape == (14,)
        L = self.left.x7_state_to_aloha(q_x7_curr[:7], q_aloha_guess[:7])
        R = self.right.x7_state_to_aloha(q_x7_curr[7:], q_aloha_guess[7:])
        return np.concatenate([L, R])
