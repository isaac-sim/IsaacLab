import numpy as np
from mapping import ArmMapper, BiArmMapper


def random_T():
    # Generate a small random transform near identity for testing
    R = np.eye(3)
    t = np.zeros(3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def test_single_arm(arm="left"):
    mapper = ArmMapper(arm=arm)
    # Example static transform: rotate 180deg around Z between frames
    T = np.eye(4)
    T[:3, :3] = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    mapper.set_static_transform(T_aloha_to_x7=T)

    q_aloha_curr = np.zeros(7)
    delta_q_aloha = 0.05 * np.ones(7)
    q_x7_curr = np.zeros(7)

    q_x7_next, q_aloha_next = mapper.aloha_delta_to_x7(q_aloha_curr, delta_q_aloha, q_x7_curr)
    assert q_x7_next.shape == (7,)
    assert q_aloha_next.shape == (7,)

    # Roundtrip X7 -> ALOHA
    q_aloha_guess = q_aloha_curr
    q_aloha_curr2 = mapper.x7_state_to_aloha(q_x7_next, q_aloha_guess)
    assert q_aloha_curr2.shape == (7,)


def test_bi_arm():
    bi = BiArmMapper()
    # Set different transforms per arm if needed
    T = np.eye(4)
    bi.left.set_static_transform(T_aloha_to_x7=T)
    bi.right.set_static_transform(T_aloha_to_x7=T)

    q_aloha_curr = np.zeros(14)
    delta_q_aloha = 0.03 * np.ones(14)
    q_x7_curr = np.zeros(14)

    q_x7_next, q_aloha_next = bi.aloha_delta_to_x7(q_aloha_curr, delta_q_aloha, q_x7_curr)
    assert q_x7_next.shape == (14,)
    assert q_aloha_next.shape == (14,)

    q_aloha_guess = q_aloha_curr
    q_aloha_curr2 = bi.x7_state_to_aloha(q_x7_next, q_aloha_guess)
    assert q_aloha_curr2.shape == (14,)


if __name__ == "__main__":
    test_single_arm("left")
    test_single_arm("right")
    test_bi_arm()
    print("Mapping sanity tests passed.")
