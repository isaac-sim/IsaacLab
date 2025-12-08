import numpy as np
import torch
from control.observation import get_observation, get_observation_with_mapping
from control.mapping import BiArmMapper
from openpi_client import websocket_client_policy


def simple_swing_control(scene, sim_time: float):
    """让双臂简单做同步摆动。"""
    action = scene["robot"].data.default_joint_pos.clone()

    # 左臂 Joint1~7
    action[:, 0:7] = 0.4 * np.sin(2 * np.pi * 0.5 * sim_time)

    # 右臂 Joint8~14
    action[:, 7:14] = 0.4 * np.sin(2 * np.pi * 0.5 * sim_time)

    scene["robot"].set_joint_position_target(action)


def pi_control(scene, sim_time, client: websocket_client_policy.WebsocketClientPolicy):
    # 1）取 observation
    obs = get_observation(scene)

    # 2）从服务器取 1 个 chunk
    action_chunk = client.infer(obs)["actions"]    # (10, 14)
    return action_chunk


def pi_control_mapped(
    scene,
    sim_time,
    client: websocket_client_policy.WebsocketClientPolicy,
    aloha_state: np.ndarray | None = None,
):
    """
    从 ALOHA 的 VLA 模型拿到动作（delta ALOHA 关节），
    映射成 X7 可用的 14 维关节目标序列（绝对关节角）。

    返回:
      x7_action_chunk: (T, 14) np.ndarray，按时间顺序的 X7 绝对关节角
      aloha_state_out: (14,) np.ndarray，映射后的“虚拟 ALOHA”末状态（供下次作为初值）
    """

    # 1) 获取给 ALOHA 模型的观测（其中 state 已是 ALOHA 空间）
    obs = get_observation_with_mapping(scene, aloha_guess=aloha_state)

    # 2) 推理得到 ALOHA 动作序列 (T, 14)
    out = client.infer(obs)
    actions = out.get("actions", None)
    if actions is None:
        raise RuntimeError("Policy result missing 'actions'.")
    actions = np.asarray(actions)
    assert actions.ndim == 2 and actions.shape[1] == 14, f"expect (T,14), got {actions.shape}"

    # 3) 当前 X7 关节（14维），去掉 batch 维
    q_x7 = scene["robot"].data.default_joint_pos
    if isinstance(q_x7, torch.Tensor):
        q_x7 = q_x7.detach().cpu().numpy()
    if isinstance(q_x7, np.ndarray) and q_x7.ndim == 2 and q_x7.shape[0] == 1:
        q_x7 = q_x7[0]
    if q_x7.shape[0] >= 14:
        q_x7 = q_x7[:14]
    else:
        q_x7 = np.concatenate([q_x7, np.zeros(14 - q_x7.shape[0])])
    q_x7 = q_x7.astype(np.float64, copy=False)

    # 4) 初始 ALOHA 状态：优先 obs 中的 state（与 X7 对齐过），否则用零
    q_aloha = obs.get("state", None)
    if q_aloha is None:
        q_aloha = np.zeros(14, dtype=np.float64)
    else:
        q_aloha = np.asarray(q_aloha, dtype=np.float64)

    # 5) 配置双臂映射器（可设置静态坐标对齐）
    bi = BiArmMapper()  # 静态变换在 mapping 层内部定义/配置

    # 6) 依次把每一帧 ALOHA delta 动作映射到 X7 绝对关节
    T = actions.shape[0]
    x7_actions = np.zeros((T, 14), dtype=np.float32)
    q_x7_curr = q_x7.copy()
    q_aloha_curr = q_aloha.copy()

    for t in range(T):
        delta_q_aloha = actions[t]
        q_x7_next, q_aloha_next = bi.aloha_delta_to_x7(
            q_aloha_curr=q_aloha_curr,
            delta_q_aloha=delta_q_aloha,
            q_x7_curr=q_x7_curr,
        )
        x7_actions[t] = q_x7_next.astype(np.float32)
        q_x7_curr = q_x7_next
        q_aloha_curr = q_aloha_next

    return x7_actions, q_aloha_curr.astype(np.float32)


def pi_control_mapped_stateless(
    scene,
    sim_time,
    client: websocket_client_policy.WebsocketClientPolicy,
):
    """
    无需保存 ALOHA 内部状态的版本：
    - 直接用当前 X7 关节，通过 IK 映射得到 ALOHA 状态作为初值，
    - 调用 ALOHA VLA 模型拿到 delta ALOHA 动作，
    - 再映射成 X7 的绝对关节动作序列。

    返回:
      x7_action_chunk: (T, 14) np.ndarray
    """
    # 当前 X7 关节（14维）
    q_x7 = scene["robot"].data.default_joint_pos
    if isinstance(q_x7, torch.Tensor):
        q_x7 = q_x7.detach().cpu().numpy()
    if isinstance(q_x7, np.ndarray) and q_x7.ndim == 2 and q_x7.shape[0] == 1:
        q_x7 = q_x7[0]
    if q_x7.shape[0] >= 14:
        q_x7 = q_x7[:14]
    else:
        q_x7 = np.concatenate([q_x7, np.zeros(14 - q_x7.shape[0])])
    q_x7 = q_x7.astype(np.float64, copy=False)

    # 准备映射器与静态变换，用于求 ALOHA 初值与后续映射
    bi = BiArmMapper()  # 静态变换在 mapping 层内部定义/配置

    # 由 X7 当下状态反推一个 ALOHA 初值（不保留历史）
    aloha_guess = np.zeros(14, dtype=np.float64)
    aloha_init = bi.x7_state_to_aloha(q_x7_curr=q_x7, q_aloha_guess=aloha_guess)

    # 生成喂给 ALOHA 模型的观测（使用上面推的初值）
    obs = get_observation_with_mapping(scene, aloha_guess=aloha_init)

    # 推理得到 ALOHA 动作序列 (T, 14)
    out = client.infer(obs)
    actions = out.get("actions", None)
    if actions is None:
        raise RuntimeError("Policy result missing 'actions'.")
    actions = np.asarray(actions)
    assert actions.ndim == 2 and actions.shape[1] == 14, f"expect (T,14), got {actions.shape}"

    # 逐步映射为 X7 的绝对关节动作
    T = actions.shape[0]
    x7_actions = np.zeros((T, 14), dtype=np.float32)
    q_x7_curr = q_x7.copy()
    q_aloha_curr = np.asarray(obs["state"], dtype=np.float64)

    for t in range(T):
        delta_q_aloha = actions[t]
        q_x7_next, q_aloha_next = bi.aloha_delta_to_x7(
            q_aloha_curr=q_aloha_curr,
            delta_q_aloha=delta_q_aloha,
            q_x7_curr=q_x7_curr,
        )
        x7_actions[t] = q_x7_next.astype(np.float32)
        q_x7_curr = q_x7_next
        q_aloha_curr = q_aloha_next

    return x7_actions

