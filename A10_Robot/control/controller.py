import numpy as np
import torch
from .observation import get_observation
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

