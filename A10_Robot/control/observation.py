from openpi_client import image_tools
import numpy as np
import torch
from control.mapping import BiArmMapper

from openpi_client import image_tools
import numpy as np
import torch

import numpy as np
import torch
import einops
from openpi_client import image_tools


def get_observation(scene):
    # ---- 准备相机 ----
    cam_high = scene["head_camera"].data.output["rgb"]
    cam_left_wrist  = scene["left_camera"].data.output["rgb"]
    cam_right_wrist = scene["right_camera"].data.output["rgb"]

    # ---- Tensor → numpy ----
    def to_numpy(img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        return img[..., :3]   # 只保留 RGB 3 通道

    cam_high = to_numpy(cam_high)
    cam_left_wrist = to_numpy(cam_left_wrist)
    cam_right_wrist = to_numpy(cam_right_wrist)

    # ---- 缺失的 cam_left_wrist → 用 0 图填充 ----
    cam_low = np.zeros_like(cam_high)

    # ---- resize + uint8 + CHW ----
    def process(img):
        # remove batch dim if present: (1, H, W, C) -> (H, W, C)
        if img.ndim == 4 and img.shape[0] == 1:
            img = img[0]

        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        return einops.rearrange(img, "h w c -> c h w")


    obs_images = {
        "cam_high": process(cam_high),
        "cam_low": process(cam_low),
        "cam_left_wrist": process(cam_left_wrist),
        "cam_right_wrist": process(cam_right_wrist),
    }


    # 使用当前关节状态（而非默认初值）
    qpos = scene["robot"].data.joint_pos
    if isinstance(qpos, torch.Tensor):
        qpos = qpos.detach().cpu().numpy()

    # 去掉可能的 batch 维: (1,14) -> (14,)
    if isinstance(qpos, np.ndarray) and qpos.ndim == 2 and qpos.shape[0] == 1:
        qpos = qpos[0]

    # 将 12 维（左臂6 + 右臂6）转换为 14 维（左臂6 + 左爪 + 右臂6 + 右爪），夹爪填充为 0
    if qpos.shape[0] == 12:
        left = qpos[0:6]
        right = qpos[6:12]
        qpos = np.concatenate([left, np.array([0.0], dtype=left.dtype), right, np.array([0.0], dtype=right.dtype)])
    elif qpos.shape[0] > 14:
        qpos = qpos[:14]
    elif qpos.shape[0] < 14:
        pad = np.zeros(14 - qpos.shape[0], dtype=qpos.dtype)
        qpos = np.concatenate([qpos, pad])

    qpos = qpos.astype(np.float32)

    # ---- 返回 ALOHA 标准格式 ----
    return {
        "state": qpos,
        "images": obs_images,
        "prompt": "Grasp the red cube on the table.",
    }


