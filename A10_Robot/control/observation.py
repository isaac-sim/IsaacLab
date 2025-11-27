from openpi_client import image_tools
import numpy as np
import torch
import einops


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

    # ---- 关节 ----
    qpos = scene["robot"].data.default_joint_pos
    if isinstance(qpos, torch.Tensor):
        qpos = qpos.detach().cpu().numpy()

    # 去掉可能的 batch 维: (1,N) -> (N,)
    if isinstance(qpos, np.ndarray) and qpos.ndim == 2 and qpos.shape[0] == 1:
        qpos = qpos[0]

    # 默认原样返回（不做 14 维截断/补零）
    qpos = qpos.astype(np.float32, copy=False)

    # ---- 返回 ALOHA 标准格式 ----
    return {
        "state": qpos,
        "images": obs_images,
        "prompt": "Grasp the red cube on the table.",
    }


def get_observation_raw(scene):
    """
    返回不做维度映射/填充的原始状态（适合 A10 双六轴：12 维）。
    结构与 get_observation 一致，只是 state 为机器人实际关节数。
    """
    obs = get_observation(scene)
    return obs


def get_observation_12(scene):
    """
    显式返回 12 维状态（双六轴：左6+右6）。
    若机器人关节数不足 12，则补零；超过则截取前 12。
    """
    obs = get_observation(scene)
    q = obs["state"]
    if q.shape[0] >= 12:
        q = q[:12]
    else:
        q = np.concatenate([q, np.zeros(12 - q.shape[0], dtype=q.dtype)])
    obs["state"] = q
    return obs


def get_observation_mapped(scene, aloha_guess: np.ndarray | None = None, static_transforms: dict | None = None):
    """
    获取观测并将 X7 的关节状态映射为 ALOHA 关节状态（14维），返回 ALOHA 标准格式：
    {"state": (14,), "images": {CHW uint8}, "prompt": str}

    参数:
    - scene: 仿真场景/设备句柄（需包含 robot 与三路相机：head/left/right）
    - aloha_guess: (14,) 可选，ALOHA IK 的初始猜测（建议使用上一次映射得到的 ALOHA 状态）。
    - static_transforms: 可选，静态坐标系对齐矩阵字典，例如：
        {
          "left": {"T_aloha_to_x7": 4x4, "T_x7_to_aloha": 4x4},
          "right": {"T_aloha_to_x7": 4x4, "T_x7_to_aloha": 4x4},
        }
      若只提供其中一个方向，另一方向将使用逆矩阵。
    """

    # ---- 相机处理（与 get_observation 一致） ----
    cam_high = scene["head_camera"].data.output["rgb"]
    cam_left_wrist = scene["left_camera"].data.output["rgb"]
    cam_right_wrist = scene["right_camera"].data.output["rgb"]

    def to_numpy(img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        return img[..., :3]

    cam_high = to_numpy(cam_high)
    cam_left_wrist = to_numpy(cam_left_wrist)
    cam_right_wrist = to_numpy(cam_right_wrist)

    cam_low = np.zeros_like(cam_high)

    import einops as _einops  # 本函数内局部导入，避免头部重复

    def process(img):
        if img.ndim == 4 and img.shape[0] == 1:
            img = img[0]
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        return _einops.rearrange(img, "h w c -> c h w")

    obs_images = {
        "cam_high": process(cam_high),
        "cam_low": process(cam_low),
        "cam_left_wrist": process(cam_left_wrist),
        "cam_right_wrist": process(cam_right_wrist),
    }

    # ---- 读取 X7 当前关节（期望 14 维：左7 + 右7） ----
    q_x7 = scene["robot"].data.default_joint_pos
    if isinstance(q_x7, torch.Tensor):
        q_x7 = q_x7.detach().cpu().numpy()
    if isinstance(q_x7, np.ndarray) and q_x7.ndim == 2 and q_x7.shape[0] == 1:
        q_x7 = q_x7[0]

    # 规范到 14 维
    if q_x7.shape[0] >= 14:
        q_x7 = q_x7[:14]
    else:
        q_x7 = np.concatenate([q_x7, np.zeros(14 - q_x7.shape[0])])
    q_x7 = q_x7.astype(np.float64, copy=False)

    # ---- ALOHA IK 初始猜测 ----
    if aloha_guess is None:
        aloha_guess = np.zeros(14, dtype=np.float64)
    else:
        aloha_guess = np.asarray(aloha_guess, dtype=np.float64)
        if aloha_guess.shape != (14,):
            aloha_guess = np.zeros(14, dtype=np.float64)

    # ---- 映射：X7 → ALOHA ----
    from A10_Robot.control.mapping import BiArmMapper
    bi = BiArmMapper()
    # 静态坐标变换（若提供）
    if static_transforms is not None:
        left_cfg = static_transforms.get("left") if isinstance(static_transforms, dict) else None
        right_cfg = static_transforms.get("right") if isinstance(static_transforms, dict) else None
        if left_cfg:
            bi.left.set_static_transform(
                T_aloha_to_x7=left_cfg.get("T_aloha_to_x7"),
                T_x7_to_aloha=left_cfg.get("T_x7_to_aloha"),
            )
        if right_cfg:
            bi.right.set_static_transform(
                T_aloha_to_x7=right_cfg.get("T_aloha_to_x7"),
                T_x7_to_aloha=right_cfg.get("T_x7_to_aloha"),
            )

    q_aloha = bi.x7_state_to_aloha(q_x7_curr=q_x7, q_aloha_guess=aloha_guess)
    q_aloha = np.asarray(q_aloha, dtype=np.float32)

    # ---- 返回 ALOHA 标准格式 ----
    return {
        "state": q_aloha,
        "images": obs_images,
        "prompt": "Grasp the red cube on the table.",
    }


# 简洁别名，便于上层调用
def get_observation_with_mapping(scene, aloha_guess: np.ndarray | None = None, static_transforms: dict | None = None):
    return get_observation_mapped(scene, aloha_guess=aloha_guess, static_transforms=static_transforms)



