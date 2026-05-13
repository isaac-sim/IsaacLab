import logging

import einops
import numpy as np
import torch

from openpi_client import image_tools

logger = logging.getLogger(__name__)

# 7-th state dim: training often used first gripper DOF only (same command mirrored to both fingers).
# If your checkpoint used (gL+gR)/2 instead, switch to averaging below.
_GRIPPER_STATE_USE_MEAN = False


def _to_numpy_rgb(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    # remove batch dim if present: (1, H, W, C) -> (H, W, C)
    if isinstance(img, np.ndarray) and img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    return img[..., :3]


def _float_rgb_to_uint8_hwc(img: np.ndarray) -> np.ndarray:
    """Isaac / Replicator may return float RGB in [0,1] or [0,255]; normalize before PIL resize."""
    if img.size == 0:
        return img
    if np.issubdtype(img.dtype, np.floating):
        mx = float(np.max(img))
        if mx <= 1.01:
            img = (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            img = np.clip(img, 0.0, 255.0).round().astype(np.uint8)
    return img


def _process_image(img):
    if not isinstance(img, np.ndarray) or img.size == 0:
        logger.warning("Wrist camera returned empty frame. Using black fallback image.")
        img = np.zeros((224, 224, 3), dtype=np.uint8)
    else:
        img = _float_rgb_to_uint8_hwc(img)
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
    return einops.rearrange(img, "h w c -> c h w")


def _get_camera_rgb_by_name(scene, cam_name: str):
    if cam_name not in scene.keys():
        return None
    try:
        output = scene[cam_name].data.output
    except Exception as e:
        logger.warning("Failed to fetch camera '%s' output (%s).", cam_name, e)
        return None
    if "rgb" not in output:
        logger.warning("Camera '%s' has no 'rgb' output. Available keys: %s", cam_name, list(output.keys()))
        return None
    return output["rgb"]


def _get_single_wrist_rgb(scene):
    # Prefer the explicit single-camera name, then fall back to common legacy names.
    for cam_name in ("wrist_camera", "right_camera", "left_camera"):
        rgb = _get_camera_rgb_by_name(scene, cam_name)
        if rgb is not None:
            return rgb
    logger.warning("No valid wrist camera available. Falling back to black image.")
    return np.zeros((224, 224, 3), dtype=np.uint8)


def _get_base_camera_rgb(scene):
    rgb = _get_camera_rgb_by_name(scene, "base_camera")
    if rgb is not None:
        return rgb
    logger.warning("No valid base_camera in scene. Falling back to black image for policy base view.")
    return np.zeros((224, 224, 3), dtype=np.uint8)


def _policy_state_7d(robot) -> np.ndarray:
    """Build (7,) state: joint1..joint6 (rad) + one gripper scalar, using articulation joint names.

    Using ``joint_pos[:7]`` is unsafe: DOF order follows the asset / PhysX, not necessarily joint1..6 then grippers.
    """
    q = robot.data.joint_pos
    if isinstance(q, torch.Tensor):
        q = q.detach().cpu().numpy()
    q = np.asarray(q)
    if q.ndim >= 2:
        q = q[0]

    try:
        arm_ids, arm_names = robot.find_joints([f"joint{i}" for i in range(1, 7)])
        if len(arm_ids) != 6:
            raise ValueError(f"expected 6 arm joints, got {len(arm_ids)}: {arm_names}")
        arm = q[arm_ids].astype(np.float32)

        gl, nl = robot.find_joints("gripper_left_joint")
        gr, nr = robot.find_joints("gripper_right_joint")
        if len(gl) != 1 or len(gr) != 1:
            raise ValueError(f"gripper joints: left={nl!r} right={nr!r}")

        if _GRIPPER_STATE_USE_MEAN:
            g = np.float32(0.5 * (q[gl[0]] + q[gr[0]]))
        else:
            g = np.float32(q[gl[0]])

        out = np.concatenate([arm, np.array([g], dtype=np.float32)])
        logger.debug(
            "observation state: arm joints %s, gripper=%s (mean=%s)",
            arm_names,
            float(g),
            float(0.5 * (q[gl[0]] + q[gr[0]])),
        )
        return out
    except (ValueError, IndexError) as e:
        logger.warning(
            "Failed to resolve policy state by joint names (%s); falling back to q[:7]. "
            "Training vs sim joint order may mismatch — fix URDF/joint names.",
            e,
        )
        if q.shape[0] >= 7:
            return q[:7].astype(np.float32)
        pad = np.zeros(7 - q.shape[0], dtype=np.float32)
        return np.concatenate([q.astype(np.float32), pad])


def get_observation(scene, prompt: str = "Reach the target fruit on the table."):
    """Pack one websocket step for OpenPI ``A10Inputs`` / ``WebsocketClientPolicy.infer``.

    Contract (server-side ``A10Inputs``):

    - ``observation/state``: ``float32``, shape ``[T, 7]`` with ``T=1`` (or ``[7]`` is also accepted there).
    - ``observation/images/right``: wrist / egocentric view, ``uint8``, **CHW** ``(3, 224, 224)``.
    - ``observation/images/top``: from scene ``base_camera`` (fixed / third-person view), same layout as ``right``.
    - ``prompt``: ``str`` (optional on server; we always send for VLA).

    Both images use the same resize-with-pad pipeline as Libero/OpenPI ``_parse_image``.
    """
    cam_wrist = _to_numpy_rgb(_get_single_wrist_rgb(scene))
    cam_wrist = _process_image(cam_wrist)
    cam_wrist = np.ascontiguousarray(cam_wrist, dtype=np.uint8)

    cam_top = _to_numpy_rgb(_get_base_camera_rgb(scene))
    cam_top = _process_image(cam_top)
    cam_top = np.ascontiguousarray(cam_top, dtype=np.uint8)

    qpos = _policy_state_7d(scene["robot"])
    state_traj = np.ascontiguousarray(qpos.reshape(1, 7), dtype=np.float32)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "observation: wrist CHW %s mean=%.3f | top CHW %s mean=%.3f | state_traj=%s | prompt=%r",
            cam_wrist.shape,
            float(np.mean(cam_wrist)),
            cam_top.shape,
            float(np.mean(cam_top)),
            np.array2string(state_traj, precision=4, suppress_small=True),
            prompt,
        )

    return {
        "observation/state": state_traj,
        "observation/images/right": cam_wrist,
        "observation/images/top": cam_top,
        "prompt": prompt,
    }
