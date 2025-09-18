# IsaacLab/source/isaaclab/isaaclab/devices/phone/se3_phone.py

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

import numpy as np
import torch

from ..device_base import DeviceBase, DeviceCfg

try:
    from teleop import Teleop
except Exception as exc:  # pragma: no cover
    Teleop = None
    _IMPORT_ERR = exc
else:
    _IMPORT_ERR = None
from dataclasses import dataclass

@dataclass
class Se3PhoneCfg(DeviceCfg):
    """Configuration for SE3 space mouse devices."""

    # pos_scale: float = 1.0          # scale factor (m / raw meter)
    # rot_scale: float = 1.0          # scale factor (rad / raw rad)
    # max_step_pos: float = 0.10      # clamp per tick (m)
    # max_step_rot: float = 0.50      # clamp per tick (rad)
    # use_euler: bool = True          # convert orientation via RPY (simple)
    # gate_key: str = "move"          # message boolean that gates motion
    # gripper_key: str = "scale"      # message float in [-1 1] for gripper
    # open_threshold: float = 0.0     # >= -> open (+1) < -> close (-1)
    # start_server: bool = True       # start Teleop server automatically
    # server_kwargs: Optional[dict] = None  # forwarded to Teleop(...)
    gripper_term: bool = True
    pos_sensitivity: float = 1.0
    rot_sensitivity: float = 1.0
    retargeters: None = None

class Se3Phone(DeviceBase):
    """Phone-based SE(3) teleop device.

    Returns a 7D tensor on `advance()`:
        [dx, dy, dz, droll, dpitch, dyaw, gripper]
    where the first 6 are *relative* deltas since the last frame (meters, radians)
    and `gripper` is in {-1.0, +1.0} (close/open).

    Notes
    -----
    - The device listens to a background `teleop.Teleop` server, which streams a 4x4
      end-effector target pose (in some chosen reference frame) and a message dict.
    - When the message indicates the move gate is OFF, the deltas are zeroed but
      the gripper command is still emitted from the message.
    """

    def __init__(self, cfg: Se3PhoneCfg):

        if Teleop is None:
            raise ImportError(
                "teleop is not available. Install it first (e.g., `pip install teleop`)."
            ) from _IMPORT_ERR

        # self._pos_scale = float(cfg.pos_scale)
        # self._rot_scale = float(cfg.rot_scale)
        # self._max_step_pos = float(cfg.max_step_pos)
        # self._max_step_rot = float(cfg.max_step_rot)
        # self._use_euler = bool(cfg.use_euler)
        # self._gate_key = cfg.gate_key
        # self._gripper_key = cfg.gripper_key
        # self._open_threshold = float(cfg.open_threshold)
        # store inputs
        self._pos_sensitivity = cfg.pos_sensitivity
        self._rot_sensitivity = cfg.rot_sensitivity
        self._gripper_term = cfg.gripper_term
        self._sim_device = cfg.sim_device
        # latest data (written by callback thread)
        self._latest_pose: Optional[np.ndarray] = None  # 4x4
        self._latest_msg: dict[str, Any] = {}
        self._enabled: bool = False

        # previous pose (read on main thread to compute deltas)
        self._prev_pose: Optional[np.ndarray] = None

        # spin Teleop server in the background so `advance()` is non-blocking
        self._teleop: Optional[Teleop] = None
        self._thread: Optional[threading.Thread] = None
        self._server_kwargs: Optional[dict] = None
        self._start_server(self._server_kwargs or {})

    # --------------------------------------------------------------------- #
    # DeviceBase required API
    # --------------------------------------------------------------------- #

    def reset(self) -> None:
        """Reset the device internals (clears reference)."""
        self._prev_pose = None
        # keep latest pose so user can re-enable without reconnect

    def add_callback(self, key: Any, func: Callable) -> None:
        """Optional: bind a callback (unused for phone device)."""
        # We could forward callbacks to Teleop if needed; noop for now.
        return

    def advance(self) -> torch.Tensor:
        """Return SE(3) delta + gripper as a 7D tensor.

        Contract matches other Isaac Lab SE(3) devices:
        first 6 entries are [dx, dy, dz, droll, dpitch, dyaw] and last is gripper. :contentReference[oaicite:1]{index=1}
        """
        pose = self._latest_pose
        msg = self._latest_msg

        # default zeros if no data yet
        if pose is None:
            return torch.zeros(7, dtype=torch.float32, device=self._sim_device)

        # compute relative motion wrt previous pose
        if self._prev_pose is None:
            self._prev_pose = pose.copy()
            return torch.tensor(
                [0, 0, 0, 0, 0, 0, self._gripper_from_msg(msg)],
                dtype=torch.float32, device=self._sim_device
            )

        dp = self._delta_pose(self._prev_pose, pose)
        self._prev_pose = pose.copy()

        # scale & clamp
        dp[:3] *= self._pos_sensitivity
        dp[3:6] *= self._rot_sensitivity
        # dp[:3] = np.clip(dp[:3], -self._max_step_pos, self._max_step_pos)
        # dp[3:6] = np.clip(dp[3:6], -self._max_step_rot, self._max_step_rot)

        command = np.append(dp, self._gripper_from_msg(msg))
        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)

    # --------------------------------------------------------------------- #
    # Teleop plumbing
    # --------------------------------------------------------------------- #

    def _start_server(self, server_kwargs: dict) -> None:
        self._teleop = Teleop(**server_kwargs)

        def _cb(pose: np.ndarray, message: dict) -> None:
            # Expect pose: (4, 4), message: dict with keys like "move", "scale"
            if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
                return
            self._latest_pose = pose.astype(np.float64, copy=True)
            self._latest_msg = dict("scale")
            print


        self._teleop.subscribe(_cb)

        self._thread = threading.Thread(
            target=self._teleop.run, name="TeleopServer", daemon=True
        )
        self._thread.start()

        # give server a moment to boot
        time.sleep(0.1)

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    def _gripper_from_msg(self, msg: dict) -> float:
        # Simple mapping: value >= threshold => open (+1), else close (-1)
        val = float(msg.get("scale", 0.0))
        print(val)
        return 1.0 if val >= 1.0 else -1.0

    def _delta_pose(self, T_prev: np.ndarray, T_curr: np.ndarray) -> np.ndarray:
        """Compute [dx, dy, dz, droll, dpitch, dyaw] from two 4x4 transforms."""
        dT = np.linalg.inv(T_prev) @ T_curr
        t = dT[:3, 3]
        # if self._use_euler:
        rpy = self._mat_to_rpy(dT[:3, :3])
        # else:
        #     # axis-angle small-angle approx
        #     rpy = self._rot_to_small_rpy(dT[:3, :3])
        return np.concatenate([t, rpy], axis=0)

    @staticmethod
    def _mat_to_rpy(R: np.ndarray) -> np.ndarray:
        """ZYX (roll-pitch-yaw) from rotation matrix, numerically safe."""
        # yaw (z)
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        # pitch (y)
        sp = -R[2, 0]
        sp = float(np.clip(sp, -1.0, 1.0))
        pitch = float(np.arcsin(sp))
        # roll (x)
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        return np.array([roll, pitch, yaw], dtype=np.float64)

    @staticmethod
    def _rot_to_small_rpy(R: np.ndarray) -> np.ndarray:
        """Small-angle rpy from rotation matrix (approx via vee(logR))."""
        w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
        return w.astype(np.float64)
