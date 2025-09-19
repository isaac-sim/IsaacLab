# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phone controller for SE(3) control."""

from __future__ import annotations

import numpy as np
import threading
import time
import torch
from collections.abc import Callable
from typing import Any

from isaaclab.utils.math import axis_angle_from_quat

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
    """Configuration for SE3 phone devices."""

    gripper_term: bool = True
    pos_sensitivity: float = 0.5
    rot_sensitivity: float = 0.4
    retargeters: None = None


class Se3Phone(DeviceBase):
    """A phone controller for sending SE(3) commands as delta poses and binary command (open/close).

    This class is designed to provide a phone controller for a robotic arm with a gripper.
    It uses the PyPi teleop package to listen to phone events and map them to robot's
    task-space commands.

    The command comprises of two parts:

    * delta pose: a 6D vector of (x, y, z, roll, pitch, yaw) in meters and radians.
    * gripper: a binary command to open or close the gripper.

    See phone controller section in the teleoperation documentation for details: `Teleop <https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html#teleoperation>`__

    .. seealso::

        PyPi teleop package documentation: `Teleop <https://pypi.org/project/teleop/>`__.

    """

    def __init__(self, cfg: Se3PhoneCfg):
        """Initialize the phone layer.

        Args:
            cfg: Configuration object for phone settings.
        """
        if Teleop is None:
            raise ImportError(
                "teleop is not available. Install it first (e.g., `pip install teleop`)."
            ) from _IMPORT_ERR

        # store inputs
        self._pos_sensitivity = cfg.pos_sensitivity
        self._rot_sensitivity = cfg.rot_sensitivity
        self._gripper_term = cfg.gripper_term
        self._sim_device = cfg.sim_device
        # latest data (written by callback thread)
        self._gripper = 1.0
        self._move_enabled = False

        self._latest_pos: torch.Tensor | None = None  # (3,)
        self._latest_rot: torch.Tensor | None = None  # (3,) (w,x,y,z)
        self._latest_msg: dict[str, Any] = {}

        # Previous sample used to compute relative deltas
        self._prev_pos: torch.Tensor | None = None  # (3,)
        self._prev_rot: torch.Tensor | None = None  # (3,)

        # spin Teleop server in the background so `advance()` is non-blocking
        self._teleop: Teleop | None = None
        self._thread: threading.Thread | None = None
        self._server_kwargs: dict | None = None
        self._start_server(self._server_kwargs or {})

    def __str__(self) -> str:
        """Returns: A string containing the information of phone."""
        msg = f"Phone Controller for SE(3): {self.__class__.__name__}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tKeep the phone upright.\n"
        msg += "\tToggle gripper (open/close): Enabled Gripper button\n"
        msg += "\tTo move or rotate the arm: Press hold and do the corresponding action\n"
        msg += "\tMove arm along x-axis: Push/Pull your phone forward/backward\n"
        msg += "\tMove arm along y-axis: Pull/Pull your phone left/right\n"
        msg += "\tMove arm along z-axis: Push/Pull your phone up/down\n"
        msg += "\tRotate arm along x-axis: Rotate your phone left/right\n"
        msg += "\tRotate arm along y-axis: Tilt your phone forward/backward\n"
        msg += "\tRotate arm along z-axis: Twist your phone left/right"
        return msg

    def reset(self) -> None:
        self._prev_pos = None
        self._prev_rot = None

    def add_callback(self, key: Any, func: Callable) -> None:
        """Optional: bind a callback (unused for phone device)."""
        return

    def advance(self) -> torch.Tensor:
        """Return SE(3) delta + gripper as a 7D tensor.

        Contract matches other Isaac Lab SE(3) devices:
        first 6 entries are [dx, dy, dz, droll, dpitch, dyaw] and last is gripper. :contentReference[oaicite:1]{index=1}
        """
        command = torch.zeros(7, dtype=torch.float32, device=self._sim_device)
        command[6] = self._gripper

        if self._latest_pos is None or self._latest_rot is None:
            return command

        if self._prev_pos is None or self._prev_rot is None:
            # First sample: initialize reference
            self._prev_pos = self._latest_pos.clone()
            self._prev_rot = self._latest_rot.clone()
            return command

        if not self._move_enabled:
            # Gate OFF: zero deltas and keep reference synced to current to avoid jumps on re-enable
            self._prev_pos = self._latest_pos.clone()
            self._prev_rot = self._latest_rot.clone()
            return command

        # Gate ON: compute SE(3) delta wrt previous
        dpos = torch.sub(self._latest_pos, self._prev_pos)
        drot = torch.sub(self._latest_rot, self._prev_rot)

        command[:3] = dpos * self._pos_sensitivity
        command[3:6] = drot * self._rot_sensitivity

        return command

    # Teleop plumbing
    def _start_server(self, server_kwargs: dict) -> None:
        self._teleop = Teleop(**server_kwargs)

        def _cb(_pose_unused: np.ndarray, message: dict) -> None:
            if not isinstance(message, dict):
                return
            self._latest_msg = dict(message)

            # --- Parse position ---
            p = message.get("position", {})
            tx = -float(p.get("z", 0.0))
            ty = -float(p.get("x", 0.0))
            tz = float(p.get("y", 0.0))

            self._latest_pos = torch.tensor([tx, ty, tz], device=self._sim_device, dtype=torch.float32)

            # --- Parse quaternion (x, y, z, w) ---
            qd = message.get("orientation", {})
            qx = float(qd.get("x", 0.0))
            qy = float(qd.get("y", 0.0))
            qz = float(qd.get("z", 0.0))
            qw = float(qd.get("w", 1.0))

            quat = torch.tensor([qw, qx, qy, qz], device=self._sim_device, dtype=torch.float32).unsqueeze(0)  # (1, 4)
            self._latest_rot = axis_angle_from_quat(quat).squeeze(0)  # (3,)
            self._latest_rot[[0, 1, 2]] = self._latest_rot[[2, 0, 1]] * torch.tensor(
                [-1, -1, 1], device=self._sim_device, dtype=torch.float32
            )

            g = message.get("gripper")
            if isinstance(g, str):
                s = g.strip().lower()
                if s == "open":
                    self._gripper = 1.0
                elif s == "close":
                    self._gripper = -1.0

            self._move_enabled = bool(message.get("move", False))

        self._teleop.subscribe(_cb)

        self._thread = threading.Thread(target=self._teleop.run, name="TeleopServer", daemon=True)
        self._thread.start()

        # give server a moment to boot
        time.sleep(0.1)
