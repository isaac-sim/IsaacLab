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
class Se2PhoneCfg(DeviceCfg):
    """Configuration for SE2 space mouse devices."""

    v_x_sensitivity: float = 0.8
    v_y_sensitivity: float = 0.4
    omega_z_sensitivity: float = 1.0


class Se2Phone(DeviceBase):
    r"""A keyboard controller for sending SE(2) commands as velocity commands.

    This class is designed to provide a keyboard controller for mobile base (such as quadrupeds).
    It uses the Omniverse keyboard interface to listen to keyboard events and map them to robot's
    task-space commands.

    The command comprises of the base linear and angular velocity: :math:`(v_x, v_y, \omega_z)`.

    Key bindings:
        ====================== ========================= ========================
        Command                Key (+ve axis)            Key (-ve axis)
        ====================== ========================= ========================
        Move along x-axis      Numpad 8 / Arrow Up       Numpad 2 / Arrow Down
        Move along y-axis      Numpad 4 / Arrow Right    Numpad 6 / Arrow Left
        Rotate along z-axis    Numpad 7 / Z              Numpad 9 / X
        ====================== ========================= ========================

    .. seealso::

        The official documentation for the phone interface: `Phone Interface <TODO: add package link>`__.

    """

    def __init__(self, cfg: Se2PhoneCfg):
        """Initialize the phone layer.

        Args:
            v_x_sensitivity: Magnitude of linear velocity along x-direction scaling. Defaults to 0.8.
            v_y_sensitivity: Magnitude of linear velocity along y-direction scaling. Defaults to 0.4.
            omega_z_sensitivity: Magnitude of angular velocity along z-direction scaling. Defaults to 1.0.
        """
        if Teleop is None:
            raise ImportError(
                "teleop is not available. Install it first (e.g., `pip install teleop`)."
            ) from _IMPORT_ERR

        # store inputs
        self._v_x_sensitivity = cfg.v_x_sensitivity
        self._v_y_sensitivity = cfg.v_y_sensitivity
        self._omega_z_sensitivity = cfg.omega_z_sensitivity
        self._sim_device = cfg.sim_device
        # latest data (written by callback thread)
        self._gripper = 1.0
        self._move_enabled = False

        self._latest_v_x: torch.Tensor | None = None
        self._latest_v_y: torch.Tensor | None = None
        self._latest_omega_z: torch.Tensor | None = None
        self._latest_msg: dict[str, Any] = {}

        # Previous sample used to compute relative deltas
        self._prev_v_x: torch.Tensor | None = None
        self._prev_v_y: torch.Tensor | None = None
        self._prev_omega_z: torch.Tensor | None = None

        # spin Teleop server in the background so `advance()` is non-blocking
        self._teleop: Teleop | None = None
        self._thread: threading.Thread | None = None
        self._server_kwargs: dict | None = None
        self._start_server(self._server_kwargs or {})

    def reset(self) -> None:
        self._prev_v_x = None
        self._prev_v_y = None
        self._prev_omega_z = None

    def add_callback(self, key: Any, func: Callable) -> None:
        """Optional: bind a callback (unused for phone device)."""
        # We could forward callbacks to Teleop if needed; noop for now.
        return

    def advance(self) -> torch.Tensor:
        """Provides the result from keyboard event state.

        Returns:
            Tensor containing the linear (x,y) and angular velocity (z).
        """
        command = torch.zeros(3, dtype=torch.float32, device=self._sim_device)

        if self._latest_v_x is None:
            return command

        # print(self._move_enabled)
        if self._prev_v_x is None:
            # First sample: initialize reference
            self._prev_v_x = self._latest_v_x.clone()
            self._prev_v_y = self._latest_v_y.clone()
            self._prev_omega_z = self._latest_omega_z.clone()
            return command

        if not self._move_enabled:
            # Gate OFF: zero deltas and keep reference synced to current to avoid jumps on re-enable
            self._prev_v_x = self._latest_v_x.clone()
            self._prev_v_y = self._latest_v_y.clone()
            self._prev_omega_z = self._latest_omega_z.clone()
            return command

        # Gate ON: compute SE(2) delta wrt previous, then update reference
        dvx = torch.sub(self._latest_v_x, self._prev_v_x)
        dvy = torch.sub(self._latest_v_y, self._prev_v_y)
        d_omega_z = torch.sub(self._latest_omega_z, self._prev_omega_z)
        print(f"dpos is {dvx, dvy}")
        print(f"drot is {d_omega_z}")

        command[0] = dvx * self._v_x_sensitivity
        command[1] = dvy * self._v_y_sensitivity
        command[2] = d_omega_z * self._omega_z_sensitivity

        return command

    # Teleop plumbing
    def _start_server(self, server_kwargs: dict) -> None:
        self._teleop = Teleop(**server_kwargs)

        def _cb(_pose_unused: np.ndarray, message: dict) -> None:
            # Expect "message" like the example in your comment.
            if not isinstance(message, dict):
                return
            self._latest_msg = dict(message)

            # --- Parse position ---
            p = message.get("position", {})
            self._latest_v_x = -float(p.get("z", 0.0))
            self._latest_v_y = -float(p.get("x", 0.0))

            # --- Parse quaternion (x, y, z, w) and normalize ---
            qd = message.get("orientation", {})
            qx = float(qd.get("x", 0.0))
            qy = float(qd.get("y", 0.0))
            qz = float(qd.get("z", 0.0))
            qw = float(qd.get("w", 1.0))

            quat = torch.tensor([qw, qx, qy, qz], device=self._sim_device, dtype=torch.float32).unsqueeze(0)  # (1, 4)
            self._latest_omega_z = axis_angle_from_quat(quat).squeeze(0)[1]

            self._move_enabled = bool(message.get("move", False))

        self._teleop.subscribe(_cb)

        self._thread = threading.Thread(target=self._teleop.run, name="TeleopServer", daemon=True)
        self._thread.start()

        # give server a moment to boot
        time.sleep(0.1)
