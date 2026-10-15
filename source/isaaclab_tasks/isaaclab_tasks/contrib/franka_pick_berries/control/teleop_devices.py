# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleoperation devices: the keyboard in the viewer window, and a gamepad read from a Linux joystick device.

:func:`keyboard_action` turns the keys held in the viewer into an action. :class:`BerryGamepad` reads
``/dev/input/js*`` events into motion commands, enabled by a deadman button (LB), with continuous gripper control by the
analog triggers (RT closes, LT opens).

Both devices command horizontal motion and tilt in the camera's view: forward, left and up. :func:`view_to_robot`
turns them into the robot's frame, so that forward moves the hand away from the camera whichever way it looks.
"""

import math
import os
import struct
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isaaclab.devices.device_base import DeviceBase, DeviceCfg
from isaaclab.utils.configclass import configclass

# Keys that move the hand forward, left and up, and turn it about those axes: (positive, negative).
_AXES = ("WS", "AD", "QE", "ZX", "TG", "CV")

CLOSING_SPEED = 0.024
"""Rate [m/s] of total aperture at which the teleoperated gripper closes: slow enough to stop short of crushing."""
OPENING_SPEED = 0.036
"""Rate [m/s] of total aperture at which the teleoperated gripper opens: fast enough to release crushed tissue."""


def view_to_robot(action: torch.Tensor, camera_yaw: float) -> torch.Tensor:
    """Turn an action's horizontal motion and tilt from the camera's view into the robot's frame.

    Args:
        action: Action of shape (1, 7) whose motion (0, 1) and rotation (3, 4) are along the camera's horizontal
            forward and left directions.
        camera_yaw: Heading [deg] of the camera's view direction about the vertical axis.
    """
    yaw = np.radians(camera_yaw)
    turn = torch.tensor([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]], dtype=action.dtype)
    action = action.clone()
    for axes in (slice(0, 2), slice(3, 5)):
        action[0, axes] = turn.to(action.device) @ action[0, axes]
    return action


def keyboard_action(viewer) -> torch.Tensor:
    """Return the environment action from the keys held in ``viewer``'s window.

    W/S, A/D and Q/E move the hand 1.5 mm per step forward, left and up in the camera's view; Z/X, T/G and C/V turn
    it 0.02 rad per step about those axes; K closes and J opens the gripper, and releasing them holds its opening.
    """
    action = torch.zeros((1, 7))
    for axis, (positive, negative) in enumerate(_AXES):
        held = int(viewer.is_key_down(positive)) - int(viewer.is_key_down(negative))
        action[0, axis] = (0.0015 if axis < 3 else 0.02) * held
    rate = OPENING_SPEED * viewer.is_key_down("J") - CLOSING_SPEED * viewer.is_key_down("K")
    viewer.keyboard_aperture = float(np.clip(viewer.keyboard_aperture + rate / 30, 0.0, 0.08))
    action[0, 6] = viewer.keyboard_aperture / 0.04 - 1
    return action


def ioctl(fd: int, request: int, buffer: bytearray) -> None:
    """Run a joystick ioctl; ``fcntl`` is imported here because it exists on Linux only."""
    import fcntl

    fcntl.ioctl(fd, request, buffer)


class BerryGamepad(DeviceBase):
    """Read a Linux joystick device (``/dev/input/js*``), without Kit, into the berry task's action.

    Holding LB enables motion: the left stick moves the hand forward and left in the view, the right stick up and turns
    it about the vertical, and the D-pad tilts it. The analog triggers drive the gripper (RT closes, LT opens, with
    their pressure setting the speed), and releasing both holds its opening. Menu calls the ``"R"`` callback.

    Args:
        cfg: The gamepad configuration.
    """

    _EVENT_STRUCT = struct.Struct("=IhBB")
    # Linux joystick ioctls: name (128-byte buffer), axis count, axis map and button map.
    _JSIOCGNAME = 0x80806A13
    _JSIOCGAXES = 0x80016A11
    _JSIOCGAXMAP = 0x80406A32
    _JSIOCGBTNMAP = 0x84006A34
    # Linux input codes of the sticks', triggers' (Z, RZ) and D-pad's (HAT0) axes, and of the LB and Menu buttons.
    _ABS_X, _ABS_Y, _ABS_Z, _ABS_RX, _ABS_RY, _ABS_RZ, _ABS_HAT0X, _ABS_HAT0Y = 0, 1, 2, 3, 4, 5, 16, 17
    _BTN_TL, _BTN_START = 310, 315

    def __init__(self, cfg: "BerryGamepadCfg"):
        super().__init__()
        self.cfg = cfg
        self._device_path = Path(cfg.device) if cfg.device is not None else self._find_device()
        self._fd = os.open(self._device_path, os.O_RDONLY | os.O_NONBLOCK)
        name = bytearray(128)
        ioctl(self._fd, self._JSIOCGNAME, name)
        self._name = bytes(name).split(b"\0", 1)[0].decode(errors="replace") or str(self._device_path)
        count = bytearray(1)
        ioctl(self._fd, self._JSIOCGAXES, count)
        self._axes = np.zeros(count[0], dtype=np.float32)
        axis_codes = bytearray(64)
        ioctl(self._fd, self._JSIOCGAXMAP, axis_codes)
        self._axis_ids = {code: i for i, code in enumerate(axis_codes[: len(self._axes)])}
        # Axes are looked up by input code: drivers order them differently (xpad puts the triggers before the right
        # stick). DirectInput-style pads report their right stick as Z/RZ, without the RX/RY that tell them apart.
        sticks_and_triggers = (self._ABS_X, self._ABS_Y, self._ABS_Z, self._ABS_RX, self._ABS_RY, self._ABS_RZ)
        if not set(sticks_and_triggers).issubset(self._axis_ids):
            raise RuntimeError("This controller needs two sticks (X/Y, RX/RY) and analog triggers (Z/RZ)")
        button_codes = bytearray(1024)
        ioctl(self._fd, self._JSIOCGBTNMAP, button_codes)
        button_codes = struct.unpack("=512H", button_codes)
        self._lb_button, self._menu_button = (
            button_codes.index(code) if code in button_codes else None for code in (self._BTN_TL, self._BTN_START)
        )
        self._callbacks: dict[Any, Callable[[], None]] = {}
        self.connected = True
        self._buttons = {}
        self._enabled = False
        self._require_enable_release = False
        self._seen_axes = set()
        self.aperture = 0.08

    @staticmethod
    def _find_device() -> Path:
        """Return the first joystick node exposed by the Linux joystick API."""
        devices = sorted(Path("/dev/input").glob("js*"))
        if not devices:
            raise RuntimeError("No Linux joystick found. Check that /dev/input/js* exists.")
        return devices[0]

    def close(self) -> None:
        """Close the joystick file descriptor."""
        fd = getattr(self, "_fd", None)
        self._fd = None
        if fd is not None:
            with suppress(OSError):
                os.close(fd)

    def __del__(self) -> None:
        self.close()

    def __str__(self) -> str:
        return (
            f"Berry gamepad: {self._name}. Hold LB/L1 to enable. "
            "Left stick: forward/left in the view; right stick: up/yaw; D-pad: tilt. "
            "RT/R2: close, LT/L2: open; release triggers to hold. Menu: reset. "
            "Trigger pressure controls closure speed; the operator decides pick versus squash."
        )

    def add_callback(self, key: Any, func: Callable[[], None]) -> None:
        """Register a callback; ``"R"`` is called when Menu is pressed."""
        self._callbacks[key] = func

    def reset(self) -> None:
        # The trigger readings are kept: joystick events arrive on changes only.
        self._require_enable_release = self._buttons.get(self._lb_button, False)
        self._enabled = False
        self.aperture = 0.08

    def _handle_event(self, value: int, event_type: int, number: int) -> None:
        initial = bool(event_type & 0x80)
        kind = event_type & 0x7F
        if kind == 2 and number < len(self._axes):
            self._seen_axes.add(number)
            self._axes[number] = np.clip(value / 32767, -1, 1)
        elif kind == 1:
            pressed = value != 0
            previous = self._buttons.get(number, False)
            self._buttons[number] = pressed
            if number == self._lb_button:
                # LB already held when the device opens must be released and pressed again.
                if initial and pressed:
                    self._require_enable_release = True
                if not pressed:
                    self._require_enable_release = False
                self._enabled = pressed and not initial and not self._require_enable_release
            if not initial and pressed and not previous and number == self._menu_button:
                callback = self._callbacks.get("R")
                if callback is not None:
                    callback()

    def _poll(self) -> None:
        """Drain pending joystick events; a failed read means the controller was disconnected."""
        if not self.connected:
            return
        while True:
            try:
                raw = os.read(self._fd, self._EVENT_STRUCT.size)
            except BlockingIOError:
                return
            except OSError:
                raw = b""
            if len(raw) != self._EVENT_STRUCT.size:
                self.connected = self._enabled = False
                self._axes.fill(0)
                return
            _, value, kind, number = self._EVENT_STRUCT.unpack(raw)
            self._handle_event(value, kind, number)

    def _axis(self, code: int) -> float:
        """Return the axis of input ``code``, past the dead zone and rescaled to the remaining range; 0 if absent."""
        index = self._axis_ids.get(code)
        value = 0.0 if index is None else float(self._axes[index])
        if abs(value) <= self.cfg.dead_zone:
            return 0.0
        return math.copysign((abs(value) - self.cfg.dead_zone) / (1.0 - self.cfg.dead_zone), value)

    def _trigger(self, code: int) -> float:
        """Return a trigger's pressure in [0, 1] past the dead zone; 0 until the trigger has reported."""
        index = self._axis_ids[code]
        if index not in self._seen_axes:
            return 0.0
        value = (float(self._axes[index]) + 1) * 0.5
        return max(0.0, (value - self.cfg.dead_zone) / (1 - self.cfg.dead_zone))

    def advance(self) -> torch.Tensor:
        """Poll the device and return the 7D command: motion, rotation and gripper opening."""
        self._poll()
        command = np.zeros(7, np.float32)
        if self.connected and self._enabled:
            pos, rot = self.cfg.pos_sensitivity, self.cfg.rot_sensitivity
            command[:6] = (
                -self._axis(self._ABS_Y) * pos,
                -self._axis(self._ABS_X) * pos,
                -self._axis(self._ABS_RY) * pos,
                -self._axis(self._ABS_HAT0X) * rot * 0.8,
                -self._axis(self._ABS_HAT0Y) * rot * 0.8,
                self._axis(self._ABS_RX) * rot,
            )
            rate = self._trigger(self._ABS_Z) * OPENING_SPEED - self._trigger(self._ABS_RZ) * CLOSING_SPEED
            self.aperture = float(np.clip(self.aperture + rate * self.cfg.control_dt, 0, 0.08))
        command[6] = self.aperture / 0.04 - 1
        return torch.tensor(command, device=self.cfg.sim_device)


@configclass
class BerryGamepadCfg(DeviceCfg):
    class_type: type = BerryGamepad
    pos_sensitivity: float = 0.003
    rot_sensitivity: float = 0.02
    dead_zone: float = 0.12
    device: str | None = None
    control_dt: float = 1 / 30
