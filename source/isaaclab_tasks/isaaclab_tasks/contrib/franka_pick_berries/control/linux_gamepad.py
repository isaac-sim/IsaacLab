# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Linux joystick-device controller for SE(3) teleoperation."""

from __future__ import annotations

import errno
import os
import struct
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isaaclab.devices.device_base import DeviceBase


def ioctl(fd: int, request: int, buffer: bytearray) -> None:
    """Run a joystick ioctl; ``fcntl`` is imported here because it exists on Linux only."""
    import fcntl

    fcntl.ioctl(fd, request, buffer)


class Se3LinuxGamepad(DeviceBase):
    """Read a conventional Linux joystick device without Kit or an Omniverse window.

    The first four axes are interpreted as left X/Y and right X/Y; axes 6/7, when
    present, are interpreted as the D-pad. This matches common Xbox and DualShock 4
    mappings. Use ``device`` to select a different joystick node.
    """

    _EVENT_STRUCT = struct.Struct("=IhBB")
    # JSIOCGNAME is _IOC(_IOC_READ, 'j', 0x13, 128) for the 128-byte name buffer.
    _JSIOCGNAME = 0x80806A13
    _JSIOCGAXES = 0x80016A11

    def __init__(
        self,
        pos_sensitivity: float = 0.1,
        rot_sensitivity: float = 0.1,
        dead_zone: float = 0.08,
        sim_device: str = "cpu",
        device: str | os.PathLike[str] | None = None,
    ) -> None:
        """Initialize a Linux joystick device."""
        super().__init__()
        if os.name != "posix" or not Path("/dev/input").is_dir():
            raise RuntimeError("Linux gamepad input requires a POSIX system with /dev/input.")
        self.pos_sensitivity = float(pos_sensitivity)
        self.rot_sensitivity = float(rot_sensitivity)
        self.dead_zone = float(dead_zone)
        self._sim_device = sim_device
        self._device_path = Path(device) if device is not None else self._find_device()
        self._fd = os.open(self._device_path, os.O_RDONLY | os.O_NONBLOCK)
        self._axis_count = self._read_axis_count()
        # Linux xpad and DualShock drivers commonly insert trigger axes before the
        # right stick: [LX, LY, LT, RX, RY, RT, HATX, HATY]. Older four-axis devices
        # use [LX, LY, RX, RY].
        self._right_x_axis = 3 if self._axis_count >= 5 else 2
        self._right_y_axis = 4 if self._axis_count >= 5 else 3
        self._axes = np.zeros(self._axis_count, dtype=np.float32)
        self._close_gripper = False
        self._callbacks: dict[Any, Callable[[], None]] = {}
        self._name = self._read_name()

    @staticmethod
    def _find_device() -> Path:
        """Return the first joystick node exposed by the Linux joystick API."""
        devices = sorted(Path("/dev/input").glob("js*"))
        if not devices:
            raise RuntimeError("No Linux joystick found. Check that /dev/input/js* exists.")
        return devices[0]

    def _read_name(self) -> str:
        """Read the kernel-provided joystick name when available."""
        name = bytearray(128)
        try:
            ioctl(self._fd, self._JSIOCGNAME, name)
            return bytes(name).split(b"\0", 1)[0].decode(errors="replace") or str(self._device_path)
        except OSError:
            return str(self._device_path)

    def _read_axis_count(self) -> int:
        """Read the number of joystick axes, falling back to the common four-axis layout."""
        count = bytearray(1)
        try:
            ioctl(self._fd, self._JSIOCGAXES, count)
            return int(count[0])
        except OSError:
            return 4

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
        """Return the controller mapping."""
        return (
            f"Linux Gamepad Controller for SE(3): {self._name}\n"
            f"\tDevice: {self._device_path}\n"
            "\tLeft stick: X/Y translation\n"
            "\tRight stick: Z translation / yaw\n"
            "\tD-pad: roll / pitch\n"
            "\tButton 0: toggle gripper"
        )

    def reset(self) -> None:
        """Clear the current command state."""
        self._axes.fill(0.0)
        self._close_gripper = False

    def add_callback(self, key: Any, func: Callable[[], None]) -> None:
        """Register a callback for a numeric button identifier."""
        self._callbacks[key] = func

    def _dispatch_button(self, button: int, pressed: bool) -> None:
        """Dispatch callbacks on a button press edge."""
        if not pressed:
            return
        if button == 0:
            self._close_gripper = not self._close_gripper
        aliases = {7: "START", 9: "R"}
        for key in (button, f"button_{button}", aliases.get(button)):
            if key is None:
                continue
            callback = self._callbacks.get(key)
            if callback is not None:
                callback()

    def _poll(self) -> None:
        """Drain pending joystick events."""
        while True:
            try:
                raw = os.read(self._fd, self._EVENT_STRUCT.size)
            except BlockingIOError:
                return
            except OSError as exc:
                if exc.errno == errno.EAGAIN:
                    return
                raise
            if len(raw) != self._EVENT_STRUCT.size:
                return
            _timestamp, value, event_type, number = self._EVENT_STRUCT.unpack(raw)
            event_type &= 0x7F
            if event_type == 0x02 and number < len(self._axes):
                self._axes[number] = np.clip(value / 32767.0, -1.0, 1.0)
            elif event_type == 0x01:
                self._dispatch_button(number, value != 0)

    def _axis(self, index: int) -> float:
        """Apply the dead zone while preserving the remaining stick range."""
        if index >= len(self._axes):
            return 0.0
        value = float(self._axes[index])
        if abs(value) <= self.dead_zone:
            return 0.0
        sign = 1.0 if value >= 0.0 else -1.0
        return sign * (abs(value) - self.dead_zone) / (1.0 - self.dead_zone)

    def advance(self) -> torch.Tensor:
        """Poll the device and return a 7D SE(3)+gripper command."""
        self._poll()
        left_x = self._axis(0)
        left_y = -self._axis(1)
        right_x = self._axis(self._right_x_axis)
        right_y = -self._axis(self._right_y_axis)
        dpad_x = self._axis(6)
        dpad_y = -self._axis(7)
        command = np.array(
            [
                left_y * self.pos_sensitivity,
                left_x * self.pos_sensitivity,
                right_y * self.pos_sensitivity,
                -dpad_x * self.rot_sensitivity * 0.8,
                dpad_y * self.rot_sensitivity * 0.8,
                right_x * self.rot_sensitivity,
            ],
            dtype=np.float32,
        )
        return torch.tensor(
            np.append(command, -1.0 if self._close_gripper else 1.0),
            dtype=torch.float32,
            device=self._sim_device,
        )
