# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleoperation devices: the keyboard in the viewer window, and a gamepad read from a Linux joystick device.

:func:`keyboard_action` turns the keys held in the viewer into an action. :class:`Se3LinuxGamepad` reads
``/dev/input/js*`` events into SE(3) motion commands, and :class:`BerryGamepad` adds a deadman button (LB) and
continuous gripper control with the analog triggers (RT closes, LT opens).
"""

import errno
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

# Keys that move the hand along x, y and z and turn it about x, y and z: (positive, negative).
_AXES = ("WS", "AD", "QE", "ZX", "TG", "CV")


def keyboard_action(viewer) -> torch.Tensor:
    """Return the environment action from the keys held in ``viewer``'s window.

    W/S, A/D and Q/E move the hand 1.5 mm per step along x, y and z; Z/X, T/G and C/V turn it 0.02 rad per step; K
    closes and J opens the gripper at 12 mm/s, and releasing them holds its opening.
    """
    action = torch.zeros((1, 7))
    for axis, (positive, negative) in enumerate(_AXES):
        held = int(viewer.is_key_down(positive)) - int(viewer.is_key_down(negative))
        action[0, axis] = (0.0015 if axis < 3 else 0.02) * held
    opening = int(viewer.is_key_down("J")) - int(viewer.is_key_down("K"))
    viewer.keyboard_aperture = float(np.clip(viewer.keyboard_aperture + 0.012 / 30 * opening, 0.0, 0.08))
    action[0, 6] = viewer.keyboard_aperture / 0.04 - 1
    return action


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


class BerryGamepad(Se3LinuxGamepad):
    """LB enables; RT closes, LT opens, release both to hold aperture [m]."""

    def __init__(self, cfg):
        super().__init__(
            pos_sensitivity=cfg.pos_sensitivity,
            rot_sensitivity=cfg.rot_sensitivity,
            dead_zone=cfg.dead_zone,
            sim_device=cfg.sim_device,
            device=cfg.device,
        )
        self.cfg = cfg
        self.connected = True
        self._buttons = {}
        self._enabled = False
        self._require_enable_release = False
        codes = bytearray(1024)
        ioctl(self._fd, 0x84006A34, codes)
        self._button_codes = dict(enumerate(struct.unpack("=512H", codes)))
        self.aperture = 0.08
        self._seen_axes = set()
        codes = bytearray(64)
        try:
            ioctl(self._fd, 0x80406A32, codes)  # JSIOCGAXMAP
        except OSError as exc:
            raise RuntimeError("Cannot identify analog triggers on this joystick") from exc
        self._axis_ids = {code: i for i, code in enumerate(codes[: self._axis_count])}
        if not {2, 5}.issubset(self._axis_ids):
            raise RuntimeError("This controller needs ABS_Z/ABS_RZ analog triggers")
        self._right_x_axis = self._axis_ids.get(3, self._right_x_axis)
        self._right_y_axis = self._axis_ids.get(4, self._right_y_axis)

    def __str__(self):
        return (
            f"Berry gamepad: {self._name}. Hold LB/L1 to enable. Sticks: XYZ/yaw; D-pad: roll/pitch. "
            "RT/R2: close, LT/L2: open; release triggers to hold. Menu: reset. "
            "Trigger pressure controls closure speed; the operator decides pick versus squash."
        )

    def _handle_event(self, value, event_type, number):
        initial = bool(event_type & 0x80)
        kind = event_type & 0x7F
        if kind == 2 and number < len(self._axes):
            self._seen_axes.add(number)
            self._axes[number] = np.clip(value / 32767, -1, 1)
        elif kind == 1:
            pressed = value != 0
            previous = self._buttons.get(number, False)
            self._buttons[number] = pressed
            code = self._button_codes.get(number)
            if code == 310:
                if initial and pressed:
                    self._require_enable_release = True
                if not pressed:
                    self._require_enable_release = False
                self._enabled = pressed and not initial and not self._require_enable_release
            if not initial and pressed and not previous and code == 315:
                callback = self._callbacks.get("R")
                if callback is not None:
                    callback()

    def _poll(self):
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

    def reset(self):
        axes = self._axes.copy()
        super().reset()
        self._require_enable_release = any(
            self._button_codes.get(i) == 310 and held for i, held in self._buttons.items()
        )
        self._enabled = False
        self.aperture = 0.08
        # Preserve physical trigger readings: joystick events arrive on changes only.
        self._axes[:] = axes

    def _trigger(self, code):
        index = self._axis_ids[code]
        if index not in self._seen_axes:
            return 0.0
        value = (float(self._axes[index]) + 1) * 0.5
        return max(0.0, (value - self.cfg.dead_zone) / (1 - self.cfg.dead_zone))

    def advance(self):
        command = super().advance()
        if not self.connected or not self._enabled:
            command[:6] = 0
        if self.connected and self._enabled:
            rate = self._trigger(2) - self._trigger(5)
            self.aperture = float(
                np.clip(
                    self.aperture + rate * self.cfg.aperture_speed * self.cfg.control_dt,
                    0,
                    0.08,
                )
            )
        command[6] = self.aperture / 0.04 - 1
        return command


@configclass
class BerryGamepadCfg(DeviceCfg):
    class_type: type = BerryGamepad
    pos_sensitivity: float = 0.0015
    rot_sensitivity: float = 0.02
    dead_zone: float = 0.12
    device: str | None = None
    aperture_speed: float = 0.012  # Total aperture change [m/s].
    control_dt: float = 1 / 30
