# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deadman-enabled incremental aperture control using Linux axis identities."""

import os
import struct

import numpy as np

from isaaclab.devices.device_base import DeviceCfg
from isaaclab.utils.configclass import configclass

from .linux_gamepad import Se3LinuxGamepad, ioctl


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
