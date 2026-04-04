# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard controller for SE(2) control."""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch

import carb
import omni

from ..device_base import DeviceBase

if TYPE_CHECKING:
    from .se2_keyboard_cfg import Se2KeyboardCfg


class Se2Keyboard(DeviceBase):
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

        The official documentation for the keyboard interface: `Carb Keyboard Interface <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

    """

    def __init__(self, cfg: Se2KeyboardCfg):
        """Initialize the keyboard layer.

        Args:
            v_x_sensitivity: Magnitude of linear velocity along x-direction scaling. Defaults to 0.8.
            v_y_sensitivity: Magnitude of linear velocity along y-direction scaling. Defaults to 0.4.
            omega_z_sensitivity: Magnitude of angular velocity along z-direction scaling. Defaults to 1.0.
        """
        # store inputs
        self.v_x_sensitivity = cfg.v_x_sensitivity
        self.v_y_sensitivity = cfg.v_y_sensitivity
        self.omega_z_sensitivity = cfg.omega_z_sensitivity
        self._sim_device = cfg.sim_device

        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        # bindings for keyboard to command
        self._create_key_bindings()
        # command buffers
        self._base_command = np.zeros(3)
        # dictionary for additional callbacks
        self._additional_callbacks = dict()

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Keyboard Controller for SE(2): {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tReset all commands: L\n"
        msg += "\tMove forward   (along x-axis): Numpad 8 / Arrow Up\n"
        msg += "\tMove backward  (along x-axis): Numpad 2 / Arrow Down\n"
        msg += "\tMove right     (along y-axis): Numpad 4 / Arrow Right\n"
        msg += "\tMove left      (along y-axis): Numpad 6 / Arrow Left\n"
        msg += "\tYaw positively (along z-axis): Numpad 7 / Z\n"
        msg += "\tYaw negatively (along z-axis): Numpad 9 / X"
        return msg

    """
    Operations
    """

    def reset(self):
        # default flags
        self._base_command.fill(0.0)

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind keyboard.

        A list of available keys are present in the
        `carb documentation <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

        Args:
            key: The keyboard button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        """Provides the result from keyboard event state.

        Returns:
            Tensor containing the linear (x,y) and angular velocity (z).
        """
        return torch.tensor(self._base_command, dtype=torch.float32, device=self._sim_device)

    """
    Internal helpers.
    """

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html
        """
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "L":
                self.reset()
            elif event.input.name in self._INPUT_KEY_MAPPING:
                self._base_command += self._INPUT_KEY_MAPPING[event.input.name]
        # remove the command when un-pressed
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._INPUT_KEY_MAPPING:
                self._base_command -= self._INPUT_KEY_MAPPING[event.input.name]
        # additional callbacks
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._additional_callbacks:
                self._additional_callbacks[event.input.name]()

        # since no error, we are fine :)
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            # forward command
            "NUMPAD_8": np.asarray([1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            "UP": np.asarray([1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            # back command
            "NUMPAD_2": np.asarray([-1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            "DOWN": np.asarray([-1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            # right command
            "NUMPAD_4": np.asarray([0.0, 1.0, 0.0]) * self.v_y_sensitivity,
            "LEFT": np.asarray([0.0, 1.0, 0.0]) * self.v_y_sensitivity,
            # left command
            "NUMPAD_6": np.asarray([0.0, -1.0, 0.0]) * self.v_y_sensitivity,
            "RIGHT": np.asarray([0.0, -1.0, 0.0]) * self.v_y_sensitivity,
            # yaw command (positive)
            "NUMPAD_7": np.asarray([0.0, 0.0, 1.0]) * self.omega_z_sensitivity,
            "Z": np.asarray([0.0, 0.0, 1.0]) * self.omega_z_sensitivity,
            # yaw command (negative)
            "NUMPAD_9": np.asarray([0.0, 0.0, -1.0]) * self.omega_z_sensitivity,
            "X": np.asarray([0.0, 0.0, -1.0]) * self.omega_z_sensitivity,
        }
