# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gamepad controller for SE(2) control."""


import numpy as np
from typing import Callable

import carb
import omni

from ..device_base import DeviceBase


class Se2Gamepad(DeviceBase):
    r"""A gamepad controller for sending SE(2) commands as velocity commands.

    This class is designed to provide a gamepad controller for mobile base (such as quadrupeds).
    It uses the Omniverse gamepad interface to listen to gamepad events and map them to robot's
    task-space commands.

    The command comprises of the base linear and angular velocity: :math:`(v_x, v_y, \omega_z)`.

    Key bindings:
        ====================== ========================= ========================
        Command                Key (+ve axis)            Key (-ve axis)
        ====================== ========================= ========================
        Move along x-axis      left stick up             left stick down
        Move along y-axis      left stick right          left stick left
        Rotate along z-axis    right stick right         right stick left
        ====================== ========================= ========================

    .. seealso::

        The official documentation for the gamepad interface: `Carb Gamepad Interface <https://docs.omniverse.nvidia.com/kit/docs/carbonite/latest/docs/python/carb.html#carb.input.Gamepad>`__.

    """

    def __init__(
        self,
        v_x_sensitivity: float = 1.0,
        v_y_sensitivity: float = 1.0,
        omega_z_sensitivity: float = 1.0,
        deadzone: float = 0.01,
    ):
        """Initialize the keyboard layer.

        Args:
            v_x_sensitivity (float): Magnitude of linear velocity along x-direction scaling. Defaults to 1.0.
            v_y_sensitivity (float): Magnitude of linear velocity along y-direction scaling. Defaults to 1.0.
            omega_z_sensitivity (float): Magnitude of angular velocity along z-direction scaling. Defaults to 1.0.
            deadzone (float): Magnitude of deadzone for gamepad. Defaults to 0.01.
        """
        # store inputs
        self.v_x_sensitivity = v_x_sensitivity
        self.v_y_sensitivity = v_y_sensitivity
        self.omega_z_sensitivity = omega_z_sensitivity
        self.deadzone = deadzone
        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._gamepad = self._appwindow.get_gamepad(0)
        self._gamepad_sub = self._input.subscribe_to_gamepad_events(self._gamepad, self._on_gamepad_event)
        # bindings for gamepad to command
        self._create_key_bindings()
        # command buffers
        self._base_command = np.zeros(3)
        # dictionary for additional callbacks
        self._additional_callbacks = dict()

    def __del__(self):
        """Unsubscribe from gamepad events."""
        self._input.unsubscribe_from_gamepad_events(self._gamepad, self._gamepad_sub)
        self._gamepad_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Gamepad Controller for SE(2): {self.__class__.__name__}\n"
        msg += f"\tDevice name: {self._input.get_gamepad_name(self._gamepad)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tMove in X-Y plane: left stick\n"
        msg += "\tRotate in Z-axis: right stick\n"
        return msg

    """
    Operations
    """

    def reset(self):
        # default flags
        self._base_command.fill(0.0)

    def add_callback(self, key: carb.input.GamepadInput, func: Callable):
        """Add additional functions to bind gamepad.

        A list of available gamepad keys are present in the
        `carb documentation <https://docs.omniverse.nvidia.com/kit/docs/carbonite/latest/docs/python/carb.html?highlight=keyboardeventtype#carb.input.GamepadInput>`__.

        Args:
            key (carb.input.GamepadInput): The gamepad button to check against.
            func (Callable): The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func

    def advance(self) -> np.ndarray:
        """Provides the result from keyboard event state.

        Returns:
            np.ndarray: A 3D array containing the linear (x,y) and angular velocity (z).
        """
        return self._base_command

    """
    Internal helpers.
    """

    def _on_gamepad_event(self, event: carb.input.GamepadEvent, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/kit/docs/carbonite/latest/docs/python/carb.html?highlight=keyboardeventtype#carb.input.GamepadInput
        """
        # the base command depend only on the current gamepad status
        # so we reset it every time
        self.reset()
        # check if the event is valid
        cur_val = event.value
        if abs(cur_val) < self.deadzone:
            cur_val = 0
        # apply the command based on the event
        if event.input in self._INPUT_STICK_VALUE_MAPPING:
            self._base_command += self._INPUT_STICK_VALUE_MAPPING[event.input] * cur_val
        # additional callbacks
        if event.input in self._additional_callbacks:
            self._additional_callbacks[event.input]()

        # since no error, we are fine :)
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_STICK_VALUE_MAPPING = {
            # forward command
            carb.input.GamepadInput.LEFT_STICK_UP: np.asarray([1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            # backward command
            carb.input.GamepadInput.LEFT_STICK_DOWN: np.asarray([-1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            # right command
            carb.input.GamepadInput.LEFT_STICK_RIGHT: np.asarray([0.0, 1.0, 0.0]) * self.v_y_sensitivity,
            # left command
            carb.input.GamepadInput.LEFT_STICK_LEFT: np.asarray([0.0, -1.0, 0.0]) * self.v_y_sensitivity,
            # yaw command (positive)
            carb.input.GamepadInput.RIGHT_STICK_RIGHT: np.asarray([0.0, 0.0, 1.0]) * self.omega_z_sensitivity,
            # yaw command (negative)
            carb.input.GamepadInput.RIGHT_STICK_LEFT: np.asarray([0.0, 0.0, -1.0]) * self.omega_z_sensitivity,
        }
