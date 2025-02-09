# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gamepad controller for SE(2) control."""

import numpy as np
import weakref
from collections.abc import Callable

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

        The official documentation for the gamepad interface: `Carb Gamepad Interface <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/gamepad.html>`__.

    """

    def __init__(
        self,
        v_x_sensitivity: float = 1.0,
        v_y_sensitivity: float = 1.0,
        omega_z_sensitivity: float = 1.0,
        dead_zone: float = 0.01,
    ):
        """Initialize the gamepad layer.

        Args:
            v_x_sensitivity: Magnitude of linear velocity along x-direction scaling. Defaults to 1.0.
            v_y_sensitivity: Magnitude of linear velocity along y-direction scaling. Defaults to 1.0.
            omega_z_sensitivity: Magnitude of angular velocity along z-direction scaling. Defaults to 1.0.
            dead_zone: Magnitude of dead zone for gamepad. An event value from the gamepad less than
                this value will be ignored. Defaults to 0.01.
        """
        # turn off simulator gamepad control
        carb_settings_iface = carb.settings.get_settings()
        carb_settings_iface.set_bool("/persistent/app/omniverse/gamepadCameraControl", False)
        # store inputs
        self.v_x_sensitivity = v_x_sensitivity
        self.v_y_sensitivity = v_y_sensitivity
        self.omega_z_sensitivity = omega_z_sensitivity
        self.dead_zone = dead_zone
        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._gamepad = self._appwindow.get_gamepad(0)
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called
        self._gamepad_sub = self._input.subscribe_to_gamepad_events(
            self._gamepad,
            lambda event, *args, obj=weakref.proxy(self): obj._on_gamepad_event(event, *args),
        )
        # bindings for gamepad to command
        self._create_key_bindings()
        # command buffers
        # When using the gamepad, two values are provided for each axis.
        # For example: when the left stick is moved down, there are two evens: `left_stick_down = 0.8`
        #   and `left_stick_up = 0.0`. If only the value of left_stick_up is used, the value will be 0.0,
        #   which is not the desired behavior. Therefore, we save both the values into the buffer and use
        #   the maximum value.
        # (positive, negative), (x, y, yaw)
        self._base_command_raw = np.zeros([2, 3])
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
        self._base_command_raw.fill(0.0)

    def add_callback(self, key: carb.input.GamepadInput, func: Callable):
        """Add additional functions to bind gamepad.

        A list of available gamepad keys are present in the
        `carb documentation <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/gamepad.html>`__.

        Args:
            key: The gamepad button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func

    def advance(self) -> np.ndarray:
        """Provides the result from gamepad event state.

        Returns:
            A 3D array containing the linear (x,y) and angular velocity (z).
        """
        return self._resolve_command_buffer(self._base_command_raw)

    """
    Internal helpers.
    """

    def _on_gamepad_event(self, event: carb.input.GamepadEvent, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/gamepad.html
        """

        # check if the event is a button press
        cur_val = event.value
        if abs(cur_val) < self.dead_zone:
            cur_val = 0
        # -- left and right stick
        if event.input in self._INPUT_STICK_VALUE_MAPPING:
            direction, axis, value = self._INPUT_STICK_VALUE_MAPPING[event.input]
            # change the value only if the stick is moved (soft press)
            self._base_command_raw[direction, axis] = value * cur_val

        # additional callbacks
        if event.input in self._additional_callbacks:
            self._additional_callbacks[event.input]()

        # since no error, we are fine :)
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_STICK_VALUE_MAPPING = {
            # forward command
            carb.input.GamepadInput.LEFT_STICK_UP: (0, 0, self.v_x_sensitivity),
            # backward command
            carb.input.GamepadInput.LEFT_STICK_DOWN: (1, 0, self.v_x_sensitivity),
            # right command
            carb.input.GamepadInput.LEFT_STICK_RIGHT: (0, 1, self.v_y_sensitivity),
            # left command
            carb.input.GamepadInput.LEFT_STICK_LEFT: (1, 1, self.v_y_sensitivity),
            # yaw command (positive)
            carb.input.GamepadInput.RIGHT_STICK_RIGHT: (0, 2, self.omega_z_sensitivity),
            # yaw command (negative)
            carb.input.GamepadInput.RIGHT_STICK_LEFT: (1, 2, self.omega_z_sensitivity),
        }

    def _resolve_command_buffer(self, raw_command: np.ndarray) -> np.ndarray:
        """Resolves the command buffer.

        Args:
            raw_command: The raw command from the gamepad. Shape is (2, 3)
                This is a 2D array since gamepad dpad/stick returns two values corresponding to
                the positive and negative direction. The first index is the direction (0: positive, 1: negative)
                and the second index is value (absolute) of the command.

        Returns:
            Resolved command. Shape is (3,)
        """
        # compare the positive and negative value decide the sign of the value
        #   if the positive value is larger, the sign is positive (i.e. False, 0)
        #   if the negative value is larger, the sign is positive (i.e. True, 1)
        command_sign = raw_command[1, :] > raw_command[0, :]
        # extract the command value
        command = raw_command.max(axis=0)
        # apply the sign
        #  if the sign is positive, the value is already positive.
        #  if the sign is negative, the value is negative after applying the sign.
        command[command_sign] *= -1

        return command
