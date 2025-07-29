# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gamepad controller for SE(3) control."""

import numpy as np
import torch
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

import carb
import omni

from ..device_base import DeviceBase, DeviceCfg


@dataclass
class Se3GamepadCfg(DeviceCfg):
    """Configuration for SE3 gamepad devices."""

    dead_zone: float = 0.01  # For gamepad devices
    pos_sensitivity: float = 1.0
    rot_sensitivity: float = 1.6
    retargeters: None = None


class Se3Gamepad(DeviceBase):
    """A gamepad controller for sending SE(3) commands as delta poses and binary command (open/close).

    This class is designed to provide a gamepad controller for a robotic arm with a gripper.
    It uses the gamepad interface to listen to gamepad events and map them to the robot's
    task-space commands.

    The command comprises of two parts:

    * delta pose: a 6D vector of (x, y, z, roll, pitch, yaw) in meters and radians.
    * gripper: a binary command to open or close the gripper.

    Stick and Button bindings:
        ============================ ========================= =========================
        Description                  Stick/Button (+ve axis)   Stick/Button (-ve axis)
        ============================ ========================= =========================
        Toggle gripper(open/close)   X Button                  X Button
        Move along x-axis            Left Stick Up             Left Stick Down
        Move along y-axis            Left Stick Left           Left Stick Right
        Move along z-axis            Right Stick Up            Right Stick Down
        Rotate along x-axis          D-Pad Left                D-Pad Right
        Rotate along y-axis          D-Pad Down                D-Pad Up
        Rotate along z-axis          Right Stick Left          Right Stick Right
        ============================ ========================= =========================

    .. seealso::

        The official documentation for the gamepad interface: `Carb Gamepad Interface <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/gamepad.html>`__.

    """

    def __init__(
        self,
        cfg: Se3GamepadCfg,
    ):
        """Initialize the gamepad layer.

        Args:
            cfg: Configuration object for gamepad settings.
        """
        # turn off simulator gamepad control
        carb_settings_iface = carb.settings.get_settings()
        carb_settings_iface.set_bool("/persistent/app/omniverse/gamepadCameraControl", False)
        # store inputs
        self.pos_sensitivity = cfg.pos_sensitivity
        self.rot_sensitivity = cfg.rot_sensitivity
        self.dead_zone = cfg.dead_zone
        self._sim_device = cfg.sim_device
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
        self._close_gripper = False
        # When using the gamepad, two values are provided for each axis.
        # For example: when the left stick is moved down, there are two evens: `left_stick_down = 0.8`
        #   and `left_stick_up = 0.0`. If only the value of left_stick_up is used, the value will be 0.0,
        #   which is not the desired behavior. Therefore, we save both the values into the buffer and use
        #   the maximum value.
        # (positive, negative), (x, y, z, roll, pitch, yaw)
        self._delta_pose_raw = np.zeros([2, 6])
        # dictionary for additional callbacks
        self._additional_callbacks = dict()

    def __del__(self):
        """Unsubscribe from gamepad events."""
        self._input.unsubscribe_to_gamepad_events(self._gamepad, self._gamepad_sub)
        self._gamepad_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Gamepad Controller for SE(3): {self.__class__.__name__}\n"
        msg += f"\tDevice name: {self._input.get_gamepad_name(self._gamepad)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tToggle gripper (open/close): X\n"
        msg += "\tMove arm along x-axis: Left Stick Up/Down\n"
        msg += "\tMove arm along y-axis: Left Stick Left/Right\n"
        msg += "\tMove arm along z-axis: Right Stick Up/Down\n"
        msg += "\tRotate arm along x-axis: D-Pad Right/Left\n"
        msg += "\tRotate arm along y-axis: D-Pad Down/Up\n"
        msg += "\tRotate arm along z-axis: Right Stick Left/Right\n"
        return msg

    """
    Operations
    """

    def reset(self):
        # default flags
        self._close_gripper = False
        self._delta_pose_raw.fill(0.0)

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

    def advance(self) -> torch.Tensor:
        """Provides the result from gamepad event state.

        Returns:
            torch.Tensor: A 7-element tensor containing:
                - delta pose: First 6 elements as [x, y, z, rx, ry, rz] in meters and radians.
                - gripper command: Last element as a binary value (+1.0 for open, -1.0 for close).
        """
        # -- resolve position command
        delta_pos = self._resolve_command_buffer(self._delta_pose_raw[:, :3])
        # -- resolve rotation command
        delta_rot = self._resolve_command_buffer(self._delta_pose_raw[:, 3:])
        # -- convert to rotation vector
        rot_vec = Rotation.from_euler("XYZ", delta_rot).as_rotvec()
        # return the command and gripper state
        gripper_value = -1.0 if self._close_gripper else 1.0
        delta_pose = np.concatenate([delta_pos, rot_vec])
        command = np.append(delta_pose, gripper_value)
        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)

    """
    Internal helpers.
    """

    def _on_gamepad_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/gamepad.html
        """
        # check if the event is a button press
        cur_val = event.value
        if abs(cur_val) < self.dead_zone:
            cur_val = 0
        # -- button
        if event.input == carb.input.GamepadInput.X:
            # toggle gripper based on the button pressed
            if cur_val > 0.5:
                self._close_gripper = not self._close_gripper
        # -- left and right stick
        if event.input in self._INPUT_STICK_VALUE_MAPPING:
            direction, axis, value = self._INPUT_STICK_VALUE_MAPPING[event.input]
            # change the value only if the stick is moved (soft press)
            self._delta_pose_raw[direction, axis] = value * cur_val
        # -- dpad (4 arrow buttons on the console)
        if event.input in self._INPUT_DPAD_VALUE_MAPPING:
            direction, axis, value = self._INPUT_DPAD_VALUE_MAPPING[event.input]
            # change the value only if button is pressed on the DPAD
            if cur_val > 0.5:
                self._delta_pose_raw[direction, axis] = value
                self._delta_pose_raw[1 - direction, axis] = 0
            else:
                self._delta_pose_raw[:, axis] = 0
        # additional callbacks
        if event.input in self._additional_callbacks:
            self._additional_callbacks[event.input]()

        # since no error, we are fine :)
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        # map gamepad input to the element in self._delta_pose_raw
        #   the first index is the direction (0: positive, 1: negative)
        #   the second index is the axis (0: x, 1: y, 2: z, 3: roll, 4: pitch, 5: yaw)
        #   the third index is the sensitivity of the command
        self._INPUT_STICK_VALUE_MAPPING = {
            # forward command
            carb.input.GamepadInput.LEFT_STICK_UP: (0, 0, self.pos_sensitivity),
            # backward command
            carb.input.GamepadInput.LEFT_STICK_DOWN: (1, 0, self.pos_sensitivity),
            # right command
            carb.input.GamepadInput.LEFT_STICK_RIGHT: (0, 1, self.pos_sensitivity),
            # left command
            carb.input.GamepadInput.LEFT_STICK_LEFT: (1, 1, self.pos_sensitivity),
            # upward command
            carb.input.GamepadInput.RIGHT_STICK_UP: (0, 2, self.pos_sensitivity),
            # downward command
            carb.input.GamepadInput.RIGHT_STICK_DOWN: (1, 2, self.pos_sensitivity),
            # yaw command (positive)
            carb.input.GamepadInput.RIGHT_STICK_RIGHT: (0, 5, self.rot_sensitivity),
            # yaw command (negative)
            carb.input.GamepadInput.RIGHT_STICK_LEFT: (1, 5, self.rot_sensitivity),
        }

        self._INPUT_DPAD_VALUE_MAPPING = {
            # pitch command (positive)
            carb.input.GamepadInput.DPAD_UP: (1, 4, self.rot_sensitivity * 0.8),
            # pitch command (negative)
            carb.input.GamepadInput.DPAD_DOWN: (0, 4, self.rot_sensitivity * 0.8),
            # roll command (positive)
            carb.input.GamepadInput.DPAD_RIGHT: (1, 3, self.rot_sensitivity * 0.8),
            # roll command (negative)
            carb.input.GamepadInput.DPAD_LEFT: (0, 3, self.rot_sensitivity * 0.8),
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
        delta_command_sign = raw_command[1, :] > raw_command[0, :]
        # extract the command value
        delta_command = raw_command.max(axis=0)
        # apply the sign
        #  if the sign is positive, the value is already positive.
        #  if the sign is negative, the value is negative after applying the sign.
        delta_command[delta_command_sign] *= -1

        return delta_command
