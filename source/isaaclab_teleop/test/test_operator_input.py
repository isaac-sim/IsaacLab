# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for the operator's ``B`` / ``P`` / ``R`` keys and SpaceMouse right button on
:class:`~isaaclab_teleop.IsaacTeleopDevice`, against a mocked session lifecycle."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from isaaclab_teleop.isaac_teleop_device import IsaacTeleopDevice

KEY_R, KEY_P, KEY_B, KEY_N = 19, 25, 48, 49


class _Buttons:
    """Stand-in for the ``spacemouse_buttons`` entry of a pipeline step result."""

    def __init__(self, *pressed: int) -> None:
        self.is_none = False
        self._bitmap = np.zeros(8, dtype=np.uint8)
        self._bitmap[list(pressed)] = 1

    def __getitem__(self, index):
        return self._bitmap


def _device(pressed_keys=(), step_result=None) -> IsaacTeleopDevice:
    device = object.__new__(IsaacTeleopDevice)
    device._session_lifecycle = MagicMock()
    device._session_lifecycle.drain_pressed_keys.return_value = list(pressed_keys)
    device._session_lifecycle.last_step_result = step_result
    device._prev_spacemouse_right_pressed = False
    device.request_start = MagicMock()
    device.request_stop = MagicMock()
    device.reset = MagicMock()
    return device


@pytest.mark.parametrize(("code", "action"), [(KEY_B, "request_start"), (KEY_P, "request_stop"), (KEY_R, "reset")])
def test_control_keys(code, action):
    device = _device(pressed_keys=[code])
    device._handle_operator_input()

    for name in ("request_start", "request_stop", "reset"):
        assert getattr(device, name).call_count == (name == action)
    if action == "reset":
        device.reset.assert_called_once_with(pause=True)


def test_other_keys_do_nothing():
    device = _device(pressed_keys=[KEY_N])
    device._handle_operator_input()

    device.request_start.assert_not_called()
    device.reset.assert_not_called()


def test_spacemouse_right_button_resets_once_per_press():
    device = _device(step_result={"spacemouse_buttons": _Buttons(1)})
    device._handle_operator_input()
    device._handle_operator_input()  # still held
    device.reset.assert_called_once_with(pause=True)

    device._session_lifecycle.last_step_result = {"spacemouse_buttons": _Buttons()}
    device._handle_operator_input()
    device._session_lifecycle.last_step_result = {"spacemouse_buttons": _Buttons(1)}
    device._handle_operator_input()
    assert device.reset.call_count == 2


def test_spacemouse_left_button_does_not_reset():
    device = _device(step_result={"spacemouse_buttons": _Buttons(0)})
    device._handle_operator_input()
    device.reset.assert_not_called()


@pytest.mark.parametrize("step_result", [None, {"action": object()}])
def test_without_spacemouse_buttons_nothing_resets(step_result):
    device = _device(step_result=step_result)
    device._handle_operator_input()
    device.reset.assert_not_called()
