# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Key-event bookkeeping of :class:`Se3Keyboard` (Kit input interfaces are mocked)."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.devices.keyboard import se3_keyboard

PRESS, RELEASE = "KEY_PRESS", "KEY_RELEASE"
STEP = 0.05
KEYS = ["W", "S", "A", "D", "Q", "E", "Z", "X", "T", "G", "C", "V", "K", "R", "L"]


class Keyboard:
    """A keyboard device on a mocked Kit input interface.

    ``down`` is the physical key state the input interface reports; ``press``/``release`` change it and
    deliver the event, ``drop_*`` change it without delivering the event (an event Kit consumed).
    """

    def __init__(self, monkeypatch):
        self.down: set[str] = set()
        carb = mock.MagicMock()
        carb.input.KeyboardEventType.KEY_PRESS = PRESS
        carb.input.KeyboardEventType.KEY_RELEASE = RELEASE
        for key in KEYS:
            setattr(carb.input.KeyboardInput, key, key)
        carb.input.acquire_input_interface.return_value.get_keyboard_value.side_effect = (
            lambda keyboard, key: 1.0 if key in self.down else 0.0
        )
        monkeypatch.setattr(se3_keyboard, "carb", carb)
        monkeypatch.setattr(se3_keyboard, "omni", mock.MagicMock())
        self.device = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=STEP, rot_sensitivity=STEP, sim_device="cpu"))

    def _send(self, kind: str, key: str):
        self.device._on_keyboard_event(SimpleNamespace(type=kind, input=SimpleNamespace(name=key)))

    def press(self, *keys: str):
        for key in keys:
            self.down.add(key)
            self._send(PRESS, key)

    def release(self, *keys: str):
        for key in keys:
            self.down.discard(key)
            self._send(RELEASE, key)

    def drop_press(self, key: str):
        self.down.add(key)

    def drop_release(self, key: str):
        self.down.discard(key)

    def motion(self) -> np.ndarray:
        """The delta pose part of the command."""
        return self.device.advance().numpy()[:6]


@pytest.fixture
def keyboard(monkeypatch):
    return Keyboard(monkeypatch)


def test_held_keys_drive_and_release_stops(keyboard):
    """Held keys add up; releasing all of them stops the arm."""
    keyboard.press("W", "A", "Z")
    np.testing.assert_allclose(keyboard.motion(), [STEP, STEP, 0.0, STEP, 0.0, 0.0], atol=1e-7)
    keyboard.release("A", "W", "Z")
    np.testing.assert_array_equal(keyboard.motion(), np.zeros(6))


def test_release_of_key_held_across_reset_leaves_arm_still(keyboard):
    """Operator holds a motion key, presses R (reset), then lets go: no motion after the reset."""
    reset_requested = []
    keyboard.device.add_callback("R", lambda: reset_requested.append(True))
    keyboard.press("E")
    keyboard.press("R")
    # teleop_se3_agent.py resets the device on the loop iteration after the R callback
    assert reset_requested
    keyboard.device.reset()
    np.testing.assert_array_equal(keyboard.motion(), np.zeros(6))
    keyboard.release("R", "E")
    np.testing.assert_array_equal(keyboard.motion(), np.zeros(6))
    # the next episode drives normally
    keyboard.press("Q")
    np.testing.assert_allclose(keyboard.motion(), [0.0, 0.0, STEP, 0.0, 0.0, 0.0], atol=1e-7)
    keyboard.release("Q")
    np.testing.assert_array_equal(keyboard.motion(), np.zeros(6))


@pytest.mark.parametrize("key", ["Q", "X"])
def test_release_of_undelivered_press_is_ignored(keyboard, key):
    """A release whose press never reached the device must not leave a constant motion command."""
    keyboard.drop_press(key)
    keyboard.release(key)
    np.testing.assert_array_equal(keyboard.motion(), np.zeros(6))
    keyboard.press(key)
    assert np.count_nonzero(keyboard.motion()) == 1
    keyboard.release(key)
    np.testing.assert_array_equal(keyboard.motion(), np.zeros(6))


@pytest.mark.parametrize("key", ["Q", "X"])
def test_undelivered_release_stops_the_key(keyboard, key):
    """A key whose release never reached the device stops once the keyboard reports it up."""
    keyboard.press(key)
    assert np.count_nonzero(keyboard.motion()) == 1
    keyboard.drop_release(key)
    np.testing.assert_array_equal(keyboard.motion(), np.zeros(6))
    # a late release event has nothing left to remove
    keyboard.release(key)
    np.testing.assert_array_equal(keyboard.motion(), np.zeros(6))


def test_repeated_press_counts_once(keyboard):
    """Two presses of a key without a release in between command its motion once."""
    keyboard.press("Q")
    keyboard.press("Q")
    np.testing.assert_allclose(keyboard.motion(), [0.0, 0.0, STEP, 0.0, 0.0, 0.0], atol=1e-7)
    keyboard.release("Q")
    np.testing.assert_array_equal(keyboard.motion(), np.zeros(6))


def test_reset_opens_gripper(keyboard):
    """K toggles the gripper; a reset opens it again."""
    assert keyboard.device.advance()[6] == 1.0
    keyboard.press("K")
    keyboard.release("K")
    assert keyboard.device.advance()[6] == -1.0
    keyboard.device.reset()
    assert keyboard.device.advance()[6] == 1.0
