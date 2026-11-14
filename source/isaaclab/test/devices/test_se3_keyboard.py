# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Key-event bookkeeping of :class:`Se3Keyboard` (Kit input interfaces are mocked)."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

# The fixture replaces carb and omni on the module, so stub them only for the import when Kit is absent.
_KIT_STUBS = {name: mock.MagicMock() for name in ("carb", "omni") if importlib.util.find_spec(name) is None}
sys.modules.update(_KIT_STUBS)
try:
    from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
    from isaaclab.devices.keyboard import se3_keyboard
finally:
    for _name in _KIT_STUBS:
        del sys.modules[_name]

PRESS, RELEASE = "KEY_PRESS", "KEY_RELEASE"
STEP = 0.05
HELD_KEYS = ["W", "A", "Q", "E", "Z", "X"]


@pytest.fixture
def keyboard(monkeypatch: pytest.MonkeyPatch) -> _Keyboard:
    return _Keyboard(monkeypatch)


def test_held_keys_drive_and_release_stops(keyboard: _Keyboard) -> None:
    """Held keys add up; releasing all of them stops the arm."""
    keyboard.press("W", "A", "Z")
    np.testing.assert_allclose(keyboard.motion(), [STEP, STEP, 0.0, STEP, 0.0, 0.0], atol=1e-7)
    keyboard.release("A", "W", "Z")
    np.testing.assert_array_equal(keyboard.motion(), np.zeros(6))


def test_release_of_key_held_across_reset_leaves_arm_still(keyboard: _Keyboard) -> None:
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
    keyboard.press("Q")
    np.testing.assert_allclose(keyboard.motion(), [0.0, 0.0, STEP, 0.0, 0.0, 0.0], atol=1e-7)
    keyboard.release("Q")
    np.testing.assert_array_equal(keyboard.motion(), np.zeros(6))


@pytest.mark.parametrize("key", ["Q", "X"])
def test_release_of_undelivered_press_is_ignored(keyboard: _Keyboard, key: str) -> None:
    """A release whose press never reached the device must not leave a constant motion command."""
    keyboard.release(key)
    np.testing.assert_array_equal(keyboard.motion(), np.zeros(6))
    keyboard.press(key)
    assert np.count_nonzero(keyboard.motion()) == 1
    keyboard.release(key)
    np.testing.assert_array_equal(keyboard.motion(), np.zeros(6))


@pytest.mark.parametrize("key", ["Q", "X"])
def test_undelivered_release_stops_the_key(keyboard: _Keyboard, key: str) -> None:
    """A key whose release never reached the device stops once the keyboard reports it up."""
    keyboard.press(key)
    assert np.count_nonzero(keyboard.motion()) == 1
    keyboard.drop_release(key)
    np.testing.assert_array_equal(keyboard.motion(), np.zeros(6))
    # a late release event has nothing left to remove
    keyboard.release(key)
    np.testing.assert_array_equal(keyboard.motion(), np.zeros(6))


def test_repeated_press_counts_once(keyboard: _Keyboard) -> None:
    """Two presses of a key without a release in between command its motion once."""
    keyboard.press("Q")
    keyboard.press("Q")
    np.testing.assert_allclose(keyboard.motion(), [0.0, 0.0, STEP, 0.0, 0.0, 0.0], atol=1e-7)
    keyboard.release("Q")
    np.testing.assert_array_equal(keyboard.motion(), np.zeros(6))


def test_reset_opens_gripper(keyboard: _Keyboard) -> None:
    """K toggles the gripper; a reset opens it again."""
    assert keyboard.device.advance()[6] == 1.0
    keyboard.press("K")
    keyboard.release("K")
    assert keyboard.device.advance()[6] == -1.0
    keyboard.device.reset()
    assert keyboard.device.advance()[6] == 1.0


class _Keyboard:
    """A keyboard device on a mocked Kit input interface.

    ``down`` is the physical key state the input interface reports; ``press``/``release`` change it and
    deliver the event through the subscribed callback, ``drop_release`` changes it without delivering the
    event (an event Kit consumed).
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.down: set[str] = set()
        carb = mock.MagicMock()
        carb.input.KeyboardEventType.KEY_PRESS = PRESS
        carb.input.KeyboardEventType.KEY_RELEASE = RELEASE
        for key in HELD_KEYS:
            setattr(carb.input.KeyboardInput, key, key)
        input_interface = carb.input.acquire_input_interface.return_value
        input_interface.get_keyboard_value.side_effect = lambda keyboard, key: 1.0 if key in self.down else 0.0
        monkeypatch.setattr(se3_keyboard, "carb", carb)
        monkeypatch.setattr(se3_keyboard, "omni", mock.MagicMock())
        self.device = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=STEP, rot_sensitivity=STEP, sim_device="cpu"))
        subscribe_call = input_interface.subscribe_to_keyboard_events.call_args
        self._on_event: Callable[[SimpleNamespace], bool] = subscribe_call.args[1]

    def press(self, *keys: str) -> None:
        for key in keys:
            self.down.add(key)
            self._send(PRESS, key)

    def release(self, *keys: str) -> None:
        for key in keys:
            self.down.discard(key)
            self._send(RELEASE, key)

    def drop_release(self, key: str) -> None:
        self.down.discard(key)

    def motion(self) -> np.ndarray:
        """The delta pose part of the command."""
        return self.device.advance().numpy()[:6]

    def _send(self, kind: str, key: str) -> None:
        self._on_event(SimpleNamespace(type=kind, input=SimpleNamespace(name=key)))
