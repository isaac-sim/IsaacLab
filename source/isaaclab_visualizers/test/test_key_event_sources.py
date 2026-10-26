# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for the Kit and Newton keyboard sources over their native windows' lifetimes.

Every backend runs against window, viewer and Kit doubles, so no display, Kit app or GPU is needed.
The contract they share is tested in ``source/isaaclab/test/visualizers/test_key_event_source.py``.
"""

from __future__ import annotations

import os
import pathlib
import runpy
import subprocess
import sys
from types import SimpleNamespace

import isaaclab_visualizers.newton.newton_key_event_source as newton_key_event_source
import pyglet
import pytest
from isaaclab_visualizers.kit.kit_key_event_source import KitKeyEventSource
from isaaclab_visualizers.newton.newton_key_event_source import NewtonKeyEventSource
from pyglet.event import EventDispatcher

from isaaclab.visualizers import KeyEventSource

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


class _Recorder:
    """Listener that records what a source reports."""

    def __init__(self) -> None:
        self.events: list[tuple[str, bool]] = []
        self.focus_lost = 0

    def on_key(self, code: str, pressed: bool) -> None:
        self.events.append((code, pressed))

    def on_focus_lost(self) -> None:
        self.focus_lost += 1


def _listen(source: KeyEventSource):
    recorder = _Recorder()
    return recorder, source.add_key_listener(recorder.on_key, recorder.on_focus_lost)


# ---------------------------------------------------------------------------
# Kit
# ---------------------------------------------------------------------------


class _CarbKeyEvent:
    def __init__(self, event_type: int, key: str | None = None) -> None:
        self.type = event_type
        self._key = key

    @property
    def input(self):
        if self._key is None:
            raise AssertionError("read the key of a CHAR event")
        return SimpleNamespace(name=self._key)


class _FakeKit:
    """``carb.input``, ``omni.appwindow`` and the event dispatcher, as the Kit source uses them."""

    KEY_PRESS, KEY_RELEASE, KEY_REPEAT, CHAR = range(4)

    def __init__(self) -> None:
        self.keyboard_callbacks: list = []
        self.focus_callbacks: list = []
        self.unsubscribed = 0
        self.observers_reset = 0
        self.fail_unsubscribe = False
        self.fail_observe = False
        self.fail_reset = False
        kit = self
        keyboard = object()

        class _Observer:
            def __init__(self, callback) -> None:
                self._callback = callback

            def reset(self) -> None:
                if kit.fail_reset:
                    raise RuntimeError("reset")
                kit.observers_reset += 1
                kit.focus_callbacks.remove(self._callback)

        def subscribe(device, callback):
            assert device is keyboard
            kit.keyboard_callbacks.append(callback)
            return callback

        def unsubscribe(device, subscription):
            assert device is keyboard
            if kit.fail_unsubscribe:
                raise RuntimeError("unsubscribe")
            kit.unsubscribed += 1
            kit.keyboard_callbacks.remove(subscription)

        def observe_event(*, filter, event_name, on_event, observer_name):
            assert (filter, event_name) == ("app-window", "window-focus")
            if kit.fail_observe:
                raise RuntimeError("observe")
            kit.focus_callbacks.append(on_event)
            return _Observer(on_event)

        interface = SimpleNamespace(subscribe_to_keyboard_events=subscribe, unsubscribe_to_keyboard_events=unsubscribe)
        window = SimpleNamespace(get_keyboard=lambda: keyboard, get_event_key=lambda: "app-window")
        self.window = window
        self.namespace = SimpleNamespace(
            input=SimpleNamespace(
                acquire_input_interface=lambda: interface,
                KeyboardEventType=SimpleNamespace(
                    KEY_PRESS=self.KEY_PRESS, KEY_RELEASE=self.KEY_RELEASE, KEY_REPEAT=self.KEY_REPEAT, CHAR=self.CHAR
                ),
            ),
            appwindow=SimpleNamespace(get_default_app_window=lambda: window, GLOBAL_EVENT_WINDOW_FOCUS="window-focus"),
            dispatcher=SimpleNamespace(observe_event=observe_event),
        )

    def key(self, event_type: int, key: str | None = None) -> list[bool]:
        return [callback(_CarbKeyEvent(event_type, key)) for callback in self.keyboard_callbacks]

    def focus(self, focused: bool) -> None:
        for callback in list(self.focus_callbacks):
            callback({"isFocused": focused})


@pytest.fixture
def kit(monkeypatch) -> _FakeKit:
    fake = _FakeKit()
    monkeypatch.setitem(sys.modules, "carb", SimpleNamespace(input=fake.namespace.input))
    monkeypatch.setitem(sys.modules, "carb.input", fake.namespace.input)
    monkeypatch.setitem(
        sys.modules, "carb.eventdispatcher", SimpleNamespace(get_eventdispatcher=lambda: fake.namespace.dispatcher)
    )
    monkeypatch.setitem(sys.modules, "omni", SimpleNamespace(appwindow=fake.namespace.appwindow))
    monkeypatch.setitem(sys.modules, "omni.appwindow", fake.namespace.appwindow)
    return fake


def test_kit_hooks_keyboard_and_focus_once(kit):
    source = KitKeyEventSource(kit.window)
    _, first = _listen(source)
    _, second = _listen(source)
    assert len(kit.keyboard_callbacks) == len(kit.focus_callbacks) == 1

    first.close()
    assert len(kit.keyboard_callbacks) == 1
    second.close()
    assert len(kit.keyboard_callbacks) == len(kit.focus_callbacks) == 1
    recorder, _ = _listen(source)
    kit.key(kit.KEY_PRESS, "W")
    assert recorder.events == [("KeyW", True)]
    source.close()
    assert kit.keyboard_callbacks == kit.focus_callbacks == []
    assert (kit.unsubscribed, kit.observers_reset) == (1, 1)


def test_kit_reports_physical_key_presses_and_releases(kit):
    source = KitKeyEventSource(kit.window)
    recorder, _ = _listen(source)
    results = [
        kit.key(kit.KEY_PRESS, "W"),
        kit.key(kit.KEY_REPEAT, "W"),
        kit.key(kit.KEY_RELEASE, "W"),
        kit.key(kit.KEY_PRESS, "UNKNOWN"),
    ]
    assert recorder.events == [("KeyW", True), ("KeyW", True), ("KeyW", False)]
    assert results == [[True]] * 4  # Kit keeps processing every event


def test_kit_ignores_char_events_without_reading_the_key(kit):
    source = KitKeyEventSource(kit.window)
    recorder, _ = _listen(source)
    assert kit.key(kit.CHAR) == [True]
    assert recorder.events == []


def test_kit_window_blur_is_a_focus_loss(kit):
    source = KitKeyEventSource(kit.window)
    recorder, _ = _listen(source)
    kit.key(kit.KEY_PRESS, "W")
    kit.focus(True)
    assert recorder.focus_lost == 0
    kit.focus(False)
    assert recorder.focus_lost == 1
    assert recorder.events == [("KeyW", True)]


def test_kit_close_drops_the_focus_hook_when_unhooking_fails(kit):
    source = KitKeyEventSource(kit.window)
    _listen(source)
    kit.fail_unsubscribe = True
    with pytest.raises(RuntimeError, match="unsubscribe"):
        source.close()
    assert kit.focus_callbacks == []
    assert kit.observers_reset == 1


def test_kit_focus_hook_failure_rolls_back_the_keyboard_hook(kit):
    kit.fail_observe = True
    with pytest.raises(RuntimeError, match="observe"):
        KitKeyEventSource(kit.window)
    assert kit.keyboard_callbacks == []

    kit.fail_observe = False
    source = KitKeyEventSource(kit.window)
    recorder, _ = _listen(source)
    kit.key(kit.KEY_PRESS, "W")
    assert recorder.events == [("KeyW", True)]


def test_kit_close_unhooks_and_releases_the_window(kit):
    source = KitKeyEventSource(kit.window)
    recorder, _ = _listen(source)
    source.close()
    assert kit.keyboard_callbacks == kit.focus_callbacks == []
    assert recorder.focus_lost == 1
    assert source._input is None and source._keyboard is None


def test_kit_failed_focus_cleanup_still_closes_and_notifies(kit):
    source = KitKeyEventSource(kit.window)
    recorder, subscription = _listen(source)
    kit.fail_reset = True
    with pytest.raises(RuntimeError, match="reset"):
        source.close()
    assert kit.keyboard_callbacks == []
    assert source.closed and subscription.closed and recorder.focus_lost == 1
    source.close()


@pytest.mark.parametrize(
    ("carb_name", "code"),
    [
        ("W", "KeyW"),
        ("KEY_7", "Digit7"),
        ("NUMPAD_3", "Numpad3"),
        ("F12", "F12"),
        ("DEL", "Delete"),
        ("NUMPAD_DEL", "NumpadDecimal"),
        ("LEFT_SHIFT", "ShiftLeft"),
        ("RIGHT_SUPER", "MetaRight"),
        ("APOSTROPHE", "Quote"),
        ("GRAVE_ACCENT", "Backquote"),
    ],
)
def test_kit_key_names_map_to_w3c_codes(kit, carb_name, code):
    source = KitKeyEventSource(kit.window)
    recorder, _ = _listen(source)
    kit.key(kit.KEY_PRESS, carb_name)
    assert recorder.events == [(code, True)]


# ---------------------------------------------------------------------------
# Newton
# ---------------------------------------------------------------------------


@pytest.fixture
def keys(monkeypatch):
    # Load the real constants without importing pyglet.window, which opens a display.
    namespace = SimpleNamespace(**runpy.run_path(str(pathlib.Path(pyglet.__file__).parent / "window/key.py")))
    monkeypatch.setitem(sys.modules, "pyglet.window", SimpleNamespace(key=namespace))
    newton_key_event_source._pyglet_symbol_to_w3c.cache_clear()
    yield namespace
    newton_key_event_source._pyglet_symbol_to_w3c.cache_clear()


class _Window(EventDispatcher):
    """Real pyglet event dispatch without a native display."""


_Window.register_event_type("on_key_press")
_Window.register_event_type("on_key_release")
_Window.register_event_type("on_deactivate")


class _FakeGui:
    def __init__(self) -> None:
        self.capturing = False

    def is_keyboard_capturing(self) -> bool:
        return self.capturing


def _newton_viewer(window: _Window, gui: _FakeGui | None = None) -> SimpleNamespace:
    return SimpleNamespace(renderer=SimpleNamespace(window=window), gui=gui, keyboard_frame_hook=None)


def test_newton_forwards_window_keys(keys):
    window = _Window()
    viewer = _newton_viewer(window, _FakeGui())
    source = NewtonKeyEventSource(viewer)
    recorder, subscription = _listen(source)
    assert len(window._event_stack) == 1 and viewer.keyboard_frame_hook is not None

    window.dispatch_event("on_key_press", keys.W, 0)
    window.dispatch_event("on_key_release", keys.W, 0)
    window.dispatch_event("on_key_press", keys.RETURN, 0)
    assert recorder.events == [("KeyW", True), ("KeyW", False), ("Enter", True)]

    subscription.close()
    assert len(window._event_stack) == 1 and viewer.keyboard_frame_hook is not None
    recorder, _ = _listen(source)
    window.dispatch_event("on_key_press", keys.W, 0)
    assert recorder.events == [("KeyW", True)]
    source.close()
    assert window._event_stack == [] and viewer.keyboard_frame_hook is None


def test_newton_ui_taking_the_keyboard_is_a_focus_loss_on_the_next_frame(keys):
    window, gui = _Window(), _FakeGui()
    viewer = _newton_viewer(window, gui)
    source = NewtonKeyEventSource(viewer)
    recorder, _ = _listen(source)
    window.dispatch_event("on_key_press", keys.W, 0)

    gui.capturing = True  # e.g. a text field was clicked: no key press needed
    viewer.keyboard_frame_hook()
    viewer.keyboard_frame_hook()
    assert recorder.focus_lost == 1

    window.dispatch_event("on_key_press", keys.A, 0)  # typed into the UI
    gui.capturing = False
    viewer.keyboard_frame_hook()
    window.dispatch_event("on_key_release", keys.A, 0)
    assert recorder.events == [("KeyW", True), ("KeyA", False)]  # the press stayed in the UI


def test_newton_viewer_without_gui_forwards_keys(keys):
    window = _Window()
    source = NewtonKeyEventSource(_newton_viewer(window, gui=None))
    recorder, _ = _listen(source)
    window.dispatch_event("on_key_press", keys.W, 0)
    assert recorder.events == [("KeyW", True)]


@pytest.mark.parametrize(
    ("unshifted", "shifted", "code"),
    [("EQUAL", "PLUS", "Equal"), ("_1", "EXCLAMATION", "Digit1"), ("SLASH", "QUESTION", "Slash")],
)
@pytest.mark.parametrize("shift_first", [False, True], ids=["shift-after-press", "shift-before-press"])
def test_newton_key_released_after_shift_changes(keys, unshifted, shifted, code, shift_first):
    """X11 names a digit or punctuation key by its shifted symbol while Shift is held."""
    window = _Window()
    source = NewtonKeyEventSource(_newton_viewer(window, _FakeGui()))
    recorder, _ = _listen(source)
    press, release = (shifted, unshifted) if shift_first else (unshifted, shifted)

    window.dispatch_event("on_key_press", getattr(keys, press), 0)
    window.dispatch_event("on_key_release", getattr(keys, release), 0)

    assert recorder.events == [(code, True), (code, False)]


def test_newton_window_deactivation_is_a_focus_loss(keys):
    window = _Window()
    source = NewtonKeyEventSource(_newton_viewer(window, _FakeGui()))
    recorder, _ = _listen(source)
    window.dispatch_event("on_key_press", keys.W, 0)
    window.dispatch_event("on_deactivate")
    assert recorder.focus_lost == 1
    assert recorder.events == [("KeyW", True)]


def test_newton_failed_unhook_still_closes_and_notifies(keys):
    window = _Window()
    viewer = _newton_viewer(window, _FakeGui())
    source = NewtonKeyEventSource(viewer)
    recorder, subscription = _listen(source)

    def fail(*handlers):
        raise RuntimeError("remove_handlers")

    window.remove_handlers = fail
    with pytest.raises(RuntimeError, match="remove_handlers"):
        source.close()
    assert viewer.keyboard_frame_hook is None  # the closed source is never polled again
    assert source.closed and subscription.closed and recorder.focus_lost == 1
    window.dispatch_event("on_key_press", keys.W, 0)  # a leftover native hook is inert
    assert recorder.events == []


def test_newton_close_unhooks_and_releases_the_window(keys):
    window = _Window()
    viewer = _newton_viewer(window, _FakeGui())
    source = NewtonKeyEventSource(viewer)
    recorder, _ = _listen(source)

    source.close()
    assert window._event_stack == [] and viewer.keyboard_frame_hook is None
    assert recorder.focus_lost == 1
    assert source._viewer is None and source._window is None


@pytest.mark.parametrize(
    ("pyglet_name", "code"),
    [
        ("W", "KeyW"),
        ("_7", "Digit7"),
        ("NUM_3", "Numpad3"),
        ("F12", "F12"),
        ("RETURN", "Enter"),
        ("LSHIFT", "ShiftLeft"),
        ("RMETA", "MetaRight"),
        ("BRACKETLEFT", "BracketLeft"),
        ("APOSTROPHE", "Quote"),
        ("GRAVE", "Backquote"),
    ],
)
def test_pyglet_symbols_map_to_w3c_codes(keys, pyglet_name, code):
    assert newton_key_event_source._pyglet_symbol_to_w3c()[getattr(keys, pyglet_name)] == code


# ---------------------------------------------------------------------------
# Import boundary
# ---------------------------------------------------------------------------


def test_backends_import_without_a_display_or_kit():
    code = (
        "import sys\n"
        "from isaaclab_visualizers.kit.kit_key_event_source import KitKeyEventSource\n"
        "from isaaclab_visualizers.newton.newton_key_event_source import NewtonKeyEventSource\n"
        "assert 'pyglet.window' not in sys.modules, 'pyglet.window'\n"
        "assert 'carb' not in sys.modules, 'carb'\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
