# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for the Kit and Newton GL keyboard sources and the visualizers that own them.

Every backend runs against window, viewer and Kit doubles, so no display, Kit app or GPU is needed.
The contract they share is tested in ``source/isaaclab/test/visualizers/test_key_event_source.py``.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import isaaclab_visualizers.kit.kit_key_event_source as kit_key_event_source
import isaaclab_visualizers.newton.newton_key_event_source as newton_key_event_source
import isaaclab_visualizers.newton.newton_visualizer as newton_visualizer
import pytest
from isaaclab_visualizers.kit.kit_key_event_source import KitKeyEventSource, _carb_key_name_to_w3c
from isaaclab_visualizers.kit.kit_visualizer import KitVisualizer
from isaaclab_visualizers.newton.newton_key_event_source import NewtonKeyEventSource, _symbol_table, viewer_window
from isaaclab_visualizers.newton.newton_visualizer import NewtonVisualizer

from isaaclab.visualizers import KeyboardCapabilities, KeyEventSource

pytestmark = [pytest.mark.unit]


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
        kit = self
        keyboard = object()

        class _Observer:
            def __init__(self, callback) -> None:
                self._callback = callback

            def reset(self) -> None:
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
    monkeypatch.setattr(kit_key_event_source, "_kit", lambda: fake.namespace)
    return fake


def test_kit_hooks_keyboard_and_focus_once(kit):
    source = KitKeyEventSource()
    _, first = _listen(source)
    _, second = _listen(source)
    assert len(kit.keyboard_callbacks) == len(kit.focus_callbacks) == 1

    first.close()
    assert len(kit.keyboard_callbacks) == 1
    second.close()
    assert kit.keyboard_callbacks == kit.focus_callbacks == []
    assert (kit.unsubscribed, kit.observers_reset) == (1, 1)


def test_kit_reports_physical_key_presses_and_releases(kit):
    source = KitKeyEventSource()
    recorder, _ = _listen(source)
    results = [
        kit.key(kit.KEY_PRESS, "W"),
        kit.key(kit.KEY_REPEAT, "W"),
        kit.key(kit.KEY_RELEASE, "W"),
        kit.key(kit.KEY_PRESS, "UNKNOWN"),
    ]
    assert recorder.events == [("KeyW", True), ("KeyW", False)]
    assert results == [[True]] * 4  # Kit keeps processing every event
    assert source.capabilities == KeyboardCapabilities(physical_keys=True, filters_ui_text_input=False)


def test_kit_ignores_char_events_without_reading_the_key(kit):
    source = KitKeyEventSource()
    recorder, _ = _listen(source)
    assert kit.key(kit.CHAR) == [True]
    assert recorder.events == []


def test_kit_window_blur_is_a_focus_loss(kit):
    source = KitKeyEventSource()
    recorder, _ = _listen(source)
    kit.key(kit.KEY_PRESS, "W")
    kit.focus(True)
    assert recorder.focus_lost == 0
    kit.focus(False)
    kit.key(kit.KEY_RELEASE, "W")
    assert recorder.focus_lost == 1
    assert recorder.events == [("KeyW", True)]


def test_kit_failed_unhook_keeps_both_hooks(kit):
    source = KitKeyEventSource()
    _, subscription = _listen(source)
    kit.fail_unsubscribe = True
    with pytest.raises(RuntimeError, match="unsubscribe"):
        subscription.close()
    assert len(kit.keyboard_callbacks) == len(kit.focus_callbacks) == 1

    # still listening: a new listener gets blur without a second hook
    recorder, _ = _listen(source)
    kit.focus(False)
    assert recorder.focus_lost == 1
    assert len(kit.keyboard_callbacks) == 1


def test_kit_close_drops_the_focus_hook_when_unhooking_fails(kit):
    source = KitKeyEventSource()
    _listen(source)
    kit.fail_unsubscribe = True
    with pytest.raises(RuntimeError, match="unsubscribe"):
        source.close()
    assert kit.focus_callbacks == []
    assert kit.observers_reset == 1


def test_kit_focus_hook_failure_rolls_back_the_keyboard_hook(kit):
    source = KitKeyEventSource()
    kit.fail_observe = True
    with pytest.raises(RuntimeError, match="observe"):
        _listen(source)
    assert kit.keyboard_callbacks == []

    kit.fail_observe = False
    recorder, _ = _listen(source)
    kit.key(kit.KEY_PRESS, "W")
    assert recorder.events == [("KeyW", True)]


def test_kit_close_unhooks_and_releases_the_window(kit):
    source = KitKeyEventSource()
    recorder, _ = _listen(source)
    source.close()
    assert kit.keyboard_callbacks == kit.focus_callbacks == []
    assert recorder.focus_lost == 1
    assert source._input is None and source._keyboard is None


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
def test_kit_key_names_map_to_w3c_codes(carb_name, code):
    assert _carb_key_name_to_w3c()[carb_name] == code


# ---------------------------------------------------------------------------
# Newton
# ---------------------------------------------------------------------------


def _pyglet_keys() -> SimpleNamespace:
    """The constants of ``pyglet/window/key.py``, read without importing ``pyglet.window`` (needs a display)."""
    spec = importlib.util.find_spec("pyglet")
    if spec is None or spec.origin is None:
        pytest.skip("pyglet is not installed")
    tree = ast.parse((pathlib.Path(spec.origin).parent / "window" / "key.py").read_text())
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and type(node.value.value) is int:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value
    return SimpleNamespace(**constants)


@pytest.fixture
def keys(monkeypatch) -> SimpleNamespace:
    namespace = _pyglet_keys()
    table = _symbol_table(namespace)
    monkeypatch.setattr(newton_key_event_source, "_pyglet_symbol_to_w3c", lambda: table)
    return namespace


class _FakeWindow:
    """pyglet window double with a handler stack."""

    def __init__(self) -> None:
        self.frames: list[dict] = []

    def push_handlers(self, **handlers) -> None:
        self.frames.append(handlers)

    def remove_handlers(self, **handlers) -> None:
        self.frames.remove(handlers)

    def dispatch(self, name: str, *args) -> None:
        for frame in reversed(self.frames):
            if name in frame:
                frame[name](*args)
                return


class _FakeGui:
    def __init__(self) -> None:
        self.capturing = False

    def is_keyboard_capturing(self) -> bool:
        return self.capturing


def _newton_viewer(gui: _FakeGui | None = None) -> SimpleNamespace:
    return SimpleNamespace(gui=gui, camera_keys_suspended=False, keyboard_frame_hook=None)


def test_newton_forwards_window_keys(keys):
    window, viewer = _FakeWindow(), _newton_viewer(_FakeGui())
    source = NewtonKeyEventSource(viewer, window)
    recorder, subscription = _listen(source)
    assert len(window.frames) == 1 and viewer.keyboard_frame_hook is not None

    window.dispatch("on_key_press", keys.W, 0)
    window.dispatch("on_key_release", keys.W, 0)
    window.dispatch("on_key_press", keys.RETURN, 0)
    assert recorder.events == [("KeyW", True), ("KeyW", False), ("Enter", True)]

    subscription.close()
    assert window.frames == [] and viewer.keyboard_frame_hook is None


def test_newton_ui_taking_the_keyboard_is_a_focus_loss_on_the_next_frame(keys):
    window, gui = _FakeWindow(), _FakeGui()
    viewer = _newton_viewer(gui)
    source = NewtonKeyEventSource(viewer, window)
    recorder, _ = _listen(source)
    window.dispatch("on_key_press", keys.W, 0)

    gui.capturing = True  # e.g. a text field was clicked: no key press needed
    viewer.keyboard_frame_hook()
    viewer.keyboard_frame_hook()
    assert recorder.focus_lost == 1

    window.dispatch("on_key_press", keys.A, 0)  # typed into the UI
    gui.capturing = False
    viewer.keyboard_frame_hook()
    window.dispatch("on_key_release", keys.A, 0)
    window.dispatch("on_key_release", keys.W, 0)
    assert recorder.events == [("KeyW", True)]


def test_newton_viewer_without_gui_forwards_keys(keys):
    window = _FakeWindow()
    source = NewtonKeyEventSource(_newton_viewer(gui=None), window)
    recorder, _ = _listen(source)
    window.dispatch("on_key_press", keys.W, 0)
    assert recorder.events == [("KeyW", True)]


@pytest.mark.parametrize(
    ("unshifted", "shifted", "code"),
    [("EQUAL", "PLUS", "Equal"), ("_1", "EXCLAMATION", "Digit1"), ("SLASH", "QUESTION", "Slash")],
)
@pytest.mark.parametrize("shift_first", [False, True], ids=["shift-after-press", "shift-before-press"])
def test_newton_key_released_after_shift_changes(keys, unshifted, shifted, code, shift_first):
    """X11 names a digit or punctuation key by its shifted symbol while Shift is held."""
    window = _FakeWindow()
    source = NewtonKeyEventSource(_newton_viewer(_FakeGui()), window)
    recorder, _ = _listen(source)
    press, release = (shifted, unshifted) if shift_first else (unshifted, shifted)

    window.dispatch("on_key_press", getattr(keys, press), 0)
    window.dispatch("on_key_release", getattr(keys, release), 0)

    assert recorder.events == [(code, True), (code, False)]


def test_newton_window_deactivation_is_a_focus_loss(keys):
    window = _FakeWindow()
    source = NewtonKeyEventSource(_newton_viewer(_FakeGui()), window)
    recorder, _ = _listen(source)
    window.dispatch("on_key_press", keys.W, 0)
    window.dispatch("on_deactivate")
    window.dispatch("on_key_release", keys.W, 0)
    assert recorder.focus_lost == 1
    assert recorder.events == [("KeyW", True)]


def test_newton_failed_unhook_keeps_the_frame_hook(keys):
    window, viewer = _FakeWindow(), _newton_viewer(_FakeGui())
    source = NewtonKeyEventSource(viewer, window)
    _, subscription = _listen(source)

    def fail(**handlers):
        raise RuntimeError("remove_handlers")

    window.remove_handlers = fail
    with pytest.raises(RuntimeError, match="remove_handlers"):
        subscription.close()
    assert len(window.frames) == 1 and viewer.keyboard_frame_hook is not None


def test_newton_capture_suspends_camera_keys_until_close(keys):
    window, viewer = _FakeWindow(), _newton_viewer(_FakeGui())
    source = NewtonKeyEventSource(viewer, window)
    recorder, _ = _listen(source)
    source.capture_keyboard()
    assert viewer.camera_keys_suspended

    source.close()
    assert not viewer.camera_keys_suspended
    assert window.frames == [] and viewer.keyboard_frame_hook is None
    assert recorder.focus_lost == 1
    assert source._viewer is None and source._window is None


def test_newton_keys_are_layout_mapped():
    assert NewtonKeyEventSource.capabilities == KeyboardCapabilities(physical_keys=False, filters_ui_text_input=True)


def test_viewer_window_is_none_without_a_window():
    window = object()
    assert viewer_window(SimpleNamespace(renderer=SimpleNamespace(headless=False, window=window))) is window
    assert viewer_window(SimpleNamespace(renderer=SimpleNamespace(headless=True, window=window))) is None
    assert viewer_window(SimpleNamespace()) is None


def test_gl_viewer_polls_keyboard_ownership_after_each_frame(monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(newton_visualizer.ViewerGL, "end_frame", lambda self: events.append("frame"))
    viewer = object.__new__(newton_visualizer.NewtonViewerGL)
    viewer._close_requested = False
    viewer.keyboard_frame_hook = lambda: events.append("poll")

    viewer.end_frame()
    viewer.keyboard_frame_hook = None
    viewer.end_frame()
    assert events == ["frame", "poll", "frame"]


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
def test_pyglet_symbols_map_to_w3c_codes(pyglet_name, code):
    keys = _pyglet_keys()
    assert _symbol_table(keys)[getattr(keys, pyglet_name)] == code


# ---------------------------------------------------------------------------
# Visualizer ownership
# ---------------------------------------------------------------------------


def _newton_gl_viewer(headless: bool) -> newton_visualizer.NewtonViewerGL:
    viewer = object.__new__(newton_visualizer.NewtonViewerGL)
    viewer.renderer = SimpleNamespace(headless=headless, window=_FakeWindow())
    viewer.gui = None
    viewer.camera_keys_suspended = False
    viewer.keyboard_frame_hook = None
    return viewer


def _newton_visualizer(viewer) -> NewtonVisualizer:
    visualizer = object.__new__(NewtonVisualizer)
    visualizer._viewer = viewer
    visualizer._key_event_source = None
    visualizer._key_input_closed = False
    visualizer._picking_enabled = False
    return visualizer


def test_newton_visualizer_key_event_source_lifecycle(keys):
    assert _newton_visualizer(None).key_event_source is None  # not initialized
    assert _newton_visualizer(_newton_gl_viewer(headless=True)).key_event_source is None
    rtx = object.__new__(newton_visualizer.NewtonViewerRTX)
    assert _newton_visualizer(rtx).key_event_source is None

    viewer = _newton_gl_viewer(headless=False)
    visualizer = _newton_visualizer(viewer)
    source = visualizer.key_event_source
    assert isinstance(source, NewtonKeyEventSource) and visualizer.key_event_source is source
    recorder, _ = _listen(source)
    source.capture_keyboard()

    visualizer._release_viewer()
    assert source.closed and recorder.focus_lost == 1
    assert not viewer.camera_keys_suspended and viewer.keyboard_frame_hook is None
    visualizer._viewer = viewer
    assert visualizer.key_event_source is None  # closed for good


def _kit_visualizer(monkeypatch, *, initialized: bool, headless: bool) -> KitVisualizer:
    visualizer = object.__new__(KitVisualizer)
    visualizer._is_initialized = initialized
    visualizer._is_closed = False
    visualizer._runtime_headless = headless
    visualizer._key_event_source = None
    visualizer._key_input_closed = False
    visualizer._streaming_camera_key = None
    visualizer._camera_sensor = None
    visualizer._camera_is_owned = False
    visualizer._generated_camera_xform_ops = {}
    visualizer._generated_camera_pose_cache = {}
    visualizer._rgb_annotator = None
    visualizer._rgb_render_product = None
    visualizer.torn_down = []
    monkeypatch.setattr(visualizer, "_teardown_backend_menubar_label", lambda: visualizer.torn_down.append("menu"))
    monkeypatch.setattr(visualizer, "_restore_env_visibility", lambda: visualizer.torn_down.append("visibility"))
    return visualizer


def test_kit_visualizer_key_event_source_lifecycle(monkeypatch, kit):
    assert _kit_visualizer(monkeypatch, initialized=False, headless=False).key_event_source is None
    assert _kit_visualizer(monkeypatch, initialized=True, headless=True).key_event_source is None

    visualizer = _kit_visualizer(monkeypatch, initialized=True, headless=False)
    source = visualizer.key_event_source
    assert isinstance(source, KitKeyEventSource) and visualizer.key_event_source is source
    recorder, _ = _listen(source)

    visualizer.close()
    assert source.closed and recorder.focus_lost == 1
    assert kit.keyboard_callbacks == []
    visualizer._is_initialized = True
    assert visualizer.key_event_source is None  # closed for good


def test_kit_visualizer_close_survives_a_failing_key_source(monkeypatch, kit, caplog):
    visualizer = _kit_visualizer(monkeypatch, initialized=True, headless=False)
    _listen(visualizer.key_event_source)
    kit.fail_unsubscribe = True

    with caplog.at_level(logging.ERROR):
        visualizer.close()
    assert visualizer.torn_down == ["menu", "visibility"]
    assert visualizer._is_closed and visualizer.key_event_source is None
    assert "Keyboard input cleanup failed" in caplog.text


def test_backends_import_without_a_display_or_kit():
    code = (
        "import sys\n"
        "from isaaclab_visualizers.kit.kit_key_event_source import KitKeyEventSource\n"
        "from isaaclab_visualizers.newton.newton_key_event_source import NewtonKeyEventSource\n"
        "KitKeyEventSource(); NewtonKeyEventSource(object(), object())\n"
        "assert 'pyglet.window' not in sys.modules, 'pyglet.window'\n"
        "assert 'carb' not in sys.modules, 'carb'\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
