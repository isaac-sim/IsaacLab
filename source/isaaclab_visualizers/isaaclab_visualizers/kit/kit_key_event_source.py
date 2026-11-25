# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard event surface for the Kit application window."""

from __future__ import annotations

from typing import Any

from isaaclab.visualizers import KeyEventSource

_CARB_KEY_CODES = {
    **{letter: f"Key{letter}" for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
    **{f"KEY_{digit}": f"Digit{digit}" for digit in "0123456789"},
    **{f"NUMPAD_{digit}": f"Numpad{digit}" for digit in "0123456789"},
    **{f"F{n}": f"F{n}" for n in range(1, 13)},
    "SPACE": "Space",
    "ENTER": "Enter",
    "ESCAPE": "Escape",
    "TAB": "Tab",
    "BACKSPACE": "Backspace",
    "DEL": "Delete",
    "INSERT": "Insert",
    "HOME": "Home",
    "END": "End",
    "PAGE_UP": "PageUp",
    "PAGE_DOWN": "PageDown",
    "UP": "ArrowUp",
    "DOWN": "ArrowDown",
    "LEFT": "ArrowLeft",
    "RIGHT": "ArrowRight",
    "LEFT_SHIFT": "ShiftLeft",
    "RIGHT_SHIFT": "ShiftRight",
    "LEFT_CONTROL": "ControlLeft",
    "RIGHT_CONTROL": "ControlRight",
    "LEFT_ALT": "AltLeft",
    "RIGHT_ALT": "AltRight",
    "LEFT_SUPER": "MetaLeft",
    "RIGHT_SUPER": "MetaRight",
    "CAPS_LOCK": "CapsLock",
    "MINUS": "Minus",
    "EQUAL": "Equal",
    "LEFT_BRACKET": "BracketLeft",
    "RIGHT_BRACKET": "BracketRight",
    "SEMICOLON": "Semicolon",
    "APOSTROPHE": "Quote",
    "GRAVE_ACCENT": "Backquote",
    "BACKSLASH": "Backslash",
    "COMMA": "Comma",
    "PERIOD": "Period",
    "SLASH": "Slash",
    "NUMPAD_ADD": "NumpadAdd",
    "NUMPAD_SUBTRACT": "NumpadSubtract",
    "NUMPAD_MULTIPLY": "NumpadMultiply",
    "NUMPAD_DIVIDE": "NumpadDivide",
    "NUMPAD_DEL": "NumpadDecimal",
    "NUMPAD_ENTER": "NumpadEnter",
    "NUMPAD_EQUAL": "NumpadEqual",
}


class KitKeyEventSource(KeyEventSource):
    """Keys typed into the Kit application window.

    Reads the supplied app window's ``carb.input`` keyboard, whose codes name physical key positions,
    for the window's lifetime, and reports a focus loss when the window loses OS focus. The raw
    keyboard stream also carries keys typed into Kit's own UI (text fields included), and Kit has no
    Python query for UI keyboard ownership, so they are reported too.
    """

    def __init__(self, app_window: Any) -> None:
        """Connect the initialized Kit window's keyboard and focus events."""
        import carb.input
        import omni.appwindow
        from carb.eventdispatcher import get_eventdispatcher

        super().__init__()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = app_window.get_keyboard()
        self._event_types = carb.input.KeyboardEventType
        self._keyboard_subscription = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        try:
            self._focus_observer = get_eventdispatcher().observe_event(
                filter=app_window.get_event_key(),
                event_name=omni.appwindow.GLOBAL_EVENT_WINDOW_FOCUS,
                on_event=self._on_focus_event,
                observer_name="isaaclab_visualizers.KitKeyEventSource",
            )
        except BaseException:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_subscription)
            raise

    def close(self) -> None:
        """Disconnect the window and notify listeners, including when native cleanup fails."""
        if self.closed:
            return
        try:
            try:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_subscription)
            finally:
                self._focus_observer.reset()
        finally:
            self._input = self._keyboard = self._keyboard_subscription = self._focus_observer = None
            super().close()

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        if self.closed:
            return True
        event_types = self._event_types
        # KeyboardEvent is a union: CHAR events carry a character, not a key.
        if event.type in (event_types.KEY_PRESS, event_types.KEY_REPEAT):
            pressed = True
        elif event.type == event_types.KEY_RELEASE:
            pressed = False
        else:
            return True
        code = _CARB_KEY_CODES.get(event.input.name)
        if code is not None:
            self._emit_key(code, pressed)
        # let Kit keep processing the event
        return True

    def _on_focus_event(self, event) -> None:
        if not event["isFocused"]:
            self._emit_focus_lost()
