# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard event surface for the Newton viewer window, shared by the GL and RTX visualizers."""

from __future__ import annotations

import functools
from typing import Any

from isaaclab.visualizers import KeyEventSource


@functools.cache
def _pyglet_symbol_to_w3c() -> dict[int, str]:
    """Map pyglet key symbols to W3C ``KeyboardEvent.code`` strings (imports ``pyglet.window``)."""
    from pyglet.window import key

    table: dict[int, str] = {}
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        table[getattr(key, letter)] = f"Key{letter}"
    for digit in "0123456789":
        table[getattr(key, f"_{digit}")] = f"Digit{digit}"
        table[getattr(key, f"NUM_{digit}")] = f"Numpad{digit}"
    for n in range(1, 13):
        table[getattr(key, f"F{n}")] = f"F{n}"
    named = {
        "SPACE": "Space",
        "RETURN": "Enter",
        "ESCAPE": "Escape",
        "TAB": "Tab",
        "BACKSPACE": "Backspace",
        "DELETE": "Delete",
        "INSERT": "Insert",
        "HOME": "Home",
        "END": "End",
        "PAGEUP": "PageUp",
        "PAGEDOWN": "PageDown",
        "UP": "ArrowUp",
        "DOWN": "ArrowDown",
        "LEFT": "ArrowLeft",
        "RIGHT": "ArrowRight",
        "LSHIFT": "ShiftLeft",
        "RSHIFT": "ShiftRight",
        "LCTRL": "ControlLeft",
        "RCTRL": "ControlRight",
        "LALT": "AltLeft",
        "RALT": "AltRight",
        "LMETA": "MetaLeft",
        "RMETA": "MetaRight",
        "CAPSLOCK": "CapsLock",
        "MINUS": "Minus",
        "EQUAL": "Equal",
        "BRACKETLEFT": "BracketLeft",
        "BRACKETRIGHT": "BracketRight",
        "SEMICOLON": "Semicolon",
        "APOSTROPHE": "Quote",
        "GRAVE": "Backquote",
        "BACKSLASH": "Backslash",
        "COMMA": "Comma",
        "PERIOD": "Period",
        "SLASH": "Slash",
        "NUM_ADD": "NumpadAdd",
        "NUM_SUBTRACT": "NumpadSubtract",
        "NUM_MULTIPLY": "NumpadMultiply",
        "NUM_DIVIDE": "NumpadDivide",
        "NUM_DECIMAL": "NumpadDecimal",
        "NUM_ENTER": "NumpadEnter",
        # X11 reports Shift+digit or punctuation as the shifted symbol (Shift+= as PLUS), so a key
        # pressed before Shift changes is released under the other symbol. Both name the same key.
        "EXCLAMATION": "Digit1",
        "AT": "Digit2",
        "HASH": "Digit3",
        "DOLLAR": "Digit4",
        "PERCENT": "Digit5",
        "ASCIICIRCUM": "Digit6",
        "AMPERSAND": "Digit7",
        "ASTERISK": "Digit8",
        "PARENLEFT": "Digit9",
        "PARENRIGHT": "Digit0",
        "UNDERSCORE": "Minus",
        "PLUS": "Equal",
        "BRACELEFT": "BracketLeft",
        "BRACERIGHT": "BracketRight",
        "COLON": "Semicolon",
        "DOUBLEQUOTE": "Quote",
        "ASCIITILDE": "Backquote",
        "BAR": "Backslash",
        "LESS": "Comma",
        "GREATER": "Period",
        "QUESTION": "Slash",
    }
    for name, code in named.items():
        table[getattr(key, name)] = code
    return table


class NewtonKeyEventSource(KeyEventSource):
    """Keys typed into a Newton viewer window.

    Hooks the initialized viewer's pyglet window until close. pyglet reports layout-mapped key
    symbols, so codes name the key that types the same character on a US layout rather than the
    physical key. Presses and releases pair up reliably on a US layout, including when Shift
    changes while a key is held; elsewhere this is best effort: a key whose shifted character
    belongs to another US key (German ``+``/``*``, French ``&``/``1``) can stay held until the
    window loses focus. Presses typed into the viewer's ImGui widgets are not forwarded, and when
    ImGui takes the keyboard, listeners see a focus loss, as when the window loses focus.
    """

    def __init__(self, viewer: Any) -> None:
        """Initialize the source.

        Args:
            viewer: The :class:`~isaaclab_visualizers.newton.newton_visualizer.NewtonViewerGL` whose
                ImGui UI may take the keyboard.
        """
        super().__init__()
        self._viewer = viewer
        self._window = viewer.renderer.window
        self._ui_has_keyboard = False
        self._window.push_handlers(self)
        self._viewer.keyboard_frame_hook = self._poll_ui_keyboard

    def close(self) -> None:
        """Disconnect the window and notify listeners, including when native cleanup fails."""
        if self.closed:
            return
        try:
            self._window.remove_handlers(self)
        finally:
            self._viewer.keyboard_frame_hook = None
            self._viewer = self._window = None
            super().close()

    def _poll_ui_keyboard(self) -> bool:
        """Track whether ImGui owns the keyboard, reporting a focus loss when it takes it."""
        if self.closed:  # a failed native unhook can leave callbacks behind
            return False
        viewer = self._viewer
        has_keyboard = viewer.gui is not None and bool(viewer.gui.is_keyboard_capturing())
        if has_keyboard and not self._ui_has_keyboard:
            self._emit_focus_lost()
        self._ui_has_keyboard = has_keyboard
        return has_keyboard

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        """Forward a pyglet key press unless the UI owns keyboard input."""
        if self._poll_ui_keyboard():
            return
        code = _pyglet_symbol_to_w3c().get(symbol)
        if code is not None:
            self._emit_key(code, True)

    def on_key_release(self, symbol: int, modifiers: int) -> None:
        """Forward a pyglet key release."""
        code = _pyglet_symbol_to_w3c().get(symbol)
        if code is not None:
            self._emit_key(code, False)

    def on_deactivate(self) -> None:
        """Release the listener's held keys when the window loses focus."""
        self._emit_focus_lost()
