# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard input from a focused visualizer window, shared by the visualizer backends."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

KeyCallback = Callable[[str, bool], None]
"""Called with a W3C ``KeyboardEvent.code`` string (e.g. ``"KeyW"``) and ``True`` on press, ``False`` on release."""

FocusLostCallback = Callable[[], None]
"""Called when the window stops delivering keys: every key the listener saw pressed is released."""


@dataclass(frozen=True)
class KeyboardCapabilities:
    """What a :class:`KeyEventSource` backend reports.

    Attributes:
        physical_keys: Whether a key code names the physical key position, independent of the
            keyboard layout. When ``False`` the code names the key that types the same character on
            a US layout, so it changes with the user's layout.
        filters_ui_text_input: Whether keys typed into the visualizer's own UI (e.g. a text field) are
            withheld from listeners, with a focus loss when the UI takes the keyboard. When ``False``
            such keys are still reported.
    """

    physical_keys: bool
    filters_ui_text_input: bool


class KeyboardSubscription:
    """A listener subscribed with :meth:`KeyEventSource.add_key_listener`.

    Close it (or use it as a context manager) to unsubscribe. Closing is idempotent and delivers no
    final focus loss: the listener chose to leave.
    """

    def __init__(self, source: KeyEventSource, on_key: KeyCallback, on_focus_lost: FocusLostCallback) -> None:
        self._source = source
        self._on_key = on_key
        self._on_focus_lost = on_focus_lost
        self._lock = threading.Lock()
        self._held: set[str] = set()
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether the subscription no longer receives callbacks."""
        return self._closed

    def close(self) -> None:
        """Unsubscribe. A callback already running on another thread may still finish."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._held.clear()
        self._source._remove_subscription(self)

    def __enter__(self) -> KeyboardSubscription:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _deliver_key(self, code: str, pressed: bool) -> None:
        # Each press reaches the listener once, and each release only follows a press it received.
        with self._lock:
            if self._closed or (code in self._held) == pressed:
                return
            if pressed:
                self._held.add(code)
            else:
                self._held.discard(code)
        _call(self._on_key, code, pressed)

    def _deliver_focus_lost(self, final: bool = False) -> None:
        with self._lock:
            if self._closed:
                return
            self._held.clear()
            if final:
                self._closed = True
        _call(self._on_focus_lost)


class KeyboardCapture:
    """A request, from :meth:`KeyEventSource.capture_keyboard`, that the backend yield its key bindings.

    Close it (or use it as a context manager) to release the request. Closing is idempotent, and a
    capture outliving its source is harmless.
    """

    def __init__(self, source: KeyEventSource) -> None:
        self._source = source
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether the request was released, directly or by closing the source."""
        return self._closed or self._source.closed

    def close(self) -> None:
        """Release the request; the backend restores its bindings once no capture is held."""
        if not self._closed:
            self._closed = True
            self._source._release_capture(self)

    def __enter__(self) -> KeyboardCapture:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class KeyEventSource:
    """Keyboard input from a focused visualizer window.

    Obtained from :attr:`~isaaclab.visualizers.BaseVisualizer.key_event_source`. Listeners receive
    each key press and release as a W3C ``KeyboardEvent.code`` string; :attr:`capabilities` says
    whether it names the physical key. For every backend:

    - A listener receives a press once (repeats are dropped) and a release only after a press it
      received. Every press it received ends with a release or with a focus loss.
    - Focus loss means the window stopped delivering keys: it lost OS focus, the visualizer's UI took
      the keyboard (when :attr:`KeyboardCapabilities.filters_ui_text_input` is set), or the source
      closed. The listener then treats every key it saw pressed as released.
    - Callbacks run on the backend's event thread (the thread that steps the visualizer), one at a
      time and in event order, never while an internal lock is held. The final focus loss runs on
      the thread that closes the source, so close it from the event thread to keep that order. A
      listener that raises is logged and does not affect other listeners.
    - Closing a subscription, a capture or the source is idempotent. Closing the source delivers a
      final focus loss to every listener and restores any captured key bindings; afterwards
      :meth:`add_key_listener` and :meth:`capture_keyboard` raise :class:`RuntimeError`.

    A backend subclasses this, sets :attr:`capabilities`, implements :meth:`_start_listening`,
    :meth:`_stop_listening`, :meth:`_set_captured` and :meth:`_release`, and reports events through
    :meth:`_emit_key` and :meth:`_emit_focus_lost`. Hooks run one at a time; a hook that raises must
    leave its backend state unchanged.
    """

    capabilities: KeyboardCapabilities
    """What this backend reports."""

    def __init__(self) -> None:
        self._lock = threading.Lock()  # guards the subscription table read by the event thread
        self._transition_lock = threading.RLock()  # serializes backend hooks and lifecycle changes
        self._subscriptions: dict[int, KeyboardSubscription] = {}
        self._captures: set[KeyboardCapture] = set()
        self._listening = False
        self._captured = False
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether the source was closed."""
        return self._closed

    def add_key_listener(self, on_key: KeyCallback, on_focus_lost: FocusLostCallback) -> KeyboardSubscription:
        """Subscribe to key presses and releases; the backend starts listening with the first listener.

        Args:
            on_key: Called with the key's W3C ``KeyboardEvent.code`` and whether it was pressed.
            on_focus_lost: Called when the window stops delivering keys.

        Returns:
            The subscription. Close it to unsubscribe.

        Raises:
            RuntimeError: If the source is closed.
        """
        with self._transition_lock:
            self._check_open()
            if not self._listening:
                self._start_listening()
                self._listening = True
            subscription = KeyboardSubscription(self, on_key, on_focus_lost)
            with self._lock:
                self._subscriptions[id(subscription)] = subscription
            return subscription

    def capture_keyboard(self) -> KeyboardCapture:
        """Ask the backend to yield its own key bindings (e.g. keyboard camera movement).

        The bindings stay yielded while any capture is held. Capturing never blocks the backend's own
        UI text input.

        Returns:
            The capture. Close it to release the request.

        Raises:
            RuntimeError: If the source is closed.
        """
        with self._transition_lock:
            self._check_open()
            if not self._captured:
                self._set_captured(True)
                self._captured = True
            capture = KeyboardCapture(self)
            self._captures.add(capture)
            return capture

    def close(self) -> None:
        """Stop listening, restore the backend's bindings, release it, and deliver a final focus loss.

        Every step is attempted even if an earlier one raises; the first failure is raised afterwards.
        """
        with self._transition_lock:
            if self._closed:
                return
            self._closed = True
            with self._lock:
                subscriptions = list(self._subscriptions.values())
                self._subscriptions.clear()
            self._captures.clear()
            steps = []
            if self._listening:
                steps.append(self._stop_listening)
            if self._captured:
                steps.append(lambda: self._set_captured(False))
            steps.append(self._release)
            errors = []
            for step in steps:
                try:
                    step()
                except Exception as error:
                    errors.append(error)
            self._listening = self._captured = False
        for subscription in subscriptions:
            subscription._deliver_focus_lost(final=True)
        for error in errors[1:]:
            logger.error("Keyboard source cleanup failed", exc_info=error)
        if errors:
            raise errors[0]

    # Backend hooks ---------------------------------------------------------------------------------

    def _start_listening(self) -> None:
        """Hook the window's key events. Called when the first listener subscribes."""

    def _stop_listening(self) -> None:
        """Unhook the window's key events. Called when the last listener unsubscribes, and on close."""

    def _set_captured(self, captured: bool) -> None:
        """Yield (``True``) or restore (``False``) the backend's own key bindings."""

    def _release(self) -> None:
        """Drop references to the backend's window and viewer. Called once, on close."""

    # Event reporting -------------------------------------------------------------------------------

    def _emit_key(self, code: str, pressed: bool) -> None:
        """Report a key press or release to every listener."""
        for subscription in self._snapshot():
            subscription._deliver_key(code, pressed)

    def _emit_focus_lost(self) -> None:
        """Tell every listener that the window stopped delivering keys."""
        for subscription in self._snapshot():
            subscription._deliver_focus_lost()

    # Internals -------------------------------------------------------------------------------------

    def _snapshot(self) -> list[KeyboardSubscription]:
        with self._lock:
            return [] if self._closed else list(self._subscriptions.values())

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")

    def _remove_subscription(self, subscription: KeyboardSubscription) -> None:
        with self._transition_lock:
            with self._lock:
                if self._subscriptions.pop(id(subscription), None) is None:
                    return
                last = not self._subscriptions
            if last and self._listening and not self._closed:
                self._stop_listening()
                self._listening = False

    def _release_capture(self, capture: KeyboardCapture) -> None:
        with self._transition_lock:
            if capture not in self._captures:
                return
            self._captures.discard(capture)
            if not self._captures and self._captured and not self._closed:
                self._set_captured(False)
                self._captured = False


def _call(callback: Callable, *args) -> None:
    try:
        callback(*args)
    except Exception:
        logger.exception("Keyboard listener %r failed", callback)
