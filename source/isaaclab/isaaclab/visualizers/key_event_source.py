# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard input from a focused visualizer window, shared by the visualizer backends."""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

KeyCallback = Callable[[str, bool], None]
"""Called with a W3C ``KeyboardEvent.code`` string (e.g. ``"KeyW"``) and ``True`` on press, ``False`` on release."""

FocusLostCallback = Callable[[], None]
"""Called when the window stops delivering keys: treat every held key as released."""


class KeyboardSubscription:
    """A listener subscribed with :meth:`KeyEventSource.add_key_listener`.

    Close it (or use it as a context manager) to unsubscribe. Closing is idempotent and delivers no
    final focus loss: the listener chose to leave.
    """

    def __init__(self, source: KeyEventSource, on_key: KeyCallback, on_focus_lost: FocusLostCallback) -> None:
        self._source = source
        self._on_key = on_key
        self._on_focus_lost = on_focus_lost
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether the subscription no longer receives callbacks."""
        return self._closed

    def close(self) -> None:
        """Unsubscribe."""
        if not self._closed:
            self._closed = True
            self._source._remove_subscription(self)

    def __enter__(self) -> KeyboardSubscription:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class KeyEventSource:
    """Keyboard input from a focused visualizer window.

    Obtained from :attr:`~isaaclab.visualizers.BaseVisualizer.key_event_source`. For every backend:

    - Subscribing, closing and every callback happen on the thread that steps the visualizer.
    - ``on_key`` receives the window's presses and releases as W3C ``KeyboardEvent.code`` strings.
      Key repeat can arrive as further presses, and a release can arrive without its press (e.g. a
      key pressed in the visualizer's UI), so a listener tracks the keys it holds.
    - ``on_focus_lost`` runs when the window stops delivering keys: it lost focus, the visualizer's
      UI took the keyboard, or the source closed. The listener treats every held key as released.
    - A listener that raises is logged and does not affect other listeners.
    - Closing the source delivers a final focus loss to every listener; afterwards
      :meth:`add_key_listener` raises :class:`RuntimeError`.

    A backend subclasses this, implements :meth:`_start_listening`, :meth:`_stop_listening` and
    :meth:`_release`, and reports events through :meth:`_emit_key` and :meth:`_emit_focus_lost`.
    """

    def __init__(self) -> None:
        self._subscriptions: list[KeyboardSubscription] = []
        self._listening = False
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
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")
        if not self._listening:
            self._start_listening()
            self._listening = True
        subscription = KeyboardSubscription(self, on_key, on_focus_lost)
        self._subscriptions.append(subscription)
        return subscription

    def close(self) -> None:
        """Stop listening, release the backend, and deliver a final focus loss to every listener."""
        if self._closed:
            return
        self._closed = True
        subscriptions, self._subscriptions = self._subscriptions, []
        try:
            if self._listening:
                self._listening = False
                self._stop_listening()
        finally:
            try:
                self._release()
            finally:
                for subscription in subscriptions:
                    subscription._closed = True
                    _call(subscription._on_focus_lost)

    # Backend hooks ---------------------------------------------------------------------------------

    def _start_listening(self) -> None:
        """Hook the window's key events. Called when the first listener subscribes."""

    def _stop_listening(self) -> None:
        """Unhook the window's key events. Called when the last listener unsubscribes, and on close."""

    def _release(self) -> None:
        """Drop references to the backend's window and viewer. Called once, on close."""

    # Event reporting -------------------------------------------------------------------------------

    def _emit_key(self, code: str, pressed: bool) -> None:
        """Report a key press or release to every listener."""
        for subscription in list(self._subscriptions):
            if not subscription.closed:
                _call(subscription._on_key, code, pressed)

    def _emit_focus_lost(self) -> None:
        """Tell every listener that the window stopped delivering keys."""
        for subscription in list(self._subscriptions):
            if not subscription.closed:
                _call(subscription._on_focus_lost)

    # Internals -------------------------------------------------------------------------------------

    def _remove_subscription(self, subscription: KeyboardSubscription) -> None:
        if subscription not in self._subscriptions:
            return
        self._subscriptions.remove(subscription)
        if not self._subscriptions and self._listening:
            # stays listening if unhooking raises, so close or the next listener does not hook twice
            self._stop_listening()
            self._listening = False


def _call(callback: Callable, *args) -> None:
    try:
        callback(*args)
    except Exception:
        logger.exception("Keyboard listener %r failed", callback)
