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
        self._source: KeyEventSource | None = source
        self._on_key = on_key
        self._on_focus_lost = on_focus_lost

    @property
    def closed(self) -> bool:
        """Whether the subscription no longer receives callbacks."""
        return self._source is None

    def close(self) -> None:
        """Unsubscribe."""
        source, self._source = self._source, None
        if source is not None and not source.closed:
            source._subscriptions.remove(self)

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

    The visualizer connects native events when its window initializes and closes the source before
    the window closes. Backend bridges report events through :meth:`_emit_key` and
    :meth:`_emit_focus_lost`; adding or removing a listener never changes native window resources.
    """

    def __init__(self) -> None:
        self._subscriptions: list[KeyboardSubscription] = []
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether the source was closed."""
        return self._closed

    def add_key_listener(self, on_key: KeyCallback, on_focus_lost: FocusLostCallback) -> KeyboardSubscription:
        """Subscribe to key presses and releases from the initialized window.

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
        subscription = KeyboardSubscription(self, on_key, on_focus_lost)
        self._subscriptions.append(subscription)
        return subscription

    def close(self) -> None:
        """Close every subscription and deliver a final focus loss to its listener."""
        if self._closed:
            return
        self._closed = True
        subscriptions, self._subscriptions = self._subscriptions, []
        for subscription in subscriptions:
            if not subscription.closed:
                subscription.close()
                _call(subscription._on_focus_lost)

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


def _call(callback: Callable, *args) -> None:
    try:
        callback(*args)
    except Exception:
        logger.exception("Keyboard listener %r failed", callback)
