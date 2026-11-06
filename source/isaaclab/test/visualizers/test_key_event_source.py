# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none


"""Window-independent event forwarding; native hook ownership is tested at each backend."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from isaaclab.visualizers import KeyEventSource, add_key_callbacks

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


def test_events_are_forwarded_as_the_window_reports_them():
    """Listeners track held keys themselves: repeats and a release without its press reach them."""
    source = KeyEventSource()
    recorder, _ = _listen(source)
    source._emit_key("KeyA", False)
    source._emit_key("KeyW", True)
    source._emit_key("KeyW", True)
    source._emit_key("KeyW", False)
    assert recorder.events == [("KeyA", False), ("KeyW", True), ("KeyW", True), ("KeyW", False)]


def test_focus_loss_reaches_every_listener():
    source = KeyEventSource()
    first, _ = _listen(source)
    second, _ = _listen(source)
    source._emit_focus_lost()
    assert first.focus_lost == second.focus_lost == 1


@pytest.mark.parametrize("event", ["key", "focus", "close"])
def test_listener_unsubscribed_during_dispatch_gets_no_more_callbacks(event):
    source = KeyEventSource()
    later = _Recorder()
    subscriptions = []
    source.add_key_listener(lambda code, pressed: subscriptions[0].close(), lambda: subscriptions[0].close())
    subscriptions.append(source.add_key_listener(later.on_key, later.on_focus_lost))

    if event == "key":
        source._emit_key("KeyW", True)
    elif event == "focus":
        source._emit_focus_lost()
    else:
        source.close()
    assert later.events == [] and later.focus_lost == 0


def test_failing_listener_does_not_affect_others(caplog):
    source = KeyEventSource()

    def fail(*args):
        raise ValueError("listener bug")

    source.add_key_listener(fail, fail)
    recorder, _ = _listen(source)
    with caplog.at_level(logging.ERROR):
        source._emit_key("KeyW", True)
        source._emit_focus_lost()
    assert recorder.events == [("KeyW", True)]
    assert recorder.focus_lost == 1
    assert "listener bug" in caplog.text


def test_close_is_terminal():
    source = KeyEventSource()
    recorder, subscription = _listen(source)
    source._emit_key("KeyW", True)

    source.close()
    source.close()  # idempotent
    assert source.closed and subscription.closed
    assert recorder.focus_lost == 1

    source._emit_key("KeyW", False)
    source._emit_focus_lost()
    subscription.close()
    assert recorder.events == [("KeyW", True)]
    assert recorder.focus_lost == 1
    with pytest.raises(RuntimeError, match="closed"):
        source.add_key_listener(recorder.on_key, recorder.on_focus_lost)


def test_subscriptions_are_independent_and_close_is_idempotent():
    source = KeyEventSource()
    first, subscription = _listen(source)
    second, _ = _listen(source)
    source._emit_key("KeyW", True)
    subscription.close()
    subscription.close()
    source._emit_key("KeyW", False)
    assert first.events == [("KeyW", True)]
    assert second.events == [("KeyW", True), ("KeyW", False)]
    assert subscription.closed and first.focus_lost == 0


def test_source_does_not_own_backend_lifecycle_or_controller_state():
    """Window setup belongs to initialize/close; held keys and camera policy belong to consumers."""
    source = KeyEventSource()
    forbidden = (
        "_start_listening",
        "_stop_listening",
        "_release",
        "_listening",
        "_held",
        "_captures",
        "_lock",
        "capture_keyboard",
    )
    assert all(not hasattr(source, name) for name in forbidden)
    subscription = source.add_key_listener(lambda code, pressed: None, lambda: None)
    assert not hasattr(subscription, "_closed"), "Subscription lifetime is owned by its source reference."


def test_script_hotkeys_track_presses_focus_and_subscription_lifetime():
    """Shared windows fire a hotkey once; focus loss releases held keys and cleanup stops delivery."""
    source = KeyEventSource()
    windows = [SimpleNamespace(key_event_source=source), SimpleNamespace(key_event_source=None)]
    pressed = []
    remove = add_key_callbacks(
        windows + windows[:1],
        {key: lambda key=key: pressed.append(key) for key in ("n", "1", "Escape")},
    )
    source._emit_key("KeyN", True)
    source._emit_key("KeyN", True)
    source._emit_key("KeyN", False)
    source._emit_key("Digit1", True)
    source._emit_key("Escape", True)
    source._emit_key("KeyM", True)
    assert pressed == ["n", "1", "Escape"]
    source._emit_key("KeyN", True)
    source._emit_focus_lost()
    source._emit_key("KeyN", True)
    assert pressed == ["n", "1", "Escape", "n", "n"]
    remove()
    remove()
    source._emit_focus_lost()
    source._emit_key("KeyN", True)
    assert pressed == ["n", "1", "Escape", "n", "n"]
