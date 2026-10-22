# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for the :class:`~isaaclab.visualizers.KeyEventSource` contract shared by every backend.

A backend double records its hooks and can fail any of them.
"""

from __future__ import annotations

import logging

import pytest

from isaaclab.visualizers import KeyEventSource

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


class _Source(KeyEventSource):
    """Backend double that records its hooks and fails the ones listed in ``fail`` once each."""

    def __init__(self, fail: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.calls: list[str] = []
        self.fail = list(fail)

    def _hook(self, name: str) -> None:
        self.calls.append(name)
        if name in self.fail:
            self.fail.remove(name)
            raise RuntimeError(name)

    def _start_listening(self) -> None:
        self._hook("start")

    def _stop_listening(self) -> None:
        self._hook("stop")

    def _release(self) -> None:
        self._hook("release")


def test_backend_listens_while_any_listener_is_subscribed():
    source = _Source()
    first, first_subscription = _listen(source)
    second, second_subscription = _listen(source)
    assert source.calls == ["start"]

    source._emit_key("KeyW", True)
    assert first.events == second.events == [("KeyW", True)]

    first_subscription.close()
    first_subscription.close()  # idempotent
    assert first_subscription.closed
    assert source.calls == ["start"]
    with second_subscription:
        pass
    assert source.calls == ["start", "stop"]

    _listen(source)
    assert source.calls == ["start", "stop", "start"]


def test_failed_start_registers_nothing_and_can_be_retried():
    source = _Source(fail=("start",))
    recorder = _Recorder()
    with pytest.raises(RuntimeError, match="start"):
        source.add_key_listener(recorder.on_key, recorder.on_focus_lost)
    source._emit_key("KeyW", True)
    assert recorder.events == []

    retried, _ = _listen(source)
    source._emit_key("KeyW", True)
    assert source.calls == ["start", "start"]
    assert retried.events == [("KeyW", True)]


def test_failed_stop_keeps_the_backend_listening():
    source = _Source(fail=("stop",))
    _, subscription = _listen(source)
    with pytest.raises(RuntimeError, match="stop"):
        subscription.close()
    assert subscription.closed

    # still hooked, so the next listener does not start it again and close unhooks it
    _listen(source)
    source.close()
    assert source.calls == ["start", "stop", "stop", "release"]


def test_events_are_forwarded_as_the_window_reports_them():
    """Listeners track held keys themselves: repeats and a release without its press reach them."""
    source = _Source()
    recorder, _ = _listen(source)
    source._emit_key("KeyA", False)
    source._emit_key("KeyW", True)
    source._emit_key("KeyW", True)
    source._emit_key("KeyW", False)
    assert recorder.events == [("KeyA", False), ("KeyW", True), ("KeyW", True), ("KeyW", False)]


def test_focus_loss_reaches_every_listener():
    source = _Source()
    first, _ = _listen(source)
    second, _ = _listen(source)
    source._emit_focus_lost()
    assert first.focus_lost == second.focus_lost == 1


def test_listener_unsubscribed_during_dispatch_gets_no_more_callbacks():
    source = _Source()
    later = _Recorder()
    subscriptions = []
    source.add_key_listener(lambda code, pressed: subscriptions[0].close(), lambda: None)
    subscriptions.append(source.add_key_listener(later.on_key, later.on_focus_lost))

    source._emit_key("KeyW", True)
    assert later.events == []


def test_failing_listener_does_not_affect_others(caplog):
    source = _Source()

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
    source = _Source()
    recorder, subscription = _listen(source)
    source._emit_key("KeyW", True)

    source.close()
    source.close()  # idempotent
    assert source.closed and subscription.closed
    assert source.calls == ["start", "stop", "release"]
    assert recorder.focus_lost == 1

    source._emit_key("KeyW", False)
    source._emit_focus_lost()
    subscription.close()
    assert recorder.events == [("KeyW", True)]
    assert recorder.focus_lost == 1
    assert source.calls == ["start", "stop", "release"]
    with pytest.raises(RuntimeError, match="closed"):
        source.add_key_listener(recorder.on_key, recorder.on_focus_lost)


def test_close_releases_and_notifies_even_when_unhooking_fails():
    source = _Source(fail=("stop",))
    recorder, _ = _listen(source)

    with pytest.raises(RuntimeError, match="stop"):
        source.close()
    assert source.calls == ["start", "stop", "release"]
    assert recorder.focus_lost == 1
    source.close()  # already closed: the failed step is not retried
    assert source.calls == ["start", "stop", "release"]
