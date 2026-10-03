# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for the :class:`~isaaclab.visualizers.KeyEventSource` contract shared by every backend.

A backend double records its hooks and can fail any of them; races are driven with events rather
than sleeps.
"""

from __future__ import annotations

import logging
import threading

import pytest

from isaaclab.visualizers import KeyboardCapabilities, KeyEventSource

pytestmark = [pytest.mark.unit]

_TIMEOUT = 5.0


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

    capabilities = KeyboardCapabilities(physical_keys=True, filters_ui_text_input=True)

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

    def _set_captured(self, captured: bool) -> None:
        self._hook(f"captured={captured}")

    def _release(self) -> None:
        self._hook("release")


class _BlockingSource(_Source):
    """Backend double whose ``block_on`` hook waits until the test lets it finish."""

    def __init__(self, block_on: str) -> None:
        super().__init__()
        self.block_on = block_on
        self.entered = threading.Event()
        self.proceed = threading.Event()

    def _hook(self, name: str) -> None:
        super()._hook(name)
        if name == self.block_on and not self.entered.is_set():
            self.entered.set()
            assert self.proceed.wait(_TIMEOUT)


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


def test_capture_leases_are_independent():
    source = _Source()
    first = source.capture_keyboard()
    second = source.capture_keyboard()
    assert source.calls == ["captured=True"]

    first.close()
    first.close()  # idempotent, and does not release the other lease
    assert first.closed and not second.closed
    assert source.calls == ["captured=True"]

    with second:
        pass
    assert source.calls == ["captured=True", "captured=False"]


def test_failed_capture_changes_are_rolled_back():
    source = _Source(fail=("captured=True", "captured=False"))
    with pytest.raises(RuntimeError):
        source.capture_keyboard()

    capture = source.capture_keyboard()
    with pytest.raises(RuntimeError):
        capture.close()
    # the bindings are still yielded, so close restores them
    source.close()
    assert source.calls == ["captured=True", "captured=True", "captured=False", "captured=False", "release"]


def test_first_subscribe_waits_for_last_unsubscribe():
    source = _BlockingSource(block_on="stop")
    _, subscription = _listen(source)
    unsubscribing = threading.Thread(target=subscription.close)
    unsubscribing.start()
    assert source.entered.wait(_TIMEOUT)

    recorder = _Recorder()
    subscribing = threading.Thread(target=source.add_key_listener, args=(recorder.on_key, recorder.on_focus_lost))
    subscribing.start()
    subscribing.join(0.05)
    assert subscribing.is_alive(), "subscribed while the backend was being unhooked"

    source.proceed.set()
    unsubscribing.join(_TIMEOUT)
    subscribing.join(_TIMEOUT)
    assert source.calls == ["start", "stop", "start"]
    source._emit_key("KeyW", True)
    assert recorder.events == [("KeyW", True)]


def test_capture_waits_for_the_last_release():
    source = _BlockingSource(block_on="captured=False")
    capture = source.capture_keyboard()
    releasing = threading.Thread(target=capture.close)
    releasing.start()
    assert source.entered.wait(_TIMEOUT)

    captures = []
    capturing = threading.Thread(target=lambda: captures.append(source.capture_keyboard()))
    capturing.start()
    capturing.join(0.05)
    assert capturing.is_alive(), "captured while the bindings were being restored"

    source.proceed.set()
    releasing.join(_TIMEOUT)
    capturing.join(_TIMEOUT)
    assert source.calls == ["captured=True", "captured=False", "captured=True"]
    assert not captures[0].closed


def test_repeats_and_orphan_releases_are_dropped():
    source = _Source()
    source._emit_key("KeyA", True)  # pressed before anyone listened
    recorder, _ = _listen(source)

    source._emit_key("KeyA", False)
    source._emit_key("KeyW", True)
    source._emit_key("KeyW", True)
    source._emit_key("KeyW", False)
    source._emit_key("KeyW", False)
    assert recorder.events == [("KeyW", True), ("KeyW", False)]


def test_focus_loss_releases_held_keys():
    source = _Source()
    recorder, _ = _listen(source)
    source._emit_key("KeyW", True)
    source._emit_focus_lost()
    source._emit_key("KeyW", False)
    source._emit_key("KeyW", True)
    assert recorder.events == [("KeyW", True), ("KeyW", True)]
    assert recorder.focus_lost == 1


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
    capture = source.capture_keyboard()
    source._emit_key("KeyW", True)

    source.close()
    source.close()  # idempotent
    assert source.closed and subscription.closed and capture.closed
    assert source.calls == ["start", "captured=True", "stop", "captured=False", "release"]
    assert recorder.focus_lost == 1

    source._emit_key("KeyW", False)
    source._emit_focus_lost()
    subscription.close()
    capture.close()
    assert recorder.events == [("KeyW", True)]
    assert recorder.focus_lost == 1
    assert source.calls == ["start", "captured=True", "stop", "captured=False", "release"]
    with pytest.raises(RuntimeError, match="closed"):
        source.add_key_listener(recorder.on_key, recorder.on_focus_lost)
    with pytest.raises(RuntimeError, match="closed"):
        source.capture_keyboard()


def test_close_attempts_every_step_and_raises_the_first_failure(caplog):
    source = _Source(fail=("stop", "release"))
    recorder, _ = _listen(source)
    source.capture_keyboard()

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="stop"):
        source.close()
    assert source.calls == ["start", "captured=True", "stop", "captured=False", "release"]
    assert recorder.focus_lost == 1
    assert "release" in caplog.text
    source.close()  # already closed: the failed steps are not retried
    assert source.calls[-1] == "release" and len(source.calls) == 5
