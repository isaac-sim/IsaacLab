# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless unit tests for the Kit application exit-status policy."""

import signal
import sys
import types

import pytest
from isaaclab_physx.app import kit_lifecycle

pytestmark = pytest.mark.unit


@pytest.fixture
def installed(monkeypatch):
    """Install the policy on a fake app, recording closes, handlers, and raised signals."""
    events = []
    handlers = {}
    exit_hooks = []

    def set_handler(signum, handler):
        handlers[signum] = handler
        events.append(("handler", signum, handler))

    app = types.SimpleNamespace(close=lambda exit_code: events.append(("close", exit_code)))
    monkeypatch.setattr(kit_lifecycle.signal, "signal", set_handler)
    monkeypatch.setattr(kit_lifecycle.signal, "raise_signal", lambda signum: events.append(("raise", signum)))
    monkeypatch.setattr(kit_lifecycle.atexit, "register", exit_hooks.append)
    kit_lifecycle.install_exit_handlers(app)
    events.clear()
    return events, handlers, exit_hooks


def test_fatal_signals_are_left_untouched(installed):
    """SIGSEGV and SIGABRT keep their default action so faults crash with a core dump."""
    _, handlers, _ = installed
    assert signal.SIGSEGV not in handlers
    assert signal.SIGABRT not in handlers
    assert handlers[signal.SIGINT] is signal.default_int_handler


def test_sigterm_disarms_then_closes_with_killed_by_signal_status(installed):
    """SIGTERM disarms itself before closing, so a repeated SIGTERM kills instead of re-entering close."""
    events, handlers, _ = installed
    handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert events == [
        ("handler", signal.SIGTERM, signal.SIG_DFL),
        ("close", 128 + signal.SIGTERM),
        ("raise", signal.SIGTERM),
    ]


@pytest.mark.parametrize("pending_exception, exit_code", [(False, 0), (True, 1)])
def test_exit_close_reports_pending_failure(installed, monkeypatch, pending_exception, exit_code):
    """Normal exit disarms SIGTERM and closes with 1 only when an exception is unhandled."""
    events, _, exit_hooks = installed
    if pending_exception:
        monkeypatch.setattr(sys, "last_exc", RuntimeError("unhandled"), raising=False)
    else:
        monkeypatch.delattr(sys, "last_exc", raising=False)
    (close_at_exit,) = exit_hooks
    close_at_exit()
    assert events == [("handler", signal.SIGTERM, signal.SIG_DFL), ("close", exit_code)]
