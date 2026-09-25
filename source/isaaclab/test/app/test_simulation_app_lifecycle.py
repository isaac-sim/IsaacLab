# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless unit tests for the AppLauncher process-lifecycle exit policy."""

import signal
import sys
import types

from isaaclab.app.app_launcher import AppLauncher


def _make_lifecycle(close_fn):
    app = types.SimpleNamespace(close=close_fn, config={"fast_shutdown": True})
    return AppLauncher._SimulationAppLifecycle(app)


def _capture_signal_actions(monkeypatch):
    actions = []
    monkeypatch.setattr(
        "isaaclab.app.app_launcher.signal.signal",
        lambda signum, handler: actions.append(("set_handler", signum, handler)),
    )
    monkeypatch.setattr(
        "isaaclab.app.app_launcher.signal.raise_signal",
        lambda signum: actions.append(("raise", signum)),
    )
    return actions


def test_abort_signal_closes_once_with_killed_by_signal_status(monkeypatch):
    """The first signal closes the app once with the conventional 128 + signum status.

    Passing the status through ``close()`` lets the app perform its full teardown and
    exit truthfully instead of the exit code 0 that Kit fast shutdown reports.
    """
    codes = []
    lifecycle = _make_lifecycle(lambda exit_code=0: codes.append(exit_code))
    actions = _capture_signal_actions(monkeypatch)

    lifecycle._on_abort_signal(signal.SIGTERM, None)

    # the app must be closed exactly once, with the killed-by-signal status
    assert codes == [128 + signal.SIGTERM]
    # if close() returns (fast shutdown disabled), the default action must re-raise
    assert ("set_handler", signal.SIGTERM, signal.SIG_DFL) in actions
    assert ("raise", signal.SIGTERM) in actions


def test_abort_signal_reentrant_signal_skips_nested_close(monkeypatch):
    """A signal delivered while ``close()`` is running must not re-enter ``close()``.

    Regression test for the infinite recursion where a SIGTERM received during
    ``SimulationApp.close()`` re-entered the abort handler, recursing until the
    stack overflowed and the process had to be SIGKILL-ed.
    """
    close_calls = []

    def _close(exit_code=0):
        close_calls.append(len(close_calls) + 1)
        # Simulate further signals delivered while the app is still closing.
        if len(close_calls) < 5:
            lifecycle._on_abort_signal(signal.SIGTERM, None)

    lifecycle = _make_lifecycle(_close)
    actions = _capture_signal_actions(monkeypatch)

    lifecycle._on_abort_signal(signal.SIGTERM, None)

    # the app must be closed exactly once despite the nested signal
    assert close_calls == [1]
    # both the nested and the outer path fall back to the default action and re-raise
    assert actions.count(("set_handler", signal.SIGTERM, signal.SIG_DFL)) >= 1
    assert actions.count(("raise", signal.SIGTERM)) >= 1


def test_abort_signal_falls_back_when_exit_code_unsupported(capfd, monkeypatch):
    """A ``close()`` without the ``exit_code`` parameter still closes the app, loudly.

    Guards against silent drift: if a SimulationApp update drops the parameter, the
    app must still be closed and the exit-status limitation must be reported.
    """
    calls = []

    def _close():  # simulates a SimulationApp.close() without the exit_code parameter
        calls.append(len(calls) + 1)

    lifecycle = _make_lifecycle(_close)
    actions = _capture_signal_actions(monkeypatch)

    lifecycle._on_abort_signal(signal.SIGTERM, None)

    assert calls == [1]
    assert "does not accept exit_code" in capfd.readouterr().err
    # the re-raise still guarantees a truthful killed-by-signal status
    assert ("raise", signal.SIGTERM) in actions


def test_atexit_close_arms_reentrancy_guard(monkeypatch):
    """A signal arriving during the atexit close must take the re-entrant path.

    The atexit callback arms the same guard flag as the signal handler, so a SIGTERM
    delivered mid-close falls back to the default signal action instead of starting a
    second, nested teardown.
    """
    close_calls = []

    def _close(exit_code=0):
        close_calls.append(len(close_calls) + 1)
        # Simulate a signal delivered while the atexit close is running.
        lifecycle._on_abort_signal(signal.SIGTERM, None)

    lifecycle = _make_lifecycle(_close)
    actions = _capture_signal_actions(monkeypatch)

    lifecycle._close_at_exit()

    # the guard must be armed before close, so the mid-close signal must not re-enter close()
    assert close_calls == [1]
    assert ("set_handler", signal.SIGTERM, signal.SIG_DFL) in actions
    assert ("raise", signal.SIGTERM) in actions


def test_atexit_close_preserves_pending_failure_status(monkeypatch):
    """The atexit close passes a nonzero exit code when an unhandled exception is pending.

    Kit fast shutdown would otherwise replace the interpreter's pending failure
    status with a successful exit code 0.
    """
    codes = []
    lifecycle = _make_lifecycle(lambda exit_code=0: codes.append(exit_code))

    # normal interpreter exit: no pending exception, close with exit code 0
    monkeypatch.delattr(sys, "last_exc", raising=False)
    lifecycle._close_at_exit()
    # unhandled exception pending: close with a failure exit code
    lifecycle._closing = False
    monkeypatch.setattr(sys, "last_exc", RuntimeError("pending failure"), raising=False)
    lifecycle._close_at_exit()

    assert codes == [0, 1]


def test_atexit_close_falls_back_when_exit_code_unsupported(capfd):
    """A ``close()`` that stops accepting ``exit_code`` still closes the app, with a warning.

    Guards the exit-code path against silent drift: if a SimulationApp update drops the
    parameter, the app must still be closed and the exit-status limitation must be
    reported instead of failing silently.
    """
    calls = []

    def _close():  # simulates a SimulationApp.close() without the exit_code parameter
        calls.append(len(calls) + 1)

    lifecycle = _make_lifecycle(_close)
    lifecycle._close_at_exit()

    assert calls == [1]
    assert "does not accept exit_code" in capfd.readouterr().err
