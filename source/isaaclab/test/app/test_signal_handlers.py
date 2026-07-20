# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless unit tests for the AppLauncher signal handlers."""

import signal
import types

from isaaclab.app.app_launcher import AppLauncher


def test_abort_handler_closes_app_once_on_reentrant_signal(monkeypatch):
    """A signal delivered while ``close()`` is running must not re-enter ``close()``.

    Regression test for the infinite recursion where a SIGTERM received during
    ``SimulationApp.close()`` re-entered the abort handler, recursing until the
    stack overflowed and the process had to be SIGKILL-ed.
    """
    close_calls = []
    fallback_actions = []

    launcher = types.SimpleNamespace(_abort_in_progress=False)

    def _close():
        close_calls.append(len(close_calls) + 1)
        # Simulate further signals delivered while the app is still closing.
        if len(close_calls) < 5:
            AppLauncher._abort_signal_handle_callback(launcher, signal.SIGTERM, None)

    launcher._app = types.SimpleNamespace(close=_close)

    monkeypatch.setattr(
        "isaaclab.app.app_launcher.signal.signal",
        lambda signum, handler: fallback_actions.append(("set_handler", signum, handler)),
    )
    monkeypatch.setattr(
        "isaaclab.app.app_launcher.signal.raise_signal",
        lambda signum: fallback_actions.append(("raise", signum)),
    )

    AppLauncher._abort_signal_handle_callback(launcher, signal.SIGTERM, None)

    # the app must be closed exactly once
    assert close_calls == [1]
    # the re-entrant signal must fall back to the default action and re-raise
    assert ("set_handler", signal.SIGTERM, signal.SIG_DFL) in fallback_actions
    assert ("raise", signal.SIGTERM) in fallback_actions
