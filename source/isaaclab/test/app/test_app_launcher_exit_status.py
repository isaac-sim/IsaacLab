# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify that a real AppLauncher process reports a truthful exit status.

Each test launches a headless CPU :class:`~isaaclab.app.AppLauncher` in a child
process and asserts on how the process ends: killed by a signal, or failed with
an unhandled exception. Kit fast shutdown previously replaced both outcomes
with a successful exit code 0.
"""

import os
import signal
import subprocess
import sys
import time

import pytest

_TRIGGER_WAIT_FOR_SIGTERM = "--wait-for-sigterm"
_TRIGGER_UNHANDLED_EXCEPTION = "--trigger-unhandled-exception"
_READY_MARKER = "SIGTERM_TEST_READY"


def _idle_after_app_launcher_initialization() -> None:
    from isaaclab.app import AppLauncher

    AppLauncher(headless=True, device="cpu")
    print(_READY_MARKER, flush=True)
    while True:
        time.sleep(0.5)


def _raise_after_app_launcher_initialization() -> None:
    from isaaclab.app import AppLauncher

    AppLauncher(headless=True, device="cpu")
    raise RuntimeError("intentional AppLauncher failure")


@pytest.mark.integration
def test_sigterm_reports_killed_by_signal_status():
    """Verify that SIGTERM tears the app down once and the process dies by SIGTERM.

    Regression test for two defects: a signal received during ``SimulationApp.close()``
    re-entered the abort handler and recursed until the stack overflowed, and the
    graceful close terminated the process with exit code 0 under Kit fast shutdown, so
    a SIGTERM-ed distributed worker was recorded as successful.
    """
    proc = subprocess.Popen(
        [sys.executable, __file__, _TRIGGER_WAIT_FOR_SIGTERM],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # wait for the app to finish starting up
        deadline = time.time() + 300
        for line in proc.stdout:
            if _READY_MARKER in line:
                break
            if time.time() > deadline:
                pytest.fail("AppLauncher child did not become ready in time")
        proc.send_signal(signal.SIGTERM)
        proc.stdout.close()
        _, stderr = proc.communicate(timeout=300)
    finally:
        if proc.poll() is None:
            proc.kill()

    # the process must die by SIGTERM, not report a successful exit
    assert proc.returncode == -signal.SIGTERM, f"returncode={proc.returncode}\n{stderr}"
    # the teardown must not recurse through the abort handler
    assert "_on_abort_signal" not in stderr


@pytest.mark.integration
def test_unhandled_exception_exits_with_failure():
    """Verify that AppLauncher shutdown does not replace an exception's exit status with zero."""
    result = subprocess.run(
        [sys.executable, __file__, _TRIGGER_UNHANDLED_EXCEPTION],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "RuntimeError: intentional AppLauncher failure" in result.stderr


if __name__ == "__main__":
    if _TRIGGER_WAIT_FOR_SIGTERM in sys.argv:
        # detach from pytest's process group so only the explicit SIGTERM reaches us
        os.setpgrp()
        _idle_after_app_launcher_initialization()
    elif _TRIGGER_UNHANDLED_EXCEPTION in sys.argv:
        _raise_after_app_launcher_initialization()
