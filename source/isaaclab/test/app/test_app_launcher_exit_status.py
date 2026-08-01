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
import threading
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


def _wait_for_ready_marker(proc: subprocess.Popen, timeout: float) -> None:
    """Wait until the child prints the ready marker, failing on timeout or early child exit.

    A reader thread consumes the child's stdout so the blocking line iteration cannot
    outlive the deadline: the test waits on an event with a timeout instead of on the
    pipe itself. On failure the collected child output is included in the report.
    """
    lines: list[str] = []
    stdout_settled = threading.Event()  # marker seen or EOF
    marker_seen = False

    def _drain_stdout() -> None:
        nonlocal marker_seen
        for line in proc.stdout:
            lines.append(line)
            if _READY_MARKER in line:
                marker_seen = True
                break
        stdout_settled.set()

    reader = threading.Thread(target=_drain_stdout, daemon=True)
    reader.start()
    settled = stdout_settled.wait(timeout)
    if not settled or not marker_seen:
        proc.kill()
        proc.wait(timeout=30)
        reader.join(timeout=30)
        stderr = proc.stderr.read()
        reason = "did not become ready in time" if not settled else "exited before becoming ready"
        pytest.fail(
            f"AppLauncher child {reason}.\n--- child stdout ---\n{''.join(lines)}\n--- child stderr ---\n{stderr}"
        )


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
        # wait for the app to finish starting up, without blocking past the deadline
        _wait_for_ready_marker(proc, timeout=300)
        proc.send_signal(signal.SIGTERM)
        proc.stdout.close()
        _, stderr = proc.communicate(timeout=300)
    finally:
        if proc.poll() is None:
            proc.kill()

    # the process must report the termination truthfully: exit status 128 + SIGTERM when
    # fast shutdown exits inside close(exit_code=...), or death by SIGTERM when close()
    # returns and the handler re-raises. Either way, never a successful exit.
    assert proc.returncode in (128 + signal.SIGTERM, -signal.SIGTERM), f"returncode={proc.returncode}\n{stderr}"
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
