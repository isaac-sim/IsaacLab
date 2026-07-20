# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify that a SIGTERM-ed AppLauncher process reports a killed-by-signal status."""

import os
import signal
import subprocess
import sys
import time

import pytest

_TRIGGER_WAIT_FOR_SIGTERM = "--wait-for-sigterm"
_READY_MARKER = "SIGTERM_TEST_READY"


def _idle_after_app_launcher_initialization() -> None:
    from isaaclab.app import AppLauncher

    AppLauncher(headless=True, device="cpu")
    print(_READY_MARKER, flush=True)
    while True:
        time.sleep(0.5)


@pytest.mark.integration
def test_sigterm_reports_killed_by_signal_status():
    """Verify that SIGTERM tears the app down once and the process dies by SIGTERM.

    Regression test for two defects: a signal received during ``SimulationApp.close()``
    re-entered the handler and recursed until the stack overflowed, and the graceful
    close terminated the process with exit code 0 under Kit fast shutdown, so a
    SIGTERM-ed distributed worker was recorded as successful.
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
    assert "_abort_signal_handle_callback" not in stderr


if __name__ == "__main__" and _TRIGGER_WAIT_FOR_SIGTERM in sys.argv:
    # detach from pytest's process group so only the explicit SIGTERM reaches us
    os.setpgrp()
    _idle_after_app_launcher_initialization()
