# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared test utilities for isaaclab_mimic dataset generation tests."""

import contextlib
import os
import signal
import subprocess

SUBPROCESS_GRACE_PERIOD = 15
"""Grace period [s] after SIGTERM before sending SIGKILL."""

SUBPROCESS_DRAIN_TIMEOUT = 5
"""Timeout [s] for draining remaining pipe output after killing a process group."""


def _kill_process_group(pgid: int, sig: int):
    """Send *sig* to every process in the group, ignoring errors if already dead."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, sig)


def run_script(command: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    """Run a script in a subprocess and return a CompletedProcess.

    The Kit / Omniverse runtime's ``simulation_app.close()`` can hang
    indefinitely when another ``SimulationApp`` instance is alive in the parent
    test process (shared GPU / IPC resources).  To avoid blocking the test
    suite we use ``Popen`` with an explicit timeout:

    1. Wait up to *timeout* seconds for the process to finish.
    2. On timeout send ``SIGTERM`` to the **entire process group** and wait
       :data:`SUBPROCESS_GRACE_PERIOD` seconds.
    3. If still alive, ``SIGKILL`` the process group and collect remaining
       output.

    The subprocess is started in its own session (``start_new_session=True``)
    so that signals reach Kit child processes, and pipe draining after a kill
    uses a bounded timeout to avoid blocking on orphaned FD holders.

    The captured *stdout* / *stderr* are returned regardless of how the process
    terminated so that callers can validate the script's printed output.

    Args:
        command: The command to execute as a list of strings.
        timeout: Maximum time [s] to wait for the process to finish.
    """
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    pgid = os.getpgid(process.pid)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(pgid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=SUBPROCESS_GRACE_PERIOD)
        except subprocess.TimeoutExpired:
            _kill_process_group(pgid, signal.SIGKILL)
            try:
                stdout, stderr = process.communicate(timeout=SUBPROCESS_DRAIN_TIMEOUT)
            except subprocess.TimeoutExpired:
                # Orphaned grandchildren may still hold pipe FDs open.
                # Close our ends so we don't block forever, then reap the
                # zombie leader.
                process.stdout.close()
                process.stderr.close()
                process.wait()
                stdout, stderr = "", ""

    return subprocess.CompletedProcess(
        args=command,
        returncode=process.returncode,
        stdout=stdout or "",
        stderr=stderr or "",
    )
