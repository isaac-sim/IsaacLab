# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared test utilities for isaaclab_mimic dataset generation tests."""

import signal
import subprocess

SUBPROCESS_GRACE_PERIOD = 15
"""Grace period [s] after SIGTERM before sending SIGKILL."""


def run_script(command: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    """Run a script in a subprocess and return a CompletedProcess.

    The Kit / Omniverse runtime's ``simulation_app.close()`` can hang
    indefinitely when another ``SimulationApp`` instance is alive in the parent
    test process (shared GPU / IPC resources).  To avoid blocking the test
    suite we use ``Popen`` with an explicit timeout:

    1. Wait up to *timeout* seconds for the process to finish.
    2. On timeout send ``SIGTERM`` and wait :data:`SUBPROCESS_GRACE_PERIOD` seconds.
    3. If still alive, ``SIGKILL`` and collect remaining output.

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
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.send_signal(signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=SUBPROCESS_GRACE_PERIOD)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()

    return subprocess.CompletedProcess(
        args=command,
        returncode=process.returncode,
        stdout=stdout or "",
        stderr=stderr or "",
    )
