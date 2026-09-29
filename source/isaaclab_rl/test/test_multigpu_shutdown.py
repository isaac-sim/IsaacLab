# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for multi-GPU launcher process isolation and shutdown."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from isaaclab.cli import multigpu

pytestmark = pytest.mark.skipif(not hasattr(os, "killpg"), reason="process groups are POSIX-only")

# Model the torchrun process and one worker in a separate child process.
_PARENT_WITH_CHILD = (
    "import subprocess, sys, time\n"
    "child = subprocess.Popen([sys.executable, '-c', {child!r}])\n"
    "print(child.pid, flush=True)\n"
    "time.sleep(300)\n"
)
_PLAIN_CHILD = "import time; time.sleep(300)"
_DEAF_CHILD = (
    "import signal, time\n"
    "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "time.sleep(300)\n"
)


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _wait_until(predicate: Callable[[], bool], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        time.sleep(0.05)
    return predicate()


def _run_and_interrupt(monkeypatch: pytest.MonkeyPatch, child_source: str) -> int:
    grandchild: dict[str, int] = {}
    real_popen = subprocess.Popen

    def _capture(command, **kwargs):
        proc = real_popen(command, stdout=subprocess.PIPE, text=True, **kwargs)
        grandchild["pid"] = int(proc.stdout.readline().strip())
        return proc

    monkeypatch.setattr(subprocess, "Popen", _capture)
    threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    multigpu.run_launch_command([sys.executable, "-c", _PARENT_WITH_CHILD.format(child=child_source)])
    return grandchild["pid"]


def test_child_session_and_exit_status(tmp_path: Path) -> None:
    """The child runs in an isolated session and returns its exit status."""
    result_file = tmp_path / "session"
    source = (
        "import os, pathlib; "
        f"pathlib.Path({str(result_file)!r}).write_text(f'{{os.getpid()}} {{os.getpgrp()}} {{os.getsid(0)}}'); "
        "raise SystemExit(7)"
    )
    assert multigpu.run_launch_command([sys.executable, "-c", source]) == 7
    pid, group, session = map(int, result_file.read_text(encoding="utf-8").split())
    assert pid == group == session
    assert group != os.getpgrp()


def test_interrupt_reaps_grandchildren(monkeypatch: pytest.MonkeyPatch) -> None:
    """Interrupting the launcher reaps descendant worker processes."""
    grandchild = _run_and_interrupt(monkeypatch, _PLAIN_CHILD)
    assert _wait_until(lambda: not _pid_is_alive(grandchild), timeout=15.0)


def test_worker_ignoring_signals_is_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shutdown escalates when a worker ignores termination signals."""
    monkeypatch.setattr(multigpu, "_GRACEFUL_SHUTDOWN_S", 1.0)
    monkeypatch.setattr(multigpu, "_FORCED_SHUTDOWN_S", 1.0)
    monkeypatch.setattr(multigpu, "_STRAGGLER_GRACE_S", 2.0)
    grandchild = _run_and_interrupt(monkeypatch, _DEAF_CHILD)
    assert _wait_until(lambda: not _pid_is_alive(grandchild), timeout=20.0)
