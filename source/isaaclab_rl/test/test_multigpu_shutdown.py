# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for multi-GPU launcher shutdown, which used to leave worker processes behind.

The launcher ran torchrun inside its own process group, so a terminal Ctrl-C reached torchrun and
every worker at the same moment the launcher forwarded a signal of its own. The extra signal landed
inside torchelastic's shutdown handler, which aborted before reaping the workers it had spawned.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from isaaclab_rl.entrypoints import multigpu

pytestmark = pytest.mark.skipif(not hasattr(os, "killpg"), reason="process groups are POSIX-only")

# A parent that spawns a child and sleeps, standing in for torchrun and its workers. The child pid is
# printed so the test can watch it independently of the parent.
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


def _wait_until(predicate, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        time.sleep(0.05)
    return predicate()


def _run_and_interrupt(monkeypatch, child_source: str) -> int:
    """Run a fake worker tree through the launcher, interrupt it, and return the grandchild pid."""
    grandchild: dict[str, int] = {}
    real_popen = subprocess.Popen

    def _capture(command, **kwargs):
        proc = real_popen(command, stdout=subprocess.PIPE, text=True, **kwargs)
        grandchild["pid"] = int(proc.stdout.readline().strip())
        return proc

    monkeypatch.setattr(subprocess, "Popen", _capture)
    threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    multigpu._run_distributed_command([sys.executable, "-c", _PARENT_WITH_CHILD.format(child=child_source)])
    return grandchild["pid"]


def test_child_runs_in_its_own_process_group():
    """The terminal must not be able to signal the worker tree directly."""
    recorded: dict[str, object] = {}
    real_popen = subprocess.Popen

    def _record(command, **kwargs):
        recorded["new_session"] = kwargs.get("start_new_session")
        return real_popen(command, **kwargs)

    subprocess.Popen = _record
    try:
        assert multigpu._run_distributed_command([sys.executable, "-c", "pass"]) == 0
    finally:
        subprocess.Popen = real_popen
    assert recorded["new_session"] is True


def test_interrupt_reaps_grandchildren(monkeypatch):
    """Workers are torchrun's children, so signalling only the direct child leaves them behind."""
    grandchild = _run_and_interrupt(monkeypatch, _PLAIN_CHILD)
    assert _wait_until(lambda: not _pid_is_alive(grandchild), timeout=15.0)


def test_worker_ignoring_signals_is_killed(monkeypatch):
    """A worker wedged in a native call never observes a signal, so escalation has to end it."""
    monkeypatch.setattr(multigpu, "_GRACEFUL_SHUTDOWN_S", 1.0)
    monkeypatch.setattr(multigpu, "_FORCED_SHUTDOWN_S", 1.0)
    monkeypatch.setattr(multigpu, "_STRAGGLER_GRACE_S", 2.0)
    grandchild = _run_and_interrupt(monkeypatch, _DEAF_CHILD)
    assert _wait_until(lambda: not _pid_is_alive(grandchild), timeout=20.0)


def test_successful_run_returns_its_exit_code():
    assert multigpu._run_distributed_command([sys.executable, "-c", "raise SystemExit(7)"]) == 7
