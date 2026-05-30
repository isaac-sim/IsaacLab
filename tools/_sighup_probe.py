# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Temporary multi-GPU CI instrumentation: identify the sender of a spurious SIGHUP.

Under concurrent multi-GPU execution a long-running, otherwise-passing test file
(newton ``test_articulation``) is killed by SIGHUP (signal 1) on the non-default
shards. The signal is external (the conftest only ever sends SIGKILL) and is not
reproducible on a single host, so this probe runs in CI to name the sender.

It blocks SIGHUP, runs the wrapped command as a child, and a watcher thread uses
``sigwaitinfo`` to log the sending PID / UID / comm / cmdline whenever a SIGHUP is
delivered to this process group. The child still inherits and may unblock SIGHUP,
so this does not change whether the file fails — it only records who signalled it.

Usage (prepended to the test command by ``tools/conftest.py``)::

    python _sighup_probe.py <real-command...>

Remove once the SIGHUP source is understood and fixed.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading


def _describe(pid: int) -> str:
    """Return ``comm (cmdline)`` for a PID, best-effort from ``/proc``."""
    comm = cmdline = "?"
    try:
        with open(f"/proc/{pid}/comm") as fh:
            comm = fh.read().strip()
    except OSError:
        pass
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            cmdline = fh.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
    except OSError:
        pass
    return f"{comm!r} ({cmdline!r})"


def _watch() -> None:
    """Log the sender of every SIGHUP delivered to this (blocked) process."""
    while True:
        try:
            info = signal.sigwaitinfo({signal.SIGHUP})
        except (OSError, ValueError, InterruptedError):
            return
        pid = getattr(info, "si_pid", -1)
        uid = getattr(info, "si_uid", -1)
        sys.stderr.write(f"\n[SIGHUP-PROBE] SIGHUP received: sender pid={pid} uid={uid} {_describe(pid)}\n")
        sys.stderr.flush()


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: _sighup_probe.py <command...>", file=sys.stderr)
        return 2
    # Block SIGHUP so sigwaitinfo can observe it instead of it terminating us.
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGHUP})
    threading.Thread(target=_watch, daemon=True).start()
    child = subprocess.Popen(sys.argv[1:])
    rc = child.wait()
    if rc < 0:
        # Child terminated by a signal; surface it as a conventional non-zero code.
        sys.stderr.write(f"\n[SIGHUP-PROBE] child terminated by signal {-rc}\n")
        sys.stderr.flush()
        return 128 - rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
