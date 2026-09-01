# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""On-demand stack dump for a test process the CI runner believes is hung.

A test that crashes reports a traceback, because ``PYTHONFAULTHANDLER=1`` (set per test file in
``tools/run_tests.py``) installs ``faulthandler`` for ``SIGSEGV`` and friends. A test that *hangs* reported
nothing: the runner detects the hang and kills the process group with ``SIGKILL``, which cannot be caught, so
no handler ever ran. This module closes that gap by giving the runner a signal to ask for a stack first.

Two constraints decide which signal that can be:

* ``SIGTERM`` and ``SIGABRT`` are unusable. :class:`~isaaclab.app.AppLauncher` binds both to a handler that
  calls ``SimulationApp.close()``, which is itself what a shutdown hang is stuck inside, so sending either
  re-enters the hang rather than reporting it. Binding ``SIGABRT`` also displaces ``faulthandler``'s own
  handler for it.
* A Python-level :mod:`signal` handler would not run anyway. Those execute between bytecodes, and a thread
  wedged in a native Kit, CUDA, or renderer call never returns to the interpreter loop -- the same reason
  ``isaaclab.cli.multigpu`` escalates to ``SIGKILL`` when reaping stragglers.

:data:`DUMP_SIGNAL` is therefore ``SIGUSR1``, which nothing else in the repo uses, and it is registered
through :func:`faulthandler.register` rather than :mod:`signal`. That installs a C-level handler which walks
every thread and writes to a file descriptor from inside the handler, so it reports a process whose GIL is
held by a native call that will never release it.

The dump goes to a *file*, not to stderr, for the same reason ``tools/ovrtx_log.py`` keeps the renderer log
in one: pytest captures at the file-descriptor level, so it has already pointed fd 2 at a temporary file of
its own by the time this plugin loads. A dump written there is discarded with the rest of the captured output
when the process is ``SIGKILL``ed, which is exactly the case this module exists to report. Writing to a file
this module owns puts the dump somewhere pytest does not redirect and the runner can read after the process
is gone.
"""

import faulthandler
import os
import signal
import sys

DUMP_SIGNAL = getattr(signal, "SIGUSR1", None)
"""Signal the CI runner sends to ask a hung test process for a stack dump.

``None`` off POSIX. ``tools/run_tests.py`` reads this so the sender and the receiver cannot disagree.
"""

DUMP_PATH_ENV_VAR = "ISAACLAB_HANG_DUMP"
"""Environment variable naming the file stacks are dumped to. Unset (the default) disables dumping.

The runner sets it per test file, mirroring the crash journal's ``ISAACLAB_TEST_JOURNAL``. Leaving it unset
outside CI keeps a local ``pytest`` run from registering a handler nothing will ever signal.
"""

_dump_file = None
"""Open handle for the dump file, held for the process lifetime.

``faulthandler`` keeps the file *descriptor*, not the object, so dropping this reference would close the fd
out from under the handler and the dump would go nowhere.
"""


def is_supported():
    """Return whether this process can register the dump handler.

    :func:`faulthandler.register` and ``SIGUSR1`` are both POSIX-only.
    """
    return DUMP_SIGNAL is not None and hasattr(faulthandler, "register")


def dump_path():
    """Return the configured dump file, or ``""`` when dumping is disabled."""
    return os.environ.get(DUMP_PATH_ENV_VAR, "")


def size(path):
    """Return the size of ``path`` in bytes, or 0 when it does not exist yet."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def read_since(path, start):
    """Return the text appended to ``path`` after ``start`` bytes.

    Args:
        path: Dump file to read.
        start: Offset the read begins at. A file shorter than this was rewritten, so the offset no longer
            describes its contents and the whole file is read instead.

    Returns:
        The appended text, or ``""`` when there is none -- the case when the process never answered.
    """
    if size(path) < start:
        start = 0
    try:
        with open(path, "rb") as handle:
            handle.seek(start)
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def register():
    """Install the dump handler, and return whether it was installed."""
    global _dump_file
    path = dump_path()
    if not path or not is_supported():
        return False
    try:
        _dump_file = open(path, "w")  # noqa: SIM115  (held open for the process lifetime, see above)
    except OSError:
        return False
    faulthandler.register(DUMP_SIGNAL, file=_dump_file, all_threads=True, chain=False)
    return True


def pytest_configure(config):
    """Register the handler before any test imports Kit, so a startup hang is reportable too."""
    if not register() and dump_path():
        print(f"[ISAACLAB] hang stack dumps unavailable on {sys.platform}", file=sys.__stderr__, flush=True)
