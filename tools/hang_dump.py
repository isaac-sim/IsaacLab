# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""On-demand stack dump for a test process the CI runner believes is hung.

A test that crashes reports a traceback, because ``PYTHONFAULTHANDLER=1`` (set per test file in
``tools/conftest.py``) installs ``faulthandler`` for ``SIGSEGV`` and friends. A test that *hangs* reported
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
"""

import faulthandler
import signal
import sys

DUMP_SIGNAL = getattr(signal, "SIGUSR1", None)
"""Signal the CI runner sends to ask a hung test process for a stack dump.

``None`` off POSIX. ``tools/conftest.py`` reads this so the sender and the receiver cannot disagree.
"""


def is_supported():
    """Return whether this process can register the dump handler.

    :func:`faulthandler.register` and ``SIGUSR1`` are both POSIX-only, and ``sys.__stderr__`` is ``None``
    when the interpreter starts without a real stderr.
    """
    return DUMP_SIGNAL is not None and hasattr(faulthandler, "register") and sys.__stderr__ is not None


def register():
    """Install the dump handler, and return whether it was installed.

    The dump is written to ``sys.__stderr__`` rather than :data:`sys.stderr` so it survives pytest's capture
    and lands on the pipe the runner is already draining -- the same reason ``AppLauncher`` prints its startup
    marker there.
    """
    if not is_supported():
        return False
    faulthandler.register(DUMP_SIGNAL, file=sys.__stderr__, all_threads=True, chain=False)
    return True


def pytest_configure(config):
    """Register the handler before any test imports Kit, so a startup hang is reportable too."""
    register()
