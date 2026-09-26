# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exit-status policy for a process that owns a Kit :class:`SimulationApp`."""

from __future__ import annotations

import atexit
import signal
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaacsim import SimulationApp


def install_exit_handlers(app: SimulationApp) -> None:
    """Make the process exit status truthful when it owns a Kit :class:`SimulationApp`.

    Kit fast shutdown exits with code 0 from inside :meth:`SimulationApp.close`, so every
    close carries the status the process should report:

    * Normal exit closes the app once, with exit code 1 if an exception is unhandled.
    * ``SIGTERM`` closes the app with ``128 + signum``; a repeated ``SIGTERM`` kills it.
    * ``SIGINT`` raises :class:`KeyboardInterrupt` so user code unwinds.
    * Fatal signals (``SIGSEGV``, ``SIGABRT``) are left untouched, keeping core dumps and
      the carb crash reporter.

    Args:
        app: The running Kit application.
    """

    def close_at_exit() -> None:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        app.close(exit_code=1 if getattr(sys, "last_exc", None) is not None else 0)

    def on_sigterm(signum: int, frame) -> None:
        signal.signal(signum, signal.SIG_DFL)
        app.close(exit_code=128 + signum)
        # close() only returns when fast shutdown is disabled
        signal.raise_signal(signum)

    atexit.register(close_at_exit)
    signal.signal(signal.SIGTERM, on_sigterm)
    signal.signal(signal.SIGINT, signal.default_int_handler)
