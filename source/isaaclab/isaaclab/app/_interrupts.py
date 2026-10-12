# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Internal helpers for handling interrupts around native application calls."""

from __future__ import annotations

import os
import signal
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType


def _force_exit(_signum: int, _frame: FrameType | None) -> None:
    """Exit immediately when graceful shutdown has already started."""
    os._exit(128 + signal.SIGINT)


@contextmanager
def defer_keyboard_interrupt() -> Iterator[None]:
    """Raise SIGINT after a native call returns instead of inside one of its Python callbacks.

    A second SIGINT forces termination in case the native call does not return.
    """
    interrupted = False

    def request_interrupt(signum: int, frame: FrameType | None) -> None:
        nonlocal interrupted
        if interrupted:
            _force_exit(signum, frame)
        interrupted = True

    previous_handler = signal.signal(signal.SIGINT, request_interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    if interrupted:
        raise KeyboardInterrupt


@contextmanager
def force_exit_on_sigint() -> Iterator[None]:
    """Force termination on SIGINT while graceful shutdown is already in progress."""
    previous_handler = signal.signal(signal.SIGINT, _force_exit)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous_handler)
