# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools

import warp as wp

##
# Capture safety — property guard.
##


def capture_unsafe(reason: str | None = None):
    """Mark a callable as not CUDA-graph-capture-safe.

    Raises ``RuntimeError`` if the decorated callable is invoked while
    ``wp.get_device().is_capturing`` is ``True``.

    Args:
        reason: Optional explanation appended to the error message.

    Usage::

        @property
        @capture_unsafe("Relies on a Python timestamp guard.")
        def projected_gravity_b(self) -> wp.array: ...
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if wp.get_device().is_capturing:
                msg = f"'{func.__qualname__}' cannot be called during CUDA graph capture."
                if reason:
                    msg = f"{msg} {reason}"
                raise RuntimeError(msg)
            return func(*args, **kwargs)

        return wrapper

    return decorator
