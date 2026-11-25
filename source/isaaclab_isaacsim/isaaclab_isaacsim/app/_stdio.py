# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flush shutdown output without locking unrelated native file streams."""

import ctypes
import os
import sys


def flush_stdio() -> None:
    """Flush Python and native stdout/stderr before Kit's fast shutdown exits the process.

    Flushing every C stream with ``fflush(NULL)`` can wait forever for a file lock held by a
    background thread. Only the two standard output streams need flushing before ``_exit``.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (ValueError, OSError, AttributeError):
            pass
    try:
        libc = ctypes.CDLL("ucrtbase" if os.name == "nt" else None)
        libc.fflush.argtypes = [ctypes.c_void_p]
        libc.fflush.restype = ctypes.c_int
        if os.name == "nt":
            libc.__acrt_iob_func.argtypes = [ctypes.c_uint]
            libc.__acrt_iob_func.restype = ctypes.c_void_p
            streams = (libc.__acrt_iob_func(1), libc.__acrt_iob_func(2))
        else:
            names = ("__stdoutp", "__stderrp") if sys.platform == "darwin" else ("stdout", "stderr")
            streams = tuple(ctypes.c_void_p.in_dll(libc, name) for name in names)
        for stream in streams:
            libc.fflush(stream)
    except (ImportError, OSError, AttributeError, TypeError, ValueError):
        pass
