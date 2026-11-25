# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shutdown output must survive fast exit without waiting for unrelated native stream locks."""

import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.kitless, pytest.mark.skipif(os.name == "nt", reason="Uses POSIX flockfile")]


def test_flush_stdio_preserves_native_output_with_an_unrelated_locked_stream():
    code = """
import ctypes
import os
import sys
import threading

from isaaclab_isaacsim.app._stdio import flush_stdio

libc = ctypes.CDLL(None)
libc.tmpfile.restype = ctypes.c_void_p
libc.flockfile.argtypes = [ctypes.c_void_p]
libc.fputs.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
names = ('__stdoutp', '__stderrp') if sys.platform == 'darwin' else ('stdout', 'stderr')
stdout, stderr = (ctypes.c_void_p.in_dll(libc, name) for name in names)
libc.fputs(b'native stdout\\n', stdout)
libc.fputs(b'native stderr\\n', stderr)
print('python stdout')
print('python stderr', file=sys.stderr)
locked = threading.Event()

def hold_file_lock():
    stream = libc.tmpfile()
    libc.flockfile(stream)
    locked.set()
    threading.Event().wait()

threading.Thread(target=hold_file_lock, daemon=True).start()
assert locked.wait(2)
flush_stdio()
os._exit(0)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode == 0, result.stderr
    assert set(result.stdout.splitlines()) == {"python stdout", "native stdout"}
    assert set(result.stderr.splitlines()) == {"python stderr", "native stderr"}
