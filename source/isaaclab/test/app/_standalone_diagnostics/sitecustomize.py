# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Report slow simulator subprocess stacks, including children of CLI wrappers."""

import faulthandler
import os
import sys

if interval := os.environ.get("ISAACLAB_STANDALONE_TRACEBACK_INTERVAL"):
    _dump_file = sys.stderr
    if path := os.environ.get("ISAACLAB_HANG_DUMP"):
        _dump_file = open(f"{path}.standalone-{os.getpid()}.log", "a")
        print(f"PID {os.getpid()}: {sys.argv}", file=_dump_file, flush=True)
    faulthandler.dump_traceback_later(float(interval), repeat=True, file=_dump_file)

    # Temporary native-stack capture for the two slow RTX launches.
    if path and sys.platform == "linux" and sys.argv[0] == "-m" and any(
        name in sys.argv for name in ("h1-locomotion", "nist")
    ):
        import ctypes
        import subprocess
        import threading

        # Permit the diagnostic child to inspect its parent under Linux's ptrace scope.
        ctypes.CDLL(None).prctl(0x59616D61, ctypes.c_ulong(-1), 0, 0, 0)

        def _dump_native():
            with open(f"{path}.native-{os.getpid()}-{threading.get_ident()}.log", "w") as output:
                subprocess.run(
                    ["uv", "tool", "run", "--from", "py-spy==0.4.1", "py-spy", "dump", "--native", "--pid", str(os.getpid())],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=45,
                    check=False,
                )

        for delay in (180, 240):
            timer = threading.Timer(delay, _dump_native)
            timer.daemon = True
            timer.start()
