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
