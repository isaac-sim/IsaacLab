# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register a stack dump in every Python process a standalone-script test launches.

The harness puts this directory on ``PYTHONPATH``. Each Python process then registers the same ``SIGUSR1``
:mod:`faulthandler` dump as ``tools/hang_dump.py``, written to ``<pid>.log`` in the directory named by
``ISAACLAB_SCRIPT_HANG_DUMP_DIR``.
"""

import faulthandler
import os
import signal

_dump_dir = os.environ.get("ISAACLAB_SCRIPT_HANG_DUMP_DIR")
if _dump_dir and hasattr(signal, "SIGUSR1"):
    # faulthandler keeps only the file descriptor, so the file object must stay referenced.
    _dump_file = open(os.path.join(_dump_dir, f"{os.getpid()}.log"), "w")  # noqa: SIM115
    faulthandler.register(signal.SIGUSR1, file=_dump_file, all_threads=True, chain=False)
