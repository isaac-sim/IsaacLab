# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locate omni.usd.libs and promote its pxr package to a regular package.

Prints the extension directory to stdout on success, nothing on failure.
Called by isaaclab.sh / isaaclab.bat to populate PYTHONPATH / LD_LIBRARY_PATH.
"""

import glob
import os
import sys

extscache = os.path.join(os.environ.get("ISAACLAB_PATH", ""), "_isaac_sim", "extscache")
candidates = sorted(glob.glob(os.path.join(extscache, "omni.usd.libs-*")))
if candidates:
    usd_libs_dir = candidates[-1]
    init_py = os.path.join(usd_libs_dir, "pxr", "__init__.py")
    if os.path.exists(os.path.join(usd_libs_dir, "pxr")) and not os.path.isfile(init_py):
        try:
            open(init_py, "w").close()
        except OSError as exc:
            print(f"[WARNING] Cannot promote omni.usd.libs/pxr to a regular package; skipping USD path setup: {exc}", file=sys.stderr)
            sys.exit(0)
    print(usd_libs_dir, end="")
