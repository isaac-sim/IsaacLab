# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locate the omni.usd.libs extension and promote its pxr package to a regular package.

Prints the resolved directory path to stdout (empty string if not found).
Called by isaaclab.sh / isaaclab.bat before launching the CLI so that the
shell can prepend the path to PYTHONPATH / LD_LIBRARY_PATH / PATH.
"""

import glob
import os

extscache = os.path.join(os.environ.get("ISAACLAB_PATH", ""), "_isaac_sim", "extscache")
candidates = sorted(glob.glob(os.path.join(extscache, "omni.usd.libs-*")))
if not candidates:
    print("", end="")
else:
    usd_libs_dir = candidates[-1]
    pxr_dir = os.path.join(usd_libs_dir, "pxr")
    init_py = os.path.join(pxr_dir, "__init__.py")
    if os.path.isdir(pxr_dir) and not os.path.isfile(init_py):
        try:
            open(init_py, "w").close()
        except OSError:
            pass
    print(usd_libs_dir, end="")
