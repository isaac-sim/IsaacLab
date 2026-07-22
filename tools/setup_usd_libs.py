# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locate omni.usd.libs and promote its pxr package to a regular package.

Prints the extension directory to stdout on success, nothing on failure.
Called by isaaclab.sh / isaaclab.bat to populate PYTHONPATH / LD_LIBRARY_PATH.
"""

import glob
import importlib.util
import os
import sys


def _find_extscache() -> str:
    # Binary / symlink install: ISAACLAB_PATH/_isaac_sim/extscache.
    # Only probe when ISAACLAB_PATH is set; an empty value would produce a
    # relative path that resolves against CWD and may accidentally match a
    # _isaac_sim symlink in an unrelated working directory.
    isaaclab_path = os.environ.get("ISAACLAB_PATH", "")
    if isaaclab_path:
        symlink_path = os.path.join(isaaclab_path, "_isaac_sim", "extscache")
        if os.path.isdir(symlink_path):
            return symlink_path

    # Wheel / pip install: locate the isaacsim package without importing it so
    # that this script can run before PYTHONPATH is fully configured.
    spec = importlib.util.find_spec("isaacsim")
    if spec is not None and spec.origin:
        return os.path.join(os.path.dirname(spec.origin), "extscache")

    return ""


extscache = _find_extscache()
candidates = sorted(glob.glob(os.path.join(extscache, "omni.usd.libs-*"))) if extscache else []
if candidates:
    usd_libs_dir = candidates[-1]
    init_py = os.path.join(usd_libs_dir, "pxr", "__init__.py")
    if os.path.exists(os.path.join(usd_libs_dir, "pxr")) and not os.path.isfile(init_py):
        try:
            open(init_py, "w").close()
        except OSError as exc:
            print(
                f"[WARNING] Cannot promote omni.usd.libs/pxr to a regular package; skipping USD path setup: {exc}",
                file=sys.stderr,
            )
            sys.exit(0)
    print(usd_libs_dir, end="")
