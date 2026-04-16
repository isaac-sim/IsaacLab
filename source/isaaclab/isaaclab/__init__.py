# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing the core framework."""

import os
import sys


def _filter_prebundle_paths():
    """Remove Isaac Sim ``pip_prebundle`` directories from ``sys.path``.

    Isaac Sim's ``setup_python_env.sh`` injects ``pip_prebundle`` directories
    (e.g. ``omni.isaac.ml_archive/pip_prebundle``) onto ``PYTHONPATH``.  These
    contain older copies of packages like torch, warp, and nvidia-cudnn that
    shadow the versions installed by Isaac Lab, causing CUDA runtime errors.

    This function strips those entries from ``sys.path`` (and ``PYTHONPATH`` in
    the environment so child processes also stay clean) early — before any
    ``import torch`` can resolve to the wrong package.

    Only paths that include *both* ``pip_prebundle`` and one of the known
    conflicting extensions are removed.  Other ``pip_prebundle`` paths (e.g.
    ``isaacsim.robot_motion.lula``) are left alone since they don't conflict.

    The removed prebundle dirs also carry NVIDIA shared libraries
    (``libcudart``, ``libcudnn``, …) that torch loads via ``ctypes.CDLL``.
    To keep those discoverable after the ``sys.path`` entry is gone, the
    function appends their ``nvidia/*/lib`` directories to
    ``LD_LIBRARY_PATH``.
    """
    import glob

    # Extensions whose prebundled packages conflict with Isaac Lab deps.
    # Only ml_archive is listed because it prebundles an older torch + nvidia
    # CUDA libs that shadow the versions installed by Isaac Lab.  Other
    # pip_prebundle directories (core_archive, pip_archive, etc.) contain
    # packages the runtime genuinely needs and must stay on the path.
    _CONFLICTING_EXTS = (
        "omni.isaac.ml_archive",
    )

    def _is_conflicting(path: str) -> bool:
        norm = path.replace("\\", "/").lower()
        return "pip_prebundle" in norm and any(ext.lower() in norm for ext in _CONFLICTING_EXTS)

    # Collect the paths we are about to remove so we can salvage their
    # nvidia shared-library directories afterwards.
    removed_paths = [p for p in sys.path if _is_conflicting(p)]

    if not removed_paths:
        return

    # Filter sys.path in-place.
    sys.path[:] = [p for p in sys.path if not _is_conflicting(p)]

    # Filter PYTHONPATH so subprocesses inherit the clean version.
    if "PYTHONPATH" in os.environ:
        parts = os.environ["PYTHONPATH"].split(os.pathsep)
        os.environ["PYTHONPATH"] = os.pathsep.join(p for p in parts if not _is_conflicting(p))

    # Preserve NVIDIA shared libraries (libcudart, libcudnn, …) that torch
    # loads at runtime via ctypes.  These live under <prebundle>/nvidia/*/lib.
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    ld_dirs = ld_path.split(os.pathsep) if ld_path else []
    ld_dirs_set = set(ld_dirs)
    for prebundle_dir in removed_paths:
        for lib_dir in glob.glob(os.path.join(prebundle_dir, "nvidia", "*", "lib")):
            if os.path.isdir(lib_dir) and lib_dir not in ld_dirs_set:
                ld_dirs.append(lib_dir)
                ld_dirs_set.add(lib_dir)
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(ld_dirs)


_filter_prebundle_paths()

# Conveniences to other module directories via relative paths.
ISAACLAB_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

# The CLI imports this module to run installation. We must handle the case where
# dependencies (like toml) are not yet installed in a fresh environment.
# This prevents ImportError during the initial bootstrap phase.
try:
    import toml
    ISAACLAB_METADATA = toml.load(os.path.join(ISAACLAB_EXT_DIR, "config", "extension.toml"))
    """Extension metadata dictionary parsed from the extension.toml file."""
    __version__ = ISAACLAB_METADATA["package"]["version"]
except ImportError:
    # Check for tomllib (Python 3.11+).
    try:
        import tomllib
        with open(os.path.join(ISAACLAB_EXT_DIR, "config", "extension.toml"), "rb") as f:
            ISAACLAB_METADATA = tomllib.load(f)
        __version__ = ISAACLAB_METADATA["package"]["version"]
    except (ImportError, FileNotFoundError):
        # Tomllib is not part of the standard library before Python 3.11.
        # Stub is good enough for installation purposes.
        ISAACLAB_METADATA = {"package": {"version": "0.0.0"}}
        __version__ = "0.0.0"
