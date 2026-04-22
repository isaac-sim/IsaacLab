# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing the core framework."""

import os
import sys


def _deprioritize_prebundle_paths():
    """Move Isaac Sim ``pip_prebundle`` and known conflicting extension directories to the end of ``sys.path``.

    Isaac Sim's ``setup_python_env.sh`` injects ``pip_prebundle`` directories
    onto ``PYTHONPATH``.  These contain older copies of packages like torch,
    warp, and nvidia-cudnn that shadow the versions installed by Isaac Lab,
    causing CUDA runtime errors.

    Additionally, certain Isaac Sim kit extensions (such as
    ``isaacsim.pip.newton``) bundle their own copies of Python packages that
    conflict with pip-installed versions.  When loaded by the extension system
    these paths can appear on ``sys.path`` before ``site-packages``, leading to
    version mismatches.

    Rather than removing these paths entirely (which would break packages like
    ``sympy`` that only exist in the prebundle), this function moves them to
    the **end** of ``sys.path`` so that pip-installed packages in
    ``site-packages`` take priority.

    The ``PYTHONPATH`` environment variable is also rewritten so that child
    processes inherit the corrected ordering.
    """

    # Extension directory fragments that are known to ship Python packages
    # which conflict with Isaac Lab's pip-installed versions.
    _CONFLICTING_EXT_FRAGMENTS = (
        "omni.isaac.ml_archive",
        "omni.isaac.core_archive",
        "omni.kit.pip_archive",
        "isaacsim.pip.newton",
    )

    def _should_demote(path: str) -> bool:
        norm = path.replace("\\", "/").lower()
        if "pip_prebundle" in norm:
            return True
        # Kit ships its own Python interpreter and installs packages into
        # kit/python/lib/python3.X/site-packages.  When a newer or differently-
        # configured version of a package (e.g. warp) is installed there, it
        # takes precedence over pip-installed packages and can cause runtime
        # failures.  Demote these site-packages to the end of sys.path while
        # leaving the Kit stdlib path (kit/python/lib/python3.X without
        # site-packages) in place so Kit's standard-library extras still work.
        if "kit/python/lib" in norm and "site-packages" in norm:
            return True
        for frag in _CONFLICTING_EXT_FRAGMENTS:
            if frag.lower() in norm:
                return True
        return False

    # Partition: keep non-conflicting in place, collect conflicting.
    clean = []
    demoted = []
    for p in sys.path:
        if _should_demote(p):
            demoted.append(p)
        else:
            clean.append(p)

    if not demoted:
        return

    # Rebuild sys.path: originals first, then demoted at the very end.
    sys.path[:] = clean + demoted

    # Rewrite PYTHONPATH with the same ordering for subprocesses.
    if "PYTHONPATH" in os.environ:
        parts = os.environ["PYTHONPATH"].split(os.pathsep)
        env_clean = []
        env_demoted = []
        for p in parts:
            if _should_demote(p):
                env_demoted.append(p)
            else:
                env_clean.append(p)
        os.environ["PYTHONPATH"] = os.pathsep.join(env_clean + env_demoted)


_deprioritize_prebundle_paths()


def _pin_warp_import():
    """Import ``warp`` now to lock ``sys.modules['warp']`` to the correct version.

    Kit's extension system may add Kit's own
    ``kit/python/lib/python3.X/site-packages`` to ``sys.path`` during
    ``SimulationApp`` startup, because Kit scans extension directories as part
    of its registry process.  Any extension that imports ``warp`` during that
    window (e.g. ``omni.replicator.core``) would set ``sys.modules['warp']`` to
    the bundled copy before our second ``_deprioritize_prebundle_paths()`` call
    in ``AppLauncher`` has a chance to run.

    By importing ``warp`` here — after ``_deprioritize_prebundle_paths()`` has
    already demoted the pip_prebundle and ``kit/python/lib`` site-packages
    paths — we ensure the pip-managed ``warp-lang`` is the one cached in
    ``sys.modules``.  Subsequent ``import warp`` calls from Kit extensions all
    return that cached module, so there is only ever one Warp runtime in the
    process.

    Failure to import (e.g. warp not yet installed during initial setup) is
    silently ignored; the import will succeed once the user has run
    ``./isaaclab.sh --install``.
    """
    try:
        import warp as _warp  # noqa: F401
    except ImportError:
        return

    # Warn if the loaded warp version is incompatible with omni.replicator.core.
    # Warp >= 1.13 deprecates warp.types.array with changed semantics; when
    # omni.replicator.core (which uses that symbol) is loaded with warp >= 1.13,
    # it triggers CUDA error 700 (illegal memory access) that poisons the CUDA
    # context and causes all subsequent warp kernel lookups to fail.
    # Run './isaaclab.sh --install' to install a compatible warp-lang version.
    import warnings

    _warp_version = getattr(_warp, "version", None)
    if _warp_version is not None:
        try:
            _parts = [int(x) for x in str(_warp_version).split(".")[:2]]
            if len(_parts) >= 2 and (_parts[0], _parts[1]) >= (1, 13):
                warnings.warn(
                    f"[IsaacLab] warp {_warp_version} is incompatible with "
                    f"omni.replicator.core.  Warp >= 1.13 deprecates "
                    f"``warp.types.array`` with changed semantics, which causes "
                    f"CUDA error 700 (illegal memory access) during rendering and "
                    f"makes all subsequent warp kernel lookups fail.  Run "
                    f"'./isaaclab.sh --install' to install warp-lang<1.13.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        except (ValueError, AttributeError):
            pass


_pin_warp_import()

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
