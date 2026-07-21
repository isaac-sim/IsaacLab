# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
from importlib.metadata import version
from importlib.util import find_spec

__version__ = version("isaaclab")

# Extend the package search path so subpackages (app/, envs/, etc.) in the
# nested source tree are importable as isaaclab.app, isaaclab.envs, etc.
__path__.append(os.path.join(os.path.dirname(__file__), "source", "isaaclab", "isaaclab"))


def _deprioritize_prebundle_paths():
    """Move Isaac Sim ``pip_prebundle`` and known conflicting extension directories to the end of ``sys.path``.

    Isaac Sim's ``setup_python_env.sh`` injects ``pip_prebundle`` directories
    onto ``PYTHONPATH``. These contain older copies of packages like torch,
    warp, and nvidia-cudnn that shadow the versions installed by Isaac Lab,
    causing CUDA runtime errors.

    Additionally, certain Isaac Sim kit extensions (such as ``omni.warp.core``)
    bundle their own copies of Python packages that conflict with pip-installed
    versions. When loaded by the extension system these paths can appear on
    ``sys.path`` before ``site-packages``, leading to version mismatches.

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
        "omni.warp.core",
        "omni.isaac.ml_archive",
        "omni.isaac.core_archive",
        "omni.kit.pip_archive",
        "isaacsim.pip.newton",
    )

    def _should_demote(path: str) -> bool:
        norm = path.replace("\\", "/").lower()
        if "pip_prebundle" in norm:
            return True
        for frag in _CONFLICTING_EXT_FRAGMENTS:
            if frag.lower() in norm:
                return True
        return False

    def _demote_paths(paths):
        # Partition: keep non-conflicting in place, collect conflicting.
        clean = []
        demoted = []
        for p in paths:
            if _should_demote(p):
                demoted.append(p)
            else:
                clean.append(p)
        return clean + demoted, bool(demoted)

    # Rebuild sys.path: originals first, then demoted at the very end.
    sys.path[:], _ = _demote_paths(sys.path)

    # Rewrite PYTHONPATH with the same ordering for subprocesses.
    if "PYTHONPATH" in os.environ:
        parts = os.environ["PYTHONPATH"].split(os.pathsep)
        os.environ["PYTHONPATH"] = os.pathsep.join(_demote_paths(parts)[0])

    # Kit can mutate a loaded package's search path directly.  If ``warp`` is
    # already imported and an older ``omni.warp.core`` path is appended to
    # ``warp.__path__``, a later ``import warp.fem`` can still mix the extension
    # package with Isaac Lab's pip-installed Warp even when ``sys.path`` is
    # clean.  Reorder loaded package paths with the same policy.
    for module in tuple(sys.modules.values()):
        package_path = getattr(module, "__path__", None)
        if package_path is None:
            continue
        try:
            reordered, changed = _demote_paths(list(package_path))
        except TypeError:
            continue
        if not changed:
            continue
        try:
            package_path[:] = reordered
        except TypeError:
            try:
                module.__path__ = reordered
            except Exception:
                pass


_deprioritize_prebundle_paths()


# TODO(myurasov-nv): bootstrap_kernel() is ported from the internal GitLab wheel builder
# for backwards compatibility. It is not called currently, but may be needed if Isaac Sim
# requires explicit kernel bootstrapping before use. Remove once confirmed unnecessary.
def bootstrap_kernel():
    # Isaac Lab path
    isaaclab_path = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

    # bootstrap kernel via Isaac Sim
    if find_spec("isaacsim") is not None:
        import isaacsim

        # log info
        if find_spec("carb") is not None:
            import carb
            carb.log_info(f"Isaac Lab path: {isaaclab_path}")


def main():
    """Entry point for the ``isaaclab`` console script (python -m isaaclab)."""
    from isaaclab.__main__ import main as _main

    sys.exit(_main())


if __name__ == "__main__":
    bootstrap_kernel()
    main()
