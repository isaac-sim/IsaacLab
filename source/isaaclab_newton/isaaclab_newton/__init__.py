# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing the Newton simulation interfaces for IsaacLab core package."""

import importlib.metadata


def _prepare_warp_imports() -> None:
    """Keep Newton imports bound to Isaac Lab's pip-installed Warp package."""
    try:
        from isaaclab import _deprioritize_prebundle_paths
    except (AttributeError, ImportError):
        def _deprioritize_prebundle_paths() -> None:
            return

    _deprioritize_prebundle_paths()
    try:
        import warp  # noqa: F401

        _deprioritize_prebundle_paths()
        import warp.fem  # noqa: F401
    except ImportError:
        # Let the concrete Newton import site surface dependency errors with
        # its normal context. This hook only corrects path ordering when Warp is
        # available.
        return


_prepare_warp_imports()


try:
    __version__ = importlib.metadata.version("isaaclab_newton")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
