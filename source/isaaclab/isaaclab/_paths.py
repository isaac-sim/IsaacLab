# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve paths shared by source checkouts and installed wheels."""

from pathlib import Path


def _resolve_isaaclab_root() -> Path:
    """Return the directory containing Isaac Lab runtime resources."""
    package_root = Path(__file__).resolve().parent
    if (package_root / "apps").is_dir():
        return package_root

    for parent in package_root.parents:
        if (parent / "apps").is_dir() and (parent / "source" / "isaaclab").is_dir():
            return parent

    raise RuntimeError(f"Could not locate the Isaac Lab root from {package_root}")


ISAACLAB_ROOT = _resolve_isaaclab_root()
"""Directory containing Isaac Lab runtime resources."""
