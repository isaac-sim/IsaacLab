# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for resolving Isaac Lab installation paths."""

import os
from pathlib import Path

import isaaclab

ISAACLAB_PATH_ENV_VAR = "ISAACLAB_PATH"


def resolve_isaaclab_path(package_file: str | Path = __file__) -> Path:
    """Resolve the Isaac Lab installation path.

    Args:
        package_file: File inside or below the ``isaaclab`` Python package.

    Returns:
        Path to the Isaac Lab installation root.
    """
    if isaaclab_path := os.environ.get(ISAACLAB_PATH_ENV_VAR):
        return Path(isaaclab_path).expanduser().resolve()

    package_path = Path(package_file).resolve()
    if checkout_root := _find_source_checkout_root(package_path):
        return checkout_root

    if isaaclab.__file__ is not None:
        package_root = Path(isaaclab.__file__).resolve().parent
        if _has_installed_resources(package_root):
            return package_root

    return package_path.parents[4]


def _find_source_checkout_root(package_path: Path) -> Path | None:
    """Find the source checkout root containing a package file."""
    for parent in package_path.parents:
        if (parent / "apps").is_dir() and (parent / "scripts").is_dir():
            return parent
    return None


def _has_installed_resources(path: Path) -> bool:
    """Return whether an installed package root carries Isaac Lab resources."""
    return (path / "apps").is_dir()
