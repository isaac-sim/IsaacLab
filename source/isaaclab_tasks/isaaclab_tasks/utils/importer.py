# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with utility for importing all modules in a package recursively."""

from __future__ import annotations

import importlib
import pkgutil
import sys


def import_packages(package_name: str, blacklist_pkgs: list[str] | None = None):
    """Import all sub-packages in a package recursively.

    Only **packages** (directories with ``__init__.py``) are imported — plain
    ``.py`` modules (e.g. ``env_cfg.py``, ``env.py``) are skipped.  This is
    sufficient because ``gym.register()`` calls live exclusively in
    ``__init__.py`` files, and avoids eagerly importing every config module
    at startup.

    Args:
        package_name: The package name.
        blacklist_pkgs: The list of blacklisted packages to skip. Defaults to None,
            which means no packages are blacklisted.
    """
    # Default blacklist
    if blacklist_pkgs is None:
        blacklist_pkgs = []
    # Import the package itself
    package = importlib.import_module(package_name)
    # Import all Python files
    for _ in _walk_packages(package.__path__, package.__name__ + ".", blacklist_pkgs=blacklist_pkgs):
        pass


"""
Internal helpers.
"""


def _walk_packages(
    path: str | None = None,
    prefix: str = "",
    onerror: callable | None = None,
    blacklist_pkgs: list[str] | None = None,
):
    """Yields ModuleInfo for all modules recursively on path, or, if path is None, all accessible modules.

    Note:
        This function is a modified version of the original ``pkgutil.walk_packages`` function. It adds
        the ``blacklist_pkgs`` argument to skip blacklisted packages. Please refer to the original
        ``pkgutil.walk_packages`` function for more details.

    """
    # Default blacklist
    if blacklist_pkgs is None:
        blacklist_pkgs = []

    def seen(p: str, m: dict[str, bool] = {}) -> bool:
        """Check if a package has been seen before."""
        if p in m:
            return True
        m[p] = True
        return False

    for info in pkgutil.iter_modules(path, prefix):
        # check blacklisted
        if any([black_pkg_name in info.name for black_pkg_name in blacklist_pkgs]):
            continue

        # Only import packages (directories with __init__.py), not plain .py
        # modules.  The walk exists to trigger gym.register() calls which live
        # exclusively in __init__.py files.  Skipping bare modules avoids
        # eagerly importing every env_cfg / env / agent config at startup.
        if not info.ispkg:
            continue

        yield info

        try:
            __import__(info.name)
        except Exception:
            if onerror is not None:
                onerror(info.name)
            else:
                raise
        else:
            path: list = getattr(sys.modules[info.name], "__path__", [])

            # don't traverse path items we've seen before
            path = [p for p in path if not seen(p)]

            yield from _walk_packages(path, info.name + ".", onerror, blacklist_pkgs)
