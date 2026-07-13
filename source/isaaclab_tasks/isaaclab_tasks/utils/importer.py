# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with utility for importing all modules in a package recursively."""

from __future__ import annotations

import importlib
import importlib.machinery
import pkgutil
import sys


def import_packages(package_name: str, blacklist_pkgs: list[str] | None = None):
    """Import all sub-packages in a package recursively.

    Only **packages** (directories with ``__init__.py``) that contain
    ``gym.register`` calls are imported — plain ``.py`` modules (e.g.
    ``env_cfg.py``, ``env.py``) and registry-free package parents are skipped.
    This is sufficient because task registration calls live exclusively in
    ``__init__.py`` files, and avoids eagerly importing config helper modules
    while walking the tree.

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

        if not info.ispkg:
            continue

        module_spec = info.module_finder.find_spec(info.name, None)
        if module_spec is None:
            continue

        child_path: list = list(module_spec.submodule_search_locations or [])

        # Only import package initializers that actually register tasks.  We
        # still recurse into registry-free parents using the package paths from
        # the module spec so nested registration packages are discovered without
        # executing every parent ``__init__``.
        if _has_gym_registration(module_spec):
            yield info

            try:
                __import__(info.name)
            except Exception:
                if onerror is not None:
                    onerror(info.name)
                else:
                    raise
            else:
                child_path = list(getattr(sys.modules[info.name], "__path__", child_path))

        child_path = [p for p in child_path if not seen(p)]
        yield from _walk_packages(child_path, info.name + ".", onerror, blacklist_pkgs)


def _has_gym_registration(module_spec: importlib.machinery.ModuleSpec) -> bool:
    """Return whether a package initializer contains Gym task registrations."""
    init_path = module_spec.origin
    if init_path is None:
        return False
    try:
        with open(init_path, "rb") as init_file:
            return b"gym.register" in init_file.read()
    except OSError:
        return False
