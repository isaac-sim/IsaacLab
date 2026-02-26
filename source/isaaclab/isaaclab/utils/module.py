# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for Python module / namespace manipulation."""

from __future__ import annotations

import importlib
import sys


def attach_cascading(
    package_name: str,
    submodules: list[str],
    packages: list[str] | tuple[str, ...] = (),
) -> tuple:
    """Create ``__getattr__`` and ``__dir__`` for a *cascading namespace* package.

    Replaces the hand-rolled scanning ``__getattr__`` boilerplate that used to
    appear in every ``mdp/__init__.py``.  Usage::

        from isaaclab.utils.module import attach_cascading

        __getattr__, __dir__ = attach_cascading(
            __name__,
            submodules=["rewards", "terminations"],
            packages=["isaaclab.envs.mdp"],
        )

    Lookup order
    ------------
    1. Local submodules (tried in the order given).
    2. Fallback packages (tried in the order given).

    Once an attribute is resolved it is written into the calling module's
    ``__dict__`` so that subsequent accesses bypass ``__getattr__`` entirely.

    Args:
        package_name: Value of ``__name__`` in the calling ``__init__.py``.
        submodules: Submodule names relative to *package_name* to scan first.
        packages: Fully-qualified package names to fall back to.

    Returns:
        A ``(__getattr__, __dir__)`` tuple ready for direct assignment.
    """

    def __getattr__(name: str):
        for mod_name in submodules:
            try:
                mod = importlib.import_module(f"{package_name}.{mod_name}")
                if hasattr(mod, name):
                    val = getattr(mod, name)
                    sys.modules[package_name].__dict__[name] = val
                    return val
            except ImportError:
                pass
        for pkg in packages:
            try:
                mod = importlib.import_module(pkg)
                if hasattr(mod, name):
                    val = getattr(mod, name)
                    sys.modules[package_name].__dict__[name] = val
                    return val
            except ImportError:
                pass
        raise AttributeError(f"module {package_name!r} has no attribute {name!r}")

    def __dir__():
        names: list[str] = []
        for mod_name in submodules:
            try:
                mod = importlib.import_module(f"{package_name}.{mod_name}")
                names.extend(n for n in dir(mod) if not n.startswith("_"))
            except ImportError:
                pass
        for pkg in packages:
            try:
                mod = importlib.import_module(pkg)
                names.extend(n for n in dir(mod) if not n.startswith("_"))
            except ImportError:
                pass
        return sorted(set(names))

    return __getattr__, __dir__
