# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for Python module / namespace manipulation."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Iterable

import lazy_loader as lazy


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
            full_mod_name = f"{package_name}.{mod_name}"
            try:
                mod = importlib.import_module(full_mod_name)
                if hasattr(mod, name):
                    val = getattr(mod, name)
                    sys.modules[package_name].__dict__[name] = val
                    return val
            except (ImportError, ModuleNotFoundError):
                continue
        for pkg in packages:
            try:
                mod = importlib.import_module(pkg)
                if hasattr(mod, name):
                    val = getattr(mod, name)
                    sys.modules[package_name].__dict__[name] = val
                    return val
            except (ImportError, ModuleNotFoundError):
                continue
        raise AttributeError(f"module {package_name!r} has no attribute {name!r}")

    def __dir__():
        names: list[str] = []
        for mod_name in submodules:
            full_mod_name = f"{package_name}.{mod_name}"
            try:
                mod = importlib.import_module(full_mod_name)
                names.extend(n for n in dir(mod) if not n.startswith("_"))
            except (ImportError, ModuleNotFoundError):
                continue
        for pkg in packages:
            try:
                mod = importlib.import_module(pkg)
                names.extend(n for n in dir(mod) if not n.startswith("_"))
            except (ImportError, ModuleNotFoundError):
                continue
        return sorted(set(names))

    return __getattr__, __dir__


def cascading_export(
    submodules: list[str],
    packages: list[str] | tuple[str, ...] = (),
) -> tuple[Callable[[str], object], Callable[[], list[str]]]:
    """Register cascading imports for the calling package's ``__init__.py``.

    This helper mirrors :func:`lazy_export` ergonomics by inferring the caller's
    package name and installing ``__getattr__`` and ``__dir__`` on that module.
    """
    caller_globals = sys._getframe(1).f_globals
    package_name = caller_globals["__name__"]
    __getattr__, __dir__ = attach_cascading(package_name, submodules=submodules, packages=packages)

    mod = sys.modules[package_name]
    setattr(mod, "__getattr__", __getattr__)
    setattr(mod, "__dir__", __dir__)
    return __getattr__, __dir__


def lazy_export(
    *imports: tuple[str, str | Iterable[str]],
    submodules: list[str] | None = None,
) -> tuple[Callable[[str], object], Callable[[], list[str]], list[str]]:
    """Register lazy imports for the calling package's ``__init__.py``.

    This helper wraps :func:`lazy_loader.attach` for explicit name→submodule
    mappings and optionally chains with :func:`attach_cascading` for star-import
    style submodules whose public names should be re-exported.
    """
    caller_globals = sys._getframe(1).f_globals
    package_name = caller_globals["__name__"]

    submod_attrs: dict[str, list[str]] = {}
    for submod, names in imports:
        entry = [names] if isinstance(names, str) else list(names)
        if submod in submod_attrs:
            submod_attrs[submod].extend(entry)
        else:
            submod_attrs[submod] = entry

    lazy_getattr, lazy_dir, __all__ = lazy.attach(package_name, submodules=[], submod_attrs=submod_attrs)

    if submodules:
        cascading_getattr, cascading_dir = attach_cascading(package_name, submodules=submodules)

        def __getattr__(name: str):
            try:
                return lazy_getattr(name)
            except AttributeError:
                return cascading_getattr(name)

        def __dir__():
            return sorted(set(lazy_dir()) | set(cascading_dir()))
    else:
        __getattr__ = lazy_getattr
        __dir__ = lazy_dir

    mod = sys.modules[package_name]
    setattr(mod, "__getattr__", __getattr__)
    setattr(mod, "__dir__", __dir__)
    setattr(mod, "__all__", __all__)
    return __getattr__, __dir__, __all__
