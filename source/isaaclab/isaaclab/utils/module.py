# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for Python module / namespace manipulation."""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable

import lazy_loader as lazy


def lazy_export(
    *,
    packages: list[str] | tuple[str, ...] | None = None,
) -> tuple[Callable[[str], object], Callable[[], list[str]], list[str]]:
    """Lazy-load names from a ``.pyi`` stub, with optional cross-package fallback.

    Call with no arguments to lazily export everything declared in the
    adjacent ``.pyi`` stub::

        from isaaclab.utils.module import lazy_export

        lazy_export()

    When a module re-exports names from another package at runtime (e.g.
    task MDP modules that fall back to ``isaaclab.envs.mdp``), pass the
    package names to scan as a fallback::

        from isaaclab.utils.module import lazy_export

        lazy_export(packages=["isaaclab.envs.mdp"])

    Args:
        packages: Fully-qualified package names to fall back to when a
            name is not found in the local ``.pyi`` stub.  When *None*
            (the default), only the stub is used.
    """
    caller_globals = sys._getframe(1).f_globals
    package_name: str = caller_globals["__name__"]
    caller_file: str = caller_globals["__file__"]

    if packages is None:
        __getattr__, __dir__, __all__ = lazy.attach_stub(package_name, caller_file)
        mod = sys.modules[package_name]
        setattr(mod, "__getattr__", __getattr__)
        setattr(mod, "__dir__", __dir__)
        setattr(mod, "__all__", __all__)
        return __getattr__, __dir__, __all__

    stub_file = f"{os.path.splitext(caller_file)[0]}.pyi"
    has_stub = os.path.exists(stub_file)

    if has_stub:
        stub_getattr, stub_dir, __all__ = lazy.attach_stub(package_name, caller_file)
    else:
        __all__: list[str] = []

    def _pkg_getattr(name: str):
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

    def _pkg_dir():
        names: list[str] = []
        for pkg in packages:
            try:
                mod = importlib.import_module(pkg)
                names.extend(n for n in dir(mod) if not n.startswith("_"))
            except (ImportError, ModuleNotFoundError):
                continue
        return sorted(set(names))

    if has_stub:

        def __getattr__(name: str):
            try:
                return stub_getattr(name)
            except AttributeError:
                return _pkg_getattr(name)

        def __dir__():
            return sorted(set(stub_dir()) | set(_pkg_dir()))

    else:

        def __getattr__(name: str):
            return _pkg_getattr(name)

        def __dir__():
            return _pkg_dir()

    mod = sys.modules[package_name]
    setattr(mod, "__getattr__", __getattr__)
    setattr(mod, "__dir__", __dir__)
    setattr(mod, "__all__", __all__)
    return __getattr__, __dir__, __all__
