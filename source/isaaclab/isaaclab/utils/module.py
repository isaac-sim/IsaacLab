# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for Python module / namespace manipulation."""

from __future__ import annotations

import ast
import importlib
import os
import sys
import tempfile
from collections.abc import Callable

import lazy_loader as lazy


def _filter_stub(stub_file: str) -> str | None:
    """Return a path to a filtered copy of *stub_file* that ``lazy_loader`` can parse.

    ``lazy_loader.attach_stub`` only supports relative (``from .x import y``)
    imports and rejects absolute imports and star (``*``) imports.  This helper
    strips those unsupported nodes from the AST so the remaining (local)
    relative imports can still be resolved through ``attach_stub``.

    Returns the path to a temporary filtered ``.pyi`` file, or *None* if no
    filtering was needed (i.e. the original stub is already compatible).
    """
    with open(stub_file) as f:
        source = f.read()

    tree = ast.parse(source)

    needs_filter = False
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level != 1 or any(alias.name == "*" for alias in node.names):
                needs_filter = True
                break

    if not needs_filter:
        return None

    filtered = ast.Module(body=[], type_ignores=[])
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.level != 1:
                continue
            if any(alias.name == "*" for alias in node.names):
                continue
        filtered.body.append(node)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pyi", delete=False) as tmp:
        tmp.write(ast.unparse(filtered))
    return tmp.name


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
        filtered_stub = _filter_stub(stub_file)
        if filtered_stub is not None:
            stub_getattr, stub_dir, __all__ = lazy.attach_stub(package_name, filtered_stub)
            os.unlink(filtered_stub)
        else:
            stub_getattr, stub_dir, __all__ = lazy.attach_stub(package_name, caller_file)

    if not has_stub:
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
