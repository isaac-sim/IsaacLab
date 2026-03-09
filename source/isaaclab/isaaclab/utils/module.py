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
import warnings
from collections.abc import Callable

import lazy_loader as lazy


def _parse_stub(stub_file: str) -> tuple[str | None, list[str], list[str]]:
    """Parse a ``.pyi`` stub in a single AST pass.

    Returns:
        A 3-tuple of ``(filtered_path, fallback_packages, relative_wildcards)``.

        *filtered_path* is a temporary ``.pyi`` containing only explicit
        relative imports (what ``lazy_loader`` can handle), or ``None`` when
        no filtering was needed.  The temporary file may be empty when the
        stub contained only wildcard imports; this is intentional — passing
        the original stub to ``lazy_loader`` would raise ``ValueError``
        because it does not support absolute or wildcard imports.

        *fallback_packages* lists fully-qualified package names extracted from
        absolute wildcard imports (``from pkg import *``).

        *relative_wildcards* lists submodule names extracted from relative
        wildcard imports (``from .mod import *``).
    """
    with open(stub_file) as f:
        source = f.read()

    tree = ast.parse(source)

    fallback_packages: list[str] = []
    relative_wildcards: list[str] = []
    filtered_body: list[ast.stmt] = []
    needs_filter = False

    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            is_star = any(alias.name == "*" for alias in node.names)
            if node.level == 1 and not is_star:
                filtered_body.append(node)
                continue
            if node.level == 0 and is_star and node.module:
                fallback_packages.append(node.module)
            elif node.level == 1 and is_star and node.module:
                relative_wildcards.append(node.module)
            needs_filter = True
        else:
            filtered_body.append(node)

    if not needs_filter:
        return None, fallback_packages, relative_wildcards

    filtered = ast.Module(body=filtered_body, type_ignores=[])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pyi", delete=False) as tmp:
        tmp.write(ast.unparse(filtered))

    return tmp.name, fallback_packages, relative_wildcards


def lazy_export(
    *,
    packages: list[str] | tuple[str, ...] | None = None,
) -> tuple[Callable[[str], object], Callable[[], list[str]], list[str]]:
    """Lazy-load names from a ``.pyi`` stub.

    The ``.pyi`` stub is the single source of truth for what a module exports.
    ``lazy_export()`` reads the stub and derives everything from it:

    * ``from .rewards import foo, bar`` — lazy-loads specific names from a
      local submodule (existing ``lazy_loader`` behaviour).
    * ``from .rewards import *`` — eagerly imports the submodule and
      re-exports all of its public names at ``lazy_export()`` time.
    * ``from isaaclab.envs.mdp import *`` — sets up a lazy fallback so that
      any name not found locally is resolved from the specified package.

    Basic usage (no wildcards)::

        from isaaclab.utils.module import lazy_export

        lazy_export()

    With a ``.pyi`` stub that contains ``from isaaclab.envs.mdp import *``
    and/or ``from .rewards import *``, no extra arguments are needed —
    ``lazy_export()`` infers the behaviour from the stub.

    Args:
        packages: **Deprecated.**  Fallback packages are now inferred from
            absolute wildcard imports in the ``.pyi`` stub.  Passing this
            argument still works but emits a :class:`DeprecationWarning`.

    Raises:
        ImportError: If the ``.pyi`` stub declares ``from pkg import *`` but
            *pkg* is not installed.
    """
    caller_globals = sys._getframe(1).f_globals
    package_name: str = caller_globals["__name__"]
    caller_file: str = caller_globals["__file__"]

    if packages is not None:
        warnings.warn(
            "The 'packages' argument to lazy_export() is deprecated. "
            "Add 'from <pkg> import *' to your .pyi stub instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    stub_file = f"{os.path.splitext(caller_file)[0]}.pyi"
    has_stub = os.path.exists(stub_file)

    fallback_packages: list[str] = list(packages) if packages else []
    relative_wildcards: list[str] = []

    if has_stub:
        filtered_path, stub_fallbacks, relative_wildcards = _parse_stub(stub_file)
        if stub_fallbacks:
            fallback_packages = list(dict.fromkeys(fallback_packages + stub_fallbacks))

        stub_path = filtered_path if filtered_path is not None else caller_file
        stub_getattr, stub_dir, __all__ = lazy.attach_stub(package_name, stub_path)
        if filtered_path is not None:
            os.unlink(filtered_path)
    else:
        __all__: list[str] = []

    mod = sys.modules[package_name]

    # -- Eagerly resolve relative wildcard imports (from .X import *) ------
    for rel_mod_name in relative_wildcards:
        fq_name = f"{package_name}.{rel_mod_name}"
        sub = importlib.import_module(fq_name)
        exported = getattr(sub, "__all__", [n for n in dir(sub) if not n.startswith("_")])
        for name in exported:
            mod.__dict__[name] = getattr(sub, name)
            if name not in __all__:
                __all__.append(name)

    # -- Build lazy fallback for absolute wildcard imports -----------------
    _sentinel = object()

    if fallback_packages:
        _resolved_pkgs: list = []
        for _pkg_name in fallback_packages:
            try:
                _resolved_pkgs.append(importlib.import_module(_pkg_name))
            except (ImportError, ModuleNotFoundError) as e:
                raise ImportError(
                    f"lazy_export() in {package_name!r}: .pyi stub declares "
                    f"'from {_pkg_name} import *' but the package is not installed."
                ) from e

        def _pkg_getattr(name: str):
            for pkg_mod in _resolved_pkgs:
                val = getattr(pkg_mod, name, _sentinel)
                if val is not _sentinel:
                    mod.__dict__[name] = val
                    return val
            raise AttributeError(f"module {package_name!r} has no attribute {name!r}")

        if has_stub:
            _stub_getattr = stub_getattr
            _stub_dir = stub_dir
            _dir_cache: list[str] | None = None

            def __getattr__(name: str):
                try:
                    return _stub_getattr(name)
                except AttributeError:
                    return _pkg_getattr(name)

            def __dir__():
                nonlocal _dir_cache
                if _dir_cache is None:
                    pkg_names = {n for p in _resolved_pkgs for n in dir(p) if not n.startswith("_")}
                    _dir_cache = sorted(set(_stub_dir()) | pkg_names)
                return _dir_cache

        else:
            _dir_cache: list[str] | None = None

            def __getattr__(name: str):
                return _pkg_getattr(name)

            def __dir__():
                nonlocal _dir_cache
                if _dir_cache is None:
                    _dir_cache = sorted(n for p in _resolved_pkgs for n in dir(p) if not n.startswith("_"))
                return _dir_cache

    elif has_stub:
        __getattr__ = stub_getattr
        __dir__ = stub_dir
    else:

        def __getattr__(name: str):
            raise AttributeError(f"module {package_name!r} has no attribute {name!r}")

        def __dir__():
            return []

    setattr(mod, "__getattr__", __getattr__)
    setattr(mod, "__dir__", __dir__)
    setattr(mod, "__all__", __all__)
    return __getattr__, __dir__, __all__
