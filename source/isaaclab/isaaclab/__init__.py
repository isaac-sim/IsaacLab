# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing the core framework."""

import importlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import os
import sys
import tomllib


def _deprioritize_prebundle_paths():
    """Move Isaac Sim bundled dependency paths behind the active Python environment.

    Isaac Sim's environment setup scripts inject Kit Python and ``pip_prebundle``
    directories onto ``PYTHONPATH``. These contain bundled copies of packages
    like torch, warp, and nvidia-cudnn that can shadow the versions installed by
    Isaac Lab, causing CUDA runtime errors.

    Additionally, certain Isaac Sim kit extensions (such as ``omni.warp.core``)
    bundle their own copies of Python packages that conflict with pip-installed
    versions.  When loaded by the extension system these paths can appear on
    ``sys.path`` before ``site-packages``, leading to version mismatches.

    Rather than removing these paths entirely (which would break packages that
    only exist in Kit), this function moves them to the **end** of ``sys.path``
    so that the active environment takes priority. When running under an
    external Python, Kit's stdlib path is also removed from ``PYTHONPATH`` for
    child processes because ``PYTHONPATH`` entries always precede the real
    standard library.

    The ``PYTHONPATH`` environment variable is also rewritten so that child
    processes inherit the corrected ordering.
    """
    using_external_python = "/kit/python/bin/" not in sys.executable.replace("\\", "/").lower()

    # Extension directory fragments that are known to ship Python packages
    # which conflict with Isaac Lab's pip-installed versions.
    _CONFLICTING_EXT_FRAGMENTS = (
        "omni.warp.core",
        "omni.isaac.ml_archive",
        "omni.isaac.core_archive",
        "omni.kit.pip_archive",
        "isaacsim.pip.newton",
    )
    _VERSIONED_PACKAGE_ROOTS = ("warp", "newton", "typing_extensions", "pydantic_core")

    def _is_kit_python_path(norm: str) -> bool:
        return "/kit/python/lib/python" in norm

    def _is_kit_python_stdlib_path(norm: str) -> bool:
        return _is_kit_python_path(norm) and "site-packages" not in norm

    def _should_demote(path: str) -> bool:
        norm = path.replace("\\", "/").lower()
        if using_external_python and _is_kit_python_path(norm):
            return True
        if "pip_prebundle" in norm:
            return True
        for frag in _CONFLICTING_EXT_FRAGMENTS:
            if frag.lower() in norm:
                return True
        return False

    def _reorder_paths(paths, *, drop_demoted: bool = False):
        clean = []
        demoted = []
        for p in paths:
            if _should_demote(p):
                demoted.append(p)
            else:
                clean.append(p)
        if drop_demoted:
            return clean, bool(demoted)
        return clean + demoted, bool(demoted)

    def _module_root(name: str) -> str:
        return name.split(".", 1)[0]

    def _module_origin_is_demoted(module) -> bool:
        origin = getattr(module, "__file__", None)
        if origin and _should_demote(origin):
            return True
        spec = getattr(module, "__spec__", None)
        spec_origin = getattr(spec, "origin", None)
        return bool(spec_origin and _should_demote(spec_origin))

    def _remove_module(name: str) -> None:
        sys.modules.pop(name, None)
        parent_name, _, child_name = name.rpartition(".")
        parent = sys.modules.get(parent_name) if parent_name else None
        if parent is not None and hasattr(parent, child_name):
            delattr(parent, child_name)

    def _install_dependency_finder() -> None:
        class _IsaacLabDependencyFinder(importlib.abc.MetaPathFinder):
            _isaaclab_dependency_finder = True

            def find_spec(self, fullname, path=None, target=None):
                if _module_root(fullname) not in _VERSIONED_PACKAGE_ROOTS:
                    return None
                search_path = sys.path if path is None else path
                clean_path = [p for p in search_path if not _should_demote(p)]
                if not clean_path:
                    return None
                spec = importlib.machinery.PathFinder.find_spec(fullname, clean_path, target)
                if spec is None:
                    return None
                if spec.origin and _should_demote(spec.origin):
                    return None
                return spec

        sys.meta_path[:] = [
            finder for finder in sys.meta_path if not getattr(finder, "_isaaclab_dependency_finder", False)
        ]
        sys.meta_path.insert(0, _IsaacLabDependencyFinder())

    # Partition: keep non-conflicting in place, collect conflicting.
    reordered_sys_path, demoted_sys_path = _reorder_paths(sys.path)

    if demoted_sys_path:
        # Rebuild sys.path: originals first, then demoted at the very end.
        sys.path[:] = reordered_sys_path

    _install_dependency_finder()

    # If a versioned dependency package has already been imported from a
    # conflicting extension path, dropping only its search path is too late: the
    # old module object can keep importing old submodules. Unload the package so
    # the next import resolves from the active environment.
    stale_roots = set()
    for name, module in tuple(sys.modules.items()):
        root = _module_root(name)
        if root in _VERSIONED_PACKAGE_ROOTS and _module_origin_is_demoted(module):
            stale_roots.add(root)
    for name in tuple(sys.modules):
        if _module_root(name) in stale_roots:
            _remove_module(name)

    # Reorder loaded package search paths too. Kit extensions can add bundled
    # package locations after a package is already imported; future submodule
    # imports then use the stale package ``__path__`` instead of ``sys.path``.
    demoted_module_paths = False
    stale_module_prefixes = set()
    for name, module in tuple(sys.modules.items()):
        module_path = getattr(module, "__path__", None)
        if module_path is None:
            continue
        try:
            original_module_path = list(module_path)
        except (TypeError, KeyError):
            continue
        drop_demoted = _module_root(name) in _VERSIONED_PACKAGE_ROOTS
        reordered_module_path, demoted_module_path = _reorder_paths(original_module_path, drop_demoted=drop_demoted)
        if not demoted_module_path or reordered_module_path == original_module_path:
            continue
        if drop_demoted and not reordered_module_path:
            stale_module_prefixes.add(name)
            continue
        demoted_module_paths = True
        try:
            module_path[:] = reordered_module_path
        except TypeError:
            module.__path__ = reordered_module_path
    for prefix in stale_module_prefixes:
        for name in tuple(sys.modules):
            if name == prefix or name.startswith(f"{prefix}."):
                _remove_module(name)

    if demoted_sys_path or demoted_module_paths or stale_roots or stale_module_prefixes:
        importlib.invalidate_caches()

    if not demoted_sys_path:
        return

    # Rewrite PYTHONPATH with the same ordering for subprocesses.
    if "PYTHONPATH" in os.environ:
        parts = os.environ["PYTHONPATH"].split(os.pathsep)
        env_clean = []
        env_demoted = []
        for p in parts:
            norm = p.replace("\\", "/").lower()
            if using_external_python and _is_kit_python_stdlib_path(norm):
                continue
            elif _should_demote(p):
                env_demoted.append(p)
            else:
                env_clean.append(p)
        os.environ["PYTHONPATH"] = os.pathsep.join(env_clean + env_demoted)


_deprioritize_prebundle_paths()

ISAACLAB_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

_ext_toml = os.path.join(ISAACLAB_EXT_DIR, "config", "extension.toml")
if os.path.exists(_ext_toml):
    with open(_ext_toml, "rb") as _f:
        ISAACLAB_METADATA = tomllib.load(_f)
else:
    ISAACLAB_METADATA = {}
"""Extension metadata dictionary parsed from the extension.toml file."""

try:
    __version__ = importlib.metadata.version("isaaclab")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
