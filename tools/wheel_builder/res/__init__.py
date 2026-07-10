# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import importlib.abc
import importlib.machinery
import os
import sys
from importlib.metadata import version
from importlib.util import find_spec

__version__ = version("isaaclab")

# Extend the package search path so subpackages (app/, envs/, etc.) in the
# nested source tree are importable as isaaclab.app, isaaclab.envs, etc.
__path__.append(os.path.join(os.path.dirname(__file__), "source", "isaaclab", "isaaclab"))


def _deprioritize_prebundle_paths():
    """Move Isaac Sim bundled dependency paths behind the active Python environment."""
    using_external_python = "/kit/python/bin/" not in sys.executable.replace("\\", "/").lower()

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

    reordered_sys_path, demoted_sys_path = _reorder_paths(sys.path)

    if demoted_sys_path:
        sys.path[:] = reordered_sys_path

    _install_dependency_finder()

    stale_roots = set()
    for name, module in tuple(sys.modules.items()):
        root = _module_root(name)
        if root in _VERSIONED_PACKAGE_ROOTS and _module_origin_is_demoted(module):
            stale_roots.add(root)
    for name in tuple(sys.modules):
        if _module_root(name) in stale_roots:
            _remove_module(name)

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


# TODO(myurasov-nv): bootstrap_kernel() is ported from the internal GitLab wheel builder
# for backwards compatibility. It is not called currently, but may be needed if Isaac Sim
# requires explicit kernel bootstrapping before use. Remove once confirmed unnecessary.
def bootstrap_kernel():
    # Isaac Lab path
    isaaclab_path = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

    # bootstrap kernel via Isaac Sim
    if find_spec("isaacsim") is not None:
        import isaacsim

        # log info
        if find_spec("carb") is not None:
            import carb
            carb.log_info(f"Isaac Lab path: {isaaclab_path}")


def main():
    """Entry point for the ``isaaclab`` console script (python -m isaaclab)."""
    from isaaclab.__main__ import main as _main

    sys.exit(_main())


if __name__ == "__main__":
    bootstrap_kernel()
    main()
