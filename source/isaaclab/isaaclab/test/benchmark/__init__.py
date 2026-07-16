# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated compatibility namespace for :mod:`isaaclab.benchmark`."""

import importlib
import sys
import types
import warnings
from typing import Any

import isaaclab.benchmark as _benchmark

__all__ = _benchmark.__all__

_LEGACY_SUBMODULES = (
    "benchmark_core",
    "benchmark_monitor",
    "builders",
    "capture",
    "formatters",
    "interfaces",
    "measurements",
    "method_benchmark",
    "metrics",
    "profiling",
    "recorders",
    "recorders.record_cpu_info",
    "recorders.record_gpu_info",
    "recorders.record_memory_info",
    "recorders.record_version_info",
    "schema",
    "serialize",
    "stepping",
)

_WARN_MESSAGE = (
    "isaaclab.test.benchmark is deprecated and will be removed in Isaac Lab 3.1. "
    "Import isaaclab.benchmark instead."
)


def _create_module_alias(legacy_name: str, target_name: str, *, is_package: bool = False) -> types.ModuleType:
    """Create a lightweight module proxy that resolves attributes lazily.

    Args:
        legacy_name: Deprecated fully qualified module name.
        target_name: Replacement fully qualified module name.
        is_package: Whether the alias represents a package.

    Returns:
        Module proxy registered under the deprecated name.
    """
    module = types.ModuleType(legacy_name)
    module.__doc__ = f"Deprecated alias for {target_name}."
    module.__file__ = None
    module.__loader__ = None
    module.__package__ = legacy_name if is_package else legacy_name.rpartition(".")[0]
    module.__spec__ = importlib.machinery.ModuleSpec(legacy_name, loader=None, is_package=is_package)
    if is_package:
        module.__path__ = []

    def _resolve(name: str) -> Any:
        target = importlib.import_module(target_name)
        if name == "__all__":
            return getattr(target, "__all__", [item for item in dir(target) if not item.startswith("_")])
        return getattr(target, name)

    def _list() -> list[str]:
        return dir(importlib.import_module(target_name))

    module.__getattr__ = _resolve
    module.__dir__ = _list
    return module


for _submodule in _LEGACY_SUBMODULES:
    _legacy_name = f"{__name__}.{_submodule}"
    _alias = _create_module_alias(
        _legacy_name,
        f"isaaclab.benchmark.{_submodule}",
        is_package=_submodule == "recorders",
    )
    sys.modules[_legacy_name] = _alias
    _parent, _, _child = _submodule.rpartition(".")
    if _parent:
        setattr(sys.modules[f"{__name__}.{_parent}"], _child, _alias)
    else:
        globals()[_child] = _alias
del _alias, _child, _legacy_name, _parent, _submodule


warnings.warn(_WARN_MESSAGE, DeprecationWarning, stacklevel=2)


def __getattr__(name: str):
    """Resolve deprecated exports lazily from :mod:`isaaclab.benchmark`."""
    if name in __all__:
        return getattr(_benchmark, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the preserved public benchmark export names."""
    return sorted(set(globals()) | set(__all__))
