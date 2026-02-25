# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities built around USD operations."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["legacy", "prims", "queries", "semantics", "stage", "transforms"],
)

_lazy_getattr = __getattr__
_SUBMODULES = ("legacy", "prims", "queries", "semantics", "stage", "transforms")


def __getattr__(name):
    try:
        return _lazy_getattr(name)
    except AttributeError:
        pass
    import importlib

    for submod_name in _SUBMODULES:
        try:
            submod = importlib.import_module(f"{__name__}.{submod_name}")
            return getattr(submod, name)
        except (ImportError, AttributeError):
            continue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
