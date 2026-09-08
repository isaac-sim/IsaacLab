# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package containing utilities for common operations and helper functions."""

import importlib

import lazy_loader as lazy

_stub_getattr, __dir__, __all__ = lazy.attach_stub(__name__, __file__)


def __getattr__(name: str):
    # ``configclass`` names both a sub-module of this package and the decorator that sub-module
    # defines, so ``isaaclab.utils.configclass`` can only ever be one of the two. Importing the
    # sub-module binds the module object here and shadows the lazily attached decorator, while
    # resolving the decorator first leaves the module unreachable through this package. Always hand
    # out the sub-module, which is callable and therefore usable as the decorator as well, so the
    # name means the same thing regardless of which was imported first. The import machinery binds
    # the sub-module here on its way out, so later lookups never reach this function.
    if name == "configclass":
        return importlib.import_module(f"{__name__}.configclass")
    return _stub_getattr(name)
