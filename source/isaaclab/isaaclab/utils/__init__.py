# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package containing utilities for common operations and helper functions."""

import sys
import types
from typing import Any

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)


class _LazyAttributePackage(types.ModuleType):
    """Module type that protects lazily attached attributes from same-named sub-modules.

    The sub-module :mod:`isaaclab.utils.configclass` and the decorator
    :func:`~isaaclab.utils.configclass.configclass` it defines share a name. Importing the
    sub-module makes the import machinery bind the module object onto this package, which shadows
    the lazily attached decorator: ``from isaaclab.utils import configclass`` then returns a module
    and ``@configclass`` raises ``TypeError: 'module' object is not callable``. Whether the name
    resolves to the decorator or to the module depends only on the order of the first imports, so
    the failure surfaces in downstream code that never imports the sub-module itself.

    Skipping that one assignment keeps the exported attribute reachable regardless of import order.
    The sub-module stays importable through ``from isaaclab.utils.configclass import ...``, which is
    how the rest of Isaac Lab refers to it.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if name in __all__ and isinstance(value, types.ModuleType) and value.__name__ == f"{__name__}.{name}":
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _LazyAttributePackage
