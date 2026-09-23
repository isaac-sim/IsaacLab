# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility for existing compute callers and subclass implementations."""

import warnings
from functools import wraps


def _compute_compat(cls: type) -> type:
    """Bind both evaluation spellings at the defining class, preserving super() dispatch."""
    for owner in cls.__mro__:
        if "__call__" in vars(owner) or "compute" in vars(owner):
            call = vars(owner).get("__call__", vars(owner).get("compute"))
            compute = vars(owner).get("compute", call)
            if owner is cls or getattr(cls, "__call__", None) is not call:
                if "__call__" not in vars(owner):
                    warnings.warn(
                        f"{cls.__qualname__} overrides compute(); define __call__() instead. "
                        "compute() compatibility will be removed in Isaac Lab 3.2.",
                        DeprecationWarning,
                        stacklevel=3,
                    )

                @wraps(compute)
                def deprecated_compute(self, *args, **kwargs):
                    warnings.warn(
                        "compute() is deprecated and will be removed in Isaac Lab 3.2. Use term(...) instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    return compute(self, *args, **kwargs)

                cls.__call__ = call
                cls.compute = deprecated_compute
            break
    return cls
