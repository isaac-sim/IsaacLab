# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility for existing compute callers and subclass implementations."""


def _compute_compat(cls: type) -> type:
    """Bind both evaluation spellings at the defining class, preserving super() dispatch."""
    for owner in cls.__mro__:
        if "__call__" in vars(owner) or "compute" in vars(owner):
            call = vars(owner).get("__call__", vars(owner).get("compute"))
            compute = vars(owner).get("compute", call)
            if owner is cls or getattr(cls, "__call__", None) is not call:
                cls.__call__ = call
                cls.compute = compute
            break
    return cls
