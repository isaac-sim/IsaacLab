# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated compatibility exports for benchmark recorders."""

import isaaclab.benchmark.recorders as _recorders

__all__ = _recorders.__all__


def __getattr__(name: str):
    """Resolve deprecated recorder exports lazily from the public package."""
    if name in __all__:
        return getattr(_recorders, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the preserved public recorder export names."""
    return sorted(set(globals()) | set(__all__))
