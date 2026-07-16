# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated compatibility namespace for :mod:`isaaclab.benchmark`."""

import warnings

import isaaclab.benchmark as _benchmark

__all__ = _benchmark.__all__

warnings.warn(
    "isaaclab.test.benchmark is deprecated; import isaaclab.benchmark instead.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str):
    """Resolve deprecated exports lazily from :mod:`isaaclab.benchmark`."""
    if name in __all__:
        return getattr(_benchmark, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the preserved public benchmark export names."""
    return sorted(set(globals()) | set(__all__))
