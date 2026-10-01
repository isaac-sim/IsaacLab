# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


def reset_timestamps(buffers: Iterable["TimestampedBuffer | None"]) -> None:
    """Invalidate cached values, skipping unallocated or unaffected buffers.

    Args:
        buffers: Timestamped buffers to invalidate. ``None`` entries are ignored.
    """
    for buffer in buffers:
        if buffer is not None:
            buffer.timestamp = -1.0


@dataclass
class TimestampedBuffer:
    """Cached data and its last successful update timestamp.

    The owner supplies storage and updates the timestamp after computing the value.
    Array caches accept Torch or Warp arrays directly; no conversion is performed.
    """

    data: Any = None
    """Cached array, native-format struct, or grouped data; None before allocation."""

    timestamp: float = -1.0
    """Source timestamp represented by the data; -1 marks it stale."""
