# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

DataT = TypeVar("DataT")


def reset_timestamps(buffers: Iterable["TimestampedBuffer | None"]) -> None:
    """Mark each non-``None`` timestamped buffer as stale so its next read recomputes.

    Each buffer is named exactly once at the call site, which avoids the
    "check one property but reset another" class of typos that arises when invalidating many
    buffers by hand. ``None`` entries are skipped so callers can inline conditional invalidations,
    e.g. ``reset_timestamps([buf_a if from_link else None, buf_b])``.

    Args:
        buffers: Timestamped buffers to invalidate. ``None`` entries are ignored.
    """
    for buffer in buffers:
        if buffer is not None:
            buffer.timestamp = -1.0


@dataclass
class TimestampedBuffer(Generic[DataT]):
    """Cached data and the source timestamp it represents; this container never allocates or computes.

    Storage and freshness are independent: ``data is None`` means no storage, while ``timestamp``
    identifies the last successful refresh. The owner allocates on first use and refreshes when
    the source timestamp differs. Same-step writes must invalidate the cache or advance the source
    timestamp; elapsed simulation time alone cannot detect them.

    Each cache records its own timestamp. Dirty flags or masks track pending work owned by one
    component; readers never clear a shared producer's dirty flag.
    Timestamps may be simulation time [s] or logical publication counters, but a cache and its
    source must use the same convention. Physical sampling periods and finite differences still
    require elapsed time, not publication counters.

    Scratch storage needs no freshness timestamp. Python timestamp checks also do not execute
    during CUDA graph replay: captured computations must execute on replay or use device-side
    invalidation.
    """

    data: DataT | None = None
    """Cached data, or None before allocation."""

    timestamp: float = -1.0
    """Source timestamp represented by the data; -1 marks it stale."""
