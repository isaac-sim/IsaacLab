# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable
from typing import Any

from ..warp.capture import capture_in_progress


def reset_timestamps(buffers: Iterable["TimestampedBuffer | None"]) -> None:
    """Invalidate cached values, skipping unallocated or unaffected buffers.

    Args:
        buffers: Timestamped buffers to invalidate. ``None`` entries are ignored.
    """
    for buffer in buffers:
        if buffer is not None:
            buffer.timestamp = -1.0


class TimestampedBuffer:
    """Cached data and its last successful update timestamp.

    The owner supplies storage and updates the timestamp after computing the value.
    Array caches accept Torch or Warp arrays directly; no conversion is performed.

    Inside any graph capture, on any device, :attr:`timestamp` reads -1 (stale) and a fresh timestamp is not
    committed: a graph replays without the owner's timestamp check, so a capture that reads the data must hold its
    update for every replay to recompute it, and the update of a capture that is abandoned never runs. Data whose
    update a capture must not hold, such as a value its owner fetches through a host call or a cache that records its
    own graph, opts out with ``update_in_graph=False``.
    """

    __slots__ = ("data", "update_in_graph", "_timestamp")

    def __init__(self, data: Any = None, timestamp: float = -1.0, *, update_in_graph: bool = True):
        """Initialize the buffer.

        Args:
            data: Cached array, native-format struct, or grouped data. Defaults to None, before allocation.
            timestamp: Source timestamp represented by the data. Defaults to -1, which marks it stale.
            update_in_graph: Whether a graph capture that reads the data updates it. Defaults to True.
        """
        self.data = data
        self.update_in_graph = update_in_graph
        self._timestamp = timestamp

    @property
    def timestamp(self) -> float:
        """Source timestamp represented by the data; -1 marks it stale."""
        if self.update_in_graph and capture_in_progress():
            return -1.0
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value: float) -> None:
        if value >= 0.0 and self.update_in_graph and capture_in_progress():
            # the update is recorded, not run: only the replays make the data fresh
            return
        self._timestamp = value
