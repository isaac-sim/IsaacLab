# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable
from typing import Any

from ..warp.capture import GraphedCallable


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

    While a :class:`~isaaclab.utils.warp.capture.GraphedCallable` records, :attr:`timestamp` reads -1 (stale) until
    the data is updated in that recording: a graph replays without the owner's timestamp check, so a recording that
    reads the data must hold its update for every replay to recompute it. Data whose update a recording must not hold,
    such as a value its owner fetches through a host call or a cache that records its own graph, opts out with
    ``update_in_graph=False``. Any other CUDA graph capture keeps the plain timestamp rule.
    """

    __slots__ = ("data", "update_in_graph", "_timestamp", "_recording")

    def __init__(self, data: Any = None, timestamp: float = -1.0, *, update_in_graph: bool = True):
        """Initialize the buffer.

        Args:
            data: Cached array, native-format struct, or grouped data. Defaults to None, before allocation.
            timestamp: Source timestamp represented by the data. Defaults to -1, which marks it stale.
            update_in_graph: Whether a :class:`~isaaclab.utils.warp.capture.GraphedCallable` recording that reads the
                data updates it. Defaults to True.
        """
        self.data = data
        self.update_in_graph = update_in_graph
        self._timestamp = timestamp
        self._recording: object = None

    @property
    def timestamp(self) -> float:
        """Source timestamp represented by the data; -1 marks it stale."""
        # is_recording(), inlined: every lazy data read checks it
        recording = GraphedCallable._recording
        if recording is not None and recording.capturing and self.update_in_graph:
            if self._recording is not recording.token:
                return -1.0
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value: float) -> None:
        self._timestamp = value
        recording = GraphedCallable._recording
        if recording is not None and recording.capturing:
            # the recording in progress now holds the update; invalidating clears it, so a later read records it again
            self._recording = recording.token if value >= 0 else None
