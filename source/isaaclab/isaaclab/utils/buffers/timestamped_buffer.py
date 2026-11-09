# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable
from typing import Any

from ..warp.capture import recording_key, recordings


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

    While Warp records a CUDA graph, :attr:`timestamp` reads -1 (stale) until the data is updated in that recording:
    a graph replays without the owner's timestamp check, so a recording that reads the data must hold its update for
    every replay to recompute it. Data that must not be updated where it is read, such as a finite difference advanced
    once per physics step or a value fetched once from the host, opts out with ``refresh_in_recordings=False``.
    """

    __slots__ = ("data", "refresh_in_recordings", "_timestamp", "_recording")

    _captures: dict | None = None
    """Warp's table of the graph recordings in progress (see :func:`~isaaclab.utils.warp.utils.recordings`)."""

    def __init__(self, data: Any = None, timestamp: float = -1.0, *, refresh_in_recordings: bool = True):
        """Initialize the buffer.

        Args:
            data: Cached array, native-format struct, or grouped data. Defaults to None, before allocation.
            timestamp: Source timestamp represented by the data. Defaults to -1, which marks it stale.
            refresh_in_recordings: Whether a CUDA graph recording that reads the data updates it. Defaults to True.
        """
        self.data = data
        self.refresh_in_recordings = refresh_in_recordings
        self._timestamp = timestamp
        self._recording: object = None
        if TimestampedBuffer._captures is None:
            # the table lives as long as the Warp runtime, which allocates the data of every buffer
            TimestampedBuffer._captures = recordings()

    @property
    def timestamp(self) -> float:
        """Source timestamp represented by the data; -1 marks it stale."""
        captures = self._captures
        if captures and self.refresh_in_recordings and self._recording != recording_key(captures):
            return -1.0
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value: float) -> None:
        self._timestamp = value
        captures = self._captures
        if captures:
            # the recording in progress now holds the update; invalidating clears it, so a later read records it again
            self._recording = recording_key(captures) if value >= 0 else None
