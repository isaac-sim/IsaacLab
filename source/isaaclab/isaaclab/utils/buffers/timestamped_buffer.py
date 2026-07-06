# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import torch


class _Timestamped(Protocol):
    """Structural type for any buffer exposing a writable :attr:`timestamp`.

    Matches both :class:`TimestampedBuffer` and ``TimestampedBufferWarp`` so the shared
    invalidation helper does not depend on the (optional) Warp buffer module.
    """

    timestamp: float


def reset_timestamps(buffers: Iterable[_Timestamped | None]) -> None:
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
class TimestampedBuffer:
    """A buffer class containing data and its timestamp.

    This class is a simple data container that stores a tensor and its timestamp. The timestamp is used to
    track the last update of the buffer. The timestamp is set to -1.0 by default, indicating that the buffer
    has not been updated yet. The timestamp should be updated whenever the data in the buffer is updated. This
    way the buffer can be used to check whether the data is outdated and needs to be refreshed.

    The buffer is useful for creating lazy buffers that only update the data when it is outdated. This can be
    useful when the data is expensive to compute or retrieve. For example usage, refer to the data classes in
    the :mod:`isaaclab.assets` module.
    """

    data: torch.Tensor = None  # type: ignore
    """The data stored in the buffer. Default is None, indicating that the buffer is empty."""

    timestamp: float = -1.0
    """Timestamp at the last update of the buffer. Default is -1.0, indicating that the buffer has not been updated."""
