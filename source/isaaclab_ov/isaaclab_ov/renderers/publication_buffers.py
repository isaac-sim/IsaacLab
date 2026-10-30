# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Producer storage retained until native scene publication completes."""

from __future__ import annotations

import contextlib
from collections import deque
from collections.abc import Iterable
from typing import TYPE_CHECKING, Generic, TypeVar

import warp as wp
from ovrtx import DataAccess

from isaaclab.scene_data import SceneDataFormat

if TYPE_CHECKING:
    from ovrtx import AttributeBinding, Operation
_BufferT = TypeVar("_BufferT", wp.array, SceneDataFormat.TransposedMatrix44d)


class _AsyncWriteBuffers(Generic[_BufferT]):
    """Retain buffers until native writes finish; reuse them in submission order."""

    def __init__(self, buffers: Iterable[_BufferT] = ()):
        self._writes: deque[tuple[_BufferT, Operation[bool] | None, wp.Stream | None]] = deque(
            (buffer, None, None) for buffer in buffers
        )

    def acquire(self, stream: wp.Stream | None) -> _BufferT:
        """Wait until the next buffer is writable on the given stream."""
        buffer, operation, producer = self._writes[0]
        if operation is not None:
            operation.wait()
            if stream is not None and producer is not None and producer != stream:
                stream.wait_stream(producer)
        return buffer

    def submit(
        self, binding: AttributeBinding, values: wp.array | list[wp.array], stream: wp.Stream | None
    ) -> Operation[bool]:
        """Submit a write and retain its inputs; a failed submission does not advance the buffers."""
        operation = binding.write_async(
            values, data_access=DataAccess.ASYNC, cuda_stream=(stream.cuda_stream or 1) if stream is not None else None
        )
        self._writes[0] = (self._writes[0][0], operation, stream)
        self._writes.rotate(-1)
        return operation

    def close(self) -> None:
        """Complete every pending write before releasing storage, including on failure."""
        with contextlib.ExitStack() as writes:
            writes.callback(self._writes.clear)
            for _, operation, producer in self._writes:
                if operation is not None:
                    if producer is not None:
                        writes.callback(wp.synchronize_stream, producer)
                    writes.callback(operation.wait)
