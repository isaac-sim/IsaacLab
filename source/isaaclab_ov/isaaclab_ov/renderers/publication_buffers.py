# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Producer storage retained until native scene publication completes."""

from __future__ import annotations

import contextlib
import re
from collections import deque
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Generic, TypeVar

import ovstage
import torch
import warp as wp
from ovrtx import DataAccess

from isaaclab.scene_data import SceneDataFormat

from isaaclab_ov.stage import points_tensor_from_warp, xform_tensor_from_warp

if TYPE_CHECKING:
    from ovrtx import AttributeBinding, Operation
_BufferT = TypeVar("_BufferT", wp.array, SceneDataFormat.TransposedMatrix44d)


class _GeometryPublicationBuffers:
    """Own completed cross-device geometry and rigid-pose snapshots through Stage copy-in."""

    def __init__(self, device: str):
        if re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device) is None:
            raise ValueError("geometry_publication_device must name an explicit CUDA device, e.g. 'cuda:0'")
        self._device = wp.get_device(device)
        self._torch_stream: torch.cuda.Stream | None = None
        self._stream: wp.Stream | None = None
        self._buffers: dict[str, wp.array] = {}
        self._sources: list[wp.array] = []
        self._source_streams: list[wp.Stream] = []
        self._failed = False

    def write(self, stage, binding, paths: Sequence[str], points: list[wp.array], ordinal: int) -> None:
        """Copy foreign-GPU points and wait for Stage to consume the batch."""
        self._write(
            stage,
            binding,
            paths,
            points,
            ordinal,
            attribute="points",
            is_array=True,
            semantic=ovstage.AttributeSemantic.POINT,
            convert=points_tensor_from_warp,
        )

    def write_transforms(self, stage, binding, matrices: wp.array, ordinal: int) -> None:
        """Publish rigid poses on the selected device through the same completed-copy boundary."""
        # Point-buffer keys are absolute prim paths; this key cannot alias a point prim.
        self._write(
            stage,
            binding,
            ("@transforms",),
            [matrices],
            ordinal,
            attribute="omni:xform",
            is_array=False,
            semantic=ovstage.AttributeSemantic.MATRIX,
            convert=xform_tensor_from_warp,
        )

    def _write(
        self,
        stage,
        binding,
        paths: Sequence[str],
        sources: list[wp.array],
        ordinal: int,
        *,
        attribute: str,
        is_array: bool,
        semantic,
        convert,
    ) -> None:
        """Keep source and staging storage alive until the native copy-in has completed."""
        if self._failed:
            raise RuntimeError("Geometry publication failed; close and recreate the renderer")
        # Reuse is safe only after the entire write completes.
        self._failed = True
        self._sources = sources
        self._source_streams = [source.device.stream for source in sources if source.device.is_cuda]
        published = list(sources)
        for index, (path, source) in enumerate(zip(paths, sources, strict=True)):
            if not source.device.is_cuda or source.device == self._device:
                continue
            if self._stream is None:
                # Retain Torch's nonblocking stream; Warp-created streams are blocking.
                self._torch_stream = torch.cuda.Stream(device=str(self._device))
                self._stream = wp.stream_from_torch(self._torch_stream)
            target = self._buffers.get(path)
            if target is None or target.shape != source.shape or target.dtype != source.dtype:
                # Warp pool allocations and their first copy must use the same stream.
                with wp.ScopedStream(self._stream, sync_enter=False):
                    target = wp.empty_like(source, device=self._device)
                self._buffers[path] = target
            producer_stream = source.device.stream
            self._stream.wait_stream(producer_stream)
            wp.copy(target, source, stream=self._stream)
            published[index] = target
        if self._stream is not None:
            wp.synchronize_stream(self._stream)
        # Include same-device producer work in Stage's input dependency.
        stream = self._device.stream if any(point.device.is_cuda for point in published) else None
        if stream is not None and self._stream is not None:
            self._stream.wait_stream(stream)
            stream = self._stream
        tensors = [convert(array) for array in published]
        stage.write_attribute(
            binding,
            attribute,
            ordinal=ordinal,
            tensors=tensors if is_array else tensors[0],
            is_array=is_array,
            semantic=semantic,
            cuda_stream=(stream.cuda_stream or 1) if stream is not None else None,
        ).wait()
        self._sources = []
        self._source_streams = []
        self._failed = False

    def close(self) -> None:
        """Drain exact transfer streams before releasing storage; retain it if draining fails."""
        for stream in self._source_streams:
            wp.synchronize_stream(stream)
        if self._stream is not None:
            wp.synchronize_stream(self._stream)
        self._sources.clear()
        self._source_streams.clear()
        self._buffers.clear()


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
