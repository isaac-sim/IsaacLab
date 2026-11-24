# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Split-GPU rendering for the OVXR renderer: GPU selection and cross-GPU geometry publication.

When :attr:`~isaaclab_teleop.OvxrRendererCfg.active_cuda_gpus` excludes the simulation GPU, rigid
poses and deformable points produced on the simulation GPU are copied to the first rendering GPU
before OVStage consumes them.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import ovstage
import torch
import warp as wp
from isaaclab_ov.stage import points_tensor_from_warp, xform_tensor_from_warp


def parse_active_cuda_gpus(value: str) -> tuple[int, ...]:
    """Parse an ``active_cuda_gpus`` selection such as ``"0"`` or ``"0,1"``.

    Args:
        value: Comma-separated CUDA device indices.

    Returns:
        The CUDA device indices in the given order.

    Raises:
        ValueError: If the value is not a comma-separated list of unique non-negative CUDA indices.
    """
    if re.fullmatch(r"(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*", value) is None:
        raise ValueError(f"active_cuda_gpus must list CUDA indices such as '0' or '0,1'; got {value!r}")
    indices = tuple(int(index) for index in value.split(","))
    if len(set(indices)) != len(indices):
        raise ValueError(f"active_cuda_gpus lists a CUDA index more than once: {value!r}")
    return indices


def _parse_cuda_device_index(value: str) -> int:
    """Return ``N`` for an explicit ``"cuda:N"`` device.

    Args:
        value: Device name, such as ``"cuda:0"``.

    Returns:
        The CUDA device index.

    Raises:
        ValueError: If the value is not an explicit ``"cuda:N"`` device.
    """
    match = re.fullmatch(r"cuda:(0|[1-9][0-9]*)", value)
    if match is None:
        raise ValueError(f"Expected an explicit CUDA device such as 'cuda:0'; got {value!r}")
    return int(match.group(1))


def _excludes_simulation_gpu(active_cuda_gpus: str | None, simulation_device: str) -> bool:
    """Whether an explicit ``active_cuda_gpus`` selection omits a ``"cuda:N"`` simulation device."""
    if active_cuda_gpus is None or not str(simulation_device).startswith("cuda:"):
        return False
    return _parse_cuda_device_index(str(simulation_device)) not in parse_active_cuda_gpus(active_cuda_gpus)


class _GeometryPublicationBuffers:
    """Own completed cross-device geometry and rigid-pose snapshots through Stage copy-in."""

    def __init__(self, device: str):
        _parse_cuda_device_index(device)
        self._device = wp.get_device(device)
        self._torch_stream: torch.cuda.Stream | None = None
        self._stream: wp.Stream | None = None
        self._buffers: dict[str, wp.array] = {}
        self._sources: list[wp.array] = []
        self._source_streams: list[wp.Stream] = []
        self._failed = False

    def write(
        self,
        stage,
        binding,
        paths: Sequence[str],
        points: list[wp.array],
        ordinal: int,
        producer: wp.Stream | None = None,
    ) -> None:
        """Copy foreign-GPU points and wait for Stage to consume the batch.

        ``producer`` is the Warp stream that produced ``points``; without it, the transfer waits on
        each source device's current stream.
        """
        self._write(
            stage,
            binding,
            paths,
            points,
            ordinal,
            producer,
            attribute="points",
            is_array=True,
            semantic=ovstage.AttributeSemantic.POINT,
            convert=points_tensor_from_warp,
        )

    def write_transforms(
        self, stage, binding, matrices: wp.array, ordinal: int, producer: wp.Stream | None = None
    ) -> None:
        """Publish rigid poses on the publication device through the same completed-copy boundary.

        ``producer`` is the Warp stream that produced ``matrices``, as in :meth:`write`.
        """
        # Point-buffer keys are absolute prim paths; this key cannot alias a point prim.
        self._write(
            stage,
            binding,
            ("@transforms",),
            [matrices],
            ordinal,
            producer,
            attribute="omni:xform",
            is_array=False,
            semantic=ovstage.AttributeSemantic.MATRIX,
            convert=xform_tensor_from_warp,
        )

    def copy_completed(self, key: str, source: wp.array) -> wp.array:
        """Copy ``source`` through the publication transfer path and wait for completion.

        Args:
            key: Reusable target-storage key.
            source: CUDA array, normally on another device.

        Returns:
            The completed copy on the publication device.
        """
        target = self._copy_to_device(key, source)
        wp.synchronize_stream(self._stream)
        return target

    def close(self) -> None:
        """Drain exact transfer streams before releasing storage; retain it if draining fails."""
        for stream in self._source_streams:
            wp.synchronize_stream(stream)
        if self._stream is not None:
            wp.synchronize_stream(self._stream)
        self._sources.clear()
        self._source_streams.clear()
        self._buffers.clear()

    def _write(
        self,
        stage,
        binding,
        paths: Sequence[str],
        sources: list[wp.array],
        ordinal: int,
        producer: wp.Stream | None,
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
        # The stream that produced the sources orders the transfer; the device's current stream may
        # have changed since.
        self._source_streams = [
            source.device.stream if producer is None else producer for source in sources if source.device.is_cuda
        ]
        published = list(sources)
        for index, (path, source) in enumerate(zip(paths, sources, strict=True)):
            if not source.device.is_cuda or source.device == self._device:
                continue
            published[index] = self._copy_to_device(path, source, producer)
        # Stage reads after the copies on the transfer stream, so no host wait precedes it. Its
        # completion wait below surfaces a failed copy before the caller seals the ordinal.
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

    def _copy_to_device(self, key: str, source: wp.array, producer: wp.Stream | None = None) -> wp.array:
        """Enqueue one foreign-GPU copy into reusable target storage on the transfer stream.

        This is the exact transfer used for publication: the target comes from Warp's default
        allocator on the publication device, peer access is left to the driver, and the copy waits
        on the stream that produced ``source``, or on the source device's current stream when the
        caller names none. ``isaaclab_teleop.tools.check_cuda_peer`` exercises it.
        """
        if self._stream is None:
            # Warp creates streams with CU_STREAM_DEFAULT, which serialize with work on the legacy
            # default stream (Torch's default). A Torch stream is created non-blocking, so transfers
            # do not wait for unrelated default-stream work; retain it for the stream's lifetime.
            self._torch_stream = torch.cuda.Stream(device=str(self._device))
            self._stream = wp.stream_from_torch(self._torch_stream)
        target = self._buffers.get(key)
        if target is None or target.shape != source.shape or target.dtype != source.dtype:
            # Warp pool allocations and their first copy must use the same stream.
            with wp.ScopedStream(self._stream, sync_enter=False):
                target = wp.empty_like(source, device=self._device)
            self._buffers[key] = target
        self._stream.wait_stream(source.device.stream if producer is None else producer)
        wp.copy(target, source, stream=self._stream)
        return target
