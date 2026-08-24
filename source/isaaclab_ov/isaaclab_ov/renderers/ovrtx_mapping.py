# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stream-safe helpers for OVRTX attribute-binding mappings.

Filling a CUDA-mapped OVRTX attribute buffer from Warp is only correct when the commit at unmap
time is ordered against the Warp stream that produced the data: OVRTX's API contract requires the
mapped data to be ready when the unmap's CUDA sync signals, and an unmap without a sync performs
no synchronization at all. The ``with binding.map(...)`` form cannot carry that sync -- its
``__exit__`` takes no arguments -- so GPU writes through a mapping should use
:func:`map_attribute_for_warp_writes` instead of the binding's own context manager.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import warp as wp


def _cuda_device_id(device: str) -> int:
    """CUDA device index parsed from a Warp device string, e.g. ``"cuda:1"`` -> ``1``.

    TODO: A bare ``"cuda"`` parses to ``0`` while Warp enqueues fill work on its *current* CUDA
    device, so the mapping and the fill can target different GPUs on multi-GPU processes. The
    split predates this helper and is kept here to avoid a behavior change; a follow-up caches
    the resolved Warp device on the renderer instead of re-deriving it from strings.

    Args:
        device: Warp CUDA device string (``"cuda"`` or ``"cuda:<index>"``).

    Returns:
        The parsed CUDA device index, ``0`` when the string carries none.
    """
    parts = device.split(":")
    return int(parts[1]) if len(parts) > 1 else 0


@contextmanager
def map_attribute_for_warp_writes(binding: Any, device: str, dtype: Any) -> Iterator[wp.array]:
    """Map ``binding`` for CUDA writes and yield its buffer as a Warp array; commit after the fill.

    The caller fills the yielded array with Warp work enqueued on ``device``'s current stream (the
    default for ``wp.launch``/``wp.copy``). On exit -- error or not -- the mapping is unmapped with
    that stream as the CUDA sync, so OVRTX's commit of the mapped data waits for the fill on the
    GPU instead of racing it. OVRTX has no discard path (unmap always commits), so a failed fill
    still publishes whatever landed in the buffer.

    Args:
        binding: OVRTX attribute binding (from ``bind_attribute``) whose buffer is written.
        device: Warp CUDA device the fill work runs on (e.g. ``"cuda:0"``).
        dtype: Warp dtype the mapped tensor is viewed as (e.g. ``wp.mat44d``).

    Yields:
        The mapped buffer as a zero-copy Warp array, valid only inside the ``with`` block.
    """
    # Deferred so importing this module (e.g. through the package's lazy exports) cannot initialize
    # ovrtx without the guarded environment ``ovrtx_renderer`` establishes (OVRTX_SKIP_USD_CHECK,
    # actionable install error). Any real ``binding`` was created through that path, so ovrtx is
    # already imported by the time this runs.
    from ovrtx import Device  # noqa: PLC0415

    attr_mapping = binding.map(device=Device.CUDA, device_id=_cuda_device_id(device))
    try:
        yield wp.from_dlpack(attr_mapping.tensor, dtype=dtype)
    finally:
        attr_mapping.unmap(stream=wp.get_stream(device).cuda_stream)
