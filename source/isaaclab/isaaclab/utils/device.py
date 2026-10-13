# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Internal utilities for selecting compute devices."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_GRAPH_MEMORY_RESERVED_DEVICES: set[str] = set()
"""Aliases of CUDA devices whose graph memory :func:`reserve_cuda_graph_memory` has already reserved."""


def set_cuda_device(device: str | int) -> None:
    """Set the process-wide CUDA device for both PyTorch and Warp.

    PyTorch must select the device before Warp so that a newly imported Warp
    runtime initializes on that device, or an existing runtime switches to it.

    Args:
        device: CUDA device index or a device string such as ``"cuda:1"``.
    """
    import torch

    torch.cuda.set_device(device)

    import warp as wp

    warp_device = f"cuda:{device}" if isinstance(device, int) else device
    wp.set_device(warp_device)


def reserve_cuda_graph_memory(device: str) -> None:
    """Reserve CUDA graph memory on a device by recording one tiny allocation in a graph.

    On Windows, the CUDA driver reserves a large virtual-address range for graph memory on the first
    allocation recorded in a CUDA graph. Once managed memory exists (cubric allocates it) and a
    rendered frame occupies the addresses the driver tries first, placing that range can take minutes,
    with the GPU idle. Recording one allocation before managed memory is allocated and before the next
    rendered frame reserves the range early; later captures reuse it. This was measured with OVRTX and
    with Kit and Newton physics. On other platforms the extra capture is harmless.
    See isaac-sim/IsaacLab#8342.

    Args:
        device: Warp device, such as ``"cuda:0"``. Devices other than CUDA devices are ignored.
    """
    import warp as wp

    warp_device = wp.get_device(device)
    if (
        not warp_device.is_cuda
        or not warp_device.is_mempool_enabled
        or warp_device.is_capturing
        or warp_device.alias in _GRAPH_MEMORY_RESERVED_DEVICES
    ):
        return
    try:
        # The array is freed inside the capture; the driver keeps the address range after the graph is destroyed.
        with wp.ScopedCapture(device=warp_device, force_module_load=False):
            wp.empty(1, dtype=wp.int32, device=warp_device)
    except Exception as error:
        logger.warning("Could not reserve CUDA graph memory on %s: %s", warp_device.alias, error)
        return
    _GRAPH_MEMORY_RESERVED_DEVICES.add(warp_device.alias)
