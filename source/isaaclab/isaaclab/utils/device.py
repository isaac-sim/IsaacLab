# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for selecting compute devices."""

from __future__ import annotations


def set_cuda_device(device: str | int) -> None:
    """Set the process-wide CUDA device for both PyTorch and Warp.

    PyTorch must select the device before Warp so that Warp initializes and
    retains the primary CUDA context for the intended device.

    Args:
        device: CUDA device index or a device string such as ``"cuda:1"``.
    """
    import torch

    torch.cuda.set_device(device)

    import warp as wp

    warp_device = f"cuda:{device}" if isinstance(device, int) else device
    wp.set_device(warp_device)
