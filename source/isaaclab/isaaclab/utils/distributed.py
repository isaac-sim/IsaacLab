# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for multi-GPU distributed training."""

from __future__ import annotations

import os

import torch


def resolve_cuda_device(local_rank: int) -> tuple[str, int]:
    """Determine the correct CUDA device for the current process in distributed training.

    When ``CUDA_VISIBLE_DEVICES`` restricts each process to a single GPU (common in
    SLURM/Kubernetes), ``local_rank`` may exceed the visible device count. In that
    case we fall back to ``cuda:0`` since the process can only see one GPU.

    Args:
        local_rank: The local rank of the current process.

    Returns:
        A tuple of ``(device_string, device_id)`` — e.g. ``("cuda:0", 0)`` or
        ``("cuda:2", 2)``.
    """
    num_visible = torch.cuda.device_count()
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if num_visible >= world_size:
        device_id = local_rank
    else:
        device_id = 0
    return f"cuda:{device_id}", device_id
