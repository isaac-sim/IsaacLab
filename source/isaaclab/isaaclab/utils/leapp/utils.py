# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warp as wp


def ensure_torch_tensor(value):
    """Convert Warp arrays to torch tensors while leaving torch tensors unchanged."""
    if isinstance(value, torch.Tensor):
        return value
    try:
        return wp.to_torch(value)
    except Exception:
        return value


def patch_warp_to_torch_passthrough() -> None:
    """Make ``wp.to_torch`` idempotent for torch tensors during export."""
    if getattr(wp.to_torch, "_leapp_passthrough_patch", False):
        return

    original_to_torch = wp.to_torch

    def patched_to_torch(value, *args, **kwargs):
        if isinstance(value, torch.Tensor):
            return value
        return original_to_torch(value, *args, **kwargs)

    patched_to_torch._leapp_passthrough_patch = True  # type: ignore[attr-defined]
    wp.to_torch = patched_to_torch
