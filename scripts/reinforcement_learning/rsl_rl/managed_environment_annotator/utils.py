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
