# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Torch bindings shared by the Newton controller wrappers."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import warp as wp


def bind_inputs(inputs, **tensors: torch.Tensor | None) -> None:
    """Point Newton input ports at torch tensors without copying.

    Each keyword names a port on the input struct; ``None`` leaves the port unchanged. Float32 contiguous
    tensors are bound in place, others are converted first.

    Raises:
        ValueError: If a tensor is given for a port the controller configuration disabled.
    """
    for name, tensor in tensors.items():
        if tensor is None:
            continue
        port = getattr(inputs, name)
        if port is None:
            raise ValueError(f"Input '{name}' is disabled by the controller configuration.")
        tensor = tensor.to(torch.float32).contiguous()
        # compact per-joint ports are flat over (env, joint)
        if port.dtype == wp.float32 and port.ndim == 1:
            tensor = tensor.reshape(-1)
        setattr(inputs, name, wp.from_torch(tensor, dtype=port.dtype))


def per_joint(value: float | Sequence[float] | None, num_envs: int, device: str) -> float | wp.array | None:
    """Convert a scalar or per-joint gain into Newton's scalar or compact per-DOF form."""
    if value is None or isinstance(value, (int, float)):
        return value
    return wp.array(list(value) * num_envs, dtype=wp.float32, device=device)


def spatial_vector(value: float | Sequence[float] | None) -> float | wp.spatial_vector | None:
    """Convert a scalar or ``(x, y, z, roll, pitch, yaw)`` tuple into Newton's per-axis form."""
    if value is None or isinstance(value, (int, float)):
        return value
    return wp.spatial_vector(*value)
