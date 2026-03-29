# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from contextlib import suppress
from typing import Any

import torch
import warp as wp


def select_element_names(names: list[str] | None, indices: Any = None) -> list[str] | None:
    """Select element names using optional runtime indices."""
    if names is None:
        return None
    if indices is None or indices == slice(None):
        return list(names)
    if isinstance(indices, slice):
        return list(names[indices])
    with suppress(AttributeError):
        indices = indices.tolist()
    if isinstance(indices, (list, tuple)):
        return [names[int(index)] for index in indices]
    if isinstance(indices, int):
        return [names[indices]]
    return None


def ensure_torch_tensor(value):
    """Convert Warp arrays to torch tensors while leaving torch tensors unchanged."""
    if isinstance(value, torch.Tensor):
        return value

    return wp.to_torch(value)


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


# ══════════════════════════════════════════════════════════════════
# Connection Builders
# ══════════════════════════════════════════════════════════════════


def build_state_connection(entity_name: str, property_name: str) -> dict[str, str]:
    """Return a compact deployment connection string for a state property."""
    return {"isaaclab_connection": f"state:{entity_name}:{property_name}"}


def build_command_connection(command_name: str) -> dict[str, str]:
    """Return a compact deployment connection string for a command term."""
    return {"isaaclab_connection": f"command:{command_name}"}


def build_write_connection(entity_name: str, method_name: str) -> dict[str, str]:
    """Return a compact deployment connection string for an articulation write target."""
    return {"isaaclab_connection": f"write:{entity_name}:{method_name}"}
