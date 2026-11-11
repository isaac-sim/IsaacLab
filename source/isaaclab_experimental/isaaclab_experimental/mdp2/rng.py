# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-world random streams, bitwise identical to Warp's ``rand_init`` and ``randf``.

A stream holds one ``int32`` seed per world. Draw ``j`` of a step is ``wp.randf(wp.rand_init(rng[world], j), low,
high)`` in Warp and element ``j`` of :func:`uniform` in Torch. ``evaluate`` and ``reset`` advance the streams of
their worlds before they draw, and ``observe`` draws without advancing, so it repeats a world's last sample until
the world is stepped or reset. Timers draw at offset ``-1``.
"""

from __future__ import annotations

import math
import zlib

import numpy as np
import torch

_UINT32 = 0xFFFFFFFF


def uniform(rng: torch.Tensor, low: float, high: float, shape: tuple[int, ...] = (), offset: int = 0) -> torch.Tensor:
    """Draw uniform samples of shape ``(num_worlds, *shape)``. Flattened element ``j`` uses draw ``offset + j``."""
    offsets = torch.arange(offset, offset + math.prod(shape), device=rng.device) & _UINT32
    state = _pcg(((rng.long()[:, None] & _UINT32) + _pcg(offsets)) & _UINT32)
    values = (_pcg(state) >> 8).float() * (1.0 / 16777216.0)
    low, span = np.float32(low), np.float32(high) - np.float32(low)
    # Warp contracts ``span * value + low`` to a fused multiply-add on CUDA only.
    if values.is_cuda:
        values = torch.addcmul(torch.full_like(values, low), torch.full_like(values, span), values)
    else:
        values = values * float(span) + float(low)
    return values.reshape(-1, *shape)


def advance(rng: torch.Tensor, mask: torch.Tensor | None = None) -> None:
    """Advance streams of shape ``(num_streams, num_worlds)`` in place, only for the worlds in ``mask`` if given."""
    if rng.numel() == 0:
        return
    advanced = _int32(_pcg(rng.long() & _UINT32))
    rng.copy_(advanced if mask is None else torch.where(mask, advanced, rng))


def streams(seed: int, names: tuple[str, ...], num_worlds: int, device: str) -> torch.Tensor:
    """Create initial streams of shape ``(len(names), num_worlds)``, seeded from each name and world index."""
    keys = torch.tensor(
        [(seed + zlib.crc32(name.encode())) & _UINT32 for name in names], dtype=torch.int64, device=device
    )
    worlds = torch.arange(num_worlds, device=device)
    return _int32(_pcg((keys[:, None] + _pcg(worlds)[None, :]) & _UINT32))


def _pcg(x: torch.Tensor) -> torch.Tensor:
    """Warp's ``rand_pcg`` on ``uint32`` values held in ``int64``."""
    b = (x * 747796405 + 2891336453) & _UINT32
    c = (((b >> ((b >> 28) + 4)) ^ b) * 277803737) & _UINT32
    return (c >> 22) ^ c


def _int32(x: torch.Tensor) -> torch.Tensor:
    return (((x + 2**31) & _UINT32) - 2**31).to(torch.int32)
