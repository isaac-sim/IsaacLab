# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-native function-based modifiers (experimental).

Each modifier takes a ``wp.array`` as its first argument and operates **in-place**
via ``wp.launch``.  The calling convention mirrors Warp MDP terms::

    modifier.func(data_wp, **params) -> None
"""

from __future__ import annotations

import warp as wp

# -- scale --------------------------------------------------------------------


@wp.kernel
def _scale_kernel(data: wp.array(dtype=wp.float32, ndim=2), multiplier: wp.float32):
    i, j = wp.tid()
    data[i, j] = data[i, j] * multiplier


def scale(data: wp.array, multiplier: float) -> None:
    """Scale all elements of *data* by *multiplier* in-place.

    Warp-native drop-in replacement for :func:`isaaclab.utils.modifiers.scale`.

    Args:
        data: The observation buffer to modify. Shape ``(num_envs, D)``.
        multiplier: Scalar multiplier.
    """
    wp.launch(_scale_kernel, dim=data.shape, inputs=[data, float(multiplier)], device=data.device)


# -- bias ---------------------------------------------------------------------


@wp.kernel
def _bias_kernel(data: wp.array(dtype=wp.float32, ndim=2), value: wp.float32):
    i, j = wp.tid()
    data[i, j] = data[i, j] + value


def bias(data: wp.array, value: float) -> None:
    """Add a uniform *value* to all elements of *data* in-place.

    Warp-native drop-in replacement for :func:`isaaclab.utils.modifiers.bias`.

    Args:
        data: The observation buffer to modify. Shape ``(num_envs, D)``.
        value: Scalar bias to add.
    """
    wp.launch(_bias_kernel, dim=data.shape, inputs=[data, float(value)], device=data.device)


# -- clip ---------------------------------------------------------------------


@wp.kernel
def _clip_kernel(data: wp.array(dtype=wp.float32, ndim=2), lo: wp.float32, hi: wp.float32):
    i, j = wp.tid()
    data[i, j] = wp.clamp(data[i, j], lo, hi)


def clip(data: wp.array, bounds: tuple[float | None, float | None]) -> None:
    """Clamp all elements of *data* to [lo, hi] in-place.

    Warp-native drop-in replacement for :func:`isaaclab.utils.modifiers.clip`.

    Args:
        data: The observation buffer to modify. Shape ``(num_envs, D)``.
        bounds: ``(min, max)`` tuple.  ``None`` means no bound on that side.
    """
    lo = float(bounds[0]) if bounds[0] is not None else float(-1e38)
    hi = float(bounds[1]) if bounds[1] is not None else float(1e38)
    wp.launch(_clip_kernel, dim=data.shape, inputs=[data, lo, hi], device=data.device)
