# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared segmentation colorization used across renderer backends.

Segmentation ids are mapped to visually distinct RGBA colors by golden-ratio hue spacing in HSV
space, matching the Isaac RTX / Replicator convention so colorized segmentation outputs are visually
consistent across the RTX, OVRTX and Newton Warp renderers. Ids ``0`` (BACKGROUND) and ``1``
(UNLABELLED) are reserved and always map to ``(0, 0, 0, 0)`` and ``(0, 0, 0, 255)`` respectively.

Two equivalent implementations are provided from a single source of truth:

* :func:`random_color_from_id` / :func:`color_hash` run on the host (Python), used to build color
  palettes and the ``(r, g, b, a)`` tuple keys of ``camera.data.info`` id-to-label maps.
* :func:`random_color_from_id_wp` / :func:`color_hash_wp` are Warp device functions callable from
  kernels that colorize an id image in place.

The two must stay in agreement; ``test_segmentation_colors.py`` asserts they produce identical output.
"""

from __future__ import annotations

import warp as wp

# Reserved ids shared by every segmentation output.
BACKGROUND_ID: int = 0
"""Id of pixels where nothing was hit (ray miss / background)."""
UNLABELLED_ID: int = 1
"""Id of geometry with no semantic label matching the active filter."""


def pack_rgba(color: tuple[int, int, int, int]) -> int:
    """Pack an ``(r, g, b, a)`` tuple into a little-endian ``r | g<<8 | b<<16 | a<<24`` uint32."""
    r, g, b, a = color
    return (r & 0xFF) | ((g & 0xFF) << 8) | ((b & 0xFF) << 16) | ((a & 0xFF) << 24)


def unpack_rgba(color: int) -> tuple[int, int, int, int]:
    """Unpack a little-endian ``r | g<<8 | b<<16 | a<<24`` uint32 into an ``(r, g, b, a)`` tuple."""
    return (color & 0xFF, (color >> 8) & 0xFF, (color >> 16) & 0xFF, (color >> 24) & 0xFF)


def color_hash(seed: int) -> int:
    """Host equivalent of :func:`color_hash_wp` (uint32 arithmetic, overflow truncates to 32 bits)."""
    mask32 = 0xFFFFFFFF
    h = seed & mask32
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & mask32
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & mask32
    h ^= h >> 16
    return h & mask32


def random_color_from_id(input_id: int) -> tuple[int, int, int, int]:
    """Host equivalent of :func:`random_color_from_id_wp`, returning an ``(r, g, b, a)`` tuple.

    ``0`` (BACKGROUND) maps to ``(0, 0, 0, 0)`` and ``1`` (UNLABELLED) to ``(0, 0, 0, 255)`` as
    special cases; all other ids get a deterministic, visually distinct opaque color.
    """
    if input_id == BACKGROUND_ID:
        return (0, 0, 0, 0)
    if input_id == UNLABELLED_ID:
        return (0, 0, 0, 255)

    hash_val = color_hash(input_id)

    golden_ratio_inv = 1.0 / 1.618033988749895
    hue = (input_id * golden_ratio_inv) % 1.0
    hue = (hue + (hash_val & 0xFFFF) / 65536.0 * 0.1) % 1.0

    saturation = 0.7 + 0.3 * (((hash_val >> 16) & 0xFF) / 255.0)
    value = 0.8 + 0.2 * (((hash_val >> 8) & 0xFF) / 255.0)

    hue_i = int(hue * 6.0)
    hue_f = hue * 6.0 - hue_i
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * hue_f)
    t = value * (1.0 - saturation * (1.0 - hue_f))

    hue_i %= 6
    if hue_i == 0:
        r, g, b = value, t, p
    elif hue_i == 1:
        r, g, b = q, value, p
    elif hue_i == 2:
        r, g, b = p, value, t
    elif hue_i == 3:
        r, g, b = p, q, value
    elif hue_i == 4:
        r, g, b = t, p, value
    else:
        r, g, b = value, p, q

    ri = min(255, max(0, int(r * 255.0)))
    gi = min(255, max(0, int(g * 255.0)))
    bi = min(255, max(0, int(b * 255.0)))
    return (ri, gi, bi, 255)


@wp.func
def color_hash_wp(seed: wp.uint32) -> wp.uint32:
    """MurmurHash3-style 32-bit finalizer matching omni.replicator's ``randomColoursCPU``.

    Arithmetic is intentionally uint32 so multiplications overflow and truncate to 32 bits.
    Computing in a wider type yields different bits and breaks parity with Replicator / Isaac RTX.
    """
    h = seed
    h = h ^ (h >> wp.uint32(16))
    h = h * wp.uint32(0x85EBCA6B)
    h = h ^ (h >> wp.uint32(13))
    h = h * wp.uint32(0xC2B2AE35)
    h = h ^ (h >> wp.uint32(16))
    return h


@wp.func
def random_color_from_id_wp(input_id: wp.uint32) -> wp.uint32:
    """Generate a deterministic, visually distinct color from a segmentation id.

    Uses golden-ratio hue spacing in HSV space then converts to RGB, packed as
    ``r | (g<<8) | (b<<16) | (a<<24)``. Id ``0`` maps to ``0`` (BACKGROUND) and id ``1`` to
    ``0xFF000000`` (UNLABELLED, opaque black).
    """
    if input_id == wp.uint32(0):
        # BACKGROUND special case
        return wp.uint32(0)
    if input_id == wp.uint32(1):
        # UNLABELLED special case
        return wp.uint32(0xFF000000)

    hash_val = color_hash_wp(input_id)

    # Golden ratio inverse = 1.0 / 1.618033988749895 (Replicator constant)
    GOLDEN_RATIO_INV = wp.float64(1.0) / wp.float64(1.618033988749895)

    # Use golden ratio spacing for maximum hue spread
    hue_tmp = wp.float64(input_id) * GOLDEN_RATIO_INV
    hue = hue_tmp - wp.floor(hue_tmp)

    # Add hash-based perturbation for better distribution
    hue_perturbation = wp.float64(hash_val & wp.uint32(0xFFFF)) / wp.float64(65536.0)
    hue_tmp = hue + hue_perturbation * wp.float64(0.1)
    hue = hue_tmp - wp.floor(hue_tmp)

    # Use hash to determine saturation and value for maximum spread
    sat_part = wp.uint32((hash_val >> wp.uint32(16)) & wp.uint32(0xFF))
    val_part = wp.uint32((hash_val >> wp.uint32(8)) & wp.uint32(0xFF))

    # Saturation: 0.7 to 1.0 for vibrant colors
    saturation = wp.float64(0.7) + wp.float64(0.3) * (wp.float64(sat_part) / wp.float64(255.0))

    # Value: 0.8 to 1.0 for bright colors
    value = wp.float64(0.8) + wp.float64(0.2) * (wp.float64(val_part) / wp.float64(255.0))

    # HSV to RGB conversion (match Replicator: ``f`` uses pre-modulo ``int(hue*6)``, then ``i %= 6``).
    hue_i = wp.int32(hue * wp.float64(6.0))
    hue_f = hue * wp.float64(6.0) - wp.float64(hue_i)
    p = value * (wp.float64(1.0) - saturation)
    q = value * (wp.float64(1.0) - saturation * hue_f)
    t = value * (wp.float64(1.0) - saturation * (wp.float64(1.0) - hue_f))

    r = wp.float64(0.0)
    g = wp.float64(0.0)
    b = wp.float64(0.0)

    hue_i = hue_i % 6

    if hue_i == 0:
        r = value
        g = t
        b = p
    elif hue_i == 1:
        r = q
        g = value
        b = p
    elif hue_i == 2:
        r = p
        g = value
        b = t
    elif hue_i == 3:
        r = p
        g = q
        b = value
    elif hue_i == 4:
        r = t
        g = p
        b = value
    else:
        r = value
        g = p
        b = q

    ri = wp.min(255, wp.max(0, wp.int32(r * wp.float64(255.0))))
    gi = wp.min(255, wp.max(0, wp.int32(g * wp.float64(255.0))))
    bi = wp.min(255, wp.max(0, wp.int32(b * wp.float64(255.0))))
    ai = wp.int32(255)

    color = (
        wp.uint32(ri)
        | (wp.uint32(gi) << wp.uint32(8))
        | (wp.uint32(bi) << wp.uint32(16))
        | (wp.uint32(ai) << wp.uint32(24))
    )
    return color
