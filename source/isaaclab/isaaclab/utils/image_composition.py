# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Choose image grids and compose explicit pixel arrays on their device."""

from __future__ import annotations

import math
from typing import Any

import warp as wp


def image_grid_columns(n_envs: int, n_gt: int, height: int, width: int, target_aspect: float = 1.0) -> int:
    """Choose complete environment rows first, then the closest display aspect ratio."""
    if not (math.isfinite(target_aspect) and target_aspect > 0):
        target_aspect = 1.0
    best_cols, best_score = 1, float("inf")
    for columns in range(1, n_envs + 1):
        rows = math.ceil(n_envs / columns)
        empty = rows * columns - n_envs
        aspect = columns * n_gt * width / (rows * height)
        score = empty * 10.0 + abs(math.log(aspect / target_aspect)) - columns * 1e-6
        if score < best_score:
            best_cols, best_score = columns, score
    return best_cols


@wp.kernel(enable_backward=False)
def _compose_image_channel(
    source: wp.array(dtype=Any, ndim=4),
    env_ids: wp.array(dtype=wp.int32),
    colors: wp.array(dtype=wp.uint8, ndim=2),
    mode: int,
    channel: int,
    num_channels: int,
    columns: int,
    depth_min: float,
    depth_span: float,
    output: wp.array(dtype=wp.uint8, ndim=3),
):
    tile, y, x = wp.tid()
    dst_y = tile // columns * source.shape[1] + y
    dst_x = (tile % columns * num_channels + channel) * source.shape[2] + x
    rgb = wp.vec3(0.0)
    if tile < env_ids.shape[0]:
        env = env_ids[tile]
        if mode == 0:
            rgb = wp.vec3(float(source[env, y, x, 0]), float(source[env, y, x, 1]), float(source[env, y, x, 2]))
        elif mode == 1:
            value = float(source[env, y, x, 0])
            if not wp.isnan(value):
                normalized = wp.clamp((value - depth_min) / depth_span, 0.0, 1.0)
                index = wp.min(int(normalized * 256.0), 255)
                rgb = wp.vec3(float(colors[index, 0]), float(colors[index, 1]), float(colors[index, 2]))
        elif mode == 2:
            for c in range(3):
                value = float(source[env, y, x, c])
                if not wp.isnan(value):
                    rgb[c] = wp.clamp((value + 1.0) * 127.5, 0.0, 255.0)
        else:
            identifier = wp.int32(source[env, y, x, 0])
            if source.shape[3] >= 3:
                identifier += wp.int32(source[env, y, x, 1]) * 256
                identifier += wp.int32(source[env, y, x, 2]) * 65536
            rgb = wp.vec3(40.0)
            if identifier != 0:
                # Match the established ID palette without enumerating IDs on the host.
                hue = wp.float64(identifier) * wp.float64(0.6180339887)
                h6 = (hue - wp.floor(hue)) * wp.float64(6.0)
                sector = int(h6) % 6
                fraction = h6 - wp.floor(h6)
                v = wp.float64(0.90)
                p = v * wp.float64(0.25)
                q = v * (wp.float64(1.0) - wp.float64(0.75) * fraction)
                t = v * (wp.float64(1.0) - wp.float64(0.75) * (wp.float64(1.0) - fraction))
                hsv = wp.vec3d(v, t, p)
                if sector == 1:
                    hsv = wp.vec3d(q, v, p)
                elif sector == 2:
                    hsv = wp.vec3d(p, v, t)
                elif sector == 3:
                    hsv = wp.vec3d(p, q, v)
                elif sector == 4:
                    hsv = wp.vec3d(t, p, v)
                elif sector == 5:
                    hsv = wp.vec3d(v, p, q)
                for c in range(3):
                    rgb[c] = float(int(hsv[c] * wp.float64(255.0)))
    for c in range(3):
        output[dst_y, dst_x, c] = wp.uint8(rgb[c])
    output[dst_y, dst_x, 3] = wp.uint8(255)


def compose_image(
    output: wp.array,
    sources: tuple[wp.array, ...],
    env_ids: wp.array,
    gt_types: tuple[str, ...],
    depth_colors: wp.array,
    *,
    depth_min: float = 0.1,
    depth_max: float = 10.0,
) -> None:
    """Write selected, colorized tiles into caller-owned RGBA storage on the source device.

    This operation allocates no pixel buffers, reads no pixels on the host, and never accesses a camera
    or renderer. The caller validates array layouts, owns stream ordering, and keeps source and output
    storage alive until the queued kernels complete.

    Args:
        output: Contiguous uint8 RGBA array sized for complete environment/channel tiles, shape [H, W, 4].
        sources: Published channel arrays with matching [N, H, W] dimensions, all on the output device.
        env_ids: Selected source rows, as an int32 array on the output device.
        gt_types: Display channels corresponding to sources: rgb, depth, normals, or segmentation.
        depth_colors: Uint8 color table on the output device, shape [256, 3] for depth or [0, 3] otherwise.
        depth_min: Near end of the depth color scale [m].
        depth_max: Far end of the depth color scale [m].
    """
    height, width = sources[0].shape[1:3]
    columns = output.shape[1] // (width * len(sources))
    tile_count = output.shape[0] // height * columns
    depth_span = max(depth_max - depth_min, 1e-6)
    for channel, (source, gt) in enumerate(zip(sources, gt_types, strict=True)):
        mode = ("rgb", "depth", "normals", "segmentation").index(gt)
        wp.launch(
            _compose_image_channel,
            dim=(tile_count, height, width),
            inputs=[source, env_ids, depth_colors, mode, channel, len(sources), columns, depth_min, depth_span, output],
            device=output.device,
        )
