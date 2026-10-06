# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare image composition parameters on the host and compose pixels on their device."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp


@dataclass(frozen=True)
class ImageCompositionParams:
    """Cached grid dimensions, channel modes, and device arrays for tiled image composition.

    Created by :func:`prepare_image_composition` and reused by :func:`compose_image`.
    The caller owns the output image and rebuilds these parameters when display settings or source layout change.
    """

    env_ids: wp.array
    channel_modes: tuple[int, ...]
    source_layout: tuple[tuple, ...]
    columns: int
    output_shape: tuple[int, int, int]
    depth_min: float
    depth_span: float
    depth_colors: wp.array


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


def prepare_image_composition(
    sources: tuple[wp.array, ...],
    gt_types: tuple[str, ...],
    env_ids: list[int],
    *,
    target_aspect: float = 1.0,
    depth_min: float = 0.1,
    depth_max: float = 10.0,
) -> ImageCompositionParams:
    """Prepare grid dimensions and channel modes, uploading indices and color tables once.

    Args:
        sources: Published arrays of shape [N, H, W, C], all on the same device.
        gt_types: Display channels corresponding to the arrays: rgb, depth, normals, or segmentation.
        env_ids: Ordered source rows to display.
        target_aspect: Desired output width divided by height.
        depth_min: Near end of the depth color scale [m].
        depth_max: Far end of the depth color scale [m].

    Returns:
        Cached parameters for composing an opaque uint8 RGBA image.
        Prepare them again when selection, source layout, or color settings change.
    """
    if not sources or len(sources) != len(gt_types) or not env_ids:
        raise ValueError("Image composition requires matching sources and channels and at least one selected row.")
    for source, gt in zip(sources, gt_types, strict=True):
        channels = 3 if gt in ("rgb", "normals") else 1
        if source.ndim != 4 or min(source.shape) < 1 or source.shape[3] < channels:
            raise ValueError(f"Channel {gt!r} requires nonempty [N, H, W, C] arrays with at least {channels} channels.")
    device = sources[0].device
    n, height, width, _ = sources[0].shape
    if any(source.device != device or source.shape[:3] != (n, height, width) for source in sources):
        raise ValueError("Image channels must have the same batch size, resolution, and device.")
    if min(env_ids) < 0 or max(env_ids) >= n:
        raise ValueError(f"Image row selection is outside the source batch of {n} rows.")
    modes = tuple(("rgb", "depth", "normals", "segmentation").index(gt) for gt in gt_types)
    columns = image_grid_columns(len(env_ids), len(sources), height, width, target_aspect)
    rows = math.ceil(len(env_ids) / columns)
    colors = np.empty((0, 3), dtype=np.uint8)
    if "depth" in gt_types:
        from matplotlib import colormaps

        colors = (colormaps["turbo"](np.arange(256) / 255.0)[..., :3] * 255).astype(np.uint8)
    return ImageCompositionParams(
        env_ids=wp.array(env_ids, dtype=wp.int32, device=device),
        channel_modes=modes,
        source_layout=tuple((source.shape, source.dtype, source.device) for source in sources),
        columns=columns,
        output_shape=(rows * height, columns * len(sources) * width, 4),
        depth_min=depth_min,
        depth_span=max(depth_max - depth_min, 1e-6),
        depth_colors=wp.array(colors, dtype=wp.uint8, device=device),
    )


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


def compose_image(output: wp.array, sources: tuple[wp.array, ...], params: ImageCompositionParams) -> None:
    """Write selected, colorized tiles into caller-owned RGBA storage on the source device.

    Arrays must match the prepared parameters. This operation allocates no pixel buffers, reads no
    pixels on the host, and never accesses a camera or renderer. The caller owns stream ordering
    and must keep source and output storage alive until the queued kernels complete.

    Args:
        output: Contiguous uint8 array with ``params.output_shape`` on the source device.
        sources: Published channel arrays, in the order used to prepare the parameters.
        params: Resolved image layout and color settings.
    """
    height, width = sources[0].shape[1:3]
    tile_count = params.output_shape[0] // height * params.columns
    for channel, (source, mode) in enumerate(zip(sources, params.channel_modes, strict=True)):
        wp.launch(
            _compose_image_channel,
            dim=(tile_count, height, width),
            inputs=[
                source,
                params.env_ids,
                params.depth_colors,
                mode,
                channel,
                len(sources),
                params.columns,
                params.depth_min,
                params.depth_span,
                output,
            ],
            device=output.device,
        )
