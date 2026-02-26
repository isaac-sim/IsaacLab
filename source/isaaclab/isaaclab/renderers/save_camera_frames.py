# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

# TODO: remove file when PR is reviewed


"""Save TiledCamera color output to PNG (RTX or Newton Warp path)."""

from __future__ import annotations

import logging
import math
import os

import torch

logger = logging.getLogger(__name__)


def _tile_camera_output_to_image(tensor: torch.Tensor, n: int, height: int, width: int) -> "np.ndarray":
    """Tile (n, H, W, C) tensor into a single (rows*H, cols*W, C) image. Returns uint8 numpy."""
    import numpy as np

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols) if n else 1
    arr = tensor.detach().cpu().numpy()
    if arr.dtype != np.uint8:
        if arr.dtype in (np.float32, np.float64) and arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    _, H, W, C = arr.shape
    out = np.zeros((rows * H, cols * W, C), dtype=np.uint8)
    for i in range(n):
        r, c = i // cols, i % cols
        out[r * H : (r + 1) * H, c * W : (c + 1) * W] = arr[i]
    return out


def save_data(camera, filename: str):
    """Save the current camera color output to a PNG.

    Prefers the camera's torch output buffer (camera._data.output["rgb"] or "rgba"]).
    Falls back to the Newton internal buffer when using isaaclab_newton's NewtonWarpRenderer.

    Args:
        camera: TiledCamera instance (must have ._data.output or ._renderer/._render_data for Warp).
        filename: Path for the PNG (e.g. "path/to/frame.png").
    """
    from PIL import Image

    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    # Prefer saving from torch output (works for RTX and Warp)
    output = getattr(getattr(camera, "_data", None), "output", None)
    if output is not None:
        rgb = output.get("rgb")
        if rgb is None and "rgba" in output:
            rgb = output["rgba"][..., :3]
        if rgb is not None and isinstance(rgb, torch.Tensor):
            try:
                n = rgb.shape[0]
                h, w = rgb.shape[1], rgb.shape[2]
                arr = _tile_camera_output_to_image(rgb, n, h, w)
                Image.fromarray(arr).save(filename)
                return
            except Exception as e:
                logger.warning("save_data: torch path failed for %s: %s", filename, e)

    # Fallback: Newton internal buffer (isaaclab_newton NewtonWarpRenderer)
    try:
        from isaaclab_newton.renderers import NewtonWarpRenderer
    except ImportError:
        return
    renderer = getattr(camera, "_renderer", None)
    if not isinstance(renderer, NewtonWarpRenderer):
        return
    render_data = getattr(camera, "_render_data", None)
    if render_data is None or not isinstance(render_data, NewtonWarpRenderer.RenderData):
        return
    color_image = getattr(render_data.outputs, "color_image", None)
    if color_image is None:
        return
    try:
        import warp as wp
        color_data = renderer.newton_sensor.render_context.utils.flatten_color_image_to_rgba(
            color_image
        )
        if hasattr(color_data, "numpy"):
            arr = wp.to_torch(color_data).cpu().numpy()
        else:
            arr = color_data
        Image.fromarray(arr).save(filename)
    except Exception as e:
        logger.warning("save_data: failed to save %s: %s", filename, e)
