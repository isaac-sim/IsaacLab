# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generic utility to save rendered/sensor data to disk for debugging and visualization.

Accepts camera output dicts (e.g. ``CameraData.output``), or any ``dict[str, torch.Tensor]``
with keys like ``"rgb"``, ``"depth"``, ``"distance_to_image_plane"``, etc. Data is saved
per data type in formats suitable for inspection: PNG for images, NPY for float/int arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def save_rendered_data(
    data: dict[str, torch.Tensor] | Any,
    output_dir: str | Path,
    *,
    env_index: int | None = None,
    frame_index: int | None = None,
    prefix: str = "",
    save_depth_as_png: bool = True,
    grid_rgb: bool = False,
) -> list[str]:
    """Save rendered/sensor data to disk for debugging and visualization.

    Accepts a dictionary of named tensors (e.g. from ``CameraData.output`` or any
    renderer output). Each key is saved in an appropriate format:

    - **rgb / rgba**: PNG (uint8). Multiple envs can be saved as a single grid or per-env files.
    - **depth, distance_to_image_plane, distance_to_camera**: NPY (float32) and optionally
      a normalized PNG for quick visualization.
    - **normals, motion_vectors**: NPY (float32).
    - **semantic_segmentation, instance_segmentation_fast, instance_id_segmentation_fast**:
      PNG if colorized (uint8), otherwise NPY (int32).

    Tensors are expected to have shape ``(N, H, W, C)`` with N the number of envs/cameras.
    They are moved to CPU if necessary.

    Args:
        data: Either a ``dict[str, torch.Tensor]`` of name -> tensor, or an object with
            an ``output`` attribute (e.g. ``CameraData``) containing such a dict.
        output_dir: Directory to write files into. Created if it does not exist.
        env_index: If set, save only this environment index; otherwise save all (or a grid).
        frame_index: Optional frame/step index appended to filenames (e.g. ``rgb_000042.png``).
        prefix: Optional string prepended to filenames (e.g. ``"cam0_"``).
        save_depth_as_png: If True, save depth-like buffers also as normalized PNGs for quick view.
        grid_rgb: If True and multiple envs, save RGB/RGBA as a single grid image; otherwise per-env.

    Returns:
        List of created file paths (absolute or as given).

    Example:
        >>> from isaaclab.utils import save_rendered_data
        >>> # After rendering, e.g. with TiledCamera
        >>> save_rendered_data(
        ...     camera.data.output,
        ...     "/tmp/debug_frames",
        ...     frame_index=step_index,
        ...     save_depth_as_png=True,
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(data, "output") and isinstance(getattr(data, "output"), dict):
        data = data.output

    if not isinstance(data, dict):
        raise TypeError("data must be a dict[str, torch.Tensor] or an object with .output dict.")

    created: list[str] = []
    frame_suffix = f"_{frame_index:06d}" if frame_index is not None else ""

    for name, tensor in data.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        # (N, H, W, C)
        t = tensor.detach().to(device="cpu")
        if t.dim() != 4:
            continue
        n, h, w, c = t.shape
        if env_index is not None:
            if env_index < 0 or env_index >= n:
                continue
            t = t[env_index : env_index + 1]
            n = 1

        if name in ("rgb", "rgba"):
            _save_rgb_like(t, output_dir, prefix, name, frame_suffix, grid_rgb, created)
        elif name in ("depth", "distance_to_image_plane", "distance_to_camera"):
            _save_depth_like(t, output_dir, prefix, name, frame_suffix, save_depth_as_png, created)
        elif name in ("normals", "motion_vectors"):
            _save_float_like(t, output_dir, prefix, name, frame_suffix, created)
        elif "semantic_segmentation" in name or "instance_segmentation" in name or "instance_id_segmentation" in name:
            _save_segmentation_like(t, output_dir, prefix, name, frame_suffix, created)
        else:
            _save_generic(t, output_dir, prefix, name, frame_suffix, created)

    return created


def _save_rgb_like(
    t: torch.Tensor,
    output_dir: Path,
    prefix: str,
    name: str,
    frame_suffix: str,
    grid_rgb: bool,
    created: list[str],
) -> None:
    """Save uint8 RGB/RGBA as PNG(s)."""
    if t.dtype != torch.uint8:
        t = (t.clamp(0.0, 1.0) * 255.0).byte() if t.dtype in (torch.float32, torch.float64) else t.byte()
    n = t.shape[0]
    if n == 1:
        f = output_dir / f"{prefix}{name}{frame_suffix}.png"
        _tensor_to_png(t[0], f)
        created.append(str(f))
        return
    if grid_rgb:
        f = output_dir / f"{prefix}{name}{frame_suffix}_grid.png"
        _tensor_grid_to_png(t, f)
        created.append(str(f))
    else:
        for i in range(n):
            f = output_dir / f"{prefix}{name}{frame_suffix}_env{i:04d}.png"
            _tensor_to_png(t[i], f)
            created.append(str(f))


def _save_depth_like(
    t: torch.Tensor,
    output_dir: Path,
    prefix: str,
    name: str,
    frame_suffix: str,
    save_as_png: bool,
    created: list[str],
) -> None:
    """Save float depth-like as NPY and optionally normalized PNG."""
    arr = t.numpy()
    n = arr.shape[0]
    for i in range(n):
        f_npy = output_dir / f"{prefix}{name}{frame_suffix}_env{i:04d}.npy"
        np.save(f_npy, arr[i])
        created.append(str(f_npy))
    if save_as_png and n > 0:
        valid = np.isfinite(arr) & (arr > 0)
        vmin = float(np.min(arr[valid])) if np.any(valid) else 0.0
        vmax = float(np.max(arr[valid])) if np.any(valid) else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        for i in range(n):
            f_png = output_dir / f"{prefix}{name}{frame_suffix}_env{i:04d}_vis.png"
            _depth_to_png(arr[i], f_png, vmin, vmax)
            created.append(str(f_png))


def _save_float_like(
    t: torch.Tensor,
    output_dir: Path,
    prefix: str,
    name: str,
    frame_suffix: str,
    created: list[str],
) -> None:
    """Save float tensors as NPY."""
    arr = t.numpy()
    n = arr.shape[0]
    for i in range(n):
        f = output_dir / f"{prefix}{name}{frame_suffix}_env{i:04d}.npy"
        np.save(f, arr[i])
        created.append(str(f))


def _save_segmentation_like(
    t: torch.Tensor,
    output_dir: Path,
    prefix: str,
    name: str,
    frame_suffix: str,
    created: list[str],
) -> None:
    """Save segmentation as PNG (colorized) or NPY (raw indices)."""
    n = t.shape[0]
    if t.shape[-1] >= 3 and t.dtype == torch.uint8:
        for i in range(n):
            f = output_dir / f"{prefix}{name}{frame_suffix}_env{i:04d}.png"
            _tensor_to_png(t[i], f)
            created.append(str(f))
    else:
        arr = t.numpy()
        for i in range(n):
            f = output_dir / f"{prefix}{name}{frame_suffix}_env{i:04d}.npy"
            np.save(f, arr[i])
            created.append(str(f))


def _save_generic(
    t: torch.Tensor,
    output_dir: Path,
    prefix: str,
    name: str,
    frame_suffix: str,
    created: list[str],
) -> None:
    """Save any other tensor as NPY."""
    arr = t.numpy()
    n = arr.shape[0]
    for i in range(n):
        f = output_dir / f"{prefix}{name}{frame_suffix}_env{i:04d}.npy"
        np.save(f, arr[i])
        created.append(str(f))


def _tensor_to_png(t: torch.Tensor, path: Path) -> None:
    """Write a single (H, W, C) tensor to PNG. C in {1,2,3,4}."""
    arr = t.numpy()
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        from PIL import Image

        Image.fromarray(arr, mode="RGBA").save(path)
    else:
        from PIL import Image

        Image.fromarray(arr, mode="RGB").save(path)


def _tensor_grid_to_png(t: torch.Tensor, path: Path) -> None:
    """Write (N, H, W, C) to a single grid PNG."""
    try:
        from torchvision.utils import make_grid, save_image
    except ImportError:
        # Fallback: tile manually and save with PIL
        n, h, w, c = t.shape
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        # (N, C, H, W) for make_grid convention
        t_nchw = t.permute(0, 3, 1, 2).float() / 255.0
        grid = torch.zeros(rows * h, cols * w, c, dtype=torch.uint8)
        for i in range(n):
            r, c_ = i // cols, i % cols
            grid[r * h : (r + 1) * h, c_ * w : (c_ + 1) * w] = t[i]
        from PIL import Image

        if c == 4:
            Image.fromarray(grid.numpy(), mode="RGBA").save(path)
        else:
            Image.fromarray(grid.numpy(), mode="RGB").save(path)
        return
    # torchvision path: (N, C, H, W), value in [0, 1] for save_image
    t_nchw = t.permute(0, 3, 1, 2)
    if t_nchw.dtype != torch.float32:
        t_nchw = t_nchw.float() / 255.0
    grid = make_grid(t_nchw, nrow=int(np.ceil(t.shape[0] ** 0.5)))
    save_image(grid, path)


def _depth_to_png(arr: np.ndarray, path: Path, vmin: float, vmax: float) -> None:
    """Save a single (H, W, 1) depth array as normalized RGB PNG."""
    from PIL import Image

    d = np.squeeze(arr)
    valid = np.isfinite(d) & (d > 0)
    norm = np.zeros_like(d, dtype=np.float64)
    norm[valid] = (d[valid] - vmin) / (vmax - vmin)
    norm = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([norm, norm, norm], axis=-1)
    Image.fromarray(rgb, mode="RGB").save(path)
