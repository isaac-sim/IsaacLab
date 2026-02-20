# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Image writing for OVRTX renderer (disk output when image_folder is set)."""

from pathlib import Path

import numpy as np
from PIL import Image

from .ovrtx_renderer_kernels import normalize_depth_to_uint8


class OVRTXImageWriter:
    """Writes OVRTX render output to disk (RGBA, depth, tiled) when image_folder is set."""

    def __init__(
        self,
        image_folder: str,
        num_envs: int,
        num_cols: int,
        num_rows: int,
    ):
        self._image_folder = Path(image_folder)
        self._image_folder.mkdir(parents=True, exist_ok=True)
        self._num_envs = num_envs
        self._num_cols = num_cols
        self._num_rows = num_rows

    def save_rgba(
        self,
        data_np: np.ndarray,
        env_idx: int,
        frame_counter: int,
        suffix: str = "",
    ) -> None:
        """Save per-env RGBA/RGB image. data_np: (H, W, 3) or (H, W, 4), float or uint8."""
        data_np = self._to_uint8_rgba(data_np)
        mode = "RGBA" if data_np.shape[2] == 4 else "RGB"
        path = self._path_env(suffix, frame_counter, env_idx, ext=".png")
        Image.fromarray(data_np, mode=mode).save(path)
        if env_idx == 0 and frame_counter <= 5:
            print(f"[OVRTX] Saved rendered image: {path}")

    def save_depth(
        self,
        depth_np: np.ndarray,
        env_idx: int,
        frame_counter: int,
    ) -> None:
        """Save per-env depth as grayscale PNG (normalized). depth_np: (H, W) or (H, W, 1)."""
        if depth_np.ndim == 3 and depth_np.shape[2] == 1:
            depth_np = depth_np[:, :, 0]
        normalized, dmin, dmax = normalize_depth_to_uint8(depth_np)
        path = self._path_env("depth", frame_counter, env_idx, ext=".png")
        Image.fromarray(normalized, mode="L").save(path)
        if env_idx == 0 and frame_counter <= 5 and dmin is not None and dmax is not None:
            print(f"[OVRTX] Saved depth image: {path} (range: {dmin:.3f} to {dmax:.3f})")

    def save_tiled_rgba(
        self,
        tiled_np: np.ndarray,
        frame_counter: int,
        suffix: str = "",
    ) -> None:
        """Save full tiled RGBA image (all envs in grid)."""
        tiled_np = self._to_uint8_rgba(tiled_np)
        path = self._path_tiled(suffix, frame_counter, ext=".png")
        Image.fromarray(tiled_np, mode="RGBA").save(path)
        if frame_counter <= 5:
            print(
                f"[OVRTX] Saved tiled {suffix + ' ' if suffix else ''}image "
                f"({self._num_envs} envs in {self._num_cols}x{self._num_rows} grid): {path}"
            )

    def save_tiled_depth(
        self,
        tiled_depth_np: np.ndarray,
        frame_counter: int,
    ) -> None:
        """Save full tiled depth as grayscale PNG (normalized)."""
        normalized, dmin, dmax = normalize_depth_to_uint8(tiled_depth_np)
        path = self._path_tiled("depth", frame_counter, ext=".png")
        Image.fromarray(normalized, mode="L").save(path)
        if frame_counter <= 5 and dmin is not None and dmax is not None:
            print(
                f"[OVRTX] Saved tiled depth image ({self._num_envs} envs in "
                f"{self._num_cols}x{self._num_rows} grid): {path} "
                f"(range: {dmin:.3f} to {dmax:.3f})"
            )

    def _path_env(self, suffix: str, frame: int, env_idx: int, ext: str = ".png") -> Path:
        if suffix:
            return self._image_folder / f"{suffix}_frame_{frame:06d}_env_{env_idx:04d}{ext}"
        return self._image_folder / f"frame_{frame:06d}_env_{env_idx:04d}{ext}"

    def _path_tiled(self, suffix: str, frame: int, ext: str = ".png") -> Path:
        if suffix:
            return self._image_folder / f"{suffix}_frame_{frame:06d}_tiled{ext}"
        return self._image_folder / f"frame_{frame:06d}_tiled{ext}"

    @staticmethod
    def _to_uint8_rgba(data: np.ndarray) -> np.ndarray:
        if data.dtype in (np.float32, np.float64):
            data = (data * 255).astype(np.uint8)
        return data
