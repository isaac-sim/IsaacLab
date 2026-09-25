# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera frame colorization utilities for visualizer streaming views."""

from __future__ import annotations

import numpy as np
import torch

SUPPORTED_GT_TYPES: frozenset[str] = frozenset({"rgb", "depth", "segmentation", "normals"})
"""GT types accepted by :class:`CameraFrameColorizer`."""

# Primary sensor output key per GT type
_GT_TO_SENSOR_KEY: dict[str, str] = {
    "rgb": "rgb",
    "depth": "depth",
    "segmentation": "semantic_segmentation",
    "normals": "normals",
}
# Fallback sensor key when the primary is absent
_GT_SENSOR_KEY_FALLBACK: dict[str, str] = {
    "depth": "distance_to_image_plane",
}


def sensor_key_for_gt_type(gt_type: str, available_keys: frozenset[str] | None = None) -> str:
    """Return the camera sensor output key for a streaming GT type.

    Args:
        gt_type: One of :data:`SUPPORTED_GT_TYPES`.
        available_keys: Keys present in ``camera.data.output``. When provided the
            fallback key for ``"depth"`` is tried if the primary is absent.

    Returns:
        The sensor output key to index into ``camera.data.output``.

    Raises:
        ValueError: If ``gt_type`` is not in :data:`SUPPORTED_GT_TYPES`.
        KeyError: If no matching key exists in ``available_keys``.
    """
    if gt_type not in SUPPORTED_GT_TYPES:
        raise ValueError(f"GT type {gt_type!r} is not supported. Valid types: {sorted(SUPPORTED_GT_TYPES)}")
    primary = _GT_TO_SENSOR_KEY[gt_type]
    if available_keys is None:
        return primary
    if primary in available_keys:
        return primary
    fallback = _GT_SENSOR_KEY_FALLBACK.get(gt_type)
    if fallback and fallback in available_keys:
        return fallback
    raise KeyError(
        f"No sensor output found for GT type {gt_type!r}. "
        f"Tried {primary!r} and {fallback!r}; available: {sorted(available_keys)}"
    )


def sensor_keys_for_gt_types(gt_types: list[str]) -> list[str]:
    """Return deduplicated ``CameraCfg.data_types`` list for the given GT types.

    Args:
        gt_types: List of streaming GT types (e.g. ``["rgb", "depth"]``).

    Returns:
        Ordered, deduplicated sensor data-type strings suitable for
        ``CameraCfg(data_types=...)``.
    """
    seen: set[str] = set()
    keys: list[str] = []
    for gt in gt_types:
        key = _GT_TO_SENSOR_KEY.get(gt)
        if key and key not in seen:
            keys.append(key)
            seen.add(key)
    return keys


class CameraFrameColorizer:
    """Colorize raw camera sensor frames for streaming display.

    Each method accepts a single-env frame tensor ``(H, W, C)`` and returns a
    ``uint8`` NumPy array with shape ``(H, W, 3)``.
    """

    @staticmethod
    def colorize(
        data: torch.Tensor,
        gt_type: str,
        *,
        depth_min: float = 0.1,
        depth_max: float = 10.0,
    ) -> np.ndarray:
        """Colorize one camera frame.

        Args:
            data: Raw sensor output for one env, shape ``(H, W, C)``.
            gt_type: One of ``"rgb"``, ``"depth"``, ``"segmentation"``, or ``"normals"``.
            depth_min: Near-clip for the turbo depth colormap [m].
            depth_max: Far-clip for the turbo depth colormap [m].

        Returns:
            Colorized ``uint8`` array with shape ``(H, W, 3)``.

        Raises:
            ValueError: If ``gt_type`` is not in :data:`SUPPORTED_GT_TYPES`.
        """
        if gt_type not in SUPPORTED_GT_TYPES:
            raise ValueError(f"GT type {gt_type!r} is not supported. Valid types: {sorted(SUPPORTED_GT_TYPES)}")
        if gt_type == "rgb":
            return CameraFrameColorizer._colorize_rgb(data)
        if gt_type == "depth":
            return CameraFrameColorizer._colorize_depth(data, depth_min, depth_max)
        if gt_type == "normals":
            return CameraFrameColorizer._colorize_normals(data)
        return CameraFrameColorizer._colorize_segmentation(data)

    # ------------------------------------------------------------------
    # Private per-type implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _colorize_rgb(data: torch.Tensor) -> np.ndarray:
        arr = data.cpu().numpy() if isinstance(data, torch.Tensor) else np.asarray(data)
        return np.ascontiguousarray(arr[..., :3]).astype(np.uint8)

    @staticmethod
    def _colorize_depth(data: torch.Tensor, depth_min: float, depth_max: float) -> np.ndarray:
        arr = (
            data.squeeze(-1).float().cpu().numpy()
            if isinstance(data, torch.Tensor)
            else np.asarray(data, dtype=np.float32).squeeze(-1)
        )
        span = max(float(depth_max - depth_min), 1e-6)
        norm = np.clip((arr - depth_min) / span, 0.0, 1.0)
        try:
            import matplotlib.cm

            rgb_f = matplotlib.cm.get_cmap("turbo")(norm)[..., :3]
            return (rgb_f * 255).astype(np.uint8)
        except Exception:
            return CameraFrameColorizer._fallback_depth_gradient(norm)

    @staticmethod
    def _colorize_segmentation(data: torch.Tensor) -> np.ndarray:
        arr = data.cpu().numpy() if isinstance(data, torch.Tensor) else np.asarray(data)
        # Class ID packed as R + G*256 + B*65536 across the first 3 channels.
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            ids = (
                arr[..., 0].astype(np.int32) + arr[..., 1].astype(np.int32) * 256 + arr[..., 2].astype(np.int32) * 65536
            )
        else:
            ids = arr.squeeze(-1).astype(np.int32)
        return CameraFrameColorizer._ids_to_colors(ids)

    @staticmethod
    def _ids_to_colors(ids: np.ndarray) -> np.ndarray:
        h, w = ids.shape
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[ids == 0] = (40, 40, 40)  # background → dark grey
        for uid in np.unique(ids[ids != 0]):
            # Golden-ratio hue shift for maximum perceptual separation between classes.
            hue = (int(uid) * 0.6180339887) % 1.0
            out[ids == uid] = _hsv_to_rgb_uint8(hue, 0.75, 0.90)
        return out

    @staticmethod
    def _colorize_normals(data: torch.Tensor) -> np.ndarray:
        """Map surface normal vectors to RGB by remapping XYZ from [-1, 1] to [0, 255].

        The standard normals-to-color convention: R=X, G=Y, B=Z with each component
        shifted from [-1, 1] → [0, 1] → uint8.  Flat upward-facing surfaces appear
        cyan/teal (X≈0, Y≈0, Z≈1 → rgb≈128, 128, 255).
        """
        arr = data.cpu().numpy() if isinstance(data, torch.Tensor) else np.asarray(data, dtype=np.float32)
        # Handle (H, W, 4) XYZW or (H, W, 3) XYZ
        xyz = arr[..., :3].astype(np.float32)
        # Remap [-1, 1] → [0, 255]; clamp to handle out-of-range normals.
        rgb = np.clip((xyz + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
        return rgb

    @staticmethod
    def _fallback_depth_gradient(norm: np.ndarray) -> np.ndarray:
        """Blue (near) → green → red (far) gradient when matplotlib is unavailable."""
        hue = (1.0 - norm) * 0.667  # 0.667 = blue, 0.0 = red
        h6 = hue * 6.0
        r = np.clip(np.abs(h6 - 3.0) - 1.0, 0.0, 1.0)
        g = np.clip(2.0 - np.abs(h6 - 2.0), 0.0, 1.0)
        b = np.clip(2.0 - np.abs(h6 - 4.0), 0.0, 1.0)
        return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _hsv_to_rgb_uint8(h: float, s: float, v: float) -> tuple[int, int, int]:
    """Convert HSV to uint8 (R, G, B)."""
    if s == 0.0:
        c = int(v * 255)
        return (c, c, c)
    i = int(h * 6.0) % 6
    f = (h * 6.0) - int(h * 6.0)
    p, q, t = v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f))
    pairs = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    r, g, b = pairs[i]
    return (int(r * 255), int(g * 255), int(b * 255))
