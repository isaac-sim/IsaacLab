# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera frame colorization utilities for visualizer streaming views."""

from __future__ import annotations

import numpy as np
import torch
import warp as wp
from matplotlib import colormaps

from ...utils.images import compose_image

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
    "rgb": "rgba",
    "depth": "distance_to_image_plane",
}


def sensor_key_for_gt_type(
    gt_type: str, available_keys: frozenset[str] | None = None, *, required: bool = True
) -> str | None:
    """Return the camera sensor output key for a streaming GT type.

    Args:
        gt_type: One of :data:`SUPPORTED_GT_TYPES`.
        available_keys: Keys present in ``camera.data.output``. When provided the
            fallback key for ``"rgb"`` or ``"depth"`` is tried if the primary is absent.
        required: Whether a missing output raises an error. False returns None for compatibility checks.

    Returns:
        The sensor output key to index into ``camera.data.output``, or None if absent and not required.

    Raises:
        ValueError: If ``gt_type`` is not in :data:`SUPPORTED_GT_TYPES`.
        KeyError: If no matching key exists in ``available_keys`` and required is True.
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
    if not required:
        return None
    raise KeyError(
        f"No sensor output found for GT type {gt_type!r}. "
        f"Tried {primary!r} and {fallback!r}; available: {sorted(available_keys)}"
    )


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
        sensor_key_for_gt_type(gt_type)
        wp.init()
        source = wp.from_torch(data) if isinstance(data, torch.Tensor) else wp.array(np.asarray(data), device="cpu")
        source = source.contiguous().reshape((1, *source.shape))
        device = source.device
        output = wp.empty((*source.shape[1:3], 4), dtype=wp.uint8, device=device)
        colors = np.empty((0, 3), dtype=np.uint8)
        if gt_type == "depth":
            colors = (colormaps["turbo"](np.arange(256) / 255.0)[..., :3] * 255).astype(np.uint8)
        compose_image(
            output, (source,), wp.array([0], dtype=wp.int32, device=device), (gt_type,),
            wp.array(colors, dtype=wp.uint8, device=device), depth_min=depth_min, depth_max=depth_max,
        )  # fmt: skip
        return np.ascontiguousarray(output.numpy()[..., :3])
