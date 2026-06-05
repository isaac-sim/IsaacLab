# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera-image processing helpers shared between DirectRLEnv and ManagerBasedEnv paths.

Provides :func:`normalize_camera_image` for data-type-aware normalization plus the ``is_*``
predicates used by callers that need to gate other behavior on the data-type family.
"""

from __future__ import annotations

import torch

from isaaclab.utils.warp.ops import normalize_image_uint8

_RGB_LIKE_PREFIXES: tuple[str, ...] = ("rgb", "albedo", "simple_shading")
_DEPTH_LIKE_PATTERNS: tuple[str, ...] = ("depth", "distance_to")
_NORMALS_PREFIXES: tuple[str, ...] = ("normals",)


def is_rgb_like(data_type: str) -> bool:
    """Whether ``data_type`` is one of the RGB-like camera outputs (rgb, albedo, simple_shading_*).

    Args:
        data_type: The camera data-type string from ``sensor.data.output`` / ``CameraCfg``.

    Returns:
        True if the data type should receive the ``(x / 255) - mean`` normalize.
    """
    return data_type.startswith(_RGB_LIKE_PREFIXES)


def is_depth_like(data_type: str) -> bool:
    """Whether ``data_type`` is one of the depth-like camera outputs (depth, distance_to_*)."""
    return any(p in data_type for p in _DEPTH_LIKE_PATTERNS)


def is_normals_like(data_type: str) -> bool:
    """Whether ``data_type`` is one of the surface-normals camera outputs."""
    return data_type.startswith(_NORMALS_PREFIXES)


def normalize_camera_image(
    images: torch.Tensor,
    data_type: str,
    out: torch.Tensor | None = None,
    channel_dim: int = -1,
) -> torch.Tensor:
    """Normalize a camera-observation tensor according to its ``data_type``.

    Dispatch (in order of check):

    - :func:`is_rgb_like` and ``images.dtype == torch.uint8`` and contiguous 4D: routes to the
      fused Warp kernel via :func:`~isaaclab.utils.warp.ops.normalize_image_uint8`. ``out`` and
      ``channel_dim`` are forwarded so callers can reuse a pre-allocated float32 buffer and
      select the image layout.
    - :func:`is_rgb_like` and any other dtype/shape: pure-PyTorch ``(x.float() / 255.0) - mean``
      with the same math. ``out`` is ignored on this branch; ``channel_dim`` selects the spatial
      reduction axes.
    - :func:`is_depth_like`: in-place ``images[images == inf] = 0``. ``images`` is returned as-is.
    - :func:`is_normals_like`: returns ``(images + 1.0) * 0.5``.
    - Otherwise: ``images`` is returned unchanged.

    Args:
        images: The camera-observation tensor. Shape and dtype vary by ``data_type``; the
            RGB-like Warp fast path requires 4D contiguous uint8 with the channel axis at
            position ``channel_dim``.
        data_type: The camera data-type string. Drives the dispatch.
        out: Optional pre-allocated float32 output for the RGB-like Warp fast path. Reused
            across steps to eliminate per-step allocation. Ignored on the PyTorch fallback
            and on non-RGB branches. Defaults to None.
        channel_dim: Position of the channel axis for the RGB-like branches. ``-1`` (BHWC,
            default) or ``-3`` / ``1`` (BCHW). Ignored on non-RGB branches.

    Returns:
        The normalized tensor. For RGB-like input this is a fresh (or pre-allocated) float32
        tensor; for depth-like input it is ``images`` itself (mutated in place); for
        normals-like input it is a new tensor; for anything else, ``images`` unchanged.
    """
    if is_rgb_like(data_type):
        if images.dtype == torch.uint8 and images.ndim == 4 and images.is_contiguous():
            return normalize_image_uint8(images, channel_dim=channel_dim, out=out)
        # PyTorch fallback for callers that pre-floated or pass a strided view.
        resolved_channel_dim = channel_dim + images.ndim if channel_dim < 0 else channel_dim
        spatial_dims = tuple(d for d in range(1, images.ndim) if d != resolved_channel_dim)
        images = images.float() / 255.0
        images -= torch.mean(images, dim=spatial_dims, keepdim=True)
        return images
    if is_depth_like(data_type):
        images[images == float("inf")] = 0
        return images
    if is_normals_like(data_type):
        return (images + 1.0) * 0.5
    return images
