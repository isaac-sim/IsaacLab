# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels and helpers for camera-related math operations.

These replace equivalent torch functions on the per-frame hot path, operating
directly on warp arrays without torch round-trips.
"""

from __future__ import annotations

from typing import Literal

import warp as wp

# Camera orientation convention conversion
#
# Every pair of (origin, target) conventions is equivalent to a single
# right-multiplication by a constant unit quaternion:
#
#   q_out[i] = q_in[i] * q_const
#
# Derivations (xyzw):
#   opengl ↔ ros   : 180° around X  →  (1, 0, 0, 0)   (self-inverse)
#   world → opengl : Rx(+90°)·Ry(−90°)  →  (0.5, −0.5, −0.5, 0.5)
#   opengl → world : inverse of above   →  (−0.5, 0.5, 0.5, 0.5)
#   ros → world    : compose ros→gl→world → (0.5, −0.5, 0.5, 0.5)
#   world → ros    : inverse of above   →  (−0.5, 0.5, −0.5, 0.5)

_CAMERA_ORIENTATION_CONST: dict[tuple[str, str], wp.quatf] = {
    ("opengl", "ros"): wp.quatf(1.0, 0.0, 0.0, 0.0),
    ("ros", "opengl"): wp.quatf(1.0, 0.0, 0.0, 0.0),
    ("world", "opengl"): wp.quatf(0.5, -0.5, -0.5, 0.5),
    ("opengl", "world"): wp.quatf(-0.5, 0.5, 0.5, 0.5),
    ("ros", "world"): wp.quatf(0.5, -0.5, 0.5, 0.5),
    ("world", "ros"): wp.quatf(-0.5, 0.5, -0.5, 0.5),
}


# TODO: Optimize these kernels with tiled ops and use wp.static
@wp.kernel
def _convert_camera_orientation_all_kernel(
    src: wp.array(dtype=wp.quatf),
    dst: wp.array(dtype=wp.quatf),
    q_const: wp.quatf,
):
    """Apply constant-quaternion convention conversion to every element."""
    i = wp.tid()
    dst[i] = src[i] * q_const


@wp.kernel
def _convert_camera_orientation_indexed_kernel(
    src: wp.array(dtype=wp.quatf),
    dst: wp.array(dtype=wp.quatf),
    indices: wp.array(dtype=wp.int32),
    q_const: wp.quatf,
):
    """Apply constant-quaternion convention conversion to indexed elements.

    Reads ``src[i]`` and writes to ``dst[indices[i]]``.  Use this for partial
    camera updates (e.g. environment resets targeting a subset of cameras).
    """
    i = wp.tid()
    dst[indices[i]] = src[i] * q_const


def convert_camera_frame_orientation_convention_wp(
    src: wp.array,
    dst: wp.array,
    origin: Literal["opengl", "ros", "world"],
    target: Literal["opengl", "ros", "world"],
    indices: wp.array | None = None,
    device: str | None = None,
) -> None:
    """Convert camera-frame quaternion orientations between conventions using a warp kernel.

    Replaces :func:`~isaaclab.utils.math.convert_camera_frame_orientation_convention` on
    the per-frame hot path. All six convention pairs collapse to a single quaternion
    right-multiplication by a pre-computed constant — no matrix round-trip, no torch.

    The operation is **in-place on** ``dst``:
    - Without ``indices``: ``dst[i] = src[i] * q_const`` for all i.
    - With ``indices``: ``dst[indices[i]] = src[i] * q_const`` for each i.

    Args:
        src: Source quaternions ``(x, y, z, w)``. Shape ``(N,)``, dtype ``wp.quatf``.
        dst: Destination quaternion array to write into. Shape ``(M,)``, dtype ``wp.quatf``.
            ``M >= N`` when ``indices`` is provided; ``M == N`` otherwise.
        origin: Source convention (``"opengl"``, ``"ros"``, or ``"world"``).
        target: Target convention (``"opengl"``, ``"ros"``, or ``"world"``).
        indices: Optional warp int32 array of shape ``(N,)`` selecting which slots of
            ``dst`` to write. If ``None`` all N elements are written sequentially.
        device: Warp device string. Defaults to ``src.device``.
    """
    if origin == target:
        if indices is None:
            wp.copy(dst, src)
        else:
            # scatter copy: dst[indices[i]] = src[i]
            wp.launch(
                _convert_camera_orientation_indexed_kernel,
                dim=indices.shape[0],
                inputs=[src, dst, indices, wp.quatf(0.0, 0.0, 0.0, 1.0)],
                device=device or src.device,
            )
        return

    q_const = _CAMERA_ORIENTATION_CONST[(origin, target)]
    dev = device or src.device

    if indices is None:
        wp.launch(
            _convert_camera_orientation_all_kernel,
            dim=src.shape[0],
            inputs=[src, dst, q_const],
            device=dev,
        )
    else:
        wp.launch(
            _convert_camera_orientation_indexed_kernel,
            dim=indices.shape[0],
            inputs=[src, dst, indices, q_const],
            device=dev,
        )


@wp.kernel
def _clamp_depth_to_inf_kernel(
    buf: wp.array(dtype=wp.float32, ndim=4),
    max_range: float,
):
    """Replace values above ``max_range`` with ``+inf``."""
    n, h, w, c = wp.tid()
    v = buf[n, h, w, c]
    if v > max_range:
        buf[n, h, w, c] = wp.inf


@wp.kernel
def _replace_inf_kernel(
    buf: wp.array(dtype=wp.float32, ndim=4),
    replacement: float,
):
    """Replace ``+inf`` values with ``replacement``."""
    n, h, w, c = wp.tid()
    if wp.isinf(buf[n, h, w, c]):
        buf[n, h, w, c] = replacement


def clamp_depth_to_inf_wp(buf: wp.array, max_range: float, device: str | None = None) -> None:
    """Replace depth values above ``max_range`` with ``+inf`` using a warp kernel.

    Replaces ``t[t > max_range] = torch.inf`` on the hot path.

    Args:
        buf: Depth buffer. Shape ``(N, H, W, C)``, dtype ``wp.float32``.
        max_range: Depth values strictly greater than this are set to ``+inf``.
        device: Warp device string. Defaults to ``buf.device``.
    """
    wp.launch(
        _clamp_depth_to_inf_kernel,
        dim=buf.shape,
        inputs=[buf, float(max_range)],
        device=device or buf.device,
    )


def replace_inf_depth_wp(buf: wp.array, replacement: float, device: str | None = None) -> None:
    """Replace ``+inf`` depth values with ``replacement`` using a warp kernel.

    Replaces ``t[torch.isinf(t)] = value`` on the hot path.

    Args:
        buf: Depth buffer. Shape ``(N, H, W, C)``, dtype ``wp.float32``.
        replacement: Value to write where ``+inf`` was found (e.g. ``0.0`` or ``max_range``).
        device: Warp device string. Defaults to ``buf.device``.
    """
    wp.launch(
        _replace_inf_kernel,
        dim=buf.shape,
        inputs=[buf, float(replacement)],
        device=device or buf.device,
    )
