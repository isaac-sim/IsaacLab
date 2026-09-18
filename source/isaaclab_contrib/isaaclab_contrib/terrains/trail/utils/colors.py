# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import numpy as np
import torch

# Imported helper
from .math import in_limits

# HSV colors
HSV_GREEN = {"h": (0.3, 0.35), "s": (0.8, 0.9), "v": (0.05, 0.15)}
HSV_BROWN = {"h": (0.05, 0.1), "s": (0.4, 0.7), "v": (0.1, 0.2)}
HSV_RED = {"h": 0.0, "s": 1.0, "v": 1.0}

# RGB colors
RGB_RED = {"r": 1.0, "g": 0.0, "b": 0.0}
RGB_BLACK = {"r": 0.0, "g": 0.0, "b": 0.0}
RGB_RED_CHANNEL = {"r": 1.0, "g": (0.0, 1.0), "b": (0.0, 1.0)}
RGB_BLUE = {"r": 0.0, "g": 0.0, "b": 1.0}
RGB_MAGANTA = {"r": 1.0, "g": 0.0, "b": 1.0}
RGB_YELLOW = {"r": 1.0, "g": 1.0, "b": 0.0}

# Color used for trail segmentation
TRAIL = HSV_BROWN
FLOOR = HSV_GREEN
START = HSV_BROWN
GOAL = RGB_MAGANTA


def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """Helper to convert an RGB tensor into an HSV tensor.

    Args:
        rgb: Tensor containing normalized RGB values in [0, 1], shape (..., 3).

    Returns:
        A tensor of normalized HSV values in [0, 1], shape (..., 3).
    """
    # rgb: (..., 3), values in [0,1]
    r, g, b = torch.unbind(rgb, dim=-1)

    maxc, _ = torch.max(rgb, dim=-1)
    minc, _ = torch.min(rgb, dim=-1)
    v = maxc
    deltac = maxc - minc
    eps = 1e-6

    # Saturation
    s = deltac / (v + eps)

    # Hue
    rc = (maxc - r) / (deltac + eps)
    gc = (maxc - g) / (deltac + eps)
    bc = (maxc - b) / (deltac + eps)

    h = torch.zeros_like(maxc)
    h[r == maxc] = bc[r == maxc] - gc[r == maxc]
    h[g == maxc] = 2.0 + rc[g == maxc] - bc[g == maxc]
    h[b == maxc] = 4.0 + gc[b == maxc] - rc[b == maxc]
    h = (h / 6.0) % 1.0

    return torch.stack([h, s, v], dim=-1)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Helper to convert an HSV numpy array into an RGB numpy array.

    Args:
        hsv: Array containing normalized HSV values in [0, 1], shape (..., 3).

    Returns:
        An array of normalized RGB values in [0, 1], shape (..., 3).
    """
    # hsv: (..., 3), values in [0,1]
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    h6 = (h % 1.0) * 6.0
    i = np.floor(h6).astype(np.int64)
    f = h6 - i.astype(h6.dtype)

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    i_mod = i % 6
    r = np.empty_like(v)
    g = np.empty_like(v)
    b = np.empty_like(v)

    mask = i_mod == 0
    r[mask], g[mask], b[mask] = v[mask], t[mask], p[mask]

    mask = i_mod == 1
    r[mask], g[mask], b[mask] = q[mask], v[mask], p[mask]

    mask = i_mod == 2
    r[mask], g[mask], b[mask] = p[mask], v[mask], t[mask]

    mask = i_mod == 3
    r[mask], g[mask], b[mask] = p[mask], q[mask], v[mask]

    mask = i_mod == 4
    r[mask], g[mask], b[mask] = t[mask], p[mask], v[mask]

    mask = i_mod == 5
    r[mask], g[mask], b[mask] = v[mask], p[mask], q[mask]

    return np.stack([r, g, b], axis=-1)


def is_color(rgb: torch.tensor, color_plate: dict[str, float | tuple[float, float]]) -> torch.tensor:
    """Check whether RGB colors belong to a pre-specified color plate.

    Args:
        rgb: Tensor containing unnormalized RGB values in [0, 1], shape (..., 3).
        color_plate: The color plate to test; can be an RGB or HSV specification.

    Returns:
        A boolean tensor indicating membership in the color plate. ``True``
        means the pixel/entry belongs to the color plate.

    Raises:
        RuntimeError: If the color plate does not contain RGB or HSV keys.
    """
    if "r" in color_plate:
        return (
            in_limits(rgb[..., 0], color_plate["r"])
            & in_limits(rgb[..., 1], color_plate["g"])
            & in_limits(rgb[..., 2], color_plate["b"])
        )
    elif "h" in color_plate:
        hsv = rgb_to_hsv(rgb)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        return (
            in_limits(h, color_plate["h"], rel_tol=0.1)
            & in_limits(s, color_plate["s"], rel_tol=0.1)
            & in_limits(v, color_plate["v"], rel_tol=0.1)
        )
    else:
        raise RuntimeError("Color plate must be either RGB or HSV.")


def check_color_plate(color_plate: dict[str, float | tuple[float, float]]) -> bool:
    """Return True if `color_plate` is a valid RGB or HSV color specification.

    Valid examples:

        RGB: {"r": 1.0, "g": 0.0, "b": 0.0}
        HSV: {"h": 0.0, "s": 1.0, "v": 1.0}
    """
    return all(k in color_plate for k in ("r", "g", "b")) or all(k in color_plate for k in ("h", "s", "v"))


def rgb_to_numpy(color_plate: dict[str, float | tuple[float, float]]) -> np.ndarray:
    """Convert a scaled RGB color plate into a NumPy uint8 RGB array.

    Args:
        color_plate: RGB color specification with components in [0, 1].

    Returns:
        NumPy array of unscaled RGB values (dtype uint8), e.g. [R, G, B].

    Raises:
        RuntimeError: If `color_plate` is not an RGB specification.
    """
    if "r" in color_plate:
        r = color_plate["r"] * 255
        g = color_plate["g"] * 255
        b = color_plate["b"] * 255
        return np.array([r, g, b], dtype=np.uint8)
    else:
        raise RuntimeError("Color plate must be RGB.")
