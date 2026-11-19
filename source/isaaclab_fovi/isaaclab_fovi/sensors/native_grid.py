# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native warped-grid rendering geometry and public image layout."""

from __future__ import annotations

import math

import torch
from fovi.sensing.coords import SamplingCoords
from fovi.sensing.projection import angular_directions


def bounded_angular_radius(radius: torch.Tensor, half_fov_rad: float) -> torch.Tensor:
    """Continue angular padding monotonically toward the antipode.

    The nominal visual disc is unchanged. Outside it, the continuation has
    matching value and derivative at the rim and never wraps around the sphere.
    """
    if not 0 < half_fov_rad < math.pi:
        raise ValueError("Spherical grid half FoV must lie strictly between zero and pi.")
    remaining = math.pi - half_fov_rad
    excess = (radius - 1).clamp_min(0) * half_fov_rad
    padding = half_fov_rad + remaining * excess / (remaining + excess)
    return torch.where(radius <= 1, radius * half_fov_rad, padding)


def unbounded_angular_radius(angle: torch.Tensor, half_fov_rad: float) -> torch.Tensor:
    """Invert :func:`bounded_angular_radius` for angles below the antipode."""
    remaining = math.pi - half_fov_rad
    fraction = ((angle - half_fov_rad) / remaining).clamp(0, 1 - torch.finfo(angle.dtype).eps)
    padding = 1 + remaining * fraction / ((1 - fraction) * half_fov_rad)
    return torch.where(angle <= half_fov_rad, angle / half_fov_rad, padding)


def native_grid_directions(coords: SamplingCoords, *, image_rotation_deg: float = 0.0) -> torch.Tensor:
    """Return canonical ordered OpenGL rays for a warped Cartesian grid.

    Grid radius is proportional to visual angle, and the padding outside the
    nominal field continues toward the antipode without wrapping.
    """

    cartesian = coords.cartesian.to(torch.float64)
    radius = torch.linalg.vector_norm(cartesian, dim=-1)
    half_fov = math.radians(coords.fov) / 2
    padded_radius = bounded_angular_radius(radius, half_fov) / half_fov
    scale = torch.where(radius > 0, padded_radius / radius.clamp_min(1e-30), 1.0)
    rays = angular_directions(cartesian * scale[:, None], coords.fov)
    x, y, z = rays.unbind(-1)
    angle = math.radians(image_rotation_deg)
    c, s = math.cos(angle), math.sin(angle)
    return torch.stack((c * x + s * y, s * x - c * y, -z), -1).float()


def native_grid_exit_uv(
    directions: torch.Tensor, coords: SamplingCoords, *, image_rotation_deg: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytically map OpenGL rays to native texture UV and visibility.

    Texture rows and columns follow canonical vector order: native x then y.
    The public image applies Fovi's upright grid permutation after rendering.
    """
    angle = math.radians(image_rotation_deg)
    x, y, z = directions.to(torch.float64).unbind(-1)
    xy = torch.stack((math.cos(angle) * x + math.sin(angle) * y, -math.sin(angle) * x + math.cos(angle) * y), -1)
    lateral = torch.linalg.vector_norm(xy, dim=-1)
    theta = torch.atan2(lateral, -z)
    radius = unbounded_angular_radius(theta, math.radians(coords.fov) / 2)
    cartesian = xy * (radius / lateral.clamp_min(1e-30))[..., None]
    visible = theta < math.pi - 1e-8
    native = coords.visual_to_native(cartesian)
    uv = (native.flip(-1) + 1) / 2
    visible = visible & (uv >= 0).all(-1) & (uv <= 1).all(-1)
    return uv.float(), visible
