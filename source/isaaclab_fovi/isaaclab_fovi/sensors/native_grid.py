# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native warped-grid rendering geometry and public image layout."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Protocol

import torch
from fovi.sensing.coords import RADIUS_NORMS, WARPED_CARTESIAN_STYLES, SamplingCoords

GRID_STYLE = "warped_cartesian_as_grid"


def validate_radius_norm(radius_norm: float, style: str, fov_type: str) -> None:
    """Reject unsupported warp configurations before renderer startup."""
    if radius_norm not in RADIUS_NORMS:
        raise ValueError(f"radius_norm must be one of {RADIUS_NORMS}, got {radius_norm!r}.")
    if radius_norm != 2.0 and style not in WARPED_CARTESIAN_STYLES:
        raise ValueError("Non-default radius_norm requires a warped_cartesian style.")
    if radius_norm == math.inf and fov_type == "wang":
        raise ValueError("radius_norm=inf does not support fov_type='wang'.")


class TorchBuffer(Protocol):
    """Renderer buffer exposing a Torch view."""

    @property
    def torch(self) -> torch.Tensor: ...


def buffer_tensor(value: torch.Tensor | TorchBuffer) -> torch.Tensor:
    """Get a renderer or public tensor without copying."""
    return value if isinstance(value, torch.Tensor) else value.torch


def _angular_padding_boundary(coords: SamplingCoords, camera_fov_rad: float | None = None) -> tuple[float, float]:
    if coords.fov_type == "circular":
        radius = 1.0
    elif coords.fov_type == "square":
        radius = math.sqrt(2)
    elif coords.fov_type == "wang":
        corner = coords.cartesian.new_tensor([[1.0, 1.0]])
        radius = float(torch.linalg.vector_norm(coords.native_to_visual(corner)).item())
    else:
        raise ValueError(f"Unsupported native grid footprint {coords.fov_type!r}.")
    fov_rad = math.radians(coords.fov) if camera_fov_rad is None else camera_fov_rad
    angle = radius * fov_rad / 2
    if angle >= math.pi:
        raise ValueError("Spherical native grid footprint reaches the antipode; reduce FoV or use circular coverage.")
    return radius, angle


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


def native_grid_directions(
    coords: SamplingCoords,
    *,
    tangent_scale: float,
    image_rotation_deg: float = 0.0,
    projection: str = "pinhole",
    camera_fov_rad: float | None = None,
) -> torch.Tensor:
    """Return canonical ordered OpenGL rays for a warped Cartesian grid."""
    cartesian = coords.cartesian.to(torch.float64)
    radius = torch.linalg.vector_norm(cartesian, dim=-1)
    if projection not in ("pinhole", "fisheye_equidistant"):
        raise ValueError(f"Unsupported native grid projection {projection!r}.")
    if coords.field_geometry == "spherical" or projection == "fisheye_equidistant":
        if coords.field_geometry != "spherical" and camera_fov_rad is None:
            raise ValueError("Planar equidistant rendering requires camera_fov_rad.")
        boundary_radius, boundary_angle = _angular_padding_boundary(
            coords, None if coords.field_geometry == "spherical" else camera_fov_rad
        )
        angle = bounded_angular_radius(radius / boundary_radius, boundary_angle)
        scale = torch.where(radius > 0, angle.sin() / radius.clamp_min(1e-30), angle.new_zeros(()))
        xy = cartesian * scale[:, None]
        z = -angle.cos()
    elif coords.field_geometry in ("planar", "legacy"):
        xy = cartesian * tangent_scale
        z = -torch.ones_like(radius)
    else:
        raise ValueError(f"Unknown field geometry {coords.field_geometry!r}.")
    angle = math.radians(image_rotation_deg)
    x, y = xy.unbind(-1)
    rays = torch.stack(
        (math.cos(angle) * x - math.sin(angle) * y, math.sin(angle) * x + math.cos(angle) * y, z),
        -1,
    )
    return torch.nn.functional.normalize(rays, dim=-1).float()


def native_grid_exit_uv(
    directions: torch.Tensor,
    coords: SamplingCoords,
    *,
    tangent_scale: float,
    image_rotation_deg: float = 0.0,
    projection: str = "pinhole",
    camera_fov_rad: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytically map OpenGL rays to native texture UV and visibility.

    Texture rows and columns follow canonical vector order: native x then y.
    The public image applies Fovi's upright grid permutation after rendering.
    """
    angle = math.radians(image_rotation_deg)
    x, y, z = directions.to(torch.float64).unbind(-1)
    xy = torch.stack((math.cos(angle) * x + math.sin(angle) * y, -math.sin(angle) * x + math.cos(angle) * y), -1)
    lateral = torch.linalg.vector_norm(xy, dim=-1)
    if projection not in ("pinhole", "fisheye_equidistant"):
        raise ValueError(f"Unsupported native grid projection {projection!r}.")
    if coords.field_geometry == "spherical" or projection == "fisheye_equidistant":
        if coords.field_geometry != "spherical" and camera_fov_rad is None:
            raise ValueError("Planar equidistant rendering requires camera_fov_rad.")
        theta = torch.atan2(lateral, -z)
        boundary_radius, boundary_angle = _angular_padding_boundary(
            coords, None if coords.field_geometry == "spherical" else camera_fov_rad
        )
        radius = unbounded_angular_radius(theta, boundary_angle) * boundary_radius
        cartesian = xy * (radius / lateral.clamp_min(1e-30))[..., None]
        visible = theta < math.pi - 1e-8
    elif coords.field_geometry in ("planar", "legacy"):
        cartesian = xy / ((-z).clamp_min(1e-30) * tangent_scale)[..., None]
        visible = z < 0
    else:
        raise ValueError(f"Unknown field geometry {coords.field_geometry!r}.")
    native = coords.visual_to_native(cartesian)
    uv = (native.flip(-1) + 1) / 2
    visible = visible & (uv >= 0).all(-1) & (uv <= 1).all(-1)
    return uv.float(), visible


def grid_output_shape(coords: SamplingCoords) -> tuple[int, int]:
    """Resolve public image dimensions, preserving vector cameras' shape."""
    if coords.style == GRID_STYLE:
        return coords.resolution, coords.resolution
    return len(coords), 1


def canonical_to_grid(value: torch.Tensor, coords: SamplingCoords) -> torch.Tensor:
    """Convert canonical BNC or BN1C values into upright BHWC values."""
    if value.ndim == 4:
        value = value.squeeze(2)
    if value.ndim != 3 or value.shape[1] != len(coords):
        raise ValueError(f"Expected canonical (B, {len(coords)}, C), got {tuple(value.shape)}.")
    return coords.as_grid(value, sample_dim=1)


def mask_grid_samples(outputs: Mapping[str, torch.Tensor | TorchBuffer], coords: SamplingCoords) -> None:
    """Zero samples outside the configured footprint before pooling or publication."""
    if coords.style != GRID_STYLE:
        return
    invalid = ~coords.valid_mask[None, :, None, None]
    for name, value in outputs.items():
        if name == "rgb" and "rgba" in outputs:
            continue
        tensor = buffer_tensor(value)
        tensor.masked_fill_(invalid, 0)


def allocate_grid_outputs(
    outputs: Mapping[str, torch.Tensor | TorchBuffer], coords: SamplingCoords
) -> dict[str, torch.Tensor]:
    """Allocate stable public BHWC buffers from flat renderer outputs."""
    result: dict[str, torch.Tensor] = {}
    for name, value in outputs.items():
        if name == "rgb" and "rgba" in outputs:
            continue
        tensor = buffer_tensor(value)
        result[name] = tensor.new_zeros((tensor.shape[0], coords.resolution, coords.resolution, tensor.shape[-1]))
    if "rgb" in outputs and "rgba" in result:
        result["rgb"] = result["rgba"][..., :3]
    return result


def grid_sample_indices(coords: SamplingCoords) -> torch.Tensor:
    """Resolve the public image permutation once, outside rendering updates."""
    return coords.as_grid(torch.arange(len(coords), device=coords.device)).reshape(-1)


def copy_grid_outputs(
    source: Mapping[str, torch.Tensor | TorchBuffer],
    target: dict[str, torch.Tensor],
    indices: torch.Tensor,
) -> None:
    """Update stable public image buffers without changing sample ordering."""
    for name, value in source.items():
        if name == "rgb" and "rgba" in source:
            continue
        tensor = buffer_tensor(value)
        flat_source = tensor.reshape(tensor.shape[0], -1, tensor.shape[-1])
        flat_target = target[name].view(tensor.shape[0], -1, tensor.shape[-1])
        torch.index_select(flat_source, 1, indices, out=flat_target)
