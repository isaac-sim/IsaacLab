# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native warped-grid rendering geometry and public image layout."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import warp as wp
from fovi.sensing.coords import SamplingCoords

from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp.memory import make_masked_copy

from .gaze_kernels import warp_stream

GRID_STYLE = "warped_cartesian_as_grid"


def buffer_tensor(value: torch.Tensor | ProxyArray) -> torch.Tensor:
    """Get a renderer or public tensor without copying."""
    return value if isinstance(value, torch.Tensor) else value.torch


def proxy_outputs(outputs: Mapping[str, torch.Tensor | ProxyArray]) -> dict[str, ProxyArray]:
    """Wrap output buffers as zero-copy :class:`ProxyArray` values, as :class:`CameraData` publishes them."""
    return {
        name: ProxyArray(wp.from_torch(value)) if isinstance(value, torch.Tensor) else value
        for name, value in outputs.items()
    }


def copy_updated_rows(source: torch.Tensor, target: torch.Tensor, rows: wp.array) -> None:
    """Copy the environments that ``rows`` selects from ``source`` into ``target``, shape (num_envs, N, 1, C).

    The other environments keep their values. The masked copy needs no full-batch temporary and no host sync.

    Args:
        source: The new samples; may be a strided view.
        target: The published samples, written in place.
        rows: Which environments to copy, shape (num_envs,) of ``bool``.
    """
    source3, target3 = source.squeeze(2), target.squeeze(2)
    dtype = wp.dtype_from_torch(target.dtype)
    wp.launch(
        make_masked_copy(dtype, 3),
        dim=tuple(target3.shape),
        inputs=[wp.from_torch(source3, dtype=dtype), rows, None],
        outputs=[wp.from_torch(target3, dtype=dtype)],
        device=str(target.device),
        stream=warp_stream(target.device),
    )


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
    _require_spherical(coords)
    cartesian = coords.cartesian.to(torch.float64)
    radius = torch.linalg.vector_norm(cartesian, dim=-1)
    boundary_radius, boundary_angle = _angular_padding_boundary(coords)
    angle = bounded_angular_radius(radius / boundary_radius, boundary_angle)
    scale = torch.where(radius > 0, angle.sin() / radius.clamp_min(1e-30), angle.new_zeros(()))
    xy = cartesian * scale[:, None]
    z = -angle.cos()
    angle = math.radians(image_rotation_deg)
    x, y = xy.unbind(-1)
    rays = torch.stack(
        (math.cos(angle) * x - math.sin(angle) * y, math.sin(angle) * x + math.cos(angle) * y, z),
        -1,
    )
    return torch.nn.functional.normalize(rays, dim=-1).float()


def native_grid_exit_uv(
    directions: torch.Tensor, coords: SamplingCoords, *, image_rotation_deg: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytically map OpenGL rays to native texture UV and visibility.

    Texture rows and columns follow canonical vector order: native x then y.
    The public image applies Fovi's upright grid permutation after rendering.
    """
    _require_spherical(coords)
    angle = math.radians(image_rotation_deg)
    x, y, z = directions.to(torch.float64).unbind(-1)
    xy = torch.stack((math.cos(angle) * x + math.sin(angle) * y, -math.sin(angle) * x + math.cos(angle) * y), -1)
    lateral = torch.linalg.vector_norm(xy, dim=-1)
    theta = torch.atan2(lateral, -z)
    boundary_radius, boundary_angle = _angular_padding_boundary(coords)
    radius = unbounded_angular_radius(theta, boundary_angle) * boundary_radius
    cartesian = xy * (radius / lateral.clamp_min(1e-30))[..., None]
    visible = theta < math.pi - 1e-8
    native = coords.visual_to_native(cartesian)
    uv = (native.flip(-1) + 1) / 2
    visible = visible & (uv >= 0).all(-1) & (uv <= 1).all(-1)
    return uv.float(), visible


def grid_output_shape(coords: SamplingCoords) -> tuple[int, int]:
    """Resolve public image dimensions, preserving vector cameras' shape."""
    if coords.style == GRID_STYLE:
        return coords.resolution, coords.resolution
    return len(coords), 1


def mask_grid_samples(outputs: Mapping[str, torch.Tensor | ProxyArray], coords: SamplingCoords) -> None:
    """Zero samples outside the configured footprint before pooling or publication."""
    if coords.style != GRID_STYLE:
        return
    invalid = ~coords.valid_mask[None, :, None, None]
    for name, value in outputs.items():
        if name == "rgb" and "rgba" in outputs:
            continue
        tensor = buffer_tensor(value)
        tensor.masked_fill_(invalid, 0)


def allocate_outputs(
    outputs: Mapping[str, torch.Tensor | ProxyArray], height: int, width: int
) -> dict[str, torch.Tensor]:
    """Allocate BHWC buffers with the source channels, dtype, device and RGB view of RGBA."""
    result: dict[str, torch.Tensor] = {}
    for name, value in outputs.items():
        if name == "rgb" and "rgba" in outputs:
            continue
        tensor = buffer_tensor(value)
        result[name] = tensor.new_zeros((tensor.shape[0], height, width, tensor.shape[-1]))
    if "rgb" in outputs and "rgba" in result:
        result["rgb"] = result["rgba"][..., :3]
    return result


def grid_sample_indices(coords: SamplingCoords) -> torch.Tensor:
    """Resolve the public image permutation once, outside rendering updates."""
    return coords.as_grid(torch.arange(len(coords), device=coords.device)).reshape(-1)


def copy_grid_outputs(
    source: Mapping[str, torch.Tensor | ProxyArray],
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


def _require_spherical(coords: SamplingCoords) -> None:
    if coords.field_geometry != "spherical":
        raise ValueError(f"Foveated cameras sample spherical coordinates, got {coords.field_geometry!r}.")


def _angular_padding_boundary(coords: SamplingCoords) -> tuple[float, float]:
    if coords.fov_type == "circular":
        radius = 1.0
    elif coords.fov_type == "square":
        radius = math.sqrt(2)
    elif coords.fov_type == "wang":
        corner = coords.cartesian.new_tensor([[1.0, 1.0]])
        radius = float(torch.linalg.vector_norm(coords.native_to_visual(corner)).item())
    else:
        raise ValueError(f"Unsupported native grid footprint {coords.fov_type!r}.")
    angle = radius * math.radians(coords.fov) / 2
    if angle >= math.pi:
        raise ValueError("Spherical native grid footprint reaches the antipode; reduce FoV or use circular coverage.")
    return radius, angle
