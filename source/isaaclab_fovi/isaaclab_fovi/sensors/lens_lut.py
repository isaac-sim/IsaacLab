# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Lens lookup tables for RTX's ``OmniLensDistortionLutAPI`` (``omni:lensdistortion:model = "lut"``).

A lookup-table lens is defined by two HDR textures attached to a camera prim:

* **rayEnterDirectionTexture** ``(H × W, RGB float32)``
  Maps each sensor texel NDC ``(u, v)`` → unit ray direction in camera space.

* **rayExitPositionTexture** ``(EW × EH, RG float32)``
  Inverse map: given a ray direction (octahedral-encoded over ``[0,1]²``),
  returns sensor NDC ``(u, v)`` in the native grid.

Isaac RTX and OVRTX both render through this API; the legacy generalized-projection camera attributes are not used.

Coordinate conventions:

* **Camera space**: X right, Y up, −Z forward (OpenGL / USD).
  Spherical grid padding may extend behind the optical hemisphere.
* **NDC**: ``u ∈ [0,1]`` = column (left → right),
  ``v ∈ [0,1]`` = row (top → bottom).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch
from fovi.sensing.coords import SamplingCoords

from .native_grid import native_grid_exit_uv

if TYPE_CHECKING:
    from pxr import Usd


def save_exr(path: str, data: torch.Tensor) -> None:
    """Save a ``(H, W, C)`` float32 tensor to an EXR file  (C ∈ {2, 3}) with OpenEXR, which the ``fovi`` extra
    installs."""
    import Imath
    import OpenEXR

    arr = np.ascontiguousarray(data.detach().cpu().numpy(), dtype=np.float32)
    h, w, c = arr.shape
    if c not in (2, 3):
        raise ValueError(f"save_exr: expected 2 or 3 channels, got {c}")
    names = ("R", "G") if c == 2 else ("R", "G", "B")
    ch_t = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    hdr = OpenEXR.Header(w, h)
    hdr["Compression"] = Imath.Compression.ZIP_COMPRESSION
    hdr["channels"] = {n: ch_t for n in names}
    exr = OpenEXR.OutputFile(path, hdr)
    exr.writePixels({n: arr[:, :, i].tobytes() for i, n in enumerate(names)})
    exr.close()


def build_native_grid_textures(
    coords: SamplingCoords, rays: torch.Tensor, *, image_rotation_deg: float, exit_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build native direction texels and an analytic octahedral inverse map."""
    decoded = torch.stack(_octahedral_decode(exit_size, exit_size, device=coords.device), -1)
    uv, valid = native_grid_exit_uv(decoded, coords, image_rotation_deg=image_rotation_deg)
    # Out-of-sensor rays must not be projected into the fovea by the inverse LUT.
    exit_texture = torch.where(valid[..., None], uv, torch.full_like(uv, -1))
    return rays.reshape(coords.resolution, coords.resolution, 3), exit_texture


def save_lut_textures(
    direction_texture: torch.Tensor,
    exit_texture: torch.Tensor,
    output_dir: str,
    basename: str,
) -> tuple[str, str]:
    """Save direction and exit LUT tensors as EXR files.

    Returns:
        ``(direction_path, exit_path)`` — absolute paths to the written files.
    """
    os.makedirs(output_dir, exist_ok=True)
    dir_path = os.path.join(output_dir, f"{basename}_ray_enter_direction.exr")
    exit_path = os.path.join(output_dir, f"{basename}_ray_exit_position.exr")
    save_exr(dir_path, direction_texture)
    save_exr(exit_path, exit_texture)
    return os.path.abspath(dir_path), os.path.abspath(exit_path)


def apply_lens_lut_to_prim(
    prim: Usd.Prim,
    direction_texture_path: str,
    exit_texture_path: str,
) -> None:
    """Apply native OmniLensDistortion LUT textures to an existing USD Camera."""
    from pxr import Sdf

    # RTX reads explicitly authored metadata and attributes without USD schema registration.
    prim.AddAppliedSchema("OmniLensDistortionLutAPI")
    prim.CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.Token).Set("lut")
    prim.CreateAttribute("omni:lensdistortion:lut:rayEnterDirectionTexture", Sdf.ValueTypeNames.Asset).Set(
        direction_texture_path
    )
    prim.CreateAttribute("omni:lensdistortion:lut:rayExitPositionTexture", Sdf.ValueTypeNames.Asset).Set(
        exit_texture_path
    )


def _octahedral_decode(
    width: int,
    height: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode octahedral-LUT texel centres to unit directions (camera −Z convention).

    Returns:
        Three ``(height, width)`` float64 tensors ``(dx, dy, dz)``.
    """
    u = (torch.arange(width, dtype=torch.float64, device=device) + 0.5) / width
    v = (torch.arange(height, dtype=torch.float64, device=device) + 0.5) / height
    px = u.unsqueeze(0).expand(height, width) * 2.0 - 1.0
    py = v.unsqueeze(1).expand(height, width) * 2.0 - 1.0
    dz = 1.0 - px.abs() - py.abs()
    sx = torch.sign(px)
    sy = torch.sign(py)
    dx = torch.where(dz <= 0.0, (1.0 - py.abs()) * sx, px)
    dy = torch.where(dz <= 0.0, (1.0 - px.abs()) * sy, py)
    n = torch.clamp((dx**2 + dy**2 + dz**2).sqrt(), min=1e-20)
    return dx / n, dy / n, -dz / n
