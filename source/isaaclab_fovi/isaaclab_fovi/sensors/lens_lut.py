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
  returns the sensor NDC ``(u, v)`` of the nearest sample.

Isaac RTX and OVRTX both render through this API; the legacy generalized-projection camera attributes are not used.

Coordinate conventions:

* **Camera space**: X right, Y up, −Z forward (OpenGL / USD).
  Spherical grid padding may extend behind the optical hemisphere.
* **NDC**: ``u ∈ [0,1]`` = column (left → right),
  ``v ∈ [0,1]`` = row (top → bottom).
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import numpy as np
import torch
from fovi.sensing.coords import SamplingCoords

from .native_grid import native_grid_directions, native_grid_exit_uv

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
    coords: SamplingCoords, *, image_rotation_deg: float, exit_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build native direction texels and an analytic octahedral inverse map."""
    directions = native_grid_directions(coords, image_rotation_deg=image_rotation_deg).reshape(
        coords.resolution, coords.resolution, 3
    )
    decoded = torch.stack(_octahedral_decode(exit_size, exit_size, device=coords.device), -1)
    uv, valid = native_grid_exit_uv(decoded, coords, image_rotation_deg=image_rotation_deg)
    # Out-of-sensor rays must not be projected into the fovea by the inverse LUT.
    exit_texture = torch.where(valid[..., None], uv, torch.full_like(uv, -1))
    return directions, exit_texture


def build_lut_textures(
    samples_uv: torch.Tensor,
    ray_directions: torch.Tensor,
    texture_width: int,
    texture_height: int,
    exit_width: int,
    exit_height: int,
    chunk: int = 8192,
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build direction and exit LUT tensors from a labelled set of sensor samples.

    Both tensors are filled by **nearest-neighbour assignment**.

    Args:
        samples_uv:     ``(N, 2)`` float32.  NDC ``(u, v)`` per sample.
        ray_directions: ``(N, 3)`` float32.  Unit ray direction per sample.
        texture_width, texture_height: Direction LUT dimensions.
        exit_width, exit_height: Exit LUT dimensions.
        chunk: NN search batch size.
        device: Torch device.

    Returns:
        ``(direction_texture, exit_texture)`` —
        ``(H, W, 3)`` and ``(EH, EW, 2)`` float32 tensors.
    """
    uv_t = _t(samples_uv, torch.float32, device)
    dirs_t = _t(ray_directions, torch.float32, device)
    N = uv_t.shape[0]

    # Direction texture.
    dir_out = torch.zeros(texture_height, texture_width, 3, dtype=torch.float32, device=device)
    dir_out[:, :, 2] = -1.0

    if N > 0:
        # Each texel takes the ray of the sample nearest it in UV, whatever order the samples come in.
        u_t = (torch.arange(texture_width, dtype=torch.float32, device=device) + 0.5) / texture_width
        v_t = (torch.arange(texture_height, dtype=torch.float32, device=device) + 0.5) / texture_height
        U = u_t.unsqueeze(0).expand(texture_height, texture_width)
        V = v_t.unsqueeze(1).expand(texture_height, texture_width)
        texel_uv = torch.stack([U.reshape(-1), V.reshape(-1)], dim=1)
        k_nn = _nn_chunk(texel_uv, uv_t, chunk)
        dir_out.reshape(-1, 3)[:] = dirs_t[k_nn]

    # Exit texture: only texels whose direction lies within the samples' field (r <= max_r) map to a sample; the rest
    # hold the invalid UV (-1, -1), as in native-grid LUTs, so rays outside the field never reach the fovea.
    d = dirs_t.to(torch.float64)
    r_s = (torch.atan2((d[:, 0] ** 2 + d[:, 1] ** 2).sqrt(), (-d[:, 2]).clamp(min=1e-20)) * (180.0 / math.pi)).float()
    th_s = torch.atan2(d[:, 1], d[:, 0]).float()
    xs = r_s * th_s.cos()
    ys = r_s * th_s.sin()
    max_r = float(r_s.max().item()) if N > 0 else 0.0

    oct_x, oct_y, oct_z = _octahedral_decode(exit_width, exit_height, device=device)
    r_q = (torch.atan2((oct_x**2 + oct_y**2).sqrt(), (-oct_z).clamp(min=1e-20)) * (180.0 / math.pi)).float()
    th_q = torch.atan2(oct_y.float(), oct_x.float())
    xq = r_q * th_q.cos()
    yq = r_q * th_q.sin()

    odz_f = oct_z.float()
    valid = (r_q <= max_r) & (odz_f < -1e-9)

    exit_u = torch.full((exit_height, exit_width), -1.0, dtype=torch.float32, device=device)
    exit_v = torch.full((exit_height, exit_width), -1.0, dtype=torch.float32, device=device)

    if N > 0 and valid.any():
        sample_cart = torch.stack([xs, ys], dim=1)
        vf = valid.reshape(-1)
        query_cart = torch.stack([xq.reshape(-1)[vf], yq.reshape(-1)[vf]], dim=1)
        k_exit = _nn_chunk(query_cart, sample_cart, chunk)
        eu_flat = exit_u.reshape(-1)
        ev_flat = exit_v.reshape(-1)
        eu_flat[vf] = uv_t[k_exit, 0]
        ev_flat[vf] = uv_t[k_exit, 1]

    return dir_out, torch.stack([exit_u, exit_v], dim=2)


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


def samples_uv_cortical_disc(
    r_deg,
    theta_rad,
    fov_deg: float,
    cmf_a: float,
    device: torch.device | str,
) -> torch.Tensor:
    """Map samples to NDC using the cortical polar disc layout.

    ``ρ = log((r + a) / a)``, mapped to a centered disc.
    Foveal centre ``r=0`` → disc centre ``(0.5, 0.5)``.

    Returns:
        ``(N, 2)`` float32 NDC tensor.
    """
    rho_max = math.log((fov_deg / 2.0 + cmf_a) / cmf_a)
    r = _t(r_deg, torch.float64, device)
    th = _t(theta_rad, torch.float64, device)
    rho = ((r + cmf_a) / cmf_a).clamp(min=1e-30).log()
    u = (rho * th.cos() / (2.0 * rho_max) + 0.5).float()
    v = (-rho * th.sin() / (2.0 * rho_max) + 0.5).float()
    return torch.stack([u, v], dim=1)


def _t(x, dtype: torch.dtype, device) -> torch.Tensor:
    """Convert array-like or tensor *x* to a ``torch.Tensor`` on *device*."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


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


def _nn_chunk(query: torch.Tensor, keys: torch.Tensor, chunk: int) -> torch.Tensor:
    """Chunked nearest-neighbour search via ``torch.cdist``.

    Chunk size is capped so the float32 distance matrix stays within 2 GiB.

    Returns:
        ``(M,)`` int64 index of the closest key for each query row.
    """
    N = keys.shape[0]
    safe_chunk = max(1, (2 << 30) // (4 * max(N, 1)))
    effective = min(chunk, safe_chunk)
    M = query.shape[0]
    idx = torch.empty(M, dtype=torch.long, device=query.device)
    for s in range(0, M, effective):
        e = min(s + effective, M)
        idx[s:e] = torch.cdist(query[s:e], keys).argmin(dim=1)
    return idx
