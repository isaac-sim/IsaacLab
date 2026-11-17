# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dense raster geometry using the same physical lens model as Fovi/Isaac Lab Fovi."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from fovi.sensing.projection import CameraModel

if TYPE_CHECKING:
    from pxr import Usd


def dense_camera_rays(camera: CameraModel, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Return OpenGL unit rays (H,W,3) and usable-pixel mask (H,W).

    Integer pixel centers refer to the original distorted raster. Invalid
    pixels receive a finite forward ray and are masked after rendering.
    """
    height, width = camera.image_size
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float64, device=device),
        torch.arange(width, dtype=torch.float64, device=device),
        indexing="ij",
    )
    directions, valid = camera.unproject(torch.stack((x, y), -1))
    directions = directions * directions.new_tensor((1, -1, -1))
    directions = torch.where(valid[..., None], directions, directions.new_tensor((0, 0, -1)))
    return directions.float().contiguous(), valid


def lens_schema_values(camera: CameraModel) -> tuple[str, str, dict[str, float]]:
    """Return the native RTX schema, model token, and OpenCV coefficient mapping."""
    fx, fy, cx, cy = camera.intrinsics
    # RTX evaluates raster samples at half-integers; OpenCV centers are integers.
    values = dict(fx=fx, fy=fy, cx=cx + 0.5, cy=cy + 0.5)
    if camera.model == "fisheye":
        schema, token = "OmniLensDistortionOpenCvFisheyeAPI", "opencvFisheye"
        names = ("k1", "k2", "k3", "k4")
    elif camera.model == "pinhole":
        schema, token = "OmniLensDistortionOpenCvPinholeAPI", "opencvPinhole"
        names = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4")
    else:
        raise ValueError(f"Unsupported lens model {camera.model!r}")
    coefficients = (*camera.distortion, *((0.0,) * (len(names) - len(camera.distortion))))
    values.update(zip(names, coefficients, strict=True))
    return schema, token, values


def apply_lens_schema(prim: Usd.Prim, camera: CameraModel) -> None:
    """Author native RTX distortion before render-product initialization."""
    from pxr import Gf, Sdf

    schema, token, values = lens_schema_values(camera)
    # Explicit metadata and attributes work even when the USD schema is not registered.
    prim.AddAppliedSchema(schema)
    prim.CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.Token).Set(token)
    prefix = f"omni:lensdistortion:{token}:"
    height, width = camera.image_size
    prim.CreateAttribute(prefix + "imageSize", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(width, height))
    for name, value in values.items():
        prim.CreateAttribute(prefix + name, Sdf.ValueTypeNames.Float).Set(float(value))
