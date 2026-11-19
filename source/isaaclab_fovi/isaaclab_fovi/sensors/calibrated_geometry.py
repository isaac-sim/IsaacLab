# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Retinal geometry shared by rendering and image sampling: the physical camera and the gaze frame."""

from __future__ import annotations

import math
from typing import Protocol

import torch
from fovi.sensing.projection import CameraCalibration, CameraModel, angular_directions

from isaaclab.sim import PinholeCameraCfg

IDEAL_IMAGE_WIDTH = 1024
"""Pixel width [px] of the image assigned to a camera derived from ideal spawn optics.

The scale is arbitrary: the retinal window and the gaze are angles, so neither depends on it."""


def camera_model_from_spawn(spawn: PinholeCameraCfg) -> CameraCalibration:
    """Describe the camera that the spawn configuration creates as :class:`CameraModel` fields.

    * An OpenCV model in ``spawn.distortion`` gives that calibrated camera. A fisheye
      model with zero coefficients is an ideal equidistant fisheye.
    * Plain pinhole optics give an ideal pinhole with the same field of view. Without
      a vertical aperture, the image is square.

    Raises:
        ValueError: The spawn configuration is not a camera's, or describes a camera fovi cannot
            model: a USD fisheye projection, thin-prism distortion, or an aperture offset.
    """
    if not isinstance(spawn, PinholeCameraCfg):
        raise ValueError(f"{type(spawn).__name__} does not spawn a camera; set camera_model instead.")
    distortion = spawn.distortion
    if distortion is not None:
        width, height = distortion.image_size
        intrinsics = (distortion.fx, distortion.fy, distortion.cx, distortion.cy)
        if distortion.model == "opencvFisheye":
            calibration = CameraCalibration(model="fisheye", max_angle_deg=math.degrees(distortion.max_fov) / 2)
            coefficients = (distortion.k1, distortion.k2, distortion.k3, distortion.k4)
        elif distortion.model == "opencvPinhole":
            calibration = CameraCalibration(model="pinhole")
            coefficients = (
                distortion.k1,
                distortion.k2,
                distortion.p1,
                distortion.p2,
                distortion.k3,
                distortion.k4,
                distortion.k5,
                distortion.k6,
            )
            thin_prism = (distortion.s1, distortion.s2, distortion.s3, distortion.s4)
            if distortion.apply_lens_distortion and any(thin_prism):
                raise ValueError("Thin-prism distortion (s1-s4) is not supported; set camera_model instead.")
        else:
            raise ValueError(f"Unsupported lens distortion model {distortion.model!r}.")
        calibration.update(image_size=(height, width), intrinsics=tuple(float(value) for value in intrinsics))
        if distortion.apply_lens_distortion and any(coefficients):
            calibration["distortion"] = tuple(float(value) for value in coefficients)
        return calibration
    if spawn.projection_type != "pinhole":
        raise ValueError(
            "Only pinhole spawn optics describe a camera; for a fisheye, set spawn.distortion to an "
            "OpenCvFisheyeDistortionCfg or set camera_model."
        )
    if spawn.horizontal_aperture_offset or spawn.vertical_aperture_offset:
        raise ValueError("Aperture offsets are not supported; set camera_model instead.")
    vertical_aperture = spawn.vertical_aperture or spawn.horizontal_aperture
    width = IDEAL_IMAGE_WIDTH
    height = round(width * vertical_aperture / spawn.horizontal_aperture)
    focal = spawn.focal_length / spawn.horizontal_aperture * width
    return CameraCalibration(
        model="pinhole", image_size=(height, width), intrinsics=(focal, focal, (width - 1) / 2, (height - 1) / 2)
    )


def resolve_camera_model(cfg: SamplingGeometryConfig) -> CameraModel:
    """Return the physical camera the retina looks through.

    This is ``cfg.camera_model`` when set, otherwise the camera that ``cfg.spawn``
    creates (see :func:`camera_model_from_spawn`).

    Raises:
        ValueError: Neither ``cfg.camera_model`` nor ``cfg.spawn`` describes the camera.
    """
    if cfg.camera_model is not None:
        return CameraModel(**cfg.camera_model)
    if cfg.spawn is None:
        raise ValueError("A foveated camera on an existing prim needs camera_model.")
    return CameraModel(**camera_model_from_spawn(cfg.spawn))


def sampling_geometry(cfg: SamplingGeometryConfig) -> tuple[float, float]:
    """Return the retinal field of view and the CMF constant, both in degrees.

    The field of view is the angular span of the retinal window in the physical
    camera. Both values are in the units :class:`fovi.sensing.coords.SamplingCoords` takes.
    """
    fov = resolve_camera_model(cfg).field_of_view(cfg.retinal_window_reference_side, cfg.retinal_window_fraction)
    return fov, cfg.cmf_a_deg


def retinal_directions(cartesian: torch.Tensor, fov_deg: float, image_rotation_deg: float = 0.0) -> torch.Tensor:
    """Return OpenGL-frame angular rays, including the sensor mount roll."""
    rays = angular_directions(cartesian, fov_deg)
    rays = rays * rays.new_tensor((1, -1, -1))
    angle = rays.new_tensor(image_rotation_deg * torch.pi / 180)
    x, y, z = rays.unbind(-1)
    return torch.stack((angle.cos() * x - angle.sin() * y, angle.sin() * x + angle.cos() * y, z), -1)


def compose_rotations(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compose small rotation matrices without TF32 tensor-core rounding.

    Camera transforms need orthonormal rotations even when the surrounding
    training process enables reduced-precision matrix multiplication.
    """
    return (left[..., :, :, None] * right[..., None, :, :]).sum(dim=-2)


class SamplingGeometryConfig(Protocol):
    """The configuration fields that place the retina in the physical camera."""

    camera_model: CameraCalibration | None
    cmf_a_deg: float
    retinal_window_reference_side: str
    retinal_window_fraction: float
    spawn: PinholeCameraCfg | None
