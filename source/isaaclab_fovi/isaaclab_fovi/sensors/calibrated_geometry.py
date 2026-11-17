# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared calibrated retinal geometry for rendering and image sampling."""

from __future__ import annotations

import math
from typing import Any, Literal, Protocol

import torch
import warp as wp
from fovi.sensing.projection import (
    CameraCalibration,
    CameraModel,
    angular_directions,
    gaze_rotation,
)

IDEAL_IMAGE_WIDTH = 1024
"""Pixel width [px] of the image assigned to a camera derived from ideal spawn optics.

The scale is arbitrary: fixations are normalized image positions and the retinal
window is an angle, so neither depends on it."""


class SamplingGeometryConfig(Protocol):
    camera_calibration: CameraCalibration | None
    cmf_a_deg: float
    retinal_window_reference_side: str
    retinal_window_short_side_fraction: float
    spawn: Any


def calibration_from_spawn(spawn: Any) -> CameraCalibration:
    """Describe the camera that the spawn configuration creates as a fovi calibration.

    * An OpenCV model in ``spawn.distortion`` gives that calibrated camera. A fisheye
      model with zero coefficients is an ideal equidistant fisheye.
    * Plain pinhole optics give an ideal pinhole with the same field of view. Without
      a vertical aperture, the image is square.

    Raises:
        ValueError: The spawn configuration describes a camera fovi cannot model: a USD
            fisheye projection, thin-prism distortion, or an aperture offset.
    """
    distortion = getattr(spawn, "distortion", None)
    if distortion is not None:
        width, height = distortion.image_size
        intrinsics = (distortion.fx, distortion.fy, distortion.cx, distortion.cy)
        if distortion.model == "opencvFisheye":
            calibration = CameraCalibration(model="fisheye", max_angle_deg=math.degrees(distortion.max_fov) / 2)
            coefficients = (distortion.k1, distortion.k2, distortion.k3, distortion.k4)
        elif distortion.model == "opencvPinhole":
            calibration = CameraCalibration(model="pinhole")
            coefficients = tuple(getattr(distortion, name) for name in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"))
            thin_prism = (distortion.s1, distortion.s2, distortion.s3, distortion.s4)
            if distortion.apply_lens_distortion and any(thin_prism):
                raise ValueError("Thin-prism distortion (s1-s4) is not supported; set camera_calibration instead.")
        else:
            raise ValueError(f"Unsupported lens distortion model {distortion.model!r}.")
        calibration.update(image_size=(height, width), intrinsics=tuple(float(value) for value in intrinsics))
        if distortion.apply_lens_distortion and any(coefficients):
            calibration["distortion"] = tuple(float(value) for value in coefficients)
        return calibration
    if getattr(spawn, "projection_type", None) != "pinhole":
        raise ValueError(
            "Only pinhole spawn optics describe a camera; for a fisheye, set spawn.distortion to an "
            "OpenCvFisheyeDistortionCfg or set camera_calibration."
        )
    if spawn.horizontal_aperture_offset or spawn.vertical_aperture_offset:
        raise ValueError("Aperture offsets are not supported; set camera_calibration instead.")
    vertical_aperture = spawn.vertical_aperture or spawn.horizontal_aperture
    width = IDEAL_IMAGE_WIDTH
    height = round(width * vertical_aperture / spawn.horizontal_aperture)
    focal = spawn.focal_length / spawn.horizontal_aperture * width
    return CameraCalibration(
        model="pinhole", image_size=(height, width), intrinsics=(focal, focal, (width - 1) / 2, (height - 1) / 2)
    )


def camera_model(cfg: SamplingGeometryConfig) -> CameraModel:
    """Return the physical camera the retina looks through.

    This is ``cfg.camera_calibration`` when set, otherwise the camera that ``cfg.spawn``
    creates (see :func:`calibration_from_spawn`).

    Raises:
        ValueError: Neither a calibration nor spawn optics describe the camera.
    """
    if cfg.camera_calibration is not None:
        return CameraModel(**cfg.camera_calibration)
    if cfg.spawn is None:
        raise ValueError("A foveated camera on an existing prim needs camera_calibration.")
    return CameraModel(**calibration_from_spawn(cfg.spawn))


def sampling_geometry(cfg: SamplingGeometryConfig) -> tuple[float, float]:
    """Return the retinal field of view and the CMF constant, both in degrees.

    The field of view is the angular span of the retinal window in the physical
    camera. Both values are in the units :class:`fovi.sensing.coords.SamplingCoords` takes.
    """
    fov = camera_model(cfg).field_of_view(cfg.retinal_window_reference_side, cfg.retinal_window_short_side_fraction)
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


def convert_camera_frame_quaternion(
    orientation: torch.Tensor,
    *,
    origin: Literal["world", "opengl"],
    target: Literal["world", "opengl", "ros"],
) -> torch.Tensor:
    """Convert world/OpenGL or OpenGL/ROS poses without a matrix round trip.

    Args:
        orientation: Camera quaternions in XYZW order, shape (..., 4).
        origin: Input camera-frame convention.
        target: Opposite world/OpenGL convention, or ROS for OpenGL input.

    Returns:
        Normalized float32 quaternions, shape (..., 4). Quaternion sign may change.
    """
    quat = orientation.float()
    quat = quat / quat.norm(dim=-1, keepdim=True)
    x, y, z, w = quat.unbind(-1)
    # Right multiplication by exact fixed-basis quaternions, without allocating
    # device constants or making the matrix round-trip sensitive to AMP/TF32.
    if origin == "world" and target == "opengl":
        components = (w + x - y + z, -w + x + y + z, -w - x - y + z, w - x + y + z)
    elif origin == "opengl" and target == "world":
        components = (-w + x + y - z, w - x + y - z, w + x + y + z, w + x - y - z)
    elif origin == "opengl" and target == "ros":
        # ROS flips the OpenGL Y/Z axes: right-multiply by a half-turn about X.
        return torch.stack((w, z, -y, -x), dim=-1)
    else:
        raise ValueError(f"Unsupported camera conversion: {origin!r} to {target!r}")
    return torch.stack(components, dim=-1) * 0.5


def calibrated_fixation_rotation(
    camera: CameraModel, fixation: torch.Tensor, convention: str = "pan_tilt"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return source-camera rotation and validity for normalized image gaze."""
    if fixation.is_cuda:
        return _compiled_fixation_rotation(camera, fixation, convention)
    return _fixation_rotation(camera, fixation, convention)


def _fixation_rotation(
    camera: CameraModel, fixation: torch.Tensor, convention: str
) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = camera.image_size
    pixels = torch.stack((fixation[..., 1] * w - 0.5, fixation[..., 0] * h - 0.5), -1)
    direction, valid = camera.unproject(pixels)
    return gaze_rotation(direction, convention), valid


def calibrated_mount_angles(camera: CameraModel, fixation: torch.Tensor) -> torch.Tensor:
    """Unproject normalized upright-image gaze to mount (tilt, pan) radians."""
    if fixation.is_cuda:
        return _compiled_mount_angles(camera, fixation)
    return _mount_angles(camera, fixation)


def _mount_angles(camera: CameraModel, fixation: torch.Tensor) -> torch.Tensor:
    h, w = camera.image_size
    pixels = torch.stack((fixation[..., 1] * w - 0.5, fixation[..., 0] * h - 0.5), -1)
    direction, _ = camera.unproject(pixels)
    x, y, z = direction.unbind(-1)
    return torch.stack((-torch.atan2(y, torch.sqrt(x * x + z * z)), -torch.atan2(x, z)), -1)


# Fuse the lens inversion's iterative pointwise math instead of launching each
# small operation separately on every simulation tick. CPU callers stay eager.
_compiled_fixation_rotation = torch.compile(_fixation_rotation, fullgraph=True)
_compiled_mount_angles = torch.compile(_mount_angles, fullgraph=True)


@wp.kernel(enable_backward=False)
def _rotate_camera_rays(
    rays: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.mat33),
    output: wp.array2d(dtype=wp.vec3),
):
    batch, sample = wp.tid()
    output[batch, sample] = rotations[batch] * rays[sample]


def rotate_camera_rays(canonical_rays: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Rotate visibility rays with IEEE float32, independent of model TF32 settings."""
    if canonical_rays.is_cuda and canonical_rays.dtype == rotation.dtype == torch.float32:
        wp.init()  # stream_from_torch requires an initialized Warp runtime.
        result = canonical_rays.new_empty((len(rotation), len(canonical_rays), 3))
        torch_stream = torch.cuda.current_stream(canonical_rays.device)
        stream = wp.get_stream(wp.device_from_torch(canonical_rays.device))
        # Temporary aliases unregister even a live Warp-owned stream when destroyed.
        if stream.cuda_stream != torch_stream.cuda_stream:
            stream = wp.stream_from_torch(torch_stream)
        with wp.ScopedStream(
            stream,
            sync_enter=False,
            sync_exit=False,
        ):
            wp.launch(
                _rotate_camera_rays,
                dim=result.shape[:2],
                inputs=[
                    wp.from_torch(canonical_rays.contiguous(), dtype=wp.vec3),
                    wp.from_torch(rotation.contiguous(), dtype=wp.mat33),
                ],
                outputs=[wp.from_torch(result, dtype=wp.vec3)],
            )
        return result
    return canonical_rays @ rotation.transpose(-1, -2)


def calibrated_visibility(camera: CameraModel, canonical_rays: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Mask OpenCV-frame canonical rays against the fixed physical sensor."""
    _, valid = camera.project(rotate_camera_rays(canonical_rays, rotation))
    return valid
