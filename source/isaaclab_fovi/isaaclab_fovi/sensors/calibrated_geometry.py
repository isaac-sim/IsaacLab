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


class SamplingGeometryConfig(Protocol):
    field_geometry: str
    camera_calibration: CameraCalibration | None
    cmf_a_deg: float
    retinal_window_reference_side: str
    retinal_window_short_side_fraction: float
    spawn: Any


def sampling_geometry(cfg: SamplingGeometryConfig) -> tuple[float, float]:
    """Return the retinal field of view and the CMF constant, both in degrees.

    These are the units :class:`fovi.sensing.coords.SamplingCoords` takes in either
    field geometry, so neither value is converted here.

    * Spherical: the field of view is the calibrated retinal window.
    * Planar: the field of view is the horizontal field of view of the spawn optics.

    Raises:
        ValueError: The geometry is unknown, or the configuration lacks the
            calibration (spherical) or spawn optics (planar) that define the window.
    """
    if cfg.field_geometry == "spherical":
        if cfg.camera_calibration is None:
            raise ValueError("Spherical sensing requires camera_calibration")
        camera = CameraModel(**cfg.camera_calibration)
        fov = camera.field_of_view(cfg.retinal_window_reference_side, cfg.retinal_window_short_side_fraction)
        return fov, cfg.cmf_a_deg
    if cfg.field_geometry == "planar":
        if cfg.spawn is None:
            raise ValueError("Planar sensing requires spawn optics to define its field of view")
        fov = math.degrees(2.0 * math.atan(cfg.spawn.horizontal_aperture / (2.0 * cfg.spawn.focal_length)))
        return fov, cfg.cmf_a_deg
    raise ValueError(f"Unknown field geometry {cfg.field_geometry!r}; use 'spherical' or 'planar'.")


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
