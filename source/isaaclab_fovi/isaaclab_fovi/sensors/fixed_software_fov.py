# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Geometry helpers for direct foveation inside a fixed software-camera FOV."""

from __future__ import annotations

import math

import torch

RETINAL_WINDOW_REFERENCE_SIDES = ("short", "long")
FIXED_FOV_ANGLE_MODELS = ("camera_xyz", "pan_tilt")


def retinal_window_reference_size(
    *,
    image_height: int,
    image_width: int,
    reference_side: str,
) -> int:
    """Resolve the image dimension used as the circular retinal diameter."""
    if image_height <= 0 or image_width <= 0:
        raise ValueError(
            f"Retinal window reference requires positive image dimensions, got {image_height}x{image_width}."
        )
    if reference_side == "short":
        return min(image_height, image_width)
    if reference_side == "long":
        return max(image_height, image_width)
    raise ValueError(
        f"retinal_window_reference_side must be one of {RETINAL_WINDOW_REFERENCE_SIDES}, got {reference_side!r}."
    )


def validate_fixed_software_fov(
    *,
    image_height: int,
    image_width: int,
    focal_length_px: float,
    principal_point: tuple[float, float],
    retinal_window_short_side_fraction: float,
    retinal_window_reference_side: str = "short",
) -> None:
    """Validate a rectilinear dense-camera geometry."""
    if image_height <= 0 or image_width <= 0:
        raise ValueError(f"fixed software FOV requires positive image dimensions, got {image_height}x{image_width}.")
    if focal_length_px <= 0.0:
        raise ValueError(f"fixed software FOV requires focal_length_px > 0, got {focal_length_px}.")
    cx, cy = principal_point
    if not 0.0 <= cx <= image_width or not 0.0 <= cy <= image_height:
        raise ValueError(
            "fixed software FOV principal point must lie inside the image, got "
            f"{principal_point} for {image_height}x{image_width}."
        )
    if not 0.0 < retinal_window_short_side_fraction <= 1.0:
        raise ValueError(
            f"retinal_window_short_side_fraction must be in (0, 1], got {retinal_window_short_side_fraction}."
        )
    retinal_window_reference_size(
        image_height=image_height,
        image_width=image_width,
        reference_side=retinal_window_reference_side,
    )


def retinal_ray_directions(
    cartesian: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    focal_length_px: float,
    retinal_window_short_side_fraction: float,
    retinal_window_reference_side: str = "short",
    image_rotation_deg: float = 0.0,
) -> torch.Tensor:
    """Convert normalized retinal coordinates into camera-space pinhole rays."""
    validate_fixed_software_fov(
        image_height=image_height,
        image_width=image_width,
        focal_length_px=focal_length_px,
        principal_point=(image_width * 0.5, image_height * 0.5),
        retinal_window_short_side_fraction=retinal_window_short_side_fraction,
        retinal_window_reference_side=retinal_window_reference_side,
    )
    rotated = cartesian
    if image_rotation_deg:
        theta = math.radians(float(image_rotation_deg))
        cosine = math.cos(theta)
        sine = math.sin(theta)
        rotation = cartesian.new_tensor(((cosine, -sine), (sine, cosine)))
        rotated = cartesian @ rotation.T
    reference_size = retinal_window_reference_size(
        image_height=image_height,
        image_width=image_width,
        reference_side=retinal_window_reference_side,
    )
    diameter_px = reference_size * retinal_window_short_side_fraction
    tangent = rotated * (0.5 * diameter_px / focal_length_px)
    directions = torch.cat(
        (tangent, -torch.ones_like(tangent[:, :1])),
        dim=-1,
    )
    return torch.nn.functional.normalize(directions, dim=-1)


def _image_fixation_tangents(
    fixation: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    focal_length_px: float,
    principal_point: tuple[float, float],
    image_rotation_deg: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_fixed_software_fov(
        image_height=image_height,
        image_width=image_width,
        focal_length_px=focal_length_px,
        principal_point=principal_point,
        retinal_window_short_side_fraction=1.0,
    )
    cx, cy = principal_point
    tangent_x = (fixation[..., 1] * image_width - cx) / focal_length_px
    tangent_y = (cy - fixation[..., 0] * image_height) / focal_length_px
    if image_rotation_deg:
        theta = math.radians(float(image_rotation_deg))
        cosine = math.cos(theta)
        sine = math.sin(theta)
        rotated_x = cosine * tangent_x - sine * tangent_y
        tangent_y = sine * tangent_x + cosine * tangent_y
        tangent_x = rotated_x
    return tangent_x, tangent_y


def image_fixation_to_camera_angles(
    fixation: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    focal_length_px: float,
    principal_point: tuple[float, float],
    image_rotation_deg: float = 0.0,
) -> torch.Tensor:
    """Aim an XYZ-rotated virtual camera through an image fixation."""
    tangent_x, tangent_y = _image_fixation_tangents(
        fixation,
        image_height=image_height,
        image_width=image_width,
        focal_length_px=focal_length_px,
        principal_point=principal_point,
        image_rotation_deg=image_rotation_deg,
    )
    roll = torch.atan(tangent_y)
    pitch = torch.atan(-tangent_x * torch.cos(roll))
    return torch.stack((roll, pitch), dim=-1)


def image_fixation_to_pan_tilt_angles(
    fixation: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    focal_length_px: float,
    principal_point: tuple[float, float],
) -> torch.Tensor:
    """Aim a +Y-forward, pan-then-tilt mount through an image fixation."""
    tangent_x, tangent_y = _image_fixation_tangents(
        fixation,
        image_height=image_height,
        image_width=image_width,
        focal_length_px=focal_length_px,
        principal_point=principal_point,
    )
    pan = torch.atan(-tangent_x)
    tilt = torch.atan(tangent_y * torch.cos(pan))
    return torch.stack((tilt, pan), dim=-1)


def fixed_frustum_validity(
    ray_directions: torch.Tensor,
    fixation_angles: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    focal_length_px: float,
    principal_point: tuple[float, float],
    angle_model: str = "camera_xyz",
    image_rotation_deg: float = 0.0,
) -> torch.Tensor:
    """Return validity after applying the configured camera or mount rotation."""
    rays = ray_directions.unsqueeze(0)
    x = rays[..., 0]
    y = rays[..., 1]
    z = rays[..., 2]
    if angle_model == "camera_xyz":
        roll = fixation_angles[:, 0:1]
        pitch = fixation_angles[:, 1:2]
        sin_r = torch.sin(roll)
        cos_r = torch.cos(roll)
        sin_p = torch.sin(pitch)
        cos_p = torch.cos(pitch)
        rotated_x = cos_p * x + sin_p * z
        rotated_y = sin_r * sin_p * x + cos_r * y - sin_r * cos_p * z
        rotated_z = -cos_r * sin_p * x + sin_r * y + cos_r * cos_p * z
    elif angle_model == "pan_tilt":
        if not math.isfinite(float(image_rotation_deg)):
            raise ValueError(f"fixed-FOV image_rotation_deg must be finite, got {image_rotation_deg!r}.")
        theta = math.radians(float(image_rotation_deg))
        cosine = math.cos(theta)
        sine = math.sin(theta)

        policy_x = cosine * x + sine * y
        policy_y = -sine * x + cosine * y
        policy_z = z
        mount_x = policy_x
        mount_y = -policy_z
        mount_z = policy_y

        tilt = fixation_angles[:, 0:1]
        pan = fixation_angles[:, 1:2]
        sin_tilt = torch.sin(tilt)
        cos_tilt = torch.cos(tilt)
        sin_pan = torch.sin(pan)
        cos_pan = torch.cos(pan)
        tilted_y = cos_tilt * mount_y - sin_tilt * mount_z
        tilted_z = sin_tilt * mount_y + cos_tilt * mount_z
        moved_mount_x = cos_pan * mount_x - sin_pan * tilted_y
        moved_mount_y = sin_pan * mount_x + cos_pan * tilted_y

        moved_policy_x = moved_mount_x
        moved_policy_y = tilted_z
        moved_policy_z = -moved_mount_y
        rotated_x = cosine * moved_policy_x - sine * moved_policy_y
        rotated_y = sine * moved_policy_x + cosine * moved_policy_y
        rotated_z = moved_policy_z
    else:
        raise ValueError(f"angle_model must be one of {FIXED_FOV_ANGLE_MODELS}, got {angle_model!r}.")

    forward = -rotated_z
    safe_forward = forward.clamp_min(torch.finfo(ray_directions.dtype).eps)
    cx, cy = principal_point
    pixel_x = focal_length_px * rotated_x / safe_forward + cx
    pixel_y = cy - focal_length_px * rotated_y / safe_forward
    return (forward > 0.0) & (pixel_x >= 0.0) & (pixel_x < image_width) & (pixel_y >= 0.0) & (pixel_y < image_height)
