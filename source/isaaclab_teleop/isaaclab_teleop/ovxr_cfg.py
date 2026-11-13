# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for an Isaac Capture-owned OVXR session.

This module has no runtime dependencies, so launchers can build and validate the
configuration without the native OVXR, OVRTX or OVStage packages.
"""

from __future__ import annotations

import math
from typing import Literal

from isaaclab.utils import configclass

_DEFAULT_OVXR_CAMERA_PATHS = (
    "/OVXRGeneratedCameras/OVXR_LeftEyeCamera",
    "/OVXRGeneratedCameras/OVXR_RightEyeCamera",
)


@configclass
class OvLibrariesXrCfg:
    """Isaac Lab configuration for an Isaac Capture-owned OVXR session."""

    application_name: str = "isaaclab-kitless-teleop"
    """Application identifier passed to the OpenXR runtime."""
    session_mode: Literal["ar", "vr"] = "ar"
    """OpenXR presentation mode: augmented or virtual reality."""
    renderer_settings: str = "--/app/hydra/aperture/conform=stretch"
    """Process-global OVXR renderer settings applied before renderer construction."""
    cloudxr_env_file: str | None = None
    """CloudXR environment-file path; None leaves runtime launch to the caller."""
    auto_launch_cloudxr: bool = True
    """Whether the session owner launches the CloudXR runtime automatically."""
    render_products: tuple[str, ...] = ()
    """Stereo RenderProduct paths; an empty selection uses OVXR's default paths."""
    camera_paths: tuple[str, ...] = _DEFAULT_OVXR_CAMERA_PATHS
    """Exactly two non-empty USD camera paths, ordered left eye then right eye."""
    create_render_products: bool = True
    """Whether to declare and configure the selected stereo rendering prims."""
    meters_per_unit: float = 1.0
    """Finite positive scene scale [m per USD unit]."""
    depth_near_z: float = 0.0
    """Near depth distance [m]; both depth distances zero select provider defaults."""
    depth_far_z: float = 0.0
    """Far depth distance [m]; explicit distances must satisfy 0 < near < far."""
    tracking_anchor_path: str = ""
    """USD tracking-anchor path; an empty path uses the stage origin."""
    host_wait_render_output: bool = True
    """Whether OVXR waits for render output before returning from frame rendering."""
    extra_openxr_extensions: tuple[str, ...] = ()
    """Additional OpenXR extension names requested when creating the session."""

    def __post_init__(self) -> None:
        if self.session_mode not in ("ar", "vr"):
            raise ValueError("session_mode must be 'ar' or 'vr'")
        if not math.isfinite(self.meters_per_unit) or self.meters_per_unit <= 0:
            raise ValueError("meters_per_unit must be finite and positive")
        if (self.depth_near_z, self.depth_far_z) != (0.0, 0.0) and (
            not math.isfinite(self.depth_near_z)
            or not math.isfinite(self.depth_far_z)
            or self.depth_near_z <= 0.0
            or self.depth_far_z <= self.depth_near_z
        ):
            raise ValueError("depth_near_z and depth_far_z must both be zero or finite with 0 < near < far")
        if len(self.camera_paths) != 2 or any(not path for path in self.camera_paths):
            raise ValueError("Kitless stereo requires exactly two non-empty OVXR camera paths")
