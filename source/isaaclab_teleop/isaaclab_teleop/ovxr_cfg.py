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
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.sensors import CameraCfg

_DEFAULT_OVXR_CAMERA_PATHS = (
    "/OVXRGeneratedCameras/OVXR_LeftEyeCamera",
    "/OVXRGeneratedCameras/OVXR_RightEyeCamera",
)


@configclass
class OvLibrariesXrCameraCfg:
    """One application-selected sensor and its camera-panel policy."""

    name: str = MISSING
    """Scene sensor name used for camera selection and as the desktop mirror's camera label."""
    camera_cfg: CameraCfg = MISSING
    """Selected sensor configuration, assigned to the session's shared renderer."""
    max_fps: float = 30.0
    """Finite positive image update cap [Hz]."""
    panel_width_m: float = 0.48
    """Physical panel width [m]; image aspect ratio determines its height."""
    position: tuple[float, float, float] = (0.0, -0.15, -0.8)
    """Panel center in OpenXR head space [m], with forward along negative Z."""
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Panel orientation in head space as an (x, y, z, w) quaternion."""
    space: Literal["head_locked"] = "head_locked"
    """Placement space; only head-locked panels are supported."""

    def __post_init__(self) -> None:
        # configclass copies field defaults, so an unset field holds a copy of MISSING.
        unset = (type(MISSING), type(None))
        if isinstance(self.name, unset) or not self.name or isinstance(self.camera_cfg, unset):
            raise ValueError("camera name and configuration are required")
        if not math.isfinite(self.max_fps) or self.max_fps <= 0:
            raise ValueError("camera max_fps must be finite and positive")
        # Mirror Capture's CameraPanelConfig checks without importing its session stack; the
        # session builds the native panel configuration with the resolved product path.
        if not math.isfinite(self.panel_width_m) or self.panel_width_m <= 0.0:
            raise ValueError("camera panel panel_width_m must be finite and positive")
        if len(self.position) != 3 or not all(math.isfinite(value) for value in self.position):
            raise ValueError("camera panel position must contain three finite values")
        if len(self.orientation_xyzw) != 4 or not all(math.isfinite(value) for value in self.orientation_xyzw):
            raise ValueError("camera panel orientation must contain four finite values")
        if math.sqrt(sum(value * value for value in self.orientation_xyzw)) <= 0.0:
            raise ValueError("camera panel orientation must have non-zero norm")
        if self.space != "head_locked":
            raise ValueError("camera panel space must be 'head_locked'")


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
    desktop_mirror_enabled: bool = False
    """Whether to display the owner-managed desktop mirror."""
    desktop_mirror_max_fps: float = 30.0
    """Finite positive desktop mirror refresh cap [Hz]."""
    auxiliary_cameras: tuple[OvLibrariesXrCameraCfg, ...] = ()
    """Up to three uniquely named camera panels; an empty selection disables panels."""
    extra_openxr_extensions: tuple[str, ...] = ()
    """Additional OpenXR extension names requested when creating the session."""

    def __post_init__(self) -> None:
        cameras = self.auxiliary_cameras
        if len(cameras) > 3 or len({camera.name for camera in cameras}) != len(cameras):
            raise ValueError("Select at most three uniquely named auxiliary cameras")
        if self.session_mode not in ("ar", "vr"):
            raise ValueError("session_mode must be 'ar' or 'vr'")
        if not math.isfinite(self.meters_per_unit) or self.meters_per_unit <= 0:
            raise ValueError("meters_per_unit must be finite and positive")
        if not math.isfinite(self.desktop_mirror_max_fps) or self.desktop_mirror_max_fps <= 0.0:
            raise ValueError("desktop_mirror_max_fps must be finite and positive")
        if (self.depth_near_z, self.depth_far_z) != (0.0, 0.0) and (
            not math.isfinite(self.depth_near_z)
            or not math.isfinite(self.depth_far_z)
            or self.depth_near_z <= 0.0
            or self.depth_far_z <= self.depth_near_z
        ):
            raise ValueError("depth_near_z and depth_far_z must both be zero or finite with 0 < near < far")
        if len(self.camera_paths) != 2 or any(not path for path in self.camera_paths):
            raise ValueError("Kitless stereo requires exactly two non-empty OVXR camera paths")
