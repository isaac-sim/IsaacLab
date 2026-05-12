# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Isaac Sim Kit perspective RGB capture (OmniverseKit_Persp + Replicator)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isaaclab.utils import configclass

if TYPE_CHECKING:
    pass


@configclass
class IsaacsimKitPerspectiveVideoCfg:
    """Settings for capturing a perspective RGB frame from the Kit viewport camera."""

    class_type: type[Any] | str = (
        "isaaclab_physx.video_recording.isaacsim_kit_perspective_video:IsaacsimKitPerspectiveVideo"
    )
    """Implementation class; default is
    :class:`~isaaclab_physx.video_recording.isaacsim_kit_perspective_video.IsaacsimKitPerspectiveVideo`."""

    camera_prim_path: str = "/OmniverseKit_Persp"
    """Viewport camera prim used for the render product."""

    eye: tuple[float, float, float] = (7.5, 7.5, 7.5)
    """Camera position in world space (metres)."""

    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Camera look-at target in world space (metres)."""

    window_width: int = 1280
    """Output width in pixels."""

    window_height: int = 720
    """Output height in pixels."""
