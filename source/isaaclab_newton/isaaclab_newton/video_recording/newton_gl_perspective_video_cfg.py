# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton GL perspective RGB capture (headless ViewerGL)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isaaclab.utils import configclass

if TYPE_CHECKING:
    pass


@configclass
class NewtonGlPerspectiveVideoCfg:
    """Settings for capturing a perspective RGB frame via ``newton.viewer.ViewerGL``."""

    class_type: type[Any] | str = "isaaclab_newton.video_recording.newton_gl_perspective_video:NewtonGlPerspectiveVideo"
    """Implementation class; default is
    :class:`~isaaclab_newton.video_recording.newton_gl_perspective_video.NewtonGlPerspectiveVideo`."""

    window_width: int = 1280
    """Viewer width in pixels."""

    window_height: int = 720
    """Viewer height in pixels."""

    camera_position: tuple[float, float, float] = (7.5, 7.5, 7.5)
    """Camera position in world space (metres)."""

    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Camera look-at target in world space (metres)."""

    horiz_fov_deg: float = 60.0
    """Horizontal FOV assumed for Kit ``/OmniverseKit_Persp``; converted to vertical FOV for GL viewer."""
