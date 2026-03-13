# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for :class:`~isaaclab.envs.utils.video_recorder.VideoRecorder`.

Captures a single wide-angle perspective view of the scene. Newton backends use the
Newton GL viewer; Kit backends use the ``/OmniverseKit_Persp`` camera via
``omni.replicator.core``.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .video_recorder import VideoRecorder


@configclass
class VideoRecorderCfg:
    """Configuration for :class:`~isaaclab.envs.utils.video_recorder.VideoRecorder`."""

    class_type: type = VideoRecorder
    """Recorder class to instantiate; must accept ``(cfg, scene)``."""

    render_mode: str | None = None
    """Render mode forwarded from the environment constructor (``"rgb_array"`` when ``--video`` is active).

    Set automatically by the environment base classes; do not set manually.
    """

    camera_eye: tuple[float, float, float] = (7.5, 7.5, 7.5)
    """Newton GL perspective camera position in world space (metres).

    Matches :attr:`~isaaclab.envs.common.ViewerCfg.eye` so the Newton GL video aligns with
    the Kit ``/OmniverseKit_Persp`` viewport. Only used by Newton backends.
    """

    camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Newton GL perspective camera look-at point in world space (metres).

    Matches :attr:`~isaaclab.envs.common.ViewerCfg.lookat`. Only used by Newton backends.
    """

    gl_viewer_width: int = 1280
    """Width in pixels of the Newton GL perspective frame."""

    gl_viewer_height: int = 720
    """Height in pixels of the Newton GL perspective frame."""
