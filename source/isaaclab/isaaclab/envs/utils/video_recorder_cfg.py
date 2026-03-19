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

    env_render_mode: str | None = None
    """Gym render mode forwarded from the environment constructor (``"rgb_array"`` when ``--video`` is active).

    Set automatically by the environment base classes; do not set manually.
    """

    camera_position: tuple[float, float, float] = (7.5, 7.5, 7.5)
    """Perspective camera position in world space (metres).

    Direct RL / MARL and manager-based RL environments overwrite this from
    :attr:`~isaaclab.envs.common.ViewerCfg.eye` before recording so ``--video`` matches the
    task viewport for both Kit (PhysX / Isaac RTX) and Newton GL (Newton / OVRTX / etc.).
    """

    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Perspective camera look-at target in world space (metres). Set from ``ViewerCfg.lookat`` at env init."""

    window_width: int = 1280
    """Width in pixels of the recorded frame."""

    window_height: int = 720
    """Height in pixels of the recorded frame."""
