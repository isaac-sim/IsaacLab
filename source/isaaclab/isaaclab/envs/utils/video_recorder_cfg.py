# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for :class:`~isaaclab.envs.utils.video_recorder.VideoRecorder`.

Two recording modes are supported (set via :attr:`VideoRecorderCfg.video_mode`):

* **Perspective view** (``"perspective"``, default) - a single wide-angle viewport
  camera.  Uses the Newton GL viewer on Newton backends; falls back to the Kit
  ``/OmniverseKit_Persp`` camera via ``omni.replicator.core`` on Kit backends.
* **Camera sensor / tiled** (``"tiled"``) - reads pixel data from a
  :class:`~isaaclab.sensors.camera.TiledCamera` sensor and arranges the per-agent
  frames into a square grid.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import TiledCameraCfg
from isaaclab.utils import configclass

from .video_recorder import VideoRecorder


DEFAULT_TILED_RECORDING_CAMERA_CFG = TiledCameraCfg(
    prim_path="/World/envs/env_0/VideoCamera",
    update_period=0.0,
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 1.0e5),
    ),
    offset=TiledCameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.0, 0.1045, 0.0, 0.9945), convention="world"),
)
"""Default :class:`~isaaclab.sensors.camera.TiledCameraCfg` for tiled state-based video recording.

Places a pinhole camera at ``(-7, 0, 3)`` m relative to env_0's origin, angled ~12° downward.
Only spawned when ``--video=tiled`` is active and no observation TiledCamera exists in the scene.

Override pose in ``__post_init__`` for tasks with different scene scales::

    self.video_recorder.fallback_camera_cfg = self.video_recorder.fallback_camera_cfg.replace(
        offset=TiledCameraCfg.OffsetCfg(pos=(-3.0, 0.0, 2.0), rot=(0.0, 0.1045, 0.0, 0.9945), convention="world"),
    )
"""


@configclass
class VideoRecorderCfg:
    """Configuration for :class:`~isaaclab.envs.utils.video_recorder.VideoRecorder`."""

    class_type: type = VideoRecorder
    """Recorder class to instantiate; must accept ``(cfg, scene)``."""

    render_mode: str | None = None
    """Render mode forwarded from the environment constructor (``"rgb_array"`` when ``--video`` is active).

    Set automatically by the environment base classes; do not set manually.
    """

    video_mode: str = "perspective"
    """Recording mode: ``"perspective"`` (default) or ``"tiled"``.

    * ``"perspective"`` - single wide-angle view of the scene. Newton backends use the Newton GL
      viewer; Kit backends use ``/OmniverseKit_Persp`` via ``omni.replicator.core``. TiledCamera
      is bypassed even when present.
    * ``"tiled"`` - square tile-grid from a :class:`~isaaclab.sensors.camera.TiledCamera`.
      Reuses the observation camera on vision-based tasks; spawns ``fallback_camera_cfg`` for
      state-based tasks. Raises ``RuntimeError`` if no TiledCamera is available.

    Set via CLI: ``--video=perspective`` / ``--video=tiled``.
    """

    video_num_tiles: int = -1
    """Max environments to include per frame (``-1`` = all).

    Tiles are arranged into a ``ceil(sqrt(N)) × ceil(sqrt(N))`` grid with black padding.
    CLI example: ``env.video_recorder.video_num_tiles=9``
    """

    fallback_camera_cfg: object = DEFAULT_TILED_RECORDING_CAMERA_CFG
    """Side-view :class:`~isaaclab.sensors.camera.TiledCameraCfg` for tiled state-based recording.

    Spawned when ``video_mode="tiled"`` and no observation TiledCamera exists in the scene.
    Set to ``None`` to disable.
    """

    camera_eye: tuple[float, float, float] = (7.5, 7.5, 7.5)
    """Newton GL perspective camera position in world space (metres).

    Matches :attr:`~isaaclab.envs.common.ViewerCfg.eye` so the Newton GL video aligns with
    the Kit ``/OmniverseKit_Persp`` viewport. Only used by Newton backends in perspective mode.
    """

    camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Newton GL perspective camera look-at point in world space (metres).

    Matches :attr:`~isaaclab.envs.common.ViewerCfg.lookat`. Only used by Newton backends in perspective mode.
    """

    gl_viewer_width: int = 1280
    """Width in pixels of the Newton GL perspective frame. Only active when ``--video`` is set."""

    gl_viewer_height: int = 720
    """Height in pixels of the Newton GL perspective frame. Only active when ``--video`` is set."""

