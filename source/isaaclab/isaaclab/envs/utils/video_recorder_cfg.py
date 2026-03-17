# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for :class:`~isaaclab.envs.utils.video_recorder.VideoRecorder`.

* **Perspective** (``video_mode="perspective"``) — Kit backends use
  :mod:`isaaclab_physx.video_recording.isaacsim_kit_perspective_video`; Newton backends use
  :mod:`isaaclab_newton.video_recording.newton_gl_perspective_video`.
* **Tiled** (``video_mode="tiled"``) — Kit backends use
  :mod:`isaaclab_physx.video_recording.isaacsim_tiled_camera_video`; Newton backends use
  :mod:`isaaclab_newton.video_recording.newton_tiled_camera_video`.
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
"""


@configclass
class VideoRecorderCfg:
    """Configuration for :class:`~isaaclab.envs.utils.video_recorder.VideoRecorder`."""

    class_type: type = VideoRecorder
    """Recorder class to instantiate; must accept ``(cfg, scene)``."""

    env_render_mode: str | None = None
    """Gym render mode forwarded from the environment constructor (``"rgb_array"`` when ``--video`` is active).

    Set automatically by the environment base classes; do not set manually.
    """

    video_mode: str = "perspective"
    """``"perspective"`` or ``"tiled"``. Set via CLI: ``--video=perspective`` / ``--video=tiled``."""

    video_num_tiles: int = -1
    """Max environments per tiled frame (``-1`` = all). CLI: ``env.video_recorder.video_num_tiles=9``."""

    fallback_camera_cfg: object | None = DEFAULT_TILED_RECORDING_CAMERA_CFG
    """Fallback :class:`~isaaclab.sensors.camera.TiledCameraCfg` for tiled mode without observation camera.

    Set to ``None`` to disable spawning. Ignored when ``video_mode="perspective"``.
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
