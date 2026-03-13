# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for :class:`~isaaclab.envs.utils.video_recorder.VideoRecorder`."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import TiledCameraCfg
from isaaclab.utils import configclass

from .video_recorder import VideoRecorder


DEFAULT_VIDEO_FALLBACK_CAMERA_CFG = TiledCameraCfg(
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
"""Default fallback :class:`~isaaclab.sensors.camera.TiledCameraCfg` for state-based video recording.

Places a pinhole camera at ``/World/envs/env_0/VideoCamera`` offset ``(-7, 0, 3)`` from
env_0's origin, angled ~12° downward in the world frame.  This matches the camera position used
by ``Isaac-Cartpole-RGB-v0`` and gives a reasonable side view for medium-scale environments
(env spacing ~4 m).

This is the **default** value of :attr:`VideoRecorderCfg.fallback_camera_cfg`.  No action is
needed in task configs - fallback cameras are automatically available for all state-based
environments.  Spawning only occurs when :attr:`VideoRecorderCfg.render_mode` is ``"rgb_array"``
(i.e. ``--video`` is active), so ordinary training runs incur zero overhead.

To customise the pose for a different environment scale, override in the task's ``__post_init__``::

    self.video_recorder.fallback_camera_cfg = self.video_recorder.fallback_camera_cfg.replace(
        offset=TiledCameraCfg.OffsetCfg(pos=(-3.0, 0.0, 2.0), rot=(0.0, 0.1045, 0.0, 0.9945), convention="world"),
    )
"""


@configclass
class VideoRecorderCfg:
    """Configuration for :class:`~isaaclab.envs.utils.video_recorder.VideoRecorder`.

    Set :attr:`class_type` to a custom subclass of
    :class:`~isaaclab.envs.utils.video_recorder.VideoRecorder` to swap the
    video-capture implementation (e.g. an Option-B pipeline that only renders
    ``video_num_tiles`` cameras on the GPU) without modifying any environment code.
    """

    class_type: type = VideoRecorder
    """The recorder class to instantiate.  Must accept ``(cfg, scene)`` as constructor arguments.
    Defaults to :class:`~isaaclab.envs.utils.video_recorder.VideoRecorder`.
    """

    render_mode: str | None = None
    """The render mode forwarded from the environment constructor.

    Populated automatically by the environment base classes from the ``render_mode`` argument
    passed to :func:`gymnasium.make` (or the environment constructor directly).  User code
    should not set this field manually.

    When ``None`` (the default, i.e. ``--video`` was **not** passed), :class:`VideoRecorder`
    skips spawning any fallback cameras so that state-based runs incur zero overhead.
    Only when this is ``"rgb_array"`` does the recorder allocate GPU resources for the
    fallback camera grid.
    """

    video_mode: str = "perspective"
    """Video recording mode.  One of ``"tiled"`` or ``"perspective"``.

    * ``"perspective"`` *(default)* - captures a single wide-angle isometric view of the
      scene.

      * **Newton backends** (Newton Warp or OVRTX renderer): a headless
        :class:`newton.viewer.ViewerGL` renders an isometric perspective of all
        environments (or the first ``video_num_tiles`` when that field is set).
      * **Kit backends** (PhysX + RTX renderer): the Kit viewport camera
        ``/OmniverseKit_Persp`` is captured via ``omni.replicator.core``.

      The TiledCamera sensor is **bypassed** entirely, even when one is present in the
      scene (e.g. vision-based tasks), giving a human-readable view instead of the
      agent's raw pixel observations.

    * ``"tiled"`` - reads pixel data from a
      :class:`~isaaclab.sensors.camera.TiledCamera`.  On vision-based tasks the agent's
      own observation camera is reused at zero extra cost and the output is a square
      tile-grid of per-agent views.  On state-based tasks with Kit-based backends a
      fallback :class:`~isaaclab.sensors.camera.TiledCamera` (``fallback_camera_cfg``) is
      spawned.  On Newton backends the Newton OpenGL perspective viewer is used instead.

    Set via the ``--video`` CLI flag (``--video=perspective`` / ``--video=tiled``), or
    as a Hydra override: ``env.video_recorder.video_mode=tiled``.
    """

    video_num_tiles: int = -1
    """Number of environment tiles to include in each video frame when using ``render_mode="rgb_array"``.
    Defaults to -1, which renders all environments.

    Environments are arranged into a square grid of size
    ``ceil(sqrt(video_num_tiles)) * ceil(sqrt(video_num_tiles))``, with unused slots filled with
    black. For example:

    * ``-1``: all environments (default)
    * ``1``: single environment (1*1)
    * ``4``: first 4 environments (2*2 grid)
    * ``9``: first 9 environments (3*3 grid)

    CLI example: ``env.video_recorder.video_num_tiles=9``
    """

    fallback_camera_cfg: object = DEFAULT_VIDEO_FALLBACK_CAMERA_CFG
    """Optional :class:`~isaaclab.sensors.camera.TiledCameraCfg` used to spawn a dedicated
    video-only camera for state-based environments (no observation ``TiledCamera`` in the scene).

    Defaults to :data:`DEFAULT_VIDEO_FALLBACK_CAMERA_CFG` - a pinhole camera placed at
    ``(-7, 0, 3)`` relative to env_0's origin, giving a reasonable side view for environments
    with ~4 m spacing.  Set to ``None`` to disable fallback cameras entirely (e.g. for
    vision-based tasks that already have an observation :class:`~isaaclab.sensors.camera.TiledCamera`).

    Spawning is **gated on** :attr:`render_mode` ``== "rgb_array"`` (i.e. ``--video`` must be
    active), so the default value causes zero overhead during ordinary training runs.

    For Newton-based backends (Newton Warp or OVRTX renderer), the Newton OpenGL perspective
    viewer is used instead of fallback TiledCameras - see :attr:`gl_viewer_width`.

    To customise the pose for a different environment scale, override in the task's ``__post_init__``::

        self.video_recorder.fallback_camera_cfg = self.video_recorder.fallback_camera_cfg.replace(
            offset=TiledCameraCfg.OffsetCfg(pos=(-3.0, 0.0, 2.0), rot=(0.0, 0.1045, 0.0, 0.9945), convention="world"),
        )

    .. note::
        The prim path in the cfg must start with ``/World/envs/env_0/`` so that the OVRTX
        renderer path check succeeds and ``TiledCamera`` correctly infers ``num_envs`` from
        the scene.
    """

    camera_eye: tuple[float, float, float] = (7.5, 7.5, 7.5)
    """World-space position of the Newton GL perspective camera (in metres).

    Defaults to ``(7.5, 7.5, 7.5)`` — the same value as :attr:`~isaaclab.envs.common.ViewerCfg.eye`
    — so the Newton GL video matches the Kit ``/OmniverseKit_Persp`` viewport exactly.

    Override to reposition the camera for tasks with a very different scene scale::

        self.video_recorder.camera_eye    = (20.0, 20.0, 20.0)
        self.video_recorder.camera_lookat = (0.0,  0.0,  0.0)

    Only used by Newton backends in perspective mode.
    """

    camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """World-space point the Newton GL perspective camera looks at (in metres).

    Defaults to ``(0.0, 0.0, 0.0)`` — the same as :attr:`~isaaclab.envs.common.ViewerCfg.lookat`.
    """

    gl_viewer_width: int = 1280
    """Width in pixels of the Newton OpenGL perspective video frame.

    Only used when the active physics/renderer backend exposes a Newton model
    (i.e. Newton Warp or OVRTX renderer presets).  In that case :class:`VideoRecorder`
    spawns a headless :class:`newton.viewer.ViewerGL` instance that renders an isometric
    perspective view of all environments (limited to :attr:`video_num_tiles` when set),
    replacing the fallback :class:`~isaaclab.sensors.camera.TiledCamera` grid.

    This perspective path is activated only when ``render_mode == "rgb_array"``
    (i.e. ``--video`` is active).  Regular training runs are unaffected.
    """

    gl_viewer_height: int = 720
    """Height in pixels of the Newton OpenGL perspective video frame.

    See :attr:`gl_viewer_width` for full description.
    """
