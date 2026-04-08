# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Video recorder implementation.

Captures a single wide-angle perspective view of the scene:

* **Kit backends** (PhysX physics or Isaac RTX renderer) — uses
  :mod:`isaaclab_physx.video_recording.isaacsim_kit_perspective_video`.
* **Newton backends** (Newton physics or Newton Warp renderer only) — uses
  :mod:`isaaclab_newton.video_recording.newton_gl_perspective_video`.

If neither a Kit nor a Newton backend is detected, construction raises so users do not
use ``--video`` on unsupported setups.

See :mod:`video_recorder_cfg` for configuration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene

    from .video_recorder_cfg import VideoRecorderCfg

logger = logging.getLogger(__name__)

_VideoBackend = Literal["kit", "newton_gl"]


def _resolve_video_backend(scene: InteractiveScene) -> _VideoBackend:
    """Resolve which video backend to use from physics and renderer configs.

    Priority: PhysX or Isaac RTX -> Kit camera; else Newton or Newton Warp -> GL viewer.
    When both are present (e.g. PhysX + Newton Warp), Kit wins.
    """
    sim = scene.sim
    physics_name = sim.physics_manager.__name__.lower()
    renderer_types: list[str] = scene._sensor_renderer_types()

    use_kit = "physx" in physics_name or "isaac_rtx" in renderer_types
    use_newton_gl = "newton" in physics_name or "newton_warp" in renderer_types

    if use_kit:
        return "kit"
    if use_newton_gl:
        return "newton_gl"
    raise RuntimeError(
        "Video recording (--video) requires a supported backend: "
        "PhysX or Isaac RTX renderer (Kit camera), or Newton physics / Newton Warp renderer (GL viewer). "
        "No supported backend detected; do not use --video for this setup."
    )


class VideoRecorder:
    """Records perspective video frames from the scene's active renderer.

    Args:
        cfg: Recorder configuration.
        scene: The interactive scene that owns the sensors.
    """

    def __init__(self, cfg: VideoRecorderCfg, scene: InteractiveScene):
        self.cfg = cfg
        self._scene = scene
        self._backend: _VideoBackend | None = None
        self._capture = None

        if cfg.env_render_mode == "rgb_array":
            self._backend = _resolve_video_backend(scene)
            if self._backend == "newton_gl":
                try:
                    import pyglet

                    if not pyglet.options.get("headless", False):
                        pyglet.options["headless"] = True
                except ImportError as e:
                    raise ImportError(
                        "The Newton GL video backend requires 'pyglet'. Install IsaacLab with './isaaclab.sh -i'."
                    ) from e
                from isaaclab_newton.video_recording.newton_gl_perspective_video import (
                    create_newton_gl_perspective_video,
                )
                from isaaclab_newton.video_recording.newton_gl_perspective_video_cfg import NewtonGlPerspectiveVideoCfg

                ncfg = NewtonGlPerspectiveVideoCfg(
                    window_width=cfg.window_width,
                    window_height=cfg.window_height,
                    camera_position=cfg.camera_position,
                    camera_target=cfg.camera_target,
                )
                self._capture = create_newton_gl_perspective_video(ncfg)
            else:
                from isaaclab_physx.video_recording.isaacsim_kit_perspective_video import (
                    create_isaacsim_kit_perspective_video,
                )
                from isaaclab_physx.video_recording.isaacsim_kit_perspective_video_cfg import (
                    IsaacsimKitPerspectiveVideoCfg,
                )

                kcfg = IsaacsimKitPerspectiveVideoCfg(
                    camera_position=cfg.camera_position,
                    camera_target=cfg.camera_target,
                    window_width=cfg.window_width,
                    window_height=cfg.window_height,
                )
                self._capture = create_isaacsim_kit_perspective_video(kcfg)

    def render_rgb_array(self) -> np.ndarray | None:
        """Return an RGB frame for the resolved backend. Fails if backend is unavailable."""
        if self._backend is None or self._capture is None:
            return None
        return self._capture.render_rgb_array()
