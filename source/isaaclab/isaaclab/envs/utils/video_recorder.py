# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Video recorder implementation.

* **Perspective** - Kit: :mod:`isaaclab_physx.video_recording.isaacsim_kit_perspective_video`;
  Newton: :mod:`isaaclab_newton.video_recording.newton_gl_perspective_video`.
* **Tiled** - Kit: :mod:`isaaclab_physx.video_recording.isaacsim_tiled_camera_video`;
  Newton: :mod:`isaaclab_newton.video_recording.newton_tiled_camera_video`.

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

# Renderer types that Kit tiled recording can read (matches IsaacsimTiledCameraVideoCfg.preferred_renderer_types).
_KIT_TILED_RENDERER_TYPES: frozenset[str] = frozenset(("isaac_rtx", "ovrtx"))


def _resolve_video_backend(scene: InteractiveScene, video_mode: str = "perspective") -> _VideoBackend:
    """PhysX or Isaac RTX -> Kit; Newton or Newton Warp -> Newton GL; Kit wins when both are present.

    Tiled exception: if the scene has no RTX cameras (all use Newton Warp), Newton GL is preferred over Kit.
    """
    physics_backend: str = scene.physics_backend
    renderer_types: list[str] = scene._sensor_renderer_types()

    use_kit = "physx" in physics_backend or "isaac_rtx" in renderer_types
    use_newton_gl = "newton" in physics_backend or "newton_warp" in renderer_types

    if use_kit and use_newton_gl and video_mode == "tiled":
        # Tie-break: Kit tiled recording requires RTX cameras.
        # If the scene has no RTX cameras but has Newton Warp cameras, then prefer Newton GL.
        has_kit_cameras = bool(_KIT_TILED_RENDERER_TYPES & set(renderer_types))
        if not has_kit_cameras:
            return "newton_gl"

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
    """Records perspective or tiled video for the active backend."""

    def __init__(self, cfg: VideoRecorderCfg, scene: InteractiveScene):
        self.cfg = cfg
        self._scene = scene
        self._backend: _VideoBackend | None = None
        self._capture = None
        self._tiled_capture = None

        if cfg.env_render_mode != "rgb_array":
            return

        video_mode = cfg.video_mode or "perspective"

        if video_mode == "tiled":
            self._backend = _resolve_video_backend(scene, video_mode="tiled")
            if self._backend == "newton_gl":
                try:
                    import pyglet

                    if not pyglet.options.get("headless", False):
                        pyglet.options["headless"] = True
                except ImportError as e:
                    raise ImportError(
                        "The Newton GL video backend requires 'pyglet'. Install IsaacLab with './isaaclab.sh -i'."
                    ) from e
                from isaaclab_newton.renderers import NewtonWarpRendererCfg
                from isaaclab_newton.video_recording.newton_tiled_camera_video import (
                    create_newton_tiled_camera_video,
                )
                from isaaclab_newton.video_recording.newton_tiled_camera_video_cfg import NewtonTiledCameraVideoCfg

                newton_fb = cfg.fallback_camera_cfg
                if newton_fb is not None:
                    newton_fb = newton_fb.replace(renderer_cfg=NewtonWarpRendererCfg())
                ncfg = NewtonTiledCameraVideoCfg(
                    video_num_tiles=cfg.video_num_tiles,
                    fallback_camera_cfg=newton_fb,
                )
                self._tiled_capture = create_newton_tiled_camera_video(ncfg, scene)
            else:
                from isaaclab_physx.video_recording.isaacsim_tiled_camera_video import (
                    create_isaacsim_tiled_camera_video,
                )
                from isaaclab_physx.video_recording.isaacsim_tiled_camera_video_cfg import (
                    IsaacsimTiledCameraVideoCfg,
                )

                kcfg = IsaacsimTiledCameraVideoCfg(
                    video_num_tiles=cfg.video_num_tiles,
                    fallback_camera_cfg=cfg.fallback_camera_cfg,
                )
                self._tiled_capture = create_isaacsim_tiled_camera_video(kcfg, scene)
            return

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
        """Return an RGB frame, or ``None`` if not in ``rgb_array`` mode; raises if mode set but no backend."""
        if self.cfg.env_render_mode != "rgb_array":
            return None
        if self._tiled_capture is not None:
            return self._tiled_capture.render_rgb_array()
        if self._capture is None:
            raise RuntimeError(
                "VideoRecorder has no capture backend despite rgb_array render mode. "
                "This is an internal error; please report it."
            )
        return self._capture.render_rgb_array()
