# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Video recorder implementation.

Backend resolution (``--video`` + ``--visualizer``):

1. **Active visualizer** - ``"kit"`` uses the Kit camera; ``"newton"`` uses the Newton GL viewer.
   ``"viser"`` / ``"rerun"`` have no capture API and fall through to rule 2.
2. **Physics/renderer stack** -
   - PhysX or Isaac RTX uses the Kit camera;
   - Newton physics or Newton Warp uses the Newton GL viewer.
   Kit wins when both signals present.  Raises if nothing resolves.

Set :attr:`~isaaclab.envs.utils.video_recorder_cfg.VideoRecorderCfg.backend_source` to ``"renderer"``
to ignore active visualizers and record from the physics/renderer stack.

Camera sync when a visualizer drives the backend: construction copies the visualizer config's
``eye`` / ``lookat`` into the recorder config; each :meth:`~VideoRecorder.render_rgb_array`
call then re-reads the Newton viewer's live ``camera.pos/pitch/yaw``. Kit video uses the
configured ``eye`` / ``lookat`` at construction time.

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

# visualizer types that map to a supported video backend.
# viser and rerun are intentionally absent - they have no video-capture API.
_VISUALIZER_TO_VIDEO_BACKEND: dict[str, _VideoBackend] = {
    "kit": "kit",
    "newton": "newton_gl",
}


def _resolve_video_backend(
    scene: InteractiveScene, backend_source: str = "visualizer"
) -> tuple[_VideoBackend, str | None]:
    """Return ``(backend, matched_visualizer_type)`` for the active scene.

    ``matched_visualizer_type`` is ``"kit"`` / ``"newton"`` when a visualizer drove the
    selection, or ``None`` when the physics/renderer preset stack was used instead.

    Args:
        scene: The interactive scene that owns the sim context.
        backend_source: ``"visualizer"`` to let active visualizers choose the backend, or ``"renderer"``
            to ignore active visualizers and use the physics/renderer stack.

    Raises:
        RuntimeError: If no supported backend is detected.
    """
    if backend_source not in ("visualizer", "renderer"):
        raise ValueError("VideoRecorderCfg.backend_source must be either 'visualizer' or 'renderer'.")

    # Prefer the visualizer backend when --visualizer is active alongside --video.
    visualizer_types: list[str] = scene.sim.resolve_visualizer_types() if backend_source == "visualizer" else []
    if visualizer_types:
        # kit takes priority when multiple visualizers are active
        for preferred in ("kit", "newton"):
            if preferred in visualizer_types:
                backend = _VISUALIZER_TO_VIDEO_BACKEND[preferred]
                logger.debug("[VideoRecorder] Using '%s' backend from active '%s' visualizer.", backend, preferred)
                return backend, preferred
        # only unsupported visualizer types (viser, rerun) are active.
        logger.warning(
            "[VideoRecorder] Active visualizer(s) %s do not support video capture; "
            "falling back to physics/renderer stack detection.",
            visualizer_types,
        )

    # fall back to physics/renderer preset stack detection.
    sim = scene.sim
    physics_name = sim.physics_manager.__name__.lower()
    renderer_types: list[str] = scene._sensor_renderer_types()

    use_kit = "physx" in physics_name or "isaac_rtx" in renderer_types
    use_newton_gl = "newton" in physics_name or "newton_warp" in renderer_types

    if use_kit:
        return "kit", None
    if use_newton_gl:
        return "newton_gl", None
    raise RuntimeError(
        "Video recording (--video) requires a supported backend: "
        "PhysX or Isaac RTX renderer (Kit camera), or Newton physics / Newton Warp renderer (GL viewer). "
        "No supported backend detected; do not use --video for this setup."
    )


def _sync_camera_from_visualizer(
    scene: InteractiveScene,
    visualizer_type: str,
    cfg: VideoRecorderCfg,
) -> None:
    """Overwrite ``cfg.eye`` and ``cfg.lookat`` from the active visualizer.

    Args:
        scene: The interactive scene that owns the sim context.
        visualizer_type: The visualizer type string matched by ``_resolve_video_backend``
            (e.g. ``"kit"`` or ``"newton"``).
        cfg: The recorder configuration to update in place.
    """
    try:
        resolved_cfgs = scene.sim._resolve_visualizer_cfgs()
    except Exception as exc:
        logger.debug("[VideoRecorder] Could not resolve visualizer cfgs for camera sync: %s", exc)
        return

    for vcfg in resolved_cfgs:
        if getattr(vcfg, "visualizer_type", None) != visualizer_type:
            continue
        pos = getattr(vcfg, "eye", None)
        tgt = getattr(vcfg, "lookat", None)
        if pos is None or tgt is None:
            break
        cfg.eye = tuple(float(x) for x in pos)
        cfg.lookat = tuple(float(x) for x in tgt)
        logger.debug(
            "[VideoRecorder] Camera synced from '%s' visualizer: position=%s, target=%s.",
            visualizer_type,
            cfg.eye,
            cfg.lookat,
        )
        return

    logger.debug(
        "[VideoRecorder] Could not find eye/lookat on '%s' visualizer cfg; keeping existing camera values.",
        visualizer_type,
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
        # visualizer type that drove backend selection (or None when using physics/renderer stack).
        self._matched_visualizer: str | None = None
        # live visualizer instance - looked up lazily on first render_rgb_array() call because
        # visualizers are initialised by sim.reset(), which runs after VideoRecorder.__init__.
        self._live_visualizer = None

        if cfg.env_render_mode == "rgb_array":
            backend_source = getattr(cfg, "backend_source", "visualizer")
            self._backend, self._matched_visualizer = _resolve_video_backend(scene, backend_source)
            if self._matched_visualizer is not None:
                _sync_camera_from_visualizer(scene, self._matched_visualizer, cfg)
            if self._backend == "newton_gl":
                try:
                    import pyglet as _pyglet  # noqa: F401 - verify pyglet is available
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
                    eye=cfg.eye,
                    lookat=cfg.lookat,
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
                    eye=cfg.eye,
                    lookat=cfg.lookat,
                    window_width=cfg.window_width,
                    window_height=cfg.window_height,
                )
                self._capture = create_isaacsim_kit_perspective_video(kcfg)

    def _sync_newton_camera(self) -> None:
        """Push the Newton visualizer's live camera pose into the capture object.

        Called once per :meth:`render_rgb_array` when a Newton visualizer is active.
        The live visualizer instance is resolved lazily (visualizers are initialised by
        ``sim.reset()``, which runs after ``VideoRecorder.__init__``).
        """
        if self._live_visualizer is None:
            for viz in self._scene.sim.visualizers:
                if getattr(getattr(viz, "cfg", None), "visualizer_type", None) == "newton":
                    self._live_visualizer = viz
                    break
            if self._live_visualizer is None:
                return

        viewer = getattr(self._live_visualizer, "_viewer", None)
        if viewer is None:
            return

        import math

        cam = viewer.camera
        pos = (float(cam.pos[0]), float(cam.pos[1]), float(cam.pos[2]))
        yaw_rad = math.radians(float(cam.yaw))
        pitch_rad = math.radians(float(cam.pitch))
        dx = math.cos(pitch_rad) * math.cos(yaw_rad)
        dy = math.cos(pitch_rad) * math.sin(yaw_rad)
        dz = math.sin(pitch_rad)
        target = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
        self._capture.update_camera(pos, target)

    def render_rgb_array(self) -> np.ndarray | None:
        """Return an RGB frame for the resolved backend. Fails if backend is unavailable."""
        if self._backend is None or self._capture is None:
            return None
        if self._matched_visualizer == "newton":
            # Newton GL camera state lives in the capture object and must be synced each frame
            # to follow interactive viewer movement.
            self._sync_newton_camera()
        # Kit capture uses the configured eye/lookat applied to the recording camera at construction time.
        return self._capture.render_rgb_array()
