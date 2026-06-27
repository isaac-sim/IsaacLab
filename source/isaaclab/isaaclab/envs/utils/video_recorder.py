# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Video recorder implementation.

An active Kit or Newton visualizer selects the capture backend by default. Newton video reuses the
active visualizer framebuffer, while renderer-selected Newton video uses a separate headless viewer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np

if TYPE_CHECKING:
    from isaaclab_newton.video_recording.newton_gl_perspective_video import NewtonGlPerspectiveVideo
    from isaaclab_physx.video_recording.isaacsim_kit_perspective_video import IsaacsimKitPerspectiveVideo
    from isaaclab_visualizers.newton import NewtonVisualizer

    from isaaclab.scene import InteractiveScene
    from isaaclab.visualizers import VisualizerCfg

    from .video_recorder_cfg import VideoRecorderCfg

_VideoBackend = Literal["kit", "newton_gl"]


def _select_video_backend(scene: InteractiveScene, backend_source: str) -> tuple[_VideoBackend, VisualizerCfg | None]:
    """Resolve the capture backend and visualizer configuration that selected it.

    Args:
        scene: The interactive scene that owns the simulation context.
        backend_source: Source used to select the capture backend.

    Returns:
        Backend identifier and the visualizer configuration that selected it, if any.

    Raises:
        ValueError: If backend_source is invalid.
        RuntimeError: If no supported backend is available.
    """
    if backend_source not in ("visualizer", "renderer"):
        raise ValueError("VideoRecorderCfg.backend_source must be either 'visualizer' or 'renderer'.")

    if backend_source == "visualizer":
        visualizer_cfgs = scene.sim._resolve_visualizer_cfgs()
        for visualizer_type, backend in (("kit", "kit"), ("newton", "newton_gl")):
            for visualizer_cfg in visualizer_cfgs:
                if visualizer_cfg.visualizer_type == visualizer_type:
                    return backend, visualizer_cfg

    physics_name = scene.sim.physics_manager.__name__.lower()
    renderer_types = scene._sensor_renderer_types()
    if "physx" in physics_name or "isaac_rtx" in renderer_types:
        return "kit", None
    if "newton" in physics_name or "newton_warp" in renderer_types:
        return "newton_gl", None
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
        self._capture: NewtonGlPerspectiveVideo | IsaacsimKitPerspectiveVideo | NewtonVisualizer | None = None
        self._use_newton_visualizer = False

        if cfg.env_render_mode != "rgb_array":
            return

        backend, visualizer_cfg = _select_video_backend(scene, cfg.backend_source)
        eye = cfg.eye if visualizer_cfg is None else visualizer_cfg.eye
        lookat = cfg.lookat if visualizer_cfg is None else visualizer_cfg.lookat

        if backend == "newton_gl" and visualizer_cfg is not None:
            self._use_newton_visualizer = True
        elif backend == "newton_gl":
            from isaaclab_newton.video_recording.newton_gl_perspective_video import (
                create_newton_gl_perspective_video,
            )
            from isaaclab_newton.video_recording.newton_gl_perspective_video_cfg import NewtonGlPerspectiveVideoCfg

            self._capture = create_newton_gl_perspective_video(
                NewtonGlPerspectiveVideoCfg(
                    window_width=cfg.window_width,
                    window_height=cfg.window_height,
                    eye=eye,
                    lookat=lookat,
                )
            )
        else:
            from isaaclab_physx.video_recording.isaacsim_kit_perspective_video import (
                create_isaacsim_kit_perspective_video,
            )
            from isaaclab_physx.video_recording.isaacsim_kit_perspective_video_cfg import (
                IsaacsimKitPerspectiveVideoCfg,
            )

            self._capture = create_isaacsim_kit_perspective_video(
                IsaacsimKitPerspectiveVideoCfg(
                    eye=eye,
                    lookat=lookat,
                    window_width=cfg.window_width,
                    window_height=cfg.window_height,
                )
            )

    def render_rgb_array(self) -> np.ndarray | None:
        """Return an RGB frame for the resolved backend."""
        if self._use_newton_visualizer and self._capture is None:
            visualizer = next(
                (
                    visualizer
                    for visualizer in self._scene.sim.visualizers
                    if visualizer.cfg.visualizer_type == "newton"
                ),
                None,
            )
            if visualizer is None:
                raise RuntimeError("The Newton visualizer was selected for video capture but is not initialized.")
            self._capture = cast("NewtonVisualizer", visualizer)
        if self._capture is None:
            return None
        return self._capture.render_rgb_array()
