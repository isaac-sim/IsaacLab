# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton ViewerRTX borrowing the simulation's authored OVStage."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import numpy as np
from isaaclab_newton.physics import NewtonBackendCfg
from isaaclab_ov.stage import OvstageBackendCfg

from isaaclab.scene_data import SceneDataFormat
from isaaclab.visualizers.base_visualizer import BaseVisualizer

from .newton_viewer import NewtonViewerRTX, _NewtonCameraControls
from .newton_visualization_markers import render_newton_visualization_markers
from .newton_visualizer_cfg import NewtonRTXVisualizerCfg

if TYPE_CHECKING:
    from isaaclab.sensors.camera import Camera
    from isaaclab.sim import SimulationContext
    from isaaclab.visualizers import PerspectiveCameraCfg


class NewtonRTXVisualizer(_NewtonCameraControls, BaseVisualizer):
    """Use Newton's native RTX camera, GPU presentation, and markers on a borrowed scene.

    The simulation owns the stage and sensors. Newton owns the perspective render product and window.
    Resizing the window scales the displayed image without resizing either source's buffers.
    """

    def __init__(self, cfg: NewtonRTXVisualizerCfg):
        """Create an unbound RTX visualizer; the simulation supplies its scene at initialization."""
        super().__init__(cfg)
        self._viewer: NewtonViewerRTX | None = None
        self._runtime_headless = cfg.headless
        self._step_counter = 0
        self.backend = None

    def initialize(self, sim: SimulationContext, *, cameras: list[PerspectiveCameraCfg | Camera]) -> None:
        """Attach to the populated scene and bind the model used to publish body poses."""
        if self._is_initialized:
            return
        if self.physics_backend in ("physx", "isaacsim_physx"):
            raise RuntimeError(
                "Newton RTX is kitless and cannot share a process with Kit physics; use Newton or OVPhysX."
            )
        super().initialize(sim, cameras=cameras)
        scene_data_provider = self._scene_data_provider
        scene = self._get_backend(OvstageBackendCfg(viewer_id=id(self)))
        scene.populate(self._scene_stage, sim.get_clone_plan())
        self.newton_cfg = NewtonBackendCfg(physics_cfg=sim.cfg.physics, device=sim.device)
        self.backend = self._get_backend(self.newton_cfg)
        self._transform_mapping = scene_data_provider.create_mapping(list(self.backend.model.body_label))
        cfg = self.cfg
        self._runtime_headless = cfg.headless or (
            sys.platform not in ("win32", "darwin") and not os.environ.get("DISPLAY")
        )
        settings = dict(cfg.render_settings)
        if cfg.background_color is not None:
            settings["omni:rtx:background:source:type"] = ("Token", "color")
            settings["omni:rtx:background:source:color"] = ("Color3f", cfg.background_color)
        self._viewer = NewtonViewerRTX(
            width=cfg.window_width,
            height=cfg.window_height,
            headless=self._runtime_headless,
            async_rendering=not self._runtime_headless,
            ovstage=scene.stage,
            render_settings=settings,
            up_axis="Z",
            metadata={
                "num_envs": scene_data_provider.num_envs,
                "physics_backend": self.physics_backend,
                "gravity": sim.cfg.gravity,
            },
            update_frequency=cfg.update_frequency,
        )
        self._viewer.set_model(self.backend.model)
        self._viewer.marker_groups = sim.vis_marker_registry.get_groups().values()
        self._viewer.picking_enabled = False
        self._setup_streaming_view(
            scene_data_provider.num_envs,
            visible_env_ids=self._env_ids,
            target_aspect=cfg.window_width / cfg.window_height,
        )
        self._viewer.register_ui_callback(self._draw_streaming_view_controls, position="side")
        self._select_camera(self._camera_index)
        self._is_initialized = True

    def _render_frame(self) -> None:
        """Update controls and publish GPU poses or sensor pixels to Newton once per frame."""
        viewer = self._viewer
        viewer.begin_frame(self._sim_time)
        try:
            self._navigate_scene_camera()
            if self._camera_sensor is not None:
                image = self._streaming_frame.data
                if image is None or self._streaming_frame.timestamp < 0 or not viewer.is_paused():
                    image = self.render_tiled_rgba()
                if image is None:
                    raise ValueError("Scene-camera display requires at least one selected camera frame.")
                viewer.log_image("Camera View", image, fullscreen=True)
            else:
                backend, provider = self.backend, self._scene_data_provider
                if not viewer.is_paused():
                    poses = SceneDataFormat.Transform()
                    if provider.get_transforms(poses, mapping=self._transform_mapping, count=backend.model.body_count):
                        backend.state_0.body_q = poses.transforms
                    if backend.geometry_offsets:
                        provider.get_geometry_points(
                            output=backend.state_0.particle_q, offsets=backend.geometry_offsets
                        )
                    viewer.log_state(backend.state_0)
                if self.cfg.enable_markers:
                    render_newton_visualization_markers(viewer, self._env_ids, num_envs=backend.model.num_envs)
        finally:
            viewer.end_frame()

    def step(self, dt: float) -> None:
        """Advance the display at the configured cadence; headless consumers capture on demand."""
        if not self._is_initialized or self._is_closed:
            return
        self._sim_time += dt
        self._step_counter += 1
        if self._runtime_headless or self._step_counter % self._viewer._update_frequency:
            return
        self._render_frame()

    def reset(self, soft: bool = False) -> None:
        """Rebind the viewer when a hard reset replaces its simulation-owned model."""
        super().reset(soft)
        if soft or not self._is_initialized or self._is_closed:
            return
        backend = self._get_backend(self.newton_cfg)
        if backend is self.backend:
            return
        self.backend = backend
        self._transform_mapping = self._scene_data_provider.create_mapping(list(backend.model.body_label))
        self._viewer.set_model(backend.model)
        self._viewer.picking_enabled = False
        self._viewer.register_ui_callback(self._viewer._render_training_controls, position="side")
        self._viewer.register_ui_callback(self._draw_streaming_view_controls, position="side")
        self._select_camera(self._camera_index)

    def render_rgb_array(self) -> np.ndarray | None:
        """Explicitly download the selected sensor or perspective image for recording."""
        if self._viewer is None:
            return None
        if self._camera_sensor is not None:
            return self.render_tiled_rgb_array()
        if self._runtime_headless:
            self._render_frame()
        return self._viewer.get_frame()

    def supports_markers(self) -> bool:
        """Return whether the native viewer accepts runtime visualization markers."""
        return True

    def is_running(self) -> bool:
        """Return whether the viewer is open."""
        return self._viewer is not None and self._viewer.is_running()

    def is_training_paused(self) -> bool:
        """Return whether the window requested a simulation pause."""
        return self._viewer is not None and self._viewer.is_training_paused()

    def is_rendering_paused(self) -> bool:
        """Return whether the window paused scene updates."""
        return self._viewer is not None and self._viewer.is_rendering_paused()

    def is_reset_requested(self) -> bool:
        """Return whether an episode reset is pending."""
        return self._viewer is not None and self._viewer.is_reset_requested()

    def consume_reset_request(self) -> bool:
        """Consume an episode reset request from the window."""
        return self._viewer is not None and self._viewer.consume_reset_request()

    def is_key_down(self, key: str) -> bool:
        """Return whether a window key is held."""
        return self._viewer is not None and self._viewer.is_key_down(key)

    def close(self) -> None:
        """Close Newton's viewer before the simulation releases the borrowed scene."""
        if self._is_closed:
            return
        try:
            if self._viewer is not None:
                self._viewer.close()
        finally:
            self._viewer = self.backend = None
            super().close()
