# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Omniverse-based visualizer using Isaac Sim viewport."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pxr import UsdGeom

from .ov_visualizer_cfg import OVVisualizerCfg
from .visualizer import Visualizer

logger = logging.getLogger(__name__)


class OVVisualizer(Visualizer):
    """Omniverse visualizer using Isaac Sim viewport."""

    def __init__(self, cfg: OVVisualizerCfg):
        super().__init__(cfg)
        self.cfg: OVVisualizerCfg = cfg

        self._simulation_app = None
        self._viewport_window = None
        self._viewport_api = None
        self._is_initialized = False
        self._sim_time = 0.0
        self._step_counter = 0

    def initialize(self, scene_data: dict[str, Any] | None = None) -> None:
        if self._is_initialized:
            logger.warning("[OVVisualizer] Already initialized.")
            return

        usd_stage = scene_data["usd_stage"]
        scene_data_provider = scene_data["scene_data_provider"]
        metadata = scene_data_provider.get_metadata()

        self._ensure_simulation_app()
        self._setup_viewport(usd_stage, metadata)

        num_envs = metadata.get("num_envs", 0)
        physics_backend = metadata.get("physics_backend", "unknown")
        logger.info(f"[OVVisualizer] Initialized ({num_envs} envs, {physics_backend} physics)")

        self._is_initialized = True

    def step(self, dt: float, state: Any | None = None) -> None:
        if not self._is_initialized:
            return
        self._sim_time += dt
        self._step_counter += 1

    def close(self) -> None:
        if not self._is_initialized:
            return
        self._simulation_app = None
        self._viewport_window = None
        self._viewport_api = None
        self._is_initialized = False

    def is_running(self) -> bool:
        if self._simulation_app is None:
            return False
        return self._simulation_app.is_running()

    def is_training_paused(self) -> bool:
        return False

    def supports_markers(self) -> bool:
        return True

    def supports_live_plots(self) -> bool:
        return True

    def set_camera_view(
        self, eye: tuple[float, float, float] | list[float], target: tuple[float, float, float] | list[float]
    ) -> None:
        if not self._is_initialized:
            logger.warning("[OVVisualizer] Cannot set camera view - visualizer not initialized.")
            return
        self._set_viewport_camera(tuple(eye), tuple(target))

    def _ensure_simulation_app(self) -> None:
        import omni.kit.app

        app = omni.kit.app.get_app()
        if app is None or not app.is_running():
            raise RuntimeError("[OVVisualizer] Isaac Sim app is not running.")

        try:
            from isaacsim import SimulationApp

            sim_app = None
            if hasattr(SimulationApp, "_instance") and SimulationApp._instance is not None:
                sim_app = SimulationApp._instance
            elif hasattr(SimulationApp, "instance") and callable(SimulationApp.instance):
                sim_app = SimulationApp.instance()

            if sim_app is not None:
                self._simulation_app = sim_app
                if self._simulation_app.config.get("headless", False):
                    logger.warning("[OVVisualizer] Running in headless mode. Viewport may not display.")
        except ImportError:
            pass

    def _setup_viewport(self, usd_stage, metadata: dict) -> None:
        import omni.kit.viewport.utility as vp_utils
        from omni.ui import DockPosition

        if self.cfg.create_viewport and self.cfg.viewport_name:
            dock_position_map = {
                "LEFT": DockPosition.LEFT,
                "RIGHT": DockPosition.RIGHT,
                "BOTTOM": DockPosition.BOTTOM,
                "SAME": DockPosition.SAME,
            }
            dock_pos = dock_position_map.get(self.cfg.dock_position.upper(), DockPosition.SAME)

            self._viewport_window = vp_utils.create_viewport_window(
                name=self.cfg.viewport_name,
                width=self.cfg.window_width,
                height=self.cfg.window_height,
                position_x=50,
                position_y=50,
                docked=True,
            )

            logger.info(f"[OVVisualizer] Created viewport '{self.cfg.viewport_name}'")
            asyncio.ensure_future(self._dock_viewport_async(self.cfg.viewport_name, dock_pos))
            self._create_and_assign_camera(usd_stage)
        else:
            if self.cfg.viewport_name:
                self._viewport_window = vp_utils.get_viewport_window_by_name(self.cfg.viewport_name)
                if self._viewport_window is None:
                    logger.warning(f"[OVVisualizer] Viewport '{self.cfg.viewport_name}' not found. Using active.")
                    self._viewport_window = vp_utils.get_active_viewport_window()
            else:
                self._viewport_window = vp_utils.get_active_viewport_window()

        self._viewport_api = self._viewport_window.viewport_api
        self._set_viewport_camera(self.cfg.camera_position, self.cfg.camera_target)

    async def _dock_viewport_async(self, viewport_name: str, dock_position) -> None:
        import omni.kit.app
        import omni.ui

        viewport_window = None
        for _ in range(10):
            viewport_window = omni.ui.Workspace.get_window(viewport_name)
            if viewport_window:
                break
            await omni.kit.app.get_app().next_update_async()

        if not viewport_window:
            logger.warning(f"[OVVisualizer] Could not find viewport window '{viewport_name}'.")
            return

        main_viewport = omni.ui.Workspace.get_window("Viewport")
        if not main_viewport:
            for alt_name in ["/OmniverseKit/Viewport", "Viewport Next"]:
                main_viewport = omni.ui.Workspace.get_window(alt_name)
                if main_viewport:
                    break

        if main_viewport and main_viewport != viewport_window:
            viewport_window.dock_in(main_viewport, dock_position, 0.5)
            await omni.kit.app.get_app().next_update_async()
            viewport_window.focus()
            viewport_window.visible = True
            await omni.kit.app.get_app().next_update_async()
            viewport_window.focus()

    def _create_and_assign_camera(self, usd_stage) -> None:
        camera_path = f"/World/Cameras/{self.cfg.viewport_name}_Camera"
        camera_prim = usd_stage.GetPrimAtPath(camera_path)
        if not camera_prim.IsValid():
            UsdGeom.Camera.Define(usd_stage, camera_path)

        if self._viewport_api:
            self._viewport_api.set_active_camera(camera_path)

    def _set_viewport_camera(self, position: tuple[float, float, float], target: tuple[float, float, float]) -> None:
        import isaacsim.core.utils.viewports as vp_utils

        camera_path = self._viewport_api.get_active_camera()
        if not camera_path:
            camera_path = "/OmniverseKit_Persp"

        vp_utils.set_camera_view(
            eye=list(position), target=list(target), camera_prim_path=camera_path, viewport_api=self._viewport_api
        )
