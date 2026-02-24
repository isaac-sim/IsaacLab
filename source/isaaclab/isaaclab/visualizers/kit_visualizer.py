# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-based visualizer using Isaac Sim viewport."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from pxr import UsdGeom

from .kit_visualizer_cfg import KitVisualizerCfg
from .visualizer import Visualizer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaclab.sim.scene_data_providers import SceneDataProvider


class KitVisualizer(Visualizer):
    """Kit visualizer using Isaac Sim viewport."""

    def __init__(self, cfg: KitVisualizerCfg):
        super().__init__(cfg)
        self.cfg: KitVisualizerCfg = cfg

        self._simulation_app = None
        self._viewport_window = None
        self._viewport_api = None
        self._is_initialized = False
        self._sim_time = 0.0
        self._step_counter = 0
        self._hidden_env_visibilities: dict[str, str] = {}

    # ---- Lifecycle ------------------------------------------------------------------------

    def initialize(self, scene_data_provider: SceneDataProvider) -> None:
        if self._is_initialized:
            logger.debug("[KitVisualizer] initialize() called while already initialized.")
            return

        if scene_data_provider is None:
            raise RuntimeError("[KitVisualizer] Requires a scene_data_provider.")
        self._scene_data_provider = scene_data_provider
        usd_stage = scene_data_provider.get_usd_stage()
        if usd_stage is None:
            raise RuntimeError("[KitVisualizer] USD stage not available from scene_data_provider.")
        metadata = scene_data_provider.get_metadata()

        self._ensure_simulation_app()
        self._setup_viewport(usd_stage)

        self._env_ids = self._compute_visualized_env_ids()
        if self._env_ids:
            logger.warning(
                "[KitVisualizer] env_filter_ids filtering is cosmetic only (no perf gain) in OV; hiding other envs."
            )
            self._apply_env_visibility(usd_stage, metadata)
        cam_pos = self.cfg.camera_position
        cam_target = self.cfg.camera_target
        logger.info("[KitVisualizer] initialized | camera_pos=%s camera_target=%s", cam_pos, cam_target)

        self._is_initialized = True

    def step(self, dt: float) -> None:
        if not self._is_initialized:
            return
        self._sim_time += dt
        self._step_counter += 1
        try:
            import omni.kit.app

            from isaaclab.app.settings_manager import get_settings_manager

            app = omni.kit.app.get_app()
            if app is not None and app.is_running():
                settings = get_settings_manager()
                settings.set_bool("/app/player/playSimulations", False)
                app.update()
                settings.set_bool("/app/player/playSimulations", True)
        except (ImportError, AttributeError) as exc:
            logger.debug("[KitVisualizer] App update skipped: %s", exc)

    def close(self) -> None:
        if not self._is_initialized:
            return
        self._restore_env_visibility()
        self._simulation_app = None
        self._viewport_window = None
        self._viewport_api = None
        self._is_initialized = False
        self._is_closed = True

    # ---- Capabilities ---------------------------------------------------------------------

    def is_running(self) -> bool:
        if self._simulation_app is not None:
            return self._simulation_app.is_running()
        try:
            import omni.kit.app

            app = omni.kit.app.get_app()
            return app is not None and app.is_running()
        except (ImportError, AttributeError):
            return False

    def is_training_paused(self) -> bool:
        return False

    def supports_markers(self) -> bool:
        return True

    def supports_live_plots(self) -> bool:
        return True

    def requires_forward_before_step(self) -> bool:
        """OV viewport relies on refreshed kinematic state before render."""
        return True

    def pumps_app_update(self) -> bool:
        """KitVisualizer calls app.update() in step(), so render() should not do it again."""
        return True

    def set_camera_view(
        self, eye: tuple[float, float, float] | list[float], target: tuple[float, float, float] | list[float]
    ) -> None:
        if not self._is_initialized:
            logger.debug("[KitVisualizer] set_camera_view() ignored because visualizer is not initialized.")
            return
        self._set_viewport_camera(tuple(eye), tuple(target))

    # ---- Viewport + camera ----------------------------------------------------------------

    def _ensure_simulation_app(self) -> None:
        import omni.kit.app

        app = omni.kit.app.get_app()
        if app is None or not app.is_running():
            raise RuntimeError("[KitVisualizer] Isaac Sim app is not running.")

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
                    logger.warning("[KitVisualizer] Running in headless mode. Viewport may not display.")
        except ImportError:
            pass

    def _setup_viewport(self, usd_stage) -> None:
        import omni.kit.viewport.utility as vp_utils
        from omni.ui import DockPosition

        if self.cfg.create_viewport and self.cfg.viewport_name:
            dock_position_name = self.cfg.dock_position.upper()
            dock_position_map = {
                "LEFT": DockPosition.LEFT,
                "RIGHT": DockPosition.RIGHT,
                "BOTTOM": DockPosition.BOTTOM,
                "SAME": DockPosition.SAME,
            }
            dock_pos = dock_position_map.get(dock_position_name, DockPosition.SAME)

            self._viewport_window = vp_utils.create_viewport_window(
                name=self.cfg.viewport_name,
                width=self.cfg.window_width,
                height=self.cfg.window_height,
                position_x=50,
                position_y=50,
                docked=True,
            )

            asyncio.ensure_future(self._dock_viewport_async(self.cfg.viewport_name, dock_pos))
            self._create_and_assign_camera(usd_stage)
        else:
            if self.cfg.viewport_name:
                self._viewport_window = vp_utils.get_viewport_window_by_name(self.cfg.viewport_name)
                if self._viewport_window is None:
                    logger.warning(f"[KitVisualizer] Viewport '{self.cfg.viewport_name}' not found. Using active.")
                    self._viewport_window = vp_utils.get_active_viewport_window()
            else:
                self._viewport_window = vp_utils.get_active_viewport_window()

        self._viewport_api = self._viewport_window.viewport_api
        # TODO: Unify camera initialization with a renderer-level rendering_cfg/camera_cfg
        # so visualizers can consume one canonical camera policy.
        if self.cfg.camera_source == "usd_path":
            if not self._set_active_camera_path(self.cfg.camera_usd_path):
                logger.warning(
                    "[KitVisualizer] camera_usd_path '%s' not found; using configured camera.",
                    self.cfg.camera_usd_path,
                )
                self._set_viewport_camera(self.cfg.camera_position, self.cfg.camera_target)
        else:
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
            logger.warning(f"[KitVisualizer] Could not find viewport window '{viewport_name}'.")
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
        # Create camera prim path based on viewport name (sanitize to ensure valid USD path).
        camera_path = f"/World/Cameras/{self.cfg.viewport_name}_Camera".replace(" ", "_")

        camera_prim = usd_stage.GetPrimAtPath(camera_path)
        if not camera_prim.IsValid():
            UsdGeom.Camera.Define(usd_stage, camera_path)

        if self._viewport_api:
            self._viewport_api.set_active_camera(camera_path)

    def _set_viewport_camera(self, position: tuple[float, float, float], target: tuple[float, float, float]) -> None:
        import isaacsim.core.utils.viewports as isaacsim_viewports

        camera_path = self._viewport_api.get_active_camera()
        if not camera_path:
            camera_path = "/OmniverseKit_Persp"

        isaacsim_viewports.set_camera_view(
            eye=list(position), target=list(target), camera_prim_path=camera_path, viewport_api=self._viewport_api
        )

    def _set_active_camera_path(self, camera_path: str) -> bool:
        if self._viewport_api is None:
            return False
        usd_stage = self._scene_data_provider.get_usd_stage() if self._scene_data_provider else None
        if usd_stage is None:
            return False
        camera_prim = usd_stage.GetPrimAtPath(camera_path)
        if not camera_prim.IsValid():
            return False
        self._viewport_api.set_active_camera(camera_path)
        return True

    def _apply_env_visibility(self, usd_stage, metadata: dict) -> None:
        if not self._env_ids:
            return
        num_envs = int(metadata.get("num_envs", 0))
        if num_envs <= 0:
            return
        visible = set(self._env_ids)
        for env_id in range(num_envs):
            if env_id in visible:
                continue
            env_path = f"/World/envs/env_{env_id}"
            prim = usd_stage.GetPrimAtPath(env_path)
            if not prim.IsValid():
                continue
            imageable = UsdGeom.Imageable(prim)
            if not imageable:
                continue
            attr = imageable.GetVisibilityAttr()
            prev = attr.Get()
            if env_path not in self._hidden_env_visibilities and prev:
                self._hidden_env_visibilities[env_path] = prev
            attr.Set(UsdGeom.Tokens.invisible)

    def _restore_env_visibility(self) -> None:
        if not self._hidden_env_visibilities:
            return
        usd_stage = self._scene_data_provider.get_usd_stage() if self._scene_data_provider else None
        if usd_stage is None:
            return
        for env_path, prev in self._hidden_env_visibilities.items():
            prim = usd_stage.GetPrimAtPath(env_path)
            if not prim.IsValid():
                continue
            imageable = UsdGeom.Imageable(prim)
            if not imageable:
                continue
            imageable.GetVisibilityAttr().Set(prev)
        self._hidden_env_visibilities.clear()
