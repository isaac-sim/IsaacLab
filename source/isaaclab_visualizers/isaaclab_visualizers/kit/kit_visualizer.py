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

from isaaclab.visualizers.base_visualizer import BaseVisualizer

from .kit_visualizer_cfg import KitVisualizerCfg

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaclab.physics import BaseSceneDataProvider


class KitVisualizer(BaseVisualizer):
    """Kit visualizer using Isaac Sim viewport."""

    def __init__(self, cfg: KitVisualizerCfg):
        """Initialize Kit visualizer state.

        Args:
            cfg: Kit visualizer configuration.
        """
        super().__init__(cfg)
        self.cfg: KitVisualizerCfg = cfg

        self._simulation_app = None
        self._viewport_window = None
        self._viewport_api = None
        self._is_initialized = False
        self._sim_time = 0.0
        self._step_counter = 0
        self._hidden_env_visibilities: dict[str, str] = {}
        self._runtime_headless = bool(cfg.headless)

    # ---- Lifecycle ------------------------------------------------------------------------

    def initialize(self, scene_data_provider: BaseSceneDataProvider) -> None:
        """Initialize viewport resources and bind scene data provider.

        Args:
            scene_data_provider: Scene data provider used by the visualizer.
        """
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
        num_visualized_envs = len(self._env_ids) if self._env_ids is not None else int(metadata.get("num_envs", 0))
        self._log_initialization_table(
            logger=logger,
            title="KitVisualizer Configuration",
            rows=[
                ("camera_position", self.cfg.camera_position),
                ("camera_target", self.cfg.camera_target),
                ("camera_source", self.cfg.camera_source),
                ("num_visualized_envs", num_visualized_envs),
                ("create_viewport", self.cfg.create_viewport),
                ("headless", self._runtime_headless),
            ],
        )

        self._is_initialized = True

    def step(self, dt: float) -> None:
        """Advance visualizer/UI updates for one simulation step.

        Args:
            dt: Simulation time-step in seconds.
        """
        if not self._is_initialized:
            return
        self._sim_time += dt
        self._step_counter += 1
        try:
            import omni.kit.app

            from isaaclab.app.settings_manager import get_settings_manager

            app = omni.kit.app.get_app()
            if app is not None and app.is_running():
                # Keep app pumping for viewport/UI updates only.
                # Simulation stepping is owned by SimulationContext.
                settings = get_settings_manager()
                settings.set_bool("/app/player/playSimulations", False)
                app.update()
                settings.set_bool("/app/player/playSimulations", True)
        except (ImportError, AttributeError) as exc:
            logger.debug("[KitVisualizer] App update skipped: %s", exc)

    def close(self) -> None:
        """Close viewport resources and restore temporary state."""
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
        """Return whether Kit app/runtime is still running.

        Returns:
            ``True`` when the visualizer can continue stepping, otherwise ``False``.
        """
        if self._simulation_app is not None:
            return self._simulation_app.is_running()
        try:
            import omni.kit.app

            app = omni.kit.app.get_app()
            return app is not None and app.is_running()
        except (ImportError, AttributeError):
            return False

    def is_training_paused(self) -> bool:
        """Return whether simulation play flag is paused in Kit settings."""
        try:
            from isaaclab.app.settings_manager import get_settings_manager

            settings = get_settings_manager()
            play_flag = settings.get("/app/player/playSimulations")
            return play_flag is False
        except Exception:
            return False

    def supports_markers(self) -> bool:
        """Kit viewport supports marker visualization through Omni UI rendering."""
        return True

    def supports_live_plots(self) -> bool:
        """Kit backend can host live plot widgets via viewport UI panels."""
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
        """Set active viewport camera eye/target.

        Args:
            eye: Camera eye position.
            target: Camera look-at target.
        """
        if not self._is_initialized:
            logger.debug("[KitVisualizer] set_camera_view() ignored because visualizer is not initialized.")
            return
        self._set_viewport_camera(tuple(eye), tuple(target))

    # ---- Viewport + camera ----------------------------------------------------------------

    def _ensure_simulation_app(self) -> None:
        """Ensure a running Isaac Sim app is available and cache runtime mode."""
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
                self._runtime_headless = bool(self.cfg.headless or self._simulation_app.config.get("headless", False))
                if self._runtime_headless:
                    logger.warning("[KitVisualizer] Running in headless mode. Viewport may not display.")
        except ImportError:
            pass

    def _setup_viewport(self, usd_stage) -> None:
        """Create/resolve viewport and configure initial camera.

        Args:
            usd_stage: USD stage used for camera prim setup.
        """
        import omni.kit.viewport.utility as vp_utils
        from omni.ui import DockPosition

        if self._runtime_headless:
            # In headless mode we keep the visualizer active but skip viewport/window setup.
            self._viewport_window = None
            self._viewport_api = None
            return

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
            self._viewport_window = vp_utils.get_active_viewport_window()

        if self._viewport_window is None:
            logger.warning("[KitVisualizer] No active viewport window found.")
            self._viewport_api = None
            return
        self._viewport_api = self._viewport_window.viewport_api
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
        """Dock a created viewport window relative to main viewport."""
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
        """Create viewport camera prim (if needed) and set it active."""
        camera_path = f"/World/Cameras/{self.cfg.viewport_name}_Camera".replace(" ", "_")

        camera_prim = usd_stage.GetPrimAtPath(camera_path)
        if not camera_prim.IsValid():
            UsdGeom.Camera.Define(usd_stage, camera_path)

        if self._viewport_api:
            self._viewport_api.set_active_camera(camera_path)

    def _set_viewport_camera(self, position: tuple[float, float, float], target: tuple[float, float, float]) -> None:
        """Apply eye/target camera view to the active viewport."""
        import isaacsim.core.utils.viewports as isaacsim_viewports

        if self._viewport_api is None:
            return
        camera_path = self._viewport_api.get_active_camera()
        if not camera_path:
            camera_path = "/OmniverseKit_Persp"

        isaacsim_viewports.set_camera_view(
            eye=list(position), target=list(target), camera_prim_path=camera_path, viewport_api=self._viewport_api
        )

    def _set_active_camera_path(self, camera_path: str) -> bool:
        """Set active camera path for viewport if the prim exists.

        Returns:
            ``True`` if camera was set, otherwise ``False``.
        """
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
        """Hide non-selected environments for cosmetic env filtering."""
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
        """Restore environment visibilities modified by env filtering."""
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
