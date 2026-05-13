# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-based visualizer using Isaac Sim viewport."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from pxr import Gf, Usd, UsdGeom, Vt

from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.visualizers.base_visualizer import BaseVisualizer

from isaaclab_visualizers.newton_adapter import resolve_visible_env_indices

from .kit_visualizer_cfg import KitVisualizerCfg

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaclab.physics import BaseSceneDataProvider

_DEFAULT_VIEWPORT_NAME = "Visualizer Viewport"


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
        # PointInstancer prim path -> (had authored invisibleIds, previous value) for partial viz restore.
        self._point_instancer_invisible_ids_backup: dict[str, tuple[bool, object]] = {}
        self._runtime_headless = bool(cfg.headless)
        # USD path for the viewport's active camera, refreshed after setup (used by CI/tests).
        self._controlled_camera_path: str | None = None

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
        self._setup_viewport()

        self._env_ids = self._compute_visualized_env_ids()
        num_envs_meta = int(metadata.get("num_envs", 0))
        self._resolved_visible_env_ids = resolve_visible_env_indices(
            self._env_ids, self.cfg.max_visible_envs, num_envs_meta
        )
        if self._resolved_visible_env_ids is not None:
            logger.warning(
                "[KitVisualizer] Partial visualization in Kit uses visibility only; unselected env prims are hidden."
            )
            self._apply_env_visibility(usd_stage, metadata, self._resolved_visible_env_ids)
        num_visualized_envs = (
            len(self._resolved_visible_env_ids) if self._resolved_visible_env_ids is not None else num_envs_meta
        )
        self._log_initialization_table(
            logger=logger,
            title="KitVisualizer Configuration",
            rows=[
                ("eye", self.cfg.eye),
                ("lookat", self.cfg.lookat),
                ("cam_source", self.cfg.cam_source),
                ("max_visible_envs", self.cfg.max_visible_envs),
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

            app = omni.kit.app.get_app()
            if app is not None and app.is_running():
                # Keep app pumping for viewport/UI updates only; physics is owned by SimulationContext.
                # Disable playSimulations around app.update() so Kit does not advance its own physics here.
                settings = get_settings_manager()
                settings.set_bool("/app/player/playSimulations", False)
                app.update()
                settings.set_bool("/app/player/playSimulations", True)
        except (ImportError, AttributeError) as exc:
            logger.debug("[KitVisualizer] App update skipped: %s", exc)
        # Markers (VisualizationMarkers) are often created or resized to num_envs only after the first
        # simulation / debug-vis step; re-apply PointInstancer invisibleIds each step when partial viz is on.
        self._refresh_partial_viz_point_instancers_if_needed()

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
            settings = get_settings_manager()
            play_flag = settings.get("/app/player/playSimulations")
            return play_flag is False
        except Exception:
            return False

    def supports_markers(self) -> bool:
        """Kit viewport supports marker visualization through Omni UI rendering."""
        return bool(self.cfg.enable_markers)

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

        When :attr:`self.cfg.cam_source` is ``"cfg"``, this is a no-op: the pose comes only from
        :attr:`self.cfg.eye` / :attr:`self.cfg.lookat` (applied in :meth:`_setup_viewport`). Otherwise
        :class:`~isaaclab.sim.simulation_context.SimulationContext` and :class:`ViewportCameraController`
        would overwrite that pose with :class:`~isaaclab.envs.common.ViewerCfg`-driven views.

        Args:
            eye: Camera eye position.
            target: Camera look-at target.
        """
        if self.cfg.cam_source == "cfg":
            return
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

    def _setup_viewport(self) -> None:
        """Create/resolve viewport and configure initial camera."""
        import omni.kit.viewport.utility as vp_utils
        from omni.ui import DockPosition

        if self._runtime_headless:
            # Headless: no viewport window; apply cfg pose to the default perspective camera path.
            self._viewport_window = None
            self._viewport_api = None
            if self.cfg.cam_source == "prim_path":
                logger.warning(
                    "[KitVisualizer] cam_source='prim_path' has limited support in headless mode; "
                    "using eye/lookat from cfg instead."
                )
            self._apply_cfg_camera_pose_if_configured()
            self._refresh_controlled_camera_path()
            return

        effective_viewport_name = (
            self.cfg.viewport_name if self.cfg.viewport_name is not None else _DEFAULT_VIEWPORT_NAME
        )

        if self.cfg.create_viewport:
            if not str(effective_viewport_name).strip():
                raise RuntimeError(
                    "[KitVisualizer] viewport_name must be a non-empty string when create_viewport=True."
                )
            dock_position_name = self.cfg.dock_position.upper()
            dock_position_map = {
                "LEFT": DockPosition.LEFT,
                "RIGHT": DockPosition.RIGHT,
                "BOTTOM": DockPosition.BOTTOM,
                "SAME": DockPosition.SAME,
            }
            dock_pos = dock_position_map.get(dock_position_name, DockPosition.SAME)

            self._viewport_window = vp_utils.create_viewport_window(
                name=effective_viewport_name,
                width=self.cfg.window_width,
                height=self.cfg.window_height,
                position_x=50,
                position_y=50,
                docked=True,
            )

            asyncio.ensure_future(self._dock_viewport_async(effective_viewport_name, dock_pos))
        else:
            self._viewport_window = vp_utils.get_active_viewport_window()

        if self._viewport_window is None:
            logger.warning("[KitVisualizer] No active viewport window found.")
            self._viewport_api = None
            self._refresh_controlled_camera_path()
            return
        self._viewport_api = self._viewport_window.viewport_api
        if self.cfg.cam_source == "prim_path":
            if not self._set_active_camera_path(self.cfg.cam_prim_path):
                raise RuntimeError(
                    "[KitVisualizer] cam_source='prim_path' requires a valid cam_prim_path. "
                    f"Camera prim not found: '{self.cfg.cam_prim_path}'."
                )
        else:
            self._apply_cfg_camera_pose_if_configured()
        self._refresh_controlled_camera_path()

    def _refresh_controlled_camera_path(self) -> None:
        """Cache :attr:`_controlled_camera_path` from the active viewport (or default persp)."""
        if self._viewport_api is not None:
            path = self._viewport_api.get_active_camera()
            self._controlled_camera_path = path if path else "/OmniverseKit_Persp"
        else:
            self._controlled_camera_path = "/OmniverseKit_Persp"

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

    def _set_viewport_camera(self, position: tuple[float, float, float], target: tuple[float, float, float]) -> None:
        """Apply eye/target camera view to the active viewport."""
        if self._viewport_api is None:
            return

        try:
            from omni.kit.viewport.utility.camera_state import ViewportCameraState
        except ImportError as exc:
            logger.warning("[KitVisualizer] Viewport camera update skipped: %s", exc)
            return

        camera_path = self._viewport_api.get_active_camera()
        if not camera_path:
            camera_path = "/OmniverseKit_Persp"

        camera_state = ViewportCameraState(camera_path, self._viewport_api)
        camera_state.set_position_world(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])), True)
        camera_state.set_target_world(Gf.Vec3d(float(target[0]), float(target[1]), float(target[2])), True)

    def _apply_cfg_camera_pose_if_configured(self) -> None:
        """Apply configured camera pose from eye/lookat."""
        self._set_viewport_camera(self.cfg.eye, self.cfg.lookat)

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

    def _apply_env_visibility(self, usd_stage, metadata: dict, visible_env_ids: list[int]) -> None:
        """Hide environments not listed in ``visible_env_ids`` (cosmetic partial visualization)."""
        num_envs = int(metadata.get("num_envs", 0))
        if num_envs <= 0:
            return
        visible = set(visible_env_ids)
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

        self._apply_visual_point_instancer_visibility(usd_stage, num_envs, visible)

    def _refresh_partial_viz_point_instancers_if_needed(self) -> None:
        """Re-apply ``invisibleIds`` for env-scaled `/Visuals` instancers (handles lazy marker creation)."""
        if self._resolved_visible_env_ids is None or self._scene_data_provider is None:
            return
        usd_stage = self._scene_data_provider.get_usd_stage()
        if usd_stage is None:
            return
        num_envs = int(self._scene_data_provider.get_metadata().get("num_envs", 0))
        if num_envs <= 0:
            return
        self._apply_visual_point_instancer_visibility(usd_stage, num_envs, set(self._resolved_visible_env_ids))

    def _apply_visual_point_instancer_visibility(self, usd_stage, num_envs: int, visible_env_ids: set[int]) -> None:
        """Set ``PointInstancer.invisibleIds`` for per-env `/Visuals` markers (e.g. velocity arrows)."""
        hidden = [i for i in range(num_envs) if i not in visible_env_ids]
        vt_hidden = Vt.Int64Array([int(i) for i in hidden])
        for root_path in ("/Visuals", "/World/Visuals"):
            root_prim = usd_stage.GetPrimAtPath(root_path)
            if not root_prim.IsValid():
                continue
            for prim in Usd.PrimRange(root_prim):
                if not prim.IsA(UsdGeom.PointInstancer):
                    continue
                pi = UsdGeom.PointInstancer(prim)
                n = self._point_instancer_instance_count(pi)
                if n is None or n != num_envs:
                    continue
                path_str = prim.GetPath().pathString
                inv_attr = pi.GetInvisibleIdsAttr()
                # Record original authorship/value once per instancer for :meth:`_restore_env_visibility`.
                if path_str not in self._point_instancer_invisible_ids_backup:
                    was_authored = inv_attr.HasAuthoredValue()
                    prev = inv_attr.Get() if was_authored else None
                    self._point_instancer_invisible_ids_backup[path_str] = (was_authored, prev)
                inv_attr.Set(vt_hidden)

    @staticmethod
    def _point_instancer_instance_count(pi: UsdGeom.PointInstancer) -> int | None:
        """Return instance count from the first authored per-instance array, if any."""
        for attr in (
            pi.GetPositionsAttr(),
            pi.GetScalesAttr(),
            pi.GetOrientationsAttr(),
            pi.GetProtoIndicesAttr(),
        ):
            if not attr.HasAuthoredValue():
                continue
            val = attr.Get()
            if val is None:
                continue
            return len(val)
        return None

    def _restore_env_visibility(self) -> None:
        """Restore environment visibilities and PointInstancer ``invisibleIds`` from partial viz."""
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

        for path_str, (was_authored, prev) in self._point_instancer_invisible_ids_backup.items():
            prim = usd_stage.GetPrimAtPath(path_str)
            if not prim.IsValid() or not prim.IsA(UsdGeom.PointInstancer):
                continue
            inv_attr = UsdGeom.PointInstancer(prim).GetInvisibleIdsAttr()
            if not was_authored:
                inv_attr.Clear()
            else:
                inv_attr.Set(prev)
        self._point_instancer_invisible_ids_backup.clear()
