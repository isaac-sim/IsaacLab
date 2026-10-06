# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-based visualizer using Isaac Sim viewport."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix
from isaaclab.utils.renderers import ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING
from isaaclab.visualizers.base_visualizer import BaseVisualizer

from isaaclab_visualizers.desktop_entry import write_desktop_entry
from isaaclab_visualizers.newton_adapter import resolve_visible_env_indices

from .kit_visualizer_cfg import KitVisualizerCfg

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaclab.cloner import ClonePlan
    from isaaclab.scene_data import SceneDataProvider

_DEFAULT_VIEWPORT_NAME = "Visualizer Viewport"
_DEFAULT_VIEWPORT_CAMERA_PATH = "/OmniverseKit_Persp"

_BACKEND_DISPLAY_NAMES = {
    "physx": "PhysX",
    "ovphysx": "OVPhysX",
    "newton": "Newton MJWarp",
}


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
        self._step_counter = 0
        self._env_ids = None
        self._resolved_visible_env_ids: list[int] | None = None
        self._hidden_env_visibilities: dict[str, str] = {}
        # PointInstancer prim path -> (had authored invisibleIds, previous value) for partial viz restore.
        self._point_instancer_invisible_ids_backup: dict[str, tuple[bool, object]] = {}
        self._runtime_headless = bool(cfg.headless)
        # USD path for the viewport's active camera, refreshed after setup (used by CI/tests).
        self._controlled_camera_path: str | None = None
        # Lazy Replicator render product + annotator for render_rgb_array().
        self._rgb_render_product = None
        self._rgb_annotator = None
        self._viewport_camera_xform_ops: dict[str, tuple[UsdGeom.XformOp, UsdGeom.XformOp]] = {}
        self._viewport_camera_pose_cache: dict[str, tuple[float, ...]] = {}
        self._camera_image_provider = None
        self._camera_image_window = None
        self._backend_menubar_label = None
        self._hid_simulation_menu = False
        # Guard flag: True once app.update() has been called in the current step() invocation.
        # render_rgb_array() skips its own pump when the step already pumped the app.
        self._app_pumped_this_step: bool = False
        # Camera tracking state (replaces ViewportCameraController)
        self._interactive_scene = None  # set from SimulationContext._interactive_scene in initialize()
        self._viewer_origin: torch.Tensor | None = None  # world-space origin offset for eye/lookat

    # ---- Lifecycle ------------------------------------------------------------------------

    @property
    def visual_material_writer(self):
        """Write material channels directly through Fabric."""
        from isaaclab_physx.renderers.visual_material import FabricVisualMaterialWriter

        return FabricVisualMaterialWriter

    def initialize(
        self,
        scene_data_provider: SceneDataProvider,
        *,
        stage: Usd.Stage | None = None,
        clone_plan: ClonePlan | None = None,
    ) -> None:
        """Initialize viewport resources and bind scene data provider.

        Args:
            scene_data_provider: Scene data provider used by the visualizer.
            stage: Authored scene stage, when available.
            clone_plan: Scene topology and environment namespace, when available.
        """
        if self._is_initialized:
            logger.debug("[KitVisualizer] initialize() called while already initialized.")
            return

        super().initialize(scene_data_provider, stage=stage, clone_plan=clone_plan)
        usd_stage = self._scene_stage
        num_envs = scene_data_provider.num_envs

        self._ensure_simulation_app()
        self._setup_viewport()
        if self._viewport_api is not None:
            self._apply_render_product_background(usd_stage, self._viewport_api.render_product_path)

        self._env_ids = self._compute_visualized_env_ids()
        self._resolved_visible_env_ids = resolve_visible_env_indices(self._env_ids, self.cfg.max_visible_envs, num_envs)
        if self._resolved_visible_env_ids is not None:
            logger.warning(
                "[KitVisualizer] Partial visualization in Kit uses visibility only; unselected env prims are hidden."
            )
            self._apply_env_visibility(usd_stage, num_envs, self._resolved_visible_env_ids)
        self._apply_viewport_camera_scene_partition(usd_stage, num_envs)
        num_visualized_envs = (
            len(self._resolved_visible_env_ids) if self._resolved_visible_env_ids is not None else num_envs
        )
        self._log_initialization_table(
            logger=logger,
            title="KitVisualizer Configuration",
            rows=[
                ("eye", self.cfg.eye),
                ("lookat", self.cfg.lookat),
                ("background_color", self.cfg.background_color),
                ("streaming_view", self.cfg.streaming_view),
                ("streaming_gt_types", list(self.cfg.streaming_gt_types)),
                ("max_visible_envs", self.cfg.max_visible_envs),
                ("num_visualized_envs", num_visualized_envs),
                ("create_viewport", self.cfg.create_viewport),
                ("headless", self._runtime_headless),
            ],
        )
        self._setup_streaming_view(num_envs)

        sim = SimulationContext.instance()
        from isaaclab_physx.renderers.fabric import FabricBackendCfg  # noqa: PLC0415 - requires Kit

        self._fabric = sim.get_or_create_backend(FabricBackendCfg(stage=sim.stage, device=sim.device))
        self._fabric.bind_transforms(scene_data_provider)
        self._is_initialized = True
        self._setup_initial_camera_view()

    def step(self, dt: float) -> None:
        """Advance visualizer/UI updates for one simulation step.

        Args:
            dt: Simulation time-step in seconds.
        """
        if not self._is_initialized:
            return
        self._app_pumped_this_step = False
        self._sim_time += dt
        self._step_counter += 1
        # Headless mode: skip the app update and camera panel refresh; rendering is
        # triggered on demand by render_rgb_array() / render_tiled_rgb_array().
        if self._runtime_headless:
            return
        self._fabric.update_transforms(self._scene_data_provider)
        self._fabric.update_geometries(self._scene_data_provider, SimulationContext.instance().render_generation)
        if self.cfg.origin_type == "asset":
            self._update_asset_tracking_camera()
        _externally_paused = self.is_training_paused()
        if not _externally_paused:
            try:
                import omni.kit.app

                app = omni.kit.app.get_app()
                if app is not None and app.is_running():
                    # Keep app pumping for viewport/UI updates only; physics is owned by SimulationContext.
                    # Disable playSimulations around app.update() so Kit does not advance its own physics here.
                    settings = get_settings_manager()
                    settings.set("/app/player/playSimulations", False)
                    app.update()
                    settings.set("/app/player/playSimulations", True)
                    self._app_pumped_this_step = True
            except (ImportError, AttributeError) as exc:
                logger.debug("[KitVisualizer] App update skipped: %s", exc)
        self._update_camera_image_panel()
        # Markers (VisualizationMarkers) are often created or resized to num_envs only after the first
        # simulation / debug-vis step; re-apply PointInstancer invisibleIds each step when partial viz is on.
        self._refresh_partial_viz_point_instancers_if_needed()

    def close(self) -> None:
        """Close viewport resources and restore temporary state."""
        if not self._is_initialized:
            return
        self._teardown_backend_menubar_label()
        self._restore_env_visibility()
        self._viewport_camera_xform_ops.clear()
        self._viewport_camera_pose_cache.clear()
        self._camera_image_provider = None
        self._camera_image_window = None
        self._simulation_app = None
        self._viewport_window = None
        self._viewport_api = None
        import contextlib

        if self._rgb_annotator is not None:
            with contextlib.suppress(Exception):
                self._rgb_annotator.detach()
        if self._rgb_render_product is not None:
            with contextlib.suppress(Exception):
                self._rgb_render_product.destroy()
        self._rgb_annotator = None
        self._rgb_render_product = None
        self._is_initialized = False
        super().close()

    def render_rgb_array(self) -> np.ndarray:
        """Return an RGB frame captured from the Kit viewport camera.

        Uses the Replicator annotator bound to the controlled camera prim
        (``/OmniverseKit_Persp`` by default). Lazily creates the render product
        and annotator on the first call. Returns a blank frame while the RTX
        pipeline warms up.

        Returns:
            RGB image array of shape ``(window_height, window_width, 3)``, dtype ``uint8``.
        """
        import omni.kit.app
        import omni.replicator.core as rep

        self._fabric.update_transforms(self._scene_data_provider)
        self._fabric.update_geometries(self._scene_data_provider, SimulationContext.instance().render_generation)
        if self._runtime_headless and self.cfg.origin_type == "asset":
            self._update_asset_tracking_camera()
        camera_path = self._controlled_camera_path or "/OmniverseKit_Persp"
        w, h = self.cfg.window_width, self.cfg.window_height

        # Create the render product and annotator before the app update so the first
        # captured frame contains real rendered output, not empty/blank data.
        if self._rgb_annotator is None:
            self._rgb_render_product = rep.create.render_product(camera_path, (w, h))
            self._apply_render_product_background(self._scene_stage, self._rgb_render_product.path)
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            self._rgb_annotator.attach([self._rgb_render_product])
        elif self._runtime_headless and self._rgb_render_product is not None:
            # In headless mode the render product is paused between captures (see below).
            # Resume it now so RTX can produce a fresh frame before we read the annotator.
            import contextlib

            with contextlib.suppress(Exception):
                self._rgb_render_product.resume()

        if not self._app_pumped_this_step:
            settings = get_settings_manager()
            play_flag = settings.get("/app/player/playSimulations")
            settings.set("/app/player/playSimulations", False)
            omni.kit.app.get_app().update()
            settings.set("/app/player/playSimulations", bool(play_flag))

        raw = self._rgb_annotator.get_data()
        if isinstance(raw, dict):
            raw = raw.get("data", np.array([], dtype=np.uint8))
        raw = np.asarray(raw, dtype=np.uint8)
        if raw.size == 0:
            return np.zeros((h, w, 3), dtype=np.uint8)
        if raw.ndim == 1:
            raw = raw.reshape(h, w, -1)
        result = raw[:, :, :3]

        # Headless on-demand: pause RTX after the frame is captured so it does not
        # render between recording windows.  resume() is called at the top of the
        # next render_rgb_array() call.
        if self._runtime_headless and self._rgb_render_product is not None:
            import contextlib

            with contextlib.suppress(Exception):
                self._rgb_render_product.pause()

        return result

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
        """Kit backend hosts live plot widgets via :class:`~isaaclab.ui.widgets.ManagerLiveVisualizer`."""
        return True

    def add_live_plots(
        self,
        managers: dict,
        scalars: dict | None = None,
        term_names: dict[str, list[str]] | None = None,
        env_idx: int = 0,
    ) -> None:
        """Register managers for live plotting using the Kit omni.ui widget path.

        Creates a :class:`~isaaclab.ui.widgets.ManagerLiveVisualizer` per manager and stores
        them in :attr:`kit_manager_visualizers` so that :class:`~isaaclab.envs.ui.BaseEnvWindow`
        can wire them into the viewport panel.  Also calls the base implementation to populate
        :attr:`_live_plot_sources` for any non-omni.ui consumers.

        Note:
            Scalar groups (e.g. episode metrics) are stored in :attr:`_live_plot_sources` via
            the base implementation but are not yet wired into the omni.ui viewport panel.

        Args:
            managers: Mapping of manager name to manager instance.
            scalars: Optional mapping of group name to a dict of ``{term_name: callable}``.
                Each callable must take no arguments and return a numeric value.
            term_names: Optional per-manager allowlists of term names to include.
            env_idx: Environment index to sample each step.  Defaults to ``0``.
        """
        super().add_live_plots(managers, scalars=scalars, term_names=term_names, env_idx=env_idx)
        from isaaclab.ui.live_plots.manager_live_plots import DirectScalarLivePlots
        from isaaclab.ui.widgets.manager_live_visualizer import (
            DirectScalarLiveVisualizer,
            ManagerLiveVisualizer,
            ManagerLiveVisualizerCfg,
        )

        self.kit_manager_visualizers: dict[str, ManagerLiveVisualizer | DirectScalarLiveVisualizer] = {
            name: ManagerLiveVisualizer(
                manager=mgr,
                cfg=ManagerLiveVisualizerCfg(
                    manager_name=name,
                    term_names=(term_names or {}).get(name),
                ),
            )
            for name, mgr in managers.items()
        }
        # Wire scalar groups (e.g. episode metrics) into the Kit UI panel.
        for source in self._live_plot_sources:
            if isinstance(source, DirectScalarLivePlots):
                self.kit_manager_visualizers[source.manager_name] = DirectScalarLiveVisualizer(source)

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

    def reapply_origin(self) -> None:
        """Recompute the camera position from the current :attr:`~KitVisualizerCfg.origin_type` and push it to
        the viewport.

        Call this after mutating :attr:`cfg.origin_type`, :attr:`cfg.origin_env_index`, or
        :attr:`cfg.origin_track_path` so the viewport reflects the new origin immediately rather than
        waiting for the next :meth:`step` call.

        For ``"asset"`` origins the camera update is deferred to the
        next :meth:`step` because asset state is not available until after
        :meth:`~isaaclab.sim.SimulationContext.reset`.
        """
        self._setup_initial_camera_view()

    @property
    def viewer_origin(self) -> torch.Tensor | None:
        """Current world-space origin offset applied to :attr:`~KitVisualizerCfg.eye` and
        :attr:`~KitVisualizerCfg.lookat` when computing the absolute camera position.

        Returns ``None`` before :meth:`initialize` is called or when no valid origin has been
        established yet (e.g. asset-tracking before the first :meth:`step`).
        """
        return self._viewer_origin

    # ---- Viewport + camera ----------------------------------------------------------------

    def _setup_backend_menubar_label(self) -> None:
        """Add a read-only backend label to the viewport menubar and hide the PhysX Simulation menu."""
        try:
            from omni.kit.viewport.menubar.core import IconMenuDelegate, ViewportMenuItem, get_menu_item
        except (ImportError, ModuleNotFoundError):
            return

        backend = self.physics_backend or "unknown"
        backend_display = _BACKEND_DISPLAY_NAMES.get(backend, backend)

        # Hide the "Simulation / PhysX" toggle menu — it only reflects the omni.physics.core
        # registry (always "PhysX") and is misleading when Newton MJWarp is active.
        if backend not in ("physx", "ovphysx"):
            sim_item = get_menu_item("Simulation")
            if sim_item is not None:
                sim_item.visible_model.set_value(False)
                self._hid_simulation_menu = True

        # Add a non-interactive backend label in the menubar. IconMenuDelegate is used (not
        # LabelMenuDelegate) because it draws the "MenuBar.Item.Background" rectangle that gives
        # other menubar items their styled box/border. width=0 suppresses the icon slot so only
        # the text is shown; has_triangle=False removes the dropdown caret.
        self._backend_menubar_label = ViewportMenuItem(
            f"Physics: {backend_display}",
            delegate=IconMenuDelegate("", text=True, width=0, has_triangle=False, enabled=False),
        )

    async def _setup_backend_menubar_label_async(self) -> None:
        """Defer backend menubar label setup by one app tick.

        Creating a :class:`ViewportMenuItem` synchronously during viewport init triggers an
        ``omni.kit.viewport.menubar.camera`` render-settings notification before Isaac Sim's
        camera collection is ready, producing a spurious ``AttributeError``.  Deferring until
        the next ``next_update_async`` tick lets the collection initialize first.
        """
        import omni.kit.app

        await omni.kit.app.get_app().next_update_async()
        self._setup_backend_menubar_label()

    def _teardown_backend_menubar_label(self) -> None:
        """Remove the backend label and restore the Simulation menu visibility."""
        if self._hid_simulation_menu:
            try:
                from omni.kit.viewport.menubar.core import get_menu_item
            except (ImportError, ModuleNotFoundError):
                self._hid_simulation_menu = False
                return
            sim_item = get_menu_item("Simulation")
            if sim_item is not None:
                sim_item.visible_model.set_value(True)
            self._hid_simulation_menu = False
        if self._backend_menubar_label is not None:
            self._backend_menubar_label.destroy()
            self._backend_menubar_label = None

    def _ensure_simulation_app(self) -> None:
        """Ensure a running Isaac Sim app is available and cache runtime mode."""
        import omni.kit.app

        app = omni.kit.app.get_app()
        if app is None or not app.is_running():
            raise RuntimeError("[KitVisualizer] Isaac Sim app is not running.")

        import os as _os

        headless_env = bool(int(_os.environ.get("HEADLESS", 0)))
        # Apply env var immediately — before sim_app lookup — so the headless guard in
        # _setup_viewport() fires even when the SimulationApp instance is not yet accessible.
        if headless_env and not self._runtime_headless:
            self._runtime_headless = True
            logger.warning("[KitVisualizer] Running in headless mode (HEADLESS=1). Viewport may not display.")

        try:
            from isaacsim import SimulationApp

            sim_app = None
            if hasattr(SimulationApp, "_instance") and SimulationApp._instance is not None:
                sim_app = SimulationApp._instance
            elif hasattr(SimulationApp, "instance") and callable(SimulationApp.instance):
                sim_app = SimulationApp.instance()

            if sim_app is not None:
                self._simulation_app = sim_app
                self._runtime_headless = bool(
                    self.cfg.headless or headless_env or self._simulation_app.config.get("headless", False)
                )
                if self._runtime_headless:
                    logger.warning("[KitVisualizer] Running in headless mode. Viewport may not display.")
        except ImportError:
            pass

    def _apply_render_product_background(self, stage: Usd.Stage, render_product_path: str | Sdf.Path) -> None:
        """Apply the configured solid background to an Isaac RTX render product."""
        if self.cfg.background_color is None:
            return
        render_product = stage.GetPrimAtPath(render_product_path)
        if not render_product.IsValid():
            logger.warning(
                "[KitVisualizer] Render product '%s' was not found; background was not applied.",
                render_product_path,
            )
            return

        with Usd.EditContext(stage, stage.GetSessionLayer()), Sdf.ChangeBlock():
            render_product.CreateAttribute("omni:rtx:background:source:type", Sdf.ValueTypeNames.Token).Set("color")
            render_product.CreateAttribute("omni:rtx:background:source:color", Sdf.ValueTypeNames.Float3).Set(
                Gf.Vec3f(*self.cfg.background_color)
            )

    def _write_desktop_entry(self) -> None:
        """Write the Linux desktop entry that lets docks show the Kit window's icon."""
        import carb.tokens

        settings = get_settings_manager()
        title = settings.get("/app/window/title")
        version = settings.get("/app/version")
        icon_path = settings.get("/app/window/iconPath")
        if title and version and icon_path:
            # Kit composes the window's WM_CLASS from the app title and version, e.g. "Isaac Lab 3.0.0".
            icon = carb.tokens.get_tokens_interface().resolve(icon_path)
            write_desktop_entry("isaaclab", title, f"{title} {version}", icon)

    def _setup_viewport(self) -> None:
        """Create/resolve viewport and configure initial camera."""
        if self._runtime_headless:
            # Headless: no viewport window; apply cfg pose to the default perspective camera path.
            # omni.kit.viewport may not be loaded when the viewport extension is disabled
            # (e.g. HEADLESS=1 without --video), so skip the import entirely.
            self._viewport_window = None
            self._viewport_api = None
            if self._uses_streaming_view():
                logger.debug("[KitVisualizer] Camera image view requested in headless mode; no UI panel is created.")
            else:
                self._apply_cfg_camera_pose_if_configured()
            self._refresh_controlled_camera_path()
            return

        self._write_desktop_entry()
        import omni.kit.viewport.utility as vp_utils
        from omni.ui import DockPosition

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
            if not self._uses_streaming_view():
                self._apply_cfg_camera_pose_if_configured()
            self._refresh_controlled_camera_path()
            return
        self._viewport_api = self._viewport_window.viewport_api
        if self._uses_streaming_view():
            # Camera sensor image views are shown in a non-interactive image panel.
            pass
        else:
            self._apply_cfg_camera_pose_if_configured()
        self._refresh_controlled_camera_path()
        asyncio.ensure_future(self._setup_backend_menubar_label_async())

    def _uses_streaming_view(self) -> bool:
        """Return whether Kit should display a streaming camera image panel."""
        return bool(self.cfg.streaming_view)

    def _setup_streaming_view(self, num_envs: int) -> None:
        """Bind a scene camera and create its Kit display panel."""
        super()._setup_streaming_view(
            num_envs,
            visible_env_ids=self._resolved_visible_env_ids,
            target_aspect=self.cfg.window_width / self.cfg.window_height,
        )
        if self._camera_sensor is not None and not self._runtime_headless:
            self._setup_camera_image_window()

    def _setup_camera_image_window(self) -> None:
        """Create a dockable Kit UI image panel for streaming camera output."""
        import omni.ui

        title = self.cfg.viewport_name or "Streaming View"
        self._camera_image_provider = omni.ui.ByteImageProvider()
        self._camera_image_window = omni.ui.Window(title, width=self.cfg.window_width, height=self.cfg.window_height)
        with self._camera_image_window.frame:
            omni.ui.ImageWithProvider(self._camera_image_provider)

        dock_position_name = self.cfg.dock_position.upper()
        dock_position_map = {
            "LEFT": omni.ui.DockPosition.LEFT,
            "RIGHT": omni.ui.DockPosition.RIGHT,
            "BOTTOM": omni.ui.DockPosition.BOTTOM,
            "SAME": omni.ui.DockPosition.SAME,
        }
        asyncio.ensure_future(
            self._dock_image_window_async(title, dock_position_map.get(dock_position_name, omni.ui.DockPosition.SAME))
        )

    async def _dock_image_window_async(self, window_name: str, dock_position) -> None:
        """Dock the camera image panel next to the main viewport."""
        import omni.kit.app
        import omni.ui

        image_window = None
        for _ in range(10):
            image_window = omni.ui.Workspace.get_window(window_name)
            if image_window:
                break
            await omni.kit.app.get_app().next_update_async()
        main_viewport = omni.ui.Workspace.get_window("Viewport")
        if image_window is not None and main_viewport is not None and image_window != main_viewport:
            image_window.dock_in(main_viewport, dock_position, 0.5)

    def _update_camera_image_panel(self) -> None:
        """Present device pixels; CPU images are uploaded only for CPU-backed sources."""
        image = self._streaming_frame.data if self.is_training_paused() else self.render_tiled_rgba()
        if image is None or self._camera_image_provider is None:
            return
        height, width = image.shape[:2]
        if image.device.is_cuda:
            import omni.gpu_foundation_factory as gf

            self._camera_image_provider.set_bytes_data_from_gpu(
                image.ptr, [width, height], gf.TextureFormat.RGBA8_UNORM
            )
        else:
            self._camera_image_provider.set_bytes_data(image.numpy().data, [width, height])

    def _refresh_controlled_camera_path(self) -> None:
        """Cache :attr:`_controlled_camera_path` from the active viewport (or default persp)."""
        if self._viewport_api is not None:
            path = self._viewport_api.get_active_camera()
            self._controlled_camera_path = path if path else _DEFAULT_VIEWPORT_CAMERA_PATH
        else:
            self._controlled_camera_path = _DEFAULT_VIEWPORT_CAMERA_PATH

    def _apply_viewport_camera_scene_partition(self, usd_stage: Usd.Stage, num_envs: int) -> None:
        """Configure the viewport camera for partitioned or all-environment viewing.

        RTX scene partitioning culls per-env geometry by the camera's non-primvar
        ``omni:scenePartition`` token. Interactive viewport cameras live outside
        ``/World/envs`` and are created by Kit, so they do not inherit the env-root
        primvar authored by the renderer. When the RTX spectator-view setting is
        enabled, leaving the viewport camera unpartitioned shows all environments.
        Otherwise, the viewport is assigned to the first visible environment.
        """

        if num_envs <= 0 or self._controlled_camera_path is None:
            return

        env_prim = usd_stage.GetPrimAtPath("/World/envs/env_0")
        env_partition_attr = env_prim.GetAttribute("primvars:omni:scenePartition")
        if not env_partition_attr.IsValid() or env_partition_attr.Get() is None:
            return
        camera_prim = usd_stage.GetPrimAtPath(self._controlled_camera_path)
        if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
            logger.debug(
                "[KitVisualizer] Scene partition token skipped for non-camera viewport prim: %s",
                self._controlled_camera_path,
            )
            return

        attr = camera_prim.GetAttribute("omni:scenePartition")
        if get_settings_manager().get(ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING, False):
            if attr.IsValid():
                camera_prim.RemoveProperty("omni:scenePartition")
            logger.debug(
                "[KitVisualizer] Leaving viewport camera '%s' unpartitioned for the all-environment spectator view.",
                self._controlled_camera_path,
            )
            return

        env_id = self._resolved_visible_env_ids[0] if self._resolved_visible_env_ids else 0
        logger.debug(
            "[KitVisualizer] Assigning viewport camera '%s' to scene partition env_%d.",
            self._controlled_camera_path,
            env_id,
        )
        if not attr.IsValid():
            attr = camera_prim.CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token)
        attr.Set(f"env_{env_id}")

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
            # Without a viewport, Kit does not create its default perspective
            # camera, so author it explicitly before render products use it.
            self._set_usd_camera_pose(_DEFAULT_VIEWPORT_CAMERA_PATH, position, target)
            return

        try:
            from omni.kit.viewport.utility.camera_state import ViewportCameraState
        except ImportError as exc:
            logger.warning("[KitVisualizer] Viewport camera update skipped: %s", exc)
            return

        camera_path = self._viewport_api.get_active_camera()
        if not camera_path:
            camera_path = _DEFAULT_VIEWPORT_CAMERA_PATH

        # ``rotate=False`` for the position set: a freshly-opened stage's default
        # ``/OmniverseKit_Persp`` has no authored ``omni:kit:centerOfInterest``,
        # which ``set_position_world(..., rotate=True)`` would feed into
        # ``Matrix4d.Transform`` as ``None`` and crash. The follow-up
        # ``set_target_world(..., rotate=True)`` performs the look-at rotation
        # and authors the COI as a side effect, so the final pose is unchanged.
        camera_state = ViewportCameraState(camera_path, self._viewport_api)
        camera_state.set_position_world(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])), False)
        camera_state.set_target_world(Gf.Vec3d(float(target[0]), float(target[1]), float(target[2])), True)

    def _set_usd_camera_pose(self, camera_path: str, position, target) -> bool:
        """Apply eye/target camera pose directly to a USD camera prim.

        Returns:
            ``True`` when authored values changed, otherwise ``False``.
        """
        # TODO: Remove this USD-side pose path once Fabric-backed camera transforms propagate reliably to Kit.
        usd_stage = self._scene_stage

        eye = torch.as_tensor(position, dtype=torch.float32, device="cpu").reshape(1, 3)
        lookat = torch.as_tensor(target, dtype=torch.float32, device="cpu").reshape(1, 3)
        up_axis = UsdGeom.GetStageUpAxis(usd_stage)
        rotation_matrix = create_rotation_matrix_from_view(eye, lookat, up_axis=up_axis, device="cpu")
        if torch.isnan(rotation_matrix).any():
            raise ValueError("[KitVisualizer] Cannot set camera pose because eye and lookat are degenerate.")
        quat_xyzw = quat_from_matrix(rotation_matrix)[0]
        pose_key = (
            float(eye[0, 0]),
            float(eye[0, 1]),
            float(eye[0, 2]),
            float(quat_xyzw[0]),
            float(quat_xyzw[1]),
            float(quat_xyzw[2]),
            float(quat_xyzw[3]),
        )
        if self._viewport_camera_pose_cache.get(camera_path) == pose_key:
            return False

        if camera_path not in self._viewport_camera_xform_ops:
            camera = UsdGeom.Camera.Define(usd_stage, camera_path)
            camera_xform = UsdGeom.Xformable(camera.GetPrim())
            camera_xform.ClearXformOpOrder()
            # Viewport eyes/targets are world-space.
            # Reset the xform stack so Kit/Fabric sees the authored pose as a world pose.
            camera_xform.SetResetXformStack(True)
            # ClearXformOpOrder removes the ordering metadata but not the prim attributes
            # themselves, so AddTranslateOp/AddOrientOp fail if they already exist on the
            # prim (e.g. for /OmniverseKit_Persp which Isaac Sim pre-populates).
            # Reuse existing attributes when present; add them only when absent.
            prim = camera.GetPrim()
            t_attr = prim.GetAttribute("xformOp:translate")
            translate_op = UsdGeom.XformOp(t_attr) if t_attr else camera_xform.AddTranslateOp()
            o_attr = prim.GetAttribute("xformOp:orient")
            orient_op = UsdGeom.XformOp(o_attr) if o_attr else camera_xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
            camera_xform.SetXformOpOrder([translate_op, orient_op], camera_xform.GetResetXformStack())
            self._viewport_camera_xform_ops[camera_path] = (translate_op, orient_op)
        else:
            translate_op, orient_op = self._viewport_camera_xform_ops[camera_path]

        quat_gf = Gf.Quatd(
            float(quat_xyzw[3]),
            Gf.Vec3d(float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])),
        )

        translate_op.Set(Gf.Vec3d(float(eye[0, 0]), float(eye[0, 1]), float(eye[0, 2])))
        orient_op.Set(quat_gf)
        self._viewport_camera_pose_cache[camera_path] = pose_key
        return True

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
        usd_stage = self._scene_stage
        camera_prim = usd_stage.GetPrimAtPath(camera_path)
        if not camera_prim.IsValid():
            return False
        self._viewport_api.set_active_camera(camera_path)
        return True

    def _apply_env_visibility(self, usd_stage, num_envs: int, visible_env_ids: list[int]) -> None:
        """Hide environments not listed in ``visible_env_ids`` (cosmetic partial visualization)."""
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
        usd_stage = self._scene_stage
        num_envs = self._scene_data_provider.num_envs
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
        usd_stage = self._scene_stage
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

    def _setup_initial_camera_view(self) -> None:
        """Position the viewport camera according to :attr:`KitVisualizerCfg.origin_type`.

        Called once at the end of :meth:`initialize`. For ``"world"`` and ``"env"`` origins the
        camera is positioned immediately. For asset-tracking origins the first update is deferred
        to :meth:`step` because asset state is not yet available at initialization time.
        """
        self._interactive_scene = self._scene_data_provider.get_interactive_scene()

        if self.cfg.origin_type == "world":
            self._viewer_origin = torch.zeros(3)
        elif self.cfg.origin_type == "env":
            scene = self._interactive_scene
            if scene is None:
                logger.warning("[KitVisualizer] origin_type='env' requested but no scene is registered yet.")
                self._viewer_origin = torch.zeros(3)
            else:
                num_envs = scene.num_envs
                if not (0 <= self.cfg.origin_env_index < num_envs):
                    raise ValueError(
                        f"[KitVisualizer] origin_env_index {self.cfg.origin_env_index} is out of range "
                        f"[0, {num_envs - 1}] for origin_type='env'."
                    )
                self._viewer_origin = scene.env_origins[self.cfg.origin_env_index]
        elif self.cfg.origin_type == "asset":
            if self.cfg.origin_track_path is None:
                raise ValueError("[KitVisualizer] origin_type='asset' requires origin_track_path to be set.")
            # Asset data is not available until after sim.reset(); defer to step().
            return
        else:
            logger.warning("[KitVisualizer] Unknown origin_type '%s'; defaulting to world.", self.cfg.origin_type)
            self._viewer_origin = torch.zeros(3)

        self._apply_viewer_origin_to_camera()

    def _update_asset_tracking_camera(self) -> None:
        """Update the viewport camera to track an asset root or body.

        Called before viewport frames when :attr:`KitVisualizerCfg.origin_type` is ``"asset"``.
        Parses :attr:`~KitVisualizerCfg.origin_track_path`: ``"asset_name"`` tracks the root,
        ``"asset_name/body_name"`` tracks a specific body.
        """
        scene = self._interactive_scene
        if scene is None or self.cfg.origin_track_path is None:
            return
        asset_name, _, body_name = self.cfg.origin_track_path.partition("/")
        try:
            asset = scene[asset_name]
        except KeyError:
            return
        if body_name:
            body_ids, _ = asset.find_bodies(body_name)
            self._viewer_origin = asset.data.body_pos_w.torch[self.cfg.origin_env_index, body_ids[0]]
        else:
            self._viewer_origin = asset.data.root_pos_w.torch[self.cfg.origin_env_index]
        self._apply_viewer_origin_to_camera()

    def _apply_viewer_origin_to_camera(self) -> None:
        """Compute absolute eye/target from :attr:`_viewer_origin` and push to the viewport."""
        if self._viewer_origin is None:
            return
        origin = self._viewer_origin.detach().cpu().numpy()
        eye = np.array(self.cfg.eye, dtype=float) + origin
        target = np.array(self.cfg.lookat, dtype=float) + origin
        self.set_camera_view(tuple(float(v) for v in eye), tuple(float(v) for v in target))
        # Keep the Isaac RTX renderer camera in sync (no-op if isaaclab_physx is not installed).
        try:
            from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view  # noqa: PLC0415

            set_kit_renderer_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")
        except (ImportError, ModuleNotFoundError):
            pass
