# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-based visualizer using Isaac Sim viewport."""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.envs.utils.camera_colorizer import (
    SUPPORTED_GT_TYPES,
    CameraFrameColorizer,
    sensor_key_for_gt_type,
    sensor_keys_for_gt_types,
)
from isaaclab.envs.utils.camera_view import (
    VISUALIZER_TILED_CAMERA_MAX_TILES,
    apply_camera_target_positions,
    camera_gt_batch,
    compose_streaming_grid,
    compute_tile_resolution,
    create_visualizer_camera,
    find_camera_by_prim_path,
    prim_world_positions,
    remove_generated_prims,
    resolve_streaming_envs,
)
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix
from isaaclab.utils.renderers import ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING
from isaaclab.visualizers.base_visualizer import BaseVisualizer

from isaaclab_visualizers.desktop_entry import write_desktop_entry
from isaaclab_visualizers.newton_adapter import resolve_visible_env_indices

from .kit_visualizer_cfg import KitVisualizerCfg

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
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

    _timeline_auto_update: ClassVar[bool | None] = None
    _timeline_control_owner_count: ClassVar[int] = 0

    def __init__(self, cfg: KitVisualizerCfg):
        """Initialize Kit visualizer state."""
        super().__init__(cfg)
        self.cfg: KitVisualizerCfg = cfg
        self._simulation_app = None
        self._viewport_window = None
        self._viewport_api = None
        self._is_initialized = False
        self._sim_time = 0.0
        self._step_counter = 0
        self._env_ids = None
        self._resolved_visible_env_ids: list[int] | None = None
        self._hidden_env_visibilities: dict[str, str] = {}
        self._point_instancer_invisible_ids_backup: dict[str, tuple[bool, object]] = {}
        self._runtime_headless = bool(cfg.headless)
        self._controlled_camera_path: str | None = None
        self._rgb_render_product = None
        self._rgb_annotator = None
        self._camera_sensor = None
        self._camera_sensor_indices: list[int] = []
        self._camera_env_indices: list[int] = []
        self._camera_is_owned = False
        self._streaming_camera_key: tuple | None = None
        self._generated_camera_prim_paths: list[str] = []
        self._last_streaming_composite: np.ndarray | None = None
        self._generated_camera_xform_ops: dict[str, tuple[UsdGeom.XformOp, UsdGeom.XformOp]] = {}
        self._generated_camera_pose_cache: dict[str, tuple[float, ...]] = {}
        self._camera_image_provider = None
        self._camera_image_window = None
        self._camera_gpu_upload_tensor = None
        self._warned_gpu_upload_failure = False
        self._backend_menubar_label = None
        self._hid_simulation_menu = False
        self._app_pumped_this_step: bool = False
        self._controls_kit_timeline = False
        self._owns_kit_timeline = False
        self._interactive_scene = None
        self._viewer_origin: torch.Tensor | None = None

    @property
    def visual_material_writer(self):
        """Write material channels directly through Fabric."""
        from isaaclab_physx.renderers.visual_material import FabricVisualMaterialWriter

        return FabricVisualMaterialWriter

    def initialize(self, scene_data_provider: SceneDataProvider) -> None:
        """Initialize viewport resources and bind scene data provider."""
        if self._is_initialized:
            logger.debug("[KitVisualizer] initialize() called while already initialized.")
            return
        scene_data_provider = self._set_scene_data_provider(scene_data_provider)
        usd_stage = SimulationContext.instance().stage
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
        from isaaclab_physx.renderers.fabric import FabricBackendCfg

        self._fabric = sim.get_or_create_backend(FabricBackendCfg(stage=sim.stage, device=sim.device))
        self._fabric.bind_transforms(scene_data_provider)
        self._is_initialized = True
        self._setup_initial_camera_view()
        self._controls_kit_timeline = sim.has_gui and self.physics_backend not in ("physx", "ovphysx")
        if self._controls_kit_timeline:
            self.play()

    def step(self, dt: float) -> None:
        """Advance visualizer/UI updates for one simulation step."""
        if not self._is_initialized:
            return
        self._app_pumped_this_step = False
        self._sim_time += dt
        self._step_counter += 1
        if self._runtime_headless:
            return
        self._fabric.update_transforms(self._scene_data_provider)
        self._fabric.update_geometries(self._scene_data_provider, SimulationContext.instance().render_generation)
        if self.cfg.origin_type == "asset":
            self._update_asset_tracking_camera()
        self._app_pumped_this_step = self._pump_kit_app_update()
        self._update_camera_image_panel(dt)
        self._refresh_partial_viz_point_instancers_if_needed()

    def close(self) -> None:
        """Close viewport resources and restore temporary state."""
        if not self._is_initialized:
            return
        try:
            if self._owns_kit_timeline:
                self.stop()
        finally:
            self._teardown_backend_menubar_label()
            self._restore_env_visibility()
            if self._streaming_camera_key is not None:
                from isaaclab.envs.utils.camera_view import evict_visualizer_camera

                evict_visualizer_camera(self._streaming_camera_key)
                self._streaming_camera_key = None
            if self._camera_sensor is not None and self._camera_is_owned:
                remove_generated_prims(self._generated_camera_prim_paths)
            self._camera_sensor = None
            self._generated_camera_xform_ops.clear()
            self._generated_camera_pose_cache.clear()
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
            self._controls_kit_timeline = False
            self._is_initialized = False
            self._is_closed = True

    def play(self) -> None:
        """Show the running state in Kit's toolbar without advancing Kit physics."""
        if not self._controls_kit_timeline:
            return
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        if not self._owns_kit_timeline:
            if KitVisualizer._timeline_control_owner_count == 0:
                KitVisualizer._timeline_auto_update = timeline.is_auto_updating()
            KitVisualizer._timeline_control_owner_count += 1
            self._owns_kit_timeline = True
        timeline.set_auto_update(False)
        timeline.play()
        self._pump_kit_app_update()

    def pause(self) -> None:
        """Show the paused state in Kit's toolbar without advancing Kit physics."""
        if not self._controls_kit_timeline:
            return
        import omni.timeline

        omni.timeline.get_timeline_interface().pause()
        self._pump_kit_app_update()

    def stop(self) -> None:
        """Show the stopped state in Kit's toolbar without advancing Kit physics."""
        if not self._controls_kit_timeline or not self._owns_kit_timeline:
            return
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        is_last_owner = KitVisualizer._timeline_control_owner_count == 1
        try:
            if is_last_owner:
                timeline.stop()
                self._pump_kit_app_update()
        finally:
            KitVisualizer._timeline_control_owner_count -= 1
            self._owns_kit_timeline = False
            if is_last_owner:
                timeline_auto_update = KitVisualizer._timeline_auto_update
                KitVisualizer._timeline_auto_update = None
                if timeline_auto_update is not None:
                    timeline.set_auto_update(timeline_auto_update)

    def render_rgb_array(self) -> np.ndarray:
        """Return an RGB frame captured from the Kit viewport camera."""
        import omni.replicator.core as rep

        self._fabric.update_transforms(self._scene_data_provider)
        self._fabric.update_geometries(self._scene_data_provider, SimulationContext.instance().render_generation)
        if self._runtime_headless and self.cfg.origin_type == "asset":
            self._update_asset_tracking_camera()
        camera_path = self._controlled_camera_path or _DEFAULT_VIEWPORT_CAMERA_PATH
        w, h = self.cfg.window_width, self.cfg.window_height
        if self._rgb_annotator is None:
            self._rgb_render_product = rep.create.render_product(camera_path, (w, h))
            self._apply_render_product_background(SimulationContext.instance().stage, self._rgb_render_product.path)
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            self._rgb_annotator.attach([self._rgb_render_product])
        elif self._runtime_headless and self._rgb_render_product is not None:
            import contextlib

            with contextlib.suppress(Exception):
                self._rgb_render_product.resume()
        if not self._app_pumped_this_step:
            self._pump_kit_app_update()
        raw = self._rgb_annotator.get_data()
        if isinstance(raw, dict):
            raw = raw.get("data", np.array([], dtype=np.uint8))
        raw = np.asarray(raw, dtype=np.uint8)
        if raw.size == 0:
            return np.zeros((h, w, 3), dtype=np.uint8)
        if raw.ndim == 1:
            raw = raw.reshape(h, w, -1)
        result = raw[:, :, :3]
        if self._runtime_headless and self._rgb_render_product is not None:
            import contextlib

            with contextlib.suppress(Exception):
                self._rgb_render_product.pause()
        return result

    def is_running(self) -> bool:
        """Return whether Kit app/runtime is still running."""
        if self._simulation_app is not None:
            return self._simulation_app.is_running()
        try:
            import omni.kit.app

            app = omni.kit.app.get_app()
            return app is not None and app.is_running()
        except (ImportError, AttributeError):
            return False

    def is_training_paused(self) -> bool:
        """Return whether the Kit timeline is paused by its toolbar controls."""
        if not self._controls_kit_timeline:
            return False
        try:
            import omni.timeline

            timeline = omni.timeline.get_timeline_interface()
            return not timeline.is_playing() and not timeline.is_stopped()
        except Exception:
            return False

    def supports_markers(self) -> bool:
        return bool(self.cfg.enable_markers)

    def supports_live_plots(self) -> bool:
        return True

    def add_live_plots(self, managers: dict, scalars: dict | None = None, term_names=None, env_idx: int = 0) -> None:
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
                cfg=ManagerLiveVisualizerCfg(manager_name=name, term_names=(term_names or {}).get(name)),
            )
            for name, mgr in managers.items()
        }
        for source in self._live_plot_sources:
            if isinstance(source, DirectScalarLivePlots):
                self.kit_manager_visualizers[source.manager_name] = DirectScalarLiveVisualizer(source)

    def pumps_app_update(self) -> bool:
        return True

    def set_camera_view(self, eye, target) -> None:
        if not self._is_initialized:
            logger.debug("[KitVisualizer] set_camera_view() ignored because visualizer is not initialized.")
            return
        self._set_viewport_camera(tuple(eye), tuple(target))

    def render_tiled_rgb_array(self) -> np.ndarray | None:
        if self._camera_sensor is None:
            return None
        if self._runtime_headless:
            self._update_camera_image_panel(0.0)
        return self._last_streaming_composite

    def reapply_origin(self) -> None:
        self._setup_initial_camera_view()

    @property
    def viewer_origin(self) -> torch.Tensor | None:
        return self._viewer_origin

    def _pump_kit_app_update(self) -> bool:
        """Process one Kit UI update without letting Kit advance physics."""
        try:
            import omni.kit.app

            app = omni.kit.app.get_app()
            if app is None or not app.is_running():
                return False
            settings = get_settings_manager()
            play_flag = settings.get("/app/player/playSimulations")
            settings.set("/app/player/playSimulations", False)
            interrupted = False

            def defer_interrupt(_signum, _frame):
                nonlocal interrupted
                interrupted = True
                signal.signal(signal.SIGINT, signal.SIG_DFL)

            previous_handler = signal.getsignal(signal.SIGINT)
            defer_interrupts = previous_handler != signal.SIG_DFL
            if defer_interrupts:
                signal.signal(signal.SIGINT, defer_interrupt)
            try:
                app.update()
            finally:
                if defer_interrupts:
                    signal.signal(signal.SIGINT, previous_handler)
                settings.set("/app/player/playSimulations", True if play_flag is None else bool(play_flag))
                if interrupted:
                    raise KeyboardInterrupt
            return True
        except (ImportError, AttributeError) as exc:
            logger.debug("[KitVisualizer] App update skipped: %s", exc)
            return False

    def _setup_backend_menubar_label(self) -> None:
        try:
            from omni.kit.viewport.menubar.core import IconMenuDelegate, ViewportMenuItem, get_menu_item
        except (ImportError, ModuleNotFoundError):
            return
        backend = self.physics_backend or "unknown"
        backend_display = _BACKEND_DISPLAY_NAMES.get(backend, backend)
        if backend not in ("physx", "ovphysx"):
            sim_item = get_menu_item("Simulation")
            if sim_item is not None:
                sim_item.visible_model.set_value(False)
                self._hid_simulation_menu = True
        self._backend_menubar_label = ViewportMenuItem(
            f"Physics: {backend_display}",
            delegate=IconMenuDelegate("", text=True, width=0, has_triangle=False, enabled=False),
        )

    async def _setup_backend_menubar_label_async(self) -> None:
        import omni.kit.app

        await omni.kit.app.get_app().next_update_async()
        self._setup_backend_menubar_label()

    def _teardown_backend_menubar_label(self) -> None:
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
        import os

        import omni.kit.app

        app = omni.kit.app.get_app()
        if app is None or not app.is_running():
            raise RuntimeError("[KitVisualizer] Isaac Sim app is not running.")
        headless_env = bool(int(os.environ.get("HEADLESS", 0)))
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
        if self.cfg.background_color is None:
            return
        render_product = stage.GetPrimAtPath(render_product_path)
        if not render_product.IsValid():
            logger.warning(
                "[KitVisualizer] Render product '%s' was not found; background was not applied.", render_product_path
            )
            return
        with Usd.EditContext(stage, stage.GetSessionLayer()), Sdf.ChangeBlock():
            render_product.CreateAttribute("omni:rtx:background:source:type", Sdf.ValueTypeNames.Token).Set("color")
            render_product.CreateAttribute("omni:rtx:background:source:color", Sdf.ValueTypeNames.Float3).Set(
                Gf.Vec3f(*self.cfg.background_color)
            )

    def _write_desktop_entry(self) -> None:
        import carb.tokens

        settings = get_settings_manager()
        title = settings.get("/app/window/title")
        version = settings.get("/app/version")
        icon_path = settings.get("/app/window/iconPath")
        if title and version and icon_path:
            icon = carb.tokens.get_tokens_interface().resolve(icon_path)
            write_desktop_entry("isaaclab", title, f"{title} {version}", icon)

    def _setup_viewport(self) -> None:
        if self._runtime_headless:
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

        name = self.cfg.viewport_name if self.cfg.viewport_name is not None else _DEFAULT_VIEWPORT_NAME
        if self.cfg.create_viewport:
            if not str(name).strip():
                raise RuntimeError(
                    "[KitVisualizer] viewport_name must be a non-empty string when create_viewport=True."
                )
            positions = {
                "LEFT": DockPosition.LEFT,
                "RIGHT": DockPosition.RIGHT,
                "BOTTOM": DockPosition.BOTTOM,
                "SAME": DockPosition.SAME,
            }
            dock_pos = positions.get(self.cfg.dock_position.upper(), DockPosition.SAME)
            self._viewport_window = vp_utils.create_viewport_window(
                name=name,
                width=self.cfg.window_width,
                height=self.cfg.window_height,
                position_x=50,
                position_y=50,
                docked=True,
            )
            asyncio.ensure_future(self._dock_viewport_async(name, dock_pos))
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
        if not self._uses_streaming_view():
            self._apply_cfg_camera_pose_if_configured()
        self._refresh_controlled_camera_path()
        asyncio.ensure_future(self._setup_backend_menubar_label_async())

    def _uses_streaming_view(self) -> bool:
        return bool(self.cfg.streaming_view)

    def _setup_streaming_view(self, num_envs: int) -> None:
        if not self._uses_streaming_view():
            return
        if not get_settings_manager().get("/isaaclab/cameras_enabled", False):
            logger.info("[KitVisualizer] Streaming view skipped: camera rendering is not enabled.")
            return
        gt_types = list(self.cfg.streaming_gt_types)
        for gt in gt_types:
            if gt not in SUPPORTED_GT_TYPES:
                raise ValueError(
                    f"[KitVisualizer] streaming_gt_types contains unsupported type {gt!r}. "
                    f"Valid types: {sorted(SUPPORTED_GT_TYPES)}"
                )
        env_ids = resolve_streaming_envs(
            num_envs,
            self.cfg.streaming_envs,
            max_tiles=VISUALIZER_TILED_CAMERA_MAX_TILES,
            sample_from=self._resolved_visible_env_ids,
        )
        self._camera_env_indices = env_ids
        if self.cfg.streaming_sensor_prim_path is not None:
            cameras = self._scene_data_provider.get_camera_sensors()
            self._camera_sensor = find_camera_by_prim_path(cameras, self.cfg.streaming_sensor_prim_path, env_ids)
            self._camera_sensor_indices = env_ids
        elif self.cfg.streaming_cam_target_prim_path is None:
            cameras = self._scene_data_provider.get_camera_sensors()
            if cameras:
                _, self._camera_sensor = next(iter(cameras.items()))
                self._camera_sensor_indices = env_ids
        else:
            count = max(1, len(env_ids))
            tile_w, tile_h = compute_tile_resolution(self.cfg.window_width, self.cfg.window_height, count)
            (
                self._camera_sensor,
                self._generated_camera_prim_paths,
                self._camera_is_owned,
                self._streaming_camera_key,
            ) = create_visualizer_camera(
                num_envs=num_envs,
                width=tile_w,
                height=tile_h,
                renderer_cfg=self.cfg.streaming_cam_renderer_cfg,
                data_types=sensor_keys_for_gt_types(gt_types),
                streaming_envs=tuple(int(i) for i in env_ids),
            )
            self._camera_sensor_indices = env_ids
            self._update_owned_camera_poses()
        if not self._runtime_headless:
            self._setup_camera_image_window()

    def _setup_camera_image_window(self) -> None:
        import omni.ui

        title = self.cfg.viewport_name or "Streaming View"
        self._camera_image_provider = omni.ui.ByteImageProvider()
        self._camera_image_window = omni.ui.Window(title, width=self.cfg.window_width, height=self.cfg.window_height)
        with self._camera_image_window.frame:
            omni.ui.ImageWithProvider(self._camera_image_provider)
        positions = {
            "LEFT": omni.ui.DockPosition.LEFT,
            "RIGHT": omni.ui.DockPosition.RIGHT,
            "BOTTOM": omni.ui.DockPosition.BOTTOM,
            "SAME": omni.ui.DockPosition.SAME,
        }
        asyncio.ensure_future(
            self._dock_image_window_async(
                title, positions.get(self.cfg.dock_position.upper(), omni.ui.DockPosition.SAME)
            )
        )

    async def _dock_image_window_async(self, window_name: str, dock_position) -> None:
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

    def _update_owned_camera_poses(self) -> None:
        if self._camera_sensor is None or not self._camera_is_owned:
            return
        target_positions = prim_world_positions(
            SimulationContext.instance().stage,
            self.cfg.streaming_cam_target_prim_path,
            self._camera_env_indices,
            scene=self._scene_data_provider.get_interactive_scene(),
        )
        apply_camera_target_positions(
            self._camera_sensor, target_positions, self.cfg.streaming_cam_eye, self._camera_env_indices
        )

    def _update_camera_image_panel(self, dt: float) -> None:
        if self._camera_sensor is None:
            return
        if self.is_training_paused():
            if self._last_streaming_composite is not None and self._camera_image_provider is not None:
                self._upload_camera_image_to_panel(self._last_streaming_composite)
            return
        if self._camera_is_owned:
            self._update_owned_camera_poses()
            self._camera_sensor.update(dt=dt, force_recompute=True)
        gt_types = list(self.cfg.streaming_gt_types)
        available = frozenset(self._camera_sensor.data.output.keys())
        gt_batches = {
            gt: camera_gt_batch(self._camera_sensor, self._camera_sensor_indices, sensor_key_for_gt_type(gt, available))
            for gt in gt_types
        }
        frames = [
            CameraFrameColorizer.colorize(
                gt_batches[gt][env_pos],
                gt,
                depth_min=self.cfg.streaming_depth_min,
                depth_max=self.cfg.streaming_depth_max,
            )
            for env_pos in range(len(self._camera_sensor_indices))
            for gt in gt_types
        ]
        target_aspect = self.cfg.window_width / self.cfg.window_height if self.cfg.window_height > 0 else 1.0
        self._last_streaming_composite = compose_streaming_grid(
            frames, len(self._camera_sensor_indices), len(gt_types), target_aspect=target_aspect
        )
        if self._camera_image_provider is not None:
            self._upload_camera_image_to_panel(self._last_streaming_composite)

    def _upload_camera_image_to_panel(self, image: np.ndarray | torch.Tensor) -> None:
        if isinstance(image, torch.Tensor):
            if image.is_cuda:
                try:
                    import omni.gpu_foundation_factory as gf

                    if image.ndim == 3 and image.shape[2] == 3:
                        alpha = torch.full((*image.shape[:2], 1), 255, dtype=torch.uint8, device=image.device)
                        image = torch.cat((image, alpha), dim=2)
                    image = image.to(dtype=torch.uint8).contiguous()
                    self._camera_gpu_upload_tensor = image
                    self._camera_image_provider.set_bytes_data_from_gpu(
                        int(image.data_ptr()), [int(image.shape[1]), int(image.shape[0])], gf.TextureFormat.RGBA8_UNORM
                    )
                    return
                except Exception as exc:
                    if not self._warned_gpu_upload_failure:
                        logger.warning("[KitVisualizer] GPU image upload failed; falling back to CPU upload: %s", exc)
                        self._warned_gpu_upload_failure = True
            image = image.detach().contiguous().cpu().numpy()
        image = image.astype("uint8", copy=False)
        if image.ndim == 3 and image.shape[2] == 3:
            image = np.concatenate((image, np.full((*image.shape[:2], 1), 255, dtype=np.uint8)), axis=2)
        image = np.ascontiguousarray(image)
        self._camera_image_provider.set_bytes_data(image.flatten().data, [image.shape[1], image.shape[0]])

    def _refresh_controlled_camera_path(self) -> None:
        if self._viewport_api is not None:
            path = self._viewport_api.get_active_camera()
            self._controlled_camera_path = path if path else _DEFAULT_VIEWPORT_CAMERA_PATH
        else:
            self._controlled_camera_path = _DEFAULT_VIEWPORT_CAMERA_PATH

    def _apply_viewport_camera_scene_partition(self, usd_stage: Usd.Stage, num_envs: int) -> None:
        if num_envs <= 0 or self._controlled_camera_path is None:
            return
        env_prim = usd_stage.GetPrimAtPath("/World/envs/env_0")
        env_partition_attr = env_prim.GetAttribute("primvars:omni:scenePartition")
        if not env_partition_attr.IsValid() or env_partition_attr.Get() is None:
            return
        camera_prim = usd_stage.GetPrimAtPath(self._controlled_camera_path)
        if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
            return
        attr = camera_prim.GetAttribute("omni:scenePartition")
        if get_settings_manager().get(ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING, False):
            if attr.IsValid():
                camera_prim.RemoveProperty("omni:scenePartition")
            return
        env_id = self._resolved_visible_env_ids[0] if self._resolved_visible_env_ids else 0
        if not attr.IsValid():
            attr = camera_prim.CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token)
        attr.Set(f"env_{env_id}")

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
            logger.warning("[KitVisualizer] Could not find viewport window '%s'.", viewport_name)
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

    def _set_viewport_camera(self, position, target) -> None:
        if self._viewport_api is None:
            self._set_usd_camera_pose(_DEFAULT_VIEWPORT_CAMERA_PATH, position, target)
            return
        try:
            from omni.kit.viewport.utility.camera_state import ViewportCameraState
        except ImportError as exc:
            logger.warning("[KitVisualizer] Viewport camera update skipped: %s", exc)
            return
        camera_path = self._viewport_api.get_active_camera() or _DEFAULT_VIEWPORT_CAMERA_PATH
        state = ViewportCameraState(camera_path, self._viewport_api)
        state.set_position_world(Gf.Vec3d(*map(float, position)), False)
        state.set_target_world(Gf.Vec3d(*map(float, target)), True)

    def _set_usd_camera_pose(self, camera_path: str, position, target) -> bool:
        usd_stage = SimulationContext.instance().stage
        eye = torch.as_tensor(position, dtype=torch.float32, device="cpu").reshape(1, 3)
        lookat = torch.as_tensor(target, dtype=torch.float32, device="cpu").reshape(1, 3)
        rotation_matrix = create_rotation_matrix_from_view(
            eye, lookat, up_axis=UsdGeom.GetStageUpAxis(usd_stage), device="cpu"
        )
        if torch.isnan(rotation_matrix).any():
            raise ValueError("[KitVisualizer] Cannot set camera pose because eye and lookat are degenerate.")
        quat_xyzw = quat_from_matrix(rotation_matrix)[0]
        pose_key = tuple(float(v) for v in (*eye[0], *quat_xyzw))
        if self._generated_camera_pose_cache.get(camera_path) == pose_key:
            return False
        if camera_path not in self._generated_camera_xform_ops:
            camera = UsdGeom.Camera.Define(usd_stage, camera_path)
            camera_xform = UsdGeom.Xformable(camera.GetPrim())
            camera_xform.ClearXformOpOrder()
            camera_xform.SetResetXformStack(True)
            prim = camera.GetPrim()
            t_attr = prim.GetAttribute("xformOp:translate")
            translate_op = UsdGeom.XformOp(t_attr) if t_attr else camera_xform.AddTranslateOp()
            o_attr = prim.GetAttribute("xformOp:orient")
            orient_op = UsdGeom.XformOp(o_attr) if o_attr else camera_xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
            camera_xform.SetXformOpOrder([translate_op, orient_op], camera_xform.GetResetXformStack())
            self._generated_camera_xform_ops[camera_path] = (translate_op, orient_op)
        else:
            translate_op, orient_op = self._generated_camera_xform_ops[camera_path]
        translate_op.Set(Gf.Vec3d(*map(float, eye[0])))
        orient_op.Set(Gf.Quatd(float(quat_xyzw[3]), Gf.Vec3d(*map(float, quat_xyzw[:3]))))
        self._generated_camera_pose_cache[camera_path] = pose_key
        return True

    def _apply_cfg_camera_pose_if_configured(self) -> None:
        self._set_viewport_camera(self.cfg.eye, self.cfg.lookat)

    def _set_active_camera_path(self, camera_path: str) -> bool:
        if self._viewport_api is None:
            return False
        if not SimulationContext.instance().stage.GetPrimAtPath(camera_path).IsValid():
            return False
        self._viewport_api.set_active_camera(camera_path)
        return True

    def _apply_env_visibility(self, usd_stage, num_envs: int, visible_env_ids: list[int]) -> None:
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
        if self._resolved_visible_env_ids is None or self._scene_data_provider is None:
            return
        num_envs = self._scene_data_provider.num_envs
        if num_envs > 0:
            self._apply_visual_point_instancer_visibility(
                SimulationContext.instance().stage, num_envs, set(self._resolved_visible_env_ids)
            )

    def _apply_visual_point_instancer_visibility(self, usd_stage, num_envs: int, visible_env_ids: set[int]) -> None:
        vt_hidden = Vt.Int64Array([i for i in range(num_envs) if i not in visible_env_ids])
        for root_path in ("/Visuals", "/World/Visuals"):
            root_prim = usd_stage.GetPrimAtPath(root_path)
            if not root_prim.IsValid():
                continue
            for prim in Usd.PrimRange(root_prim):
                if not prim.IsA(UsdGeom.PointInstancer):
                    continue
                pi = UsdGeom.PointInstancer(prim)
                if self._point_instancer_instance_count(pi) != num_envs:
                    continue
                path_str = prim.GetPath().pathString
                inv_attr = pi.GetInvisibleIdsAttr()
                if path_str not in self._point_instancer_invisible_ids_backup:
                    was_authored = inv_attr.HasAuthoredValue()
                    self._point_instancer_invisible_ids_backup[path_str] = (
                        was_authored,
                        inv_attr.Get() if was_authored else None,
                    )
                inv_attr.Set(vt_hidden)

    @staticmethod
    def _point_instancer_instance_count(pi: UsdGeom.PointInstancer) -> int | None:
        for attr in (pi.GetPositionsAttr(), pi.GetScalesAttr(), pi.GetOrientationsAttr(), pi.GetProtoIndicesAttr()):
            if attr.HasAuthoredValue():
                value = attr.Get()
                if value is not None:
                    return len(value)
        return None

    def _restore_env_visibility(self) -> None:
        usd_stage = SimulationContext.instance().stage
        for env_path, prev in self._hidden_env_visibilities.items():
            prim = usd_stage.GetPrimAtPath(env_path)
            if prim.IsValid() and UsdGeom.Imageable(prim):
                UsdGeom.Imageable(prim).GetVisibilityAttr().Set(prev)
        self._hidden_env_visibilities.clear()
        for path_str, (was_authored, prev) in self._point_instancer_invisible_ids_backup.items():
            prim = usd_stage.GetPrimAtPath(path_str)
            if not prim.IsValid() or not prim.IsA(UsdGeom.PointInstancer):
                continue
            attr = UsdGeom.PointInstancer(prim).GetInvisibleIdsAttr()
            attr.Set(prev) if was_authored else attr.Clear()
        self._point_instancer_invisible_ids_backup.clear()

    def _setup_initial_camera_view(self) -> None:
        self._interactive_scene = getattr(SimulationContext.instance(), "_interactive_scene", None)
        if self.cfg.origin_type == "world":
            self._viewer_origin = torch.zeros(3)
        elif self.cfg.origin_type == "env":
            scene = self._interactive_scene
            if scene is None:
                logger.warning("[KitVisualizer] origin_type='env' requested but no scene is registered yet.")
                self._viewer_origin = torch.zeros(3)
            else:
                if not (0 <= self.cfg.origin_env_index < scene.num_envs):
                    raise ValueError(
                        f"[KitVisualizer] origin_env_index {self.cfg.origin_env_index} is out of range "
                        f"[0, {scene.num_envs - 1}] for origin_type='env'."
                    )
                self._viewer_origin = scene.env_origins[self.cfg.origin_env_index]
        elif self.cfg.origin_type == "asset":
            if self.cfg.origin_track_path is None:
                raise ValueError("[KitVisualizer] origin_type='asset' requires origin_track_path to be set.")
            return
        else:
            logger.warning("[KitVisualizer] Unknown origin_type '%s'; defaulting to world.", self.cfg.origin_type)
            self._viewer_origin = torch.zeros(3)
        self._apply_viewer_origin_to_camera()

    def _update_asset_tracking_camera(self) -> None:
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
        if self._viewer_origin is None:
            return
        origin = self._viewer_origin.detach().cpu().numpy()
        eye = np.array(self.cfg.eye, dtype=float) + origin
        target = np.array(self.cfg.lookat, dtype=float) + origin
        self.set_camera_view(tuple(float(v) for v in eye), tuple(float(v) for v in target))
        try:
            from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view

            set_kit_renderer_camera_view(eye=eye, target=target, camera_prim_path=_DEFAULT_VIEWPORT_CAMERA_PATH)
        except (ImportError, ModuleNotFoundError):
            pass
