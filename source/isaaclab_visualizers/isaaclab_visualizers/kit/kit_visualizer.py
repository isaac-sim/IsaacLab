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
from isaaclab.envs.utils.camera_view import (
    VISUALIZER_TILED_CAMERA_MAX_TILES,
    apply_camera_target_positions,
    camera_rgb_batch,
    compose_rgb_grid_tensor,
    compute_tile_resolution,
    create_visualizer_camera,
    find_camera_by_prim_path,
    prim_world_positions,
    remove_generated_prims,
    resolve_tiled_env_indices,
)
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix
from isaaclab.utils.renderers import isaac_rtx_per_env_scene_partition_enabled
from isaaclab.visualizers.base_visualizer import BaseVisualizer

from isaaclab_visualizers.newton_adapter import resolve_visible_env_indices

from .kit_visualizer_cfg import KitVisualizerCfg

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaclab.scene_data import SceneDataProvider

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
        self._env_ids = None
        self._resolved_visible_env_ids: list[int] | None = None
        self._hidden_env_visibilities: dict[str, str] = {}
        # PointInstancer prim path -> (had authored invisibleIds, previous value) for partial viz restore.
        self._point_instancer_invisible_ids_backup: dict[str, tuple[bool, object]] = {}
        self._runtime_headless = bool(cfg.headless)
        # USD path for the viewport's active camera, refreshed after setup (used by CI/tests).
        self._controlled_camera_path: str | None = None
        self._camera_sensor = None
        self._camera_sensor_indices: list[int] = []
        self._camera_env_indices: list[int] = []
        self._camera_is_owned = False
        self._generated_camera_prim_paths: list[str] = []
        self._generated_camera_xform_ops: dict[str, tuple[UsdGeom.XformOp, UsdGeom.XformOp]] = {}
        self._generated_camera_pose_cache: dict[str, tuple[float, ...]] = {}
        self._generated_camera_poses_dirty = False
        self._camera_image_provider = None
        self._camera_image_window = None
        self._camera_gpu_upload_tensor = None
        self._warned_gpu_upload_failure = False

    # ---- Lifecycle ------------------------------------------------------------------------

    def initialize(self, scene_data_provider: SceneDataProvider) -> None:
        """Initialize viewport resources and bind scene data provider.

        Args:
            scene_data_provider: Scene data provider used by the visualizer.
        """
        if self._is_initialized:
            logger.debug("[KitVisualizer] initialize() called while already initialized.")
            return

        scene_data_provider = self._set_scene_data_provider(scene_data_provider)
        usd_stage = scene_data_provider.usd_stage
        if usd_stage is None:
            raise RuntimeError("[KitVisualizer] USD stage not available from scene_data_provider.")
        num_envs = scene_data_provider.num_envs

        self._ensure_simulation_app()
        self._setup_viewport()

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
                ("tiled_cam_view", self.cfg.tiled_cam_view),
                ("tiled_cam_num", self.cfg.tiled_cam_num),
                ("max_visible_envs", self.cfg.max_visible_envs),
                ("num_visualized_envs", num_visualized_envs),
                ("create_viewport", self.cfg.create_viewport),
                ("headless", self._runtime_headless),
            ],
        )
        self._setup_camera_sensor_view(num_envs)

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
        self._update_camera_image_panel(dt)
        # Markers (VisualizationMarkers) are often created or resized to num_envs only after the first
        # simulation / debug-vis step; re-apply PointInstancer invisibleIds each step when partial viz is on.
        self._refresh_partial_viz_point_instancers_if_needed()

    def close(self) -> None:
        """Close viewport resources and restore temporary state."""
        if not self._is_initialized:
            return
        self._restore_env_visibility()
        if self._camera_sensor is not None and self._camera_is_owned:
            remove_generated_prims(self._generated_camera_prim_paths)
        self._camera_sensor = None
        self._generated_camera_xform_ops.clear()
        self._generated_camera_pose_cache.clear()
        self._generated_camera_poses_dirty = False
        self._camera_image_provider = None
        self._camera_image_window = None
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

    def _setup_viewport(self) -> None:
        """Create/resolve viewport and configure initial camera."""
        import omni.kit.viewport.utility as vp_utils
        from omni.ui import DockPosition

        if self._runtime_headless:
            # Headless: no viewport window; apply cfg pose to the default perspective camera path.
            self._viewport_window = None
            self._viewport_api = None
            if self._uses_camera_sensor_view():
                logger.debug("[KitVisualizer] Camera image view requested in headless mode; no UI panel is created.")
            else:
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
        if self._uses_camera_sensor_view():
            # Camera sensor image views are shown in a non-interactive image panel.
            pass
        else:
            self._apply_cfg_camera_pose_if_configured()
        self._refresh_controlled_camera_path()

    def _uses_camera_sensor_view(self) -> bool:
        """Return whether Kit should display a camera sensor image instead of an interactive viewport camera."""
        return bool(self.cfg.tiled_cam_view)

    def _setup_camera_sensor_view(self, num_envs: int) -> None:
        """Resolve or create the Camera sensor backing non-interactive image views."""
        if not self._uses_camera_sensor_view():
            return
        if self._runtime_headless:
            return
        if not get_settings_manager().get("/isaaclab/cameras_enabled", False):
            raise RuntimeError(
                "[KitVisualizer] tiled_cam_view=True requires camera rendering support. "
                "Rerun with --enable_cameras, or disable tiled_cam_view for this visualizer config."
            )
        logger.debug(
            "[KitVisualizer] Setting up camera image view: tiled=%s source=%s num_envs=%s",
            self.cfg.tiled_cam_view,
            "prim_path" if self.cfg.tiled_cam_prim_path is not None else "generated",
            num_envs,
        )
        env_ids = resolve_tiled_env_indices(
            num_envs,
            self.cfg.tiled_cam_num,
            self.cfg.tiled_cam_env_indices,
            max_tiles=VISUALIZER_TILED_CAMERA_MAX_TILES,
            sample_from=self._resolved_visible_env_ids,
        )
        self._camera_env_indices = env_ids
        if self.cfg.tiled_cam_prim_path is not None:
            logger.debug(
                "[KitVisualizer] tiled_cam_prim_path uses existing camera sensor output; "
                "generated tiled camera pose fields are ignored."
            )
            cameras = self._scene_data_provider.get_camera_sensors()
            self._camera_sensor = find_camera_by_prim_path(cameras, self.cfg.tiled_cam_prim_path, env_ids)
            self._camera_sensor_indices = env_ids
        else:
            from isaaclab_physx.renderers import IsaacRtxRendererCfg

            count = max(1, len(env_ids))
            tile_w, tile_h = compute_tile_resolution(self.cfg.window_width, self.cfg.window_height, count)
            logger.debug(
                "[KitVisualizer] Creating generated camera sensor: env_ids=%s tile=%sx%s",
                env_ids,
                tile_w,
                tile_h,
            )
            self._camera_sensor, self._generated_camera_prim_paths = create_visualizer_camera(
                num_envs=num_envs,
                width=tile_w,
                height=tile_h,
                renderer_cfg=IsaacRtxRendererCfg(),
            )
            logger.debug("[KitVisualizer] Generated camera sensor initialized.")
            self._camera_sensor_indices = env_ids
            self._camera_is_owned = True
            self._update_owned_camera_poses()
            logger.debug("[KitVisualizer] Generated camera poses initialized.")
        self._setup_camera_image_window()
        logger.debug("[KitVisualizer] Camera image window initialized.")

    def _setup_camera_image_window(self) -> None:
        """Create a dockable Kit UI image panel for camera sensor RGB output."""
        import omni.ui

        title = self.cfg.viewport_name or "Visualizer Tiled Camera"
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

    def _update_owned_camera_poses(self) -> None:
        """Update generated camera poses from env origins or follow prims."""
        if self._camera_sensor is None or not self._camera_is_owned:
            return
        target_positions = prim_world_positions(
            self._scene_data_provider.get_usd_stage(),
            self.cfg.tiled_cam_target_prim_path,
            self._camera_env_indices,
            scene=self._scene_data_provider.get_interactive_scene(),
        )
        eyes, targets = apply_camera_target_positions(
            self._camera_sensor, target_positions, self.cfg.tiled_cam_eye, self._camera_env_indices
        )
        self._set_generated_usd_camera_poses(eyes, targets)

    def _update_camera_image_panel(self, dt: float) -> None:
        """Refresh the non-interactive Kit image panel from camera RGB output."""
        if self._camera_sensor is None or self._camera_image_provider is None:
            return
        if self._camera_is_owned:
            self._update_owned_camera_poses()
            if self._generated_camera_poses_dirty:
                self._sync_camera_pose_updates_to_kit()
                self._generated_camera_poses_dirty = False
        if self._camera_is_owned:
            self._camera_sensor.update(dt=dt, force_recompute=True)
        rgb = camera_rgb_batch(self._camera_sensor, self._camera_sensor_indices)
        image = compose_rgb_grid_tensor(rgb) if self.cfg.tiled_cam_view else rgb[0].contiguous()
        self._upload_camera_image_to_panel(image)

    def _upload_camera_image_to_panel(self, image: np.ndarray | torch.Tensor) -> None:
        """Upload an RGB/RGBA image to the Kit image provider."""
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
            alpha = np.full((*image.shape[:2], 1), 255, dtype=np.uint8)
            image = np.concatenate((image, alpha), axis=2)
        image = np.ascontiguousarray(image)
        self._camera_image_provider.set_bytes_data(image.flatten().data, [image.shape[1], image.shape[0]])

    def _sync_camera_pose_updates_to_kit(self) -> None:
        """Flush generated camera pose writes before camera RGB is sampled."""
        try:
            import omni.kit.app

            app = omni.kit.app.get_app()
            if app is None or not app.is_running():
                return
            settings = get_settings_manager()
            play_flag = settings.get("/app/player/playSimulations")
            settings.set_bool("/app/player/playSimulations", False)
            app.update()
            if play_flag is not None:
                settings.set_bool("/app/player/playSimulations", bool(play_flag))
        except Exception as exc:
            logger.debug("[KitVisualizer] Camera pose Kit sync skipped: %s", exc)

    def _refresh_controlled_camera_path(self) -> None:
        """Cache :attr:`_controlled_camera_path` from the active viewport (or default persp)."""
        if self._viewport_api is not None:
            path = self._viewport_api.get_active_camera()
            self._controlled_camera_path = path if path else "/OmniverseKit_Persp"
        else:
            self._controlled_camera_path = "/OmniverseKit_Persp"

    def _apply_viewport_camera_scene_partition(self, usd_stage: Usd.Stage, num_envs: int) -> None:
        """Tag the viewport camera with the first visible env partition.

        RTX scene partitioning culls per-env geometry by the camera's non-primvar
        ``omni:scenePartition`` token. Interactive viewport cameras live outside
        ``/World/envs`` and are created by Kit, so they do not inherit the env-root
        primvar authored by :class:`~isaaclab.scene.InteractiveScene`.

        This method is a no-op unless ``ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=1``,
        matching the opt-in behaviour of
        :meth:`~isaaclab_physx.renderers.IsaacRtxRenderer.prepare_stage`.
        """

        if not isaac_rtx_per_env_scene_partition_enabled():
            return

        if num_envs <= 0 or self._controlled_camera_path is None:
            return

        logger.debug(
            "[KitVisualizer] Per-environment Isaac RTX scene partitioning is enabled"
            " (ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=1)."
            " Authoring omni:scenePartition attribute onto viewport camera '%s'.",
            self._controlled_camera_path,
        )

        env_id = self._resolved_visible_env_ids[0] if self._resolved_visible_env_ids else 0
        camera_prim = usd_stage.GetPrimAtPath(self._controlled_camera_path)
        if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
            logger.debug(
                "[KitVisualizer] Scene partition token skipped for non-camera viewport prim: %s",
                self._controlled_camera_path,
            )
            return
        attr = camera_prim.GetAttribute("omni:scenePartition")
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
            return

        try:
            from omni.kit.viewport.utility.camera_state import ViewportCameraState
        except ImportError as exc:
            logger.warning("[KitVisualizer] Viewport camera update skipped: %s", exc)
            return

        camera_path = self._viewport_api.get_active_camera()
        if not camera_path:
            camera_path = "/OmniverseKit_Persp"

        # ``rotate=False`` for the position set: a freshly-opened stage's default
        # ``/OmniverseKit_Persp`` has no authored ``omni:kit:centerOfInterest``,
        # which ``set_position_world(..., rotate=True)`` would feed into
        # ``Matrix4d.Transform`` as ``None`` and crash. The follow-up
        # ``set_target_world(..., rotate=True)`` performs the look-at rotation
        # and authors the COI as a side effect, so the final pose is unchanged.
        camera_state = ViewportCameraState(camera_path, self._viewport_api)
        camera_state.set_position_world(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])), False)
        camera_state.set_target_world(Gf.Vec3d(float(target[0]), float(target[1]), float(target[2])), True)

    def _set_generated_usd_camera_poses(self, eyes: torch.Tensor, targets: torch.Tensor) -> None:
        """Author generated camera poses directly on USD camera prims for Kit/Fabric visibility."""
        # TODO: Remove this USD-side pose path once Fabric-backed camera transforms propagate reliably to Kit.
        for local_idx, env_id in enumerate(self._camera_env_indices):
            if local_idx >= eyes.shape[0]:
                break
            camera_path = (
                self._generated_camera_prim_paths[env_id]
                if 0 <= env_id < len(self._generated_camera_prim_paths)
                else f"/World/envs/env_{env_id}/VisualizerCamera"
            )
            self._generated_camera_poses_dirty |= self._set_usd_camera_pose(
                camera_path, eyes[local_idx], targets[local_idx]
            )

    def _set_usd_camera_pose(self, camera_path: str, position, target) -> bool:
        """Apply eye/target camera pose directly to a USD camera prim.

        Returns:
            ``True`` when authored values changed, otherwise ``False``.
        """
        # TODO: Remove this USD-side pose path once Fabric-backed camera transforms propagate reliably to Kit.
        usd_stage = self._scene_data_provider.usd_stage if self._scene_data_provider else None
        if usd_stage is None:
            return False

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
        if self._generated_camera_pose_cache.get(camera_path) == pose_key:
            return False

        if camera_path not in self._generated_camera_xform_ops:
            camera = UsdGeom.Camera.Define(usd_stage, camera_path)
            camera_xform = UsdGeom.Xformable(camera.GetPrim())
            camera_xform.ClearXformOpOrder()
            # Generated visualizer cameras live under env prims, but eyes/targets are world-space.
            # Reset the xform stack so Kit/Fabric sees the authored pose as a world pose.
            camera_xform.SetResetXformStack(True)
            translate_op = camera_xform.AddTranslateOp()
            orient_op = camera_xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
            self._generated_camera_xform_ops[camera_path] = (translate_op, orient_op)
        else:
            translate_op, orient_op = self._generated_camera_xform_ops[camera_path]

        quat_gf = Gf.Quatd(
            float(quat_xyzw[3]),
            Gf.Vec3d(float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])),
        )

        translate_op.Set(Gf.Vec3d(float(eye[0, 0]), float(eye[0, 1]), float(eye[0, 2])))
        orient_op.Set(quat_gf)
        self._generated_camera_pose_cache[camera_path] = pose_key
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
        usd_stage = self._scene_data_provider.usd_stage if self._scene_data_provider else None
        if usd_stage is None:
            return False
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
        usd_stage = self._scene_data_provider.usd_stage
        if usd_stage is None:
            return
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
        usd_stage = self._scene_data_provider.usd_stage if self._scene_data_provider else None
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
