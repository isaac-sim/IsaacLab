# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rerun visualizer implementation for Isaac Lab."""

from __future__ import annotations

import atexit
import contextlib
import inspect
import logging
import socket
import webbrowser
from typing import TYPE_CHECKING
from urllib.parse import quote

import newton
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from newton.viewer import ViewerRerun

from isaaclab.visualizers.base_visualizer import BaseVisualizer

from isaaclab_visualizers.newton.newton_visualization_markers import render_newton_visualization_markers
from isaaclab_visualizers.newton_adapter import (
    apply_viewer_visible_worlds,
    log_geo_with_expanded_plane_scale,
    resolve_visible_env_indices,
)

from .rerun_visualizer_cfg import RerunVisualizerCfg

if TYPE_CHECKING:
    from isaaclab.scene_data import SceneDataProvider

logger = logging.getLogger(__name__)


def _preload_ovrtx_native_deps() -> None:
    """Pre-load ``libosdCPU.so`` from ``ovstage`` so ``ovrtx.Renderer`` can resolve it.

    ``libovrtx.dylib.so`` depends on ``libosdCPU.so.3.6.0`` which ships inside the
    ``ovstage`` wheel but is not on the system ``LD_LIBRARY_PATH``.  Loading it
    explicitly places it in the process-wide ``dlopen`` cache.
    """
    import ctypes
    import importlib.util
    import pathlib

    spec = importlib.util.find_spec("ovstage")
    if spec is None:
        return
    lib = pathlib.Path(spec.origin).parent / "bin" / "plugins" / "libosdCPU.so.3.6.0"
    if lib.exists():
        with contextlib.suppress(OSError):
            ctypes.CDLL(str(lib))


def _resolve_streaming_renderer_cfg(renderer_name: str | None):
    """Return a renderer cfg for the auto-created streaming camera."""
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    if renderer_name is None or renderer_name == "newton_warp":
        return NewtonWarpRendererCfg()
    if renderer_name == "ovrtx":
        _preload_ovrtx_native_deps()
        from isaaclab_ov.renderers import OVRTXRendererCfg

        return OVRTXRendererCfg()
    if renderer_name == "isaac_rtx":
        try:
            from isaaclab_physx.renderers import IsaacRtxRendererCfg

            import omni.replicator.core  # noqa: F401

            return IsaacRtxRendererCfg()
        except ModuleNotFoundError:
            logger.info(
                "[RerunVisualizer] streaming_cam_renderer='isaac_rtx' unavailable (kitless); using newton_warp."
            )
            return NewtonWarpRendererCfg()
    raise ValueError(
        f"streaming_cam_renderer={renderer_name!r} unsupported. Use 'newton_warp', 'ovrtx', 'isaac_rtx', or None."
    )


_BACKEND_DISPLAY_NAMES = {
    "physx": "PhysX",
    "ovphysx": "OVPhysX",
    "newton": "Newton MJWarp",
}


def _is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    """Return whether a TCP port can be bound on host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, int(port)))
            return True
        except OSError:
            return False


def _is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    """Return whether a TCP port is currently accepting connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, int(port))) == 0


def _normalize_host(addr: str) -> str:
    """Normalize bind host to loopback-friendly address for client URLs."""
    if addr in ("0.0.0.0", "127.0.0.1", "localhost"):
        return "127.0.0.1"
    return addr


def _ensure_rerun_server(app_id: str, bind_address: str, grpc_port: int, web_port: int) -> tuple[str, bool]:
    """Resolve rerun endpoint and whether viewer should start web/grpc server."""
    del app_id
    connect_host = _normalize_host(bind_address)
    expected_uri = f"rerun+http://{connect_host}:{int(grpc_port)}/proxy"

    if _is_port_open(grpc_port, host=connect_host):
        # Reuse existing endpoint; do not create a new server here.
        return expected_uri, False

    if not _is_port_free(web_port, host=connect_host):
        raise RuntimeError(f"Rerun web port {web_port} is in use. Free the port or choose a different `web_port`.")

    # No existing gRPC server: NewtonViewerRerun should start and own it.
    return expected_uri, True


def _open_rerun_web_viewer(host: str, web_port: int, connect_to: str) -> None:
    """Open rerun web UI and prefill endpoint connection URL."""
    url = _rerun_web_viewer_url(host, web_port, connect_to)
    try:
        if not webbrowser.open_new_tab(url):
            logger.info("[RerunVisualizer] Could not auto-open browser tab. Open manually: %s", url)
    except Exception:
        logger.info("[RerunVisualizer] Could not auto-open browser tab. Open manually: %s", url)


def _rerun_web_viewer_url(host: str, web_port: int, connect_to: str) -> str:
    """Return rerun web UI URL with prefilled endpoint."""
    # Keep the nested URL readable while still encoding '+' in the rerun+http scheme.
    return f"http://{host}:{int(web_port)}/?url={quote(connect_to, safe=':/')}"


class NewtonViewerRerun(ViewerRerun):
    """Wrapper around Newton's ViewerRerun with rendering pause controls."""

    #: Manager names set by :meth:`RerunVisualizer.add_live_plots`; when non-empty,
    #: ``_get_blueprint`` produces one ``TimeSeriesView`` per manager instead of the
    #: default single view.
    _live_plot_manager_names: list[str]

    def __init__(self, *args, open_browser: bool = False, **kwargs):
        """Initialize viewer wrapper and Isaac Lab pause state."""
        self._live_plot_manager_names = []
        self._camera_pose: tuple | None = None
        self._streaming_view_active: bool = False
        if open_browser:
            super().__init__(*args, **kwargs)
        else:
            original_serve_web_viewer = rr.serve_web_viewer

            # Rerun Viewer launches a browser automatically, so here we suppress that behavior
            def _serve_web_viewer_without_browser(*serve_args, **serve_kwargs):
                with contextlib.suppress(TypeError, ValueError):
                    supports_open_browser = "open_browser" in inspect.signature(original_serve_web_viewer).parameters
                    if supports_open_browser:
                        serve_kwargs.setdefault("open_browser", False)
                return original_serve_web_viewer(*serve_args, **serve_kwargs)

            with contextlib.ExitStack() as stack:
                rr.serve_web_viewer = _serve_web_viewer_without_browser
                stack.callback(setattr, rr, "serve_web_viewer", original_serve_web_viewer)
                super().__init__(*args, **kwargs)
        self._paused_rendering = False
        self._reset_requested = False

    def _get_blueprint(self):
        """Return a Rerun blueprint.

        When ``streaming_view`` is active the streaming composite
        (``Spatial2DView``) is the primary full-width panel and live-plot
        time-series views are appended as a narrow right column when registered.

        When streaming is **not** active the standard 3D Newton view is used,
        with live-plot time-series views appended when registered.

        The stored :attr:`_camera_pose` is forwarded to
        :class:`~rerun.blueprint.EyeControls3D` when the 3D view is included.
        """
        manager_views = (
            [rrb.TimeSeriesView(name=name, origin=f"/{name}") for name in self._live_plot_manager_names]
            if self._live_plot_manager_names
            else []
        )

        # Streaming-view blueprint: 2D composite panel is dominant.
        if self._streaming_view_active:
            streaming_panel = rrb.Spatial2DView(name="Streaming View", origin="streaming/view")
            if manager_views:
                return rrb.Blueprint(
                    rrb.Horizontal(
                        streaming_panel,
                        rrb.Vertical(*manager_views),
                        column_shares=[4, 1],
                    ),
                    rrb.TimePanel(timeline="time", state="collapsed"),
                    collapse_panels=True,
                )
            return rrb.Blueprint(
                streaming_panel,
                rrb.TimePanel(timeline="time", state="collapsed"),
                collapse_panels=True,
            )

        # Standard 3D blueprint (no streaming).
        eye_controls = (
            rrb.EyeControls3D(position=self._camera_pose[0], look_target=self._camera_pose[1])
            if self._camera_pose
            else None
        )
        view_3d = (
            rrb.Spatial3DView(name="3D View", origin="/", eye_controls=eye_controls)
            if eye_controls
            else rrb.Spatial3DView(name="3D View", origin="/")
        )
        if manager_views:
            return rrb.Blueprint(
                rrb.Horizontal(
                    view_3d,
                    rrb.Vertical(*manager_views),
                    column_shares=[4, 1],
                ),
                rrb.TimePanel(timeline="time", state="collapsed"),
                collapse_panels=True,
            )
        return rrb.Blueprint(
            view_3d,
            rrb.TimePanel(timeline="time", state="collapsed"),
            collapse_panels=True,
        )

    def is_rendering_paused(self) -> bool:
        """Return whether rendering is paused by viewer controls."""
        return self._paused_rendering

    def is_reset_requested(self) -> bool:
        """Return whether an episode reset was requested without clearing the flag."""
        return self._reset_requested

    def consume_reset_request(self) -> bool:
        """Return whether an episode reset was requested and clear the flag."""
        requested = self._reset_requested
        self._reset_requested = False
        return requested

    def _render_ui(self):
        """Extend base UI with Isaac Lab rendering pause toggle."""
        super()._render_ui()

        if not self._has_imgui:
            return

        imgui = self._imgui
        if not imgui:
            return

        if imgui.collapsing_header("IsaacLab Controls"):
            if imgui.button("Pause Rendering" if not self._paused_rendering else "Resume Rendering"):
                self._paused_rendering = not self._paused_rendering
            if imgui.button("Reset Episode"):
                self._reset_requested = True

    def log_geo(
        self,
        name: str,
        geo_type: int,
        geo_scale: tuple[float, ...],
        geo_thickness: float,
        geo_is_solid: bool,
        geo_src=None,
        hidden: bool = False,
    ):
        """Log geometry, preserving large render extents for infinite ground planes."""
        return log_geo_with_expanded_plane_scale(
            super().log_geo,
            newton.GeoType.PLANE,
            name,
            geo_type,
            geo_scale,
            geo_thickness,
            geo_is_solid,
            geo_src,
            hidden,
        )


class RerunVisualizer(BaseVisualizer):
    """Rerun visualizer for Isaac Lab."""

    def __init__(self, cfg: RerunVisualizerCfg):
        """Initialize Rerun visualizer state.

        Args:
            cfg: Rerun visualizer configuration.
        """
        super().__init__(cfg)
        self.cfg: RerunVisualizerCfg = cfg
        self._viewer: NewtonViewerRerun | None = None
        self._backend_display: str | None = None
        self._sim_time = 0.0
        self._step_counter = 0
        self._model = None
        self._state = None
        self._last_camera_pose: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
        self._resolved_visible_env_ids: list[int] | None = None
        self._camera_sensor = None
        self._camera_sensor_indices: list[int] = []
        self._camera_env_indices: list[int] = []
        self._camera_is_owned = False
        self._generated_camera_prim_paths: list[str] = []
        self._streaming_view_active: bool = False
        self._streaming_camera_key: tuple | None = None
        self._last_streaming_composite: np.ndarray | None = None

    def initialize(self, scene_data_provider: SceneDataProvider) -> None:
        """Initialize rerun viewer and bind scene data provider.

        Args:
            scene_data_provider: Scene data provider used to fetch model/state data.
        """
        from isaaclab_newton.physics import NewtonManager

        if self._is_initialized:
            return

        scene_data_provider = self._set_scene_data_provider(scene_data_provider)
        num_envs = scene_data_provider.num_envs
        self._env_ids = self._compute_visualized_env_ids()
        self._model = NewtonManager.get_model()
        self._state = NewtonManager.get_state(self._scene_data_provider)

        grpc_port = int(self.cfg.grpc_port)
        web_port = int(self.cfg.web_port)
        bind_address = self.cfg.bind_address or "0.0.0.0"
        rerun_address, start_server_in_viewer = _ensure_rerun_server(
            app_id=self.cfg.app_id,
            bind_address=bind_address,
            grpc_port=grpc_port,
            web_port=web_port,
        )
        if not start_server_in_viewer:
            logger.info("[RerunVisualizer] Reusing existing rerun server at %s.", rerun_address)

        viewer_address = None if start_server_in_viewer else rerun_address
        self._viewer = NewtonViewerRerun(
            app_id=self.cfg.app_id,
            address=viewer_address,
            serve_web_viewer=start_server_in_viewer,
            web_port=web_port,
            grpc_port=grpc_port,
            keep_historical_data=self.cfg.keep_historical_data,
            keep_scalar_history=self.cfg.keep_scalar_history or self.cfg.enable_live_plots,
            record_to_rrd=self.cfg.record_to_rrd,
            open_browser=self.cfg.open_browser,
        )
        if start_server_in_viewer:
            rerun_address = getattr(self._viewer, "_grpc_server_uri", rerun_address)
        viewer_host = _normalize_host(bind_address)
        viewer_url = _rerun_web_viewer_url(viewer_host, web_port, rerun_address)
        print()
        self._log_viewer_url("RerunVisualizer", viewer_url)
        if self.cfg.open_browser and not start_server_in_viewer:
            _open_rerun_web_viewer(viewer_host, web_port, rerun_address)
        self._viewer.set_model(self._model)
        self._viewer.show_particles = self.cfg.show_particles
        apply_viewer_visible_worlds(
            self._viewer,
            env_ids=self._env_ids,
            max_visible_envs=self.cfg.max_visible_envs,
            num_envs=num_envs,
        )
        # Preserve simulation world positions (env_spacing) rather than adding viewer-side offsets.
        self._viewer.set_world_offsets((0.0, 0.0, 0.0))
        backend = self.physics_backend or "unknown"
        self._backend_display = _BACKEND_DISPLAY_NAMES.get(backend, backend)
        initial_pose = self._resolve_initial_camera_pose()
        self._apply_camera_pose(initial_pose)
        self._viewer.up_axis = 2
        self._viewer.scaling = 1.0
        self._viewer._paused = False

        self._resolved_visible_env_ids = resolve_visible_env_indices(self._env_ids, self.cfg.max_visible_envs, num_envs)
        num_visualized_envs = (
            len(self._resolved_visible_env_ids) if self._resolved_visible_env_ids is not None else num_envs
        )
        self._log_initialization_table(
            logger=logger,
            title="RerunVisualizer Configuration",
            rows=[
                ("eye", self.cfg.eye),
                ("lookat", self.cfg.lookat),
                ("focal_length", f"{self.cfg.focal_length} (not applied: Rerun EyeControls3D has no FOV field)"),
                ("num_visualized_envs", num_visualized_envs),
                ("endpoint", f"http://{viewer_host}:{web_port}"),
                ("bind_address", bind_address),
                ("grpc_port", grpc_port),
                ("web_port", web_port),
                ("open_browser", self.cfg.open_browser),
                ("show_particles", self.cfg.show_particles),
                ("record_to_rrd", self.cfg.record_to_rrd or "<none>"),
            ],
        )

        rr.log("info/physics_backend", rr.TextDocument(""), static=True)

        self._setup_streaming_view(num_envs)
        self._is_initialized = True
        atexit.register(self.close)

    def step(self, dt: float) -> None:
        """Advance visualization by one simulation step.

        Args:
            dt: Simulation time-step in seconds.
        """
        from isaaclab_newton.physics import NewtonManager

        if not self._is_initialized or self._is_closed or self._viewer is None:
            return

        self._sim_time += dt
        self._step_counter += 1

        self._state = NewtonManager.get_state(self._scene_data_provider)
        num_envs = NewtonManager.get_num_envs()

        if not self._viewer.is_paused():
            self._viewer.begin_frame(self._sim_time)
            try:
                if self._state is not None:
                    body_q = getattr(self._state, "body_q", None)
                    # Skip log_state for empty body arrays but do not return: _push_streaming_frame
                    # must still run after end_frame() so the streaming panel stays live.
                    if not (hasattr(body_q, "shape") and body_q.shape[0] == 0):
                        self._viewer.log_state(self._state)
                        if self.cfg.enable_markers:
                            render_newton_visualization_markers(
                                self._viewer, self._resolved_visible_env_ids, num_envs=num_envs
                            )
                self._render_live_plots()
            finally:
                self._viewer.end_frame()

        # Push streaming outside the pause-gate so it updates even when the
        # Newton viewer is paused, and outside begin/end_frame so the rr.log
        # call is not constrained to the viewer's internal time context.
        # When paused, only compose (update _last_streaming_composite for any
        # render_tiled_rgb_array() consumer) without re-logging to Rerun —
        # the viewer already holds the last frame.
        if self._viewer.is_paused():
            self._compose_streaming_frame()
        else:
            self._push_streaming_frame()

    def close(self) -> None:
        """Close viewer/session resources."""
        if self._is_closed:
            return

        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception as exc:
                logger.warning("[RerunVisualizer] Failed while closing viewer: %s", exc)
            finally:
                self._viewer = None

        if self._camera_sensor is not None and self._camera_is_owned:
            from isaaclab.envs.utils.camera_view import evict_visualizer_camera, remove_generated_prims

            evict_visualizer_camera(self._streaming_camera_key)
            remove_generated_prims(self._generated_camera_prim_paths)
        self._camera_sensor = None

        try:
            rr.disconnect()
        except Exception as exc:
            logger.warning("[RerunVisualizer] Failed while disconnecting rerun: %s", exc)
        self._is_closed = True

    def is_running(self) -> bool:
        """Return whether the visualizer should continue stepping.

        Returns:
            ``True`` while the visualizer is active, otherwise ``False``.
        """
        if not self._is_initialized or self._is_closed:
            return False
        if self._viewer is None:
            return False
        return self._viewer.is_running()

    # ------------------------------------------------------------------
    # Streaming view
    # ------------------------------------------------------------------

    def _setup_streaming_view(self, num_envs: int) -> None:
        """Resolve or create the streaming camera sensor."""
        from isaaclab.envs.utils.camera_colorizer import SUPPORTED_GT_TYPES, sensor_keys_for_gt_types
        from isaaclab.envs.utils.camera_view import (
            VISUALIZER_TILED_CAMERA_MAX_TILES,
            create_visualizer_camera,
            find_camera_by_prim_path,
            resolve_streaming_envs,
        )

        if not self.cfg.streaming_view:
            return

        gt_types = list(self.cfg.streaming_gt_types)
        for gt in gt_types:
            if gt not in SUPPORTED_GT_TYPES:
                raise ValueError(
                    f"[RerunVisualizer] streaming_gt_types contains unsupported type {gt!r}. "
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
            self._streaming_view_active = True
            self._viewer._streaming_view_active = True
            return

        # Auto-detect fallback: with Newton MJWarp replicate_physics=True, post-init prim
        # spawning only survives at env_0. Reuse the first scene camera with matching
        # renderer_type (or any scene camera with the right count as secondary fallback).
        renderer_cfg = _resolve_streaming_renderer_cfg(self.cfg.streaming_cam_renderer)
        renderer_type = getattr(renderer_cfg, "renderer_type", None)
        scene_cameras = self._scene_data_provider.get_camera_sensors()
        _fallback_cam = None
        for cam in scene_cameras.values():
            if cam._view.count != num_envs:
                continue
            if getattr(getattr(cam.cfg, "renderer_cfg", None), "renderer_type", None) == renderer_type:
                _fallback_cam = cam
                break
            if _fallback_cam is None:
                _fallback_cam = cam
        if _fallback_cam is not None:
            self._camera_sensor = _fallback_cam
            self._camera_sensor_indices = env_ids
            self._streaming_view_active = True
            self._viewer._streaming_view_active = True
            return

        tile_w, tile_h = 320, 240  # default resolution for Rerun stream (no window size)
        try:
            result = create_visualizer_camera(
                num_envs=num_envs,
                width=tile_w,
                height=tile_h,
                renderer_cfg=renderer_cfg,
                data_types=sensor_keys_for_gt_types(gt_types),
                streaming_envs=tuple(int(i) for i in env_ids),
            )
        except Exception as e:
            logger.warning("[RerunVisualizer] Streaming view disabled: could not auto-create a camera sensor (%s).", e)
            return
        self._camera_sensor, self._generated_camera_prim_paths, self._camera_is_owned, self._streaming_camera_key = (
            result
        )
        self._camera_sensor_indices = env_ids
        self._streaming_view_active = True
        self._viewer._streaming_view_active = True
        self._apply_streaming_camera_pose(env_ids)

    def _apply_streaming_camera_pose(self, env_ids: list[int]) -> None:
        """Position the auto-created streaming camera using the cfg target prim and eye offset."""
        if not self._camera_is_owned or self._camera_sensor is None:
            return
        from isaaclab.envs.utils.camera_view import apply_camera_target_positions, prim_world_positions
        from isaaclab.sim import get_current_stage

        try:
            stage = get_current_stage()
            scene = self._scene_data_provider.get_interactive_scene() if self._scene_data_provider else None
            target_positions = prim_world_positions(
                stage, self.cfg.streaming_cam_target_prim_path, env_ids, scene=scene
            )
            apply_camera_target_positions(self._camera_sensor, target_positions, self.cfg.streaming_cam_eye, env_ids)
        except Exception as exc:
            logger.debug("[RerunVisualizer] streaming camera pose: %s", exc)

    def _compose_streaming_frame(self) -> None:
        """Colorize camera tiles and store the result in ``_last_streaming_composite``.

        This is the compute-only half of streaming frame production.  It updates
        ``_last_streaming_composite`` but does **not** log the image to Rerun.
        Call :meth:`_push_streaming_frame` to compose *and* log in a single pass.
        """
        from isaaclab.envs.utils.camera_colorizer import CameraFrameColorizer, sensor_key_for_gt_type
        from isaaclab.envs.utils.camera_view import camera_gt_batch, compose_streaming_grid

        if self._camera_sensor is None:
            return
        if self._camera_is_owned:
            self._apply_streaming_camera_pose(self._camera_sensor_indices)
            self._camera_sensor.update(dt=0.0, force_recompute=True)

        gt_types = list(self.cfg.streaming_gt_types)
        available = frozenset(self._camera_sensor.data.output.keys())
        frames = []
        for env_idx in self._camera_sensor_indices:
            for gt in gt_types:
                key = sensor_key_for_gt_type(gt, available)
                raw = camera_gt_batch(self._camera_sensor, [env_idx], key)[0]
                frames.append(
                    CameraFrameColorizer.colorize(
                        raw,
                        gt,
                        depth_min=self.cfg.streaming_depth_min,
                        depth_max=self.cfg.streaming_depth_max,
                    )
                )

        n_envs = len(self._camera_sensor_indices)
        self._last_streaming_composite = compose_streaming_grid(frames, n_envs, len(gt_types))

    def _push_streaming_frame(self) -> None:
        """Compose the streaming frame and log it to Rerun."""
        self._compose_streaming_frame()
        if self._last_streaming_composite is not None:
            rr.log("streaming/view", rr.Image(self._last_streaming_composite))

    def render_tiled_rgb_array(self) -> np.ndarray | None:
        """Return the last composited streaming frame (all GT types side-by-side).

        If no frame has been composited yet (e.g. the viewer is paused and no
        push has occurred), compositing is triggered on demand so that a
        :class:`VideoRecorder` can capture headless frames.

        Returns:
            ``uint8 (H, W, 3)`` composite array, or ``None`` if streaming view
            is not active or camera data is unavailable.
        """
        if self._last_streaming_composite is None:
            self._compose_streaming_frame()
        return self._last_streaming_composite

    def _resolve_initial_camera_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Resolve initial camera pose from config."""
        return self._resolve_cfg_camera_pose("RerunVisualizer")

    def _apply_camera_pose(self, pose: tuple[tuple[float, float, float], tuple[float, float, float]]) -> None:
        """Apply camera pose to rerun's 3D view controls.

        Args:
            pose: Camera eye and target tuples.
        """
        if self._viewer is None:
            return
        cam_pos, cam_target = pose
        self._viewer._camera_pose = pose
        # Do not send a Spatial3DView blueprint when the streaming composite is active:
        # the streaming blueprint (Spatial2DView) from _get_blueprint() would be replaced
        # by a 3D view, hiding the streaming composite panel entirely.
        if self._streaming_view_active:
            return
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Vertical(
                    rrb.Spatial3DView(
                        name="3D View",
                        origin="/",
                        eye_controls=rrb.EyeControls3D(
                            position=cam_pos,
                            look_target=cam_target,
                        ),
                    ),
                    rrb.TextDocumentView(
                        name=f"Physics: {self._backend_display or 'unknown'}",
                        origin="info/physics_backend",
                    ),
                    row_shares=[20, 1],
                ),
                collapse_panels=True,
            )
        )
        self._last_camera_pose = (cam_pos, cam_target)

    def supports_markers(self) -> bool:
        """Rerun backend supports Isaac Lab markers through Newton viewer primitives."""
        return bool(self.cfg.enable_markers)

    def supports_live_plots(self) -> bool:
        """Rerun backend supports live plots via :meth:`newton.Viewer.log_scalar` (mapped to ``rr.Scalars``)."""
        return True

    def add_live_plots(
        self,
        managers: dict,
        scalars: dict | None = None,
        term_names: dict[str, list[str]] | None = None,
        env_idx: int = 0,
    ) -> None:
        """Register managers for live plotting and send a per-manager blueprint.

        Calls the base implementation to populate :attr:`_live_plot_sources`, then sends a
        Rerun blueprint with one :class:`rerun.blueprint.TimeSeriesView` per manager so that
        each manager's terms appear in a separate chart panel rather than all sharing a single
        time-series view.

        Args:
            managers: Mapping of manager name to manager instance.
            scalars: Optional mapping of group name to a dict of ``{term_name: callable}``.
                Each callable must take no arguments and return a numeric value.
            term_names: Optional per-manager allowlists of term names to include.
            env_idx: Environment index to sample each step.  Defaults to ``0``.
        """
        super().add_live_plots(managers, scalars=scalars, term_names=term_names, env_idx=env_idx)
        if self._viewer is None or not self._live_plot_sources:
            return
        # Store manager names on the viewer so _get_blueprint() returns the per-manager
        # layout.  ViewerRerun.log_scalar calls _get_blueprint() on the first scalar logged,
        # which would overwrite any blueprint we send here — so we inject the layout into
        # the viewer's own blueprint factory instead of calling rr.send_blueprint directly.
        # Build the list of Rerun series-view names.  For manager sources, one view per
        # manager groups all their terms together.  For DirectScalarLivePlots (e.g. episode
        # metrics), each scalar gets its own view so they have independent Y axes — otherwise
        # episode_length (~160) and mean_reward (~0-1) share an axis, hiding the smaller one.
        from isaaclab.ui.live_plots.manager_live_plots import DirectScalarLivePlots

        names = []
        for source in self._live_plot_sources:
            if isinstance(source, DirectScalarLivePlots):
                for term in source._scalars:
                    names.append(f"{source.manager_name}/{term}")
            else:
                names.append(source.manager_name)
        self._viewer._live_plot_manager_names = names

    def _render_live_plots(self) -> None:
        """Push manager-term scalars to Rerun as time-series scalars."""
        if self._viewer is None or not self._live_plot_sources:
            return
        self._live_plots_step_counter += 1
        if self._live_plots_step_counter % max(1, getattr(self.cfg, "live_plots_update_interval", 10)) != 0:
            return
        for source in self._live_plot_sources:
            for term_name, values in source.collect(self._live_plot_env_idx).items():
                if len(values) == 1:
                    self._viewer.log_scalar(f"{source.manager_name}/{term_name}", values[0])
                else:
                    for i, v in enumerate(values):
                        self._viewer.log_scalar(f"{source.manager_name}/{term_name}[{i}]", v)

    def is_training_paused(self) -> bool:
        """Return whether training is paused.

        Rerun viewer exposes rendering pause only.
        """
        return False

    def is_rendering_paused(self) -> bool:
        """Return whether rendering is paused from viewer controls."""
        if not self._is_initialized or self._viewer is None:
            return False
        return self._viewer.is_rendering_paused()

    def is_reset_requested(self) -> bool:
        """Return whether an episode reset was requested from viewer controls without clearing the flag."""
        if not self._is_initialized or self._viewer is None:
            return False
        return self._viewer.is_reset_requested()

    def consume_reset_request(self) -> bool:
        """Return whether an episode reset was requested from viewer controls and clear the flag."""
        if not self._is_initialized or self._viewer is None:
            return False
        return self._viewer.consume_reset_request()
