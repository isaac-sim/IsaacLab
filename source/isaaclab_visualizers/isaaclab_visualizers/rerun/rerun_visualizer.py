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
import rerun as rr
import rerun.blueprint as rrb
from isaaclab_newton.physics import NewtonBackendCfg
from newton.viewer import ViewerRerun

from isaaclab.scene_data import SceneDataFormat
from isaaclab.sim import SimulationContext
from isaaclab.visualizers.base_visualizer import BaseVisualizer

from isaaclab_visualizers.newton.newton_visualization_markers import render_newton_visualization_markers
from isaaclab_visualizers.newton_adapter import (
    apply_viewer_visible_worlds,
    log_geo_with_expanded_plane_scale,
    resolve_visible_env_indices,
)

from .rerun_visualizer_cfg import RerunVisualizerCfg

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.cloner import ClonePlan
    from isaaclab.scene_data import SceneDataProvider

logger = logging.getLogger(__name__)


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

    def __init__(self, *args, open_browser: bool = False, streaming_view: bool = False, **kwargs):
        """Initialize viewer wrapper and Isaac Lab pause state."""
        self._live_plot_manager_names = []
        self._camera_pose: tuple | None = None
        self._streaming_view_active = streaming_view
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
        # TimePanel is always hidden (this viewer has no scrubbing UI use case).
        panel_states = [rrb.TimePanel(state="hidden")]

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
                    *panel_states,
                    collapse_panels=True,
                )
            return rrb.Blueprint(
                streaming_panel,
                *panel_states,
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
                *panel_states,
                collapse_panels=True,
            )
        return rrb.Blueprint(
            view_3d,
            *panel_states,
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
        self._step_counter = 0
        self.backend = None
        self._last_camera_pose: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
        self._resolved_visible_env_ids: list[int] | None = None

    def initialize(
        self,
        scene_data_provider: SceneDataProvider,
        *,
        stage: Usd.Stage | None = None,
        clone_plan: ClonePlan | None = None,
    ) -> None:
        """Initialize rerun viewer and bind scene data provider.

        Args:
            scene_data_provider: Scene data provider used to fetch model/state data.
            stage: Authored scene stage, when available.
            clone_plan: Scene topology and environment namespace, when available.
        """
        if self._is_initialized:
            return

        super().initialize(scene_data_provider, stage=stage, clone_plan=clone_plan)
        num_envs = scene_data_provider.num_envs
        self._env_ids = self._compute_visualized_env_ids()
        sim = SimulationContext.instance()
        self.newton_cfg = NewtonBackendCfg(physics_cfg=sim.cfg.physics, device=sim.device)
        self.backend = sim.get_or_create_backend(self.newton_cfg)
        self._transform_mapping = scene_data_provider.create_mapping(list(self.backend.model.body_label))

        self._resolved_visible_env_ids = resolve_visible_env_indices(self._env_ids, self.cfg.max_visible_envs, num_envs)
        self._setup_streaming_view(num_envs, visible_env_ids=self._resolved_visible_env_ids)
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
            streaming_view=self._camera_sensor is not None,
        )
        if start_server_in_viewer:
            rerun_address = getattr(self._viewer, "_grpc_server_uri", rerun_address)
        viewer_host = _normalize_host(bind_address)
        viewer_url = _rerun_web_viewer_url(viewer_host, web_port, rerun_address)
        print()
        self._log_viewer_url("RerunVisualizer", viewer_url)
        if self.cfg.open_browser and not start_server_in_viewer:
            _open_rerun_web_viewer(viewer_host, web_port, rerun_address)
        self._viewer.set_model(self.backend.model)
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

        self._is_initialized = True
        atexit.register(self.close)

    def step(self, dt: float) -> None:
        """Advance visualization by one simulation step.

        Args:
            dt: Simulation time-step in seconds.
        """
        if not self._is_initialized or self._is_closed or self._viewer is None:
            return

        self._sim_time += dt
        self._step_counter += 1

        num_envs = self.backend.model.num_envs

        if not self._viewer.is_paused():
            backend, provider = self.backend, self._scene_data_provider
            poses = SceneDataFormat.Transform()
            if provider.get_transforms(poses, mapping=self._transform_mapping, count=backend.model.body_count):
                backend.state_0.body_q = poses.transforms
            if backend.geometry_offsets:
                provider.get_geometry_points(output=backend.state_0.particle_q, offsets=backend.geometry_offsets)
            self._viewer.begin_frame(self._sim_time)
            try:
                # Empty body arrays skip log_state, but streaming still runs after end_frame.
                body_q = backend.state_0.body_q
                if body_q is None or body_q.shape[0]:
                    self._viewer.log_state(backend.state_0)
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
        # When paused, only compose (update _streaming_frame for any
        # render_tiled_rgb_array() consumer) without re-logging to Rerun —
        # the viewer already holds the last frame.
        if self._viewer.is_paused():
            self.render_tiled_rgb_array()
        else:
            self._push_streaming_frame()

    def reset(self, soft: bool = False) -> None:
        """Rebind the viewer when a hard reset replaces the shared native model."""
        super().reset(soft)
        if soft or not self._is_initialized or self._is_closed:
            return
        sim = SimulationContext.instance()
        backend = sim.get_or_create_backend(self.newton_cfg)
        if backend is self.backend:
            return
        self.backend = backend
        self._transform_mapping = self._scene_data_provider.create_mapping(list(backend.model.body_label))
        self._viewer.set_model(backend.model)
        self._viewer.set_visible_worlds(self._resolved_visible_env_ids)
        self._viewer.set_world_offsets((0.0, 0.0, 0.0))

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

        try:
            rr.disconnect()
        except Exception as exc:
            logger.warning("[RerunVisualizer] Failed while disconnecting rerun: %s", exc)
        self.backend = self._transform_mapping = None
        super().close()

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

    def _push_streaming_frame(self) -> None:
        """Compose the streaming frame and log it to Rerun."""
        composite = self.render_tiled_rgb_array()
        if composite is not None:
            rr.log("streaming/view", rr.Image(composite))

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
        if self._camera_sensor is not None:
            return
        panel_states = [rrb.TimePanel(state="hidden")]
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
                *panel_states,
                collapse_panels=True,
            )
        )
        self._last_camera_pose = (cam_pos, cam_target)

    def set_camera_view(
        self, eye: tuple[float, float, float] | list[float], target: tuple[float, float, float] | list[float]
    ) -> None:
        """Set the 3D view's camera eye/target.

        Args:
            eye: Camera eye position.
            target: Camera look-at target.
        """
        eye_t = (float(eye[0]), float(eye[1]), float(eye[2]))
        target_t = (float(target[0]), float(target[1]), float(target[2]))
        self._apply_camera_pose((eye_t, target_t))

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
