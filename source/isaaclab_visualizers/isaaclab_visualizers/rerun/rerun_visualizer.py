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
import warp as wp
from isaaclab_newton.physics import NewtonBackendCfg
from newton.viewer import ViewerRerun

from isaaclab.scene_data import SceneDataFormat
from isaaclab.sim import SimulationContext
from isaaclab.visualizers import VisualizerCfg
from isaaclab.visualizers.base_visualizer import BaseVisualizer
from isaaclab.visualizers.visualizer_cfg import PerspectiveCameraCfg

from isaaclab_visualizers.newton.newton_visualization_markers import render_newton_visualization_markers
from isaaclab_visualizers.newton_adapter import log_geo_with_expanded_plane_scale

if TYPE_CHECKING:
    from isaaclab.sensors.camera import Camera

logger = logging.getLogger(__name__)


_BACKEND_DISPLAY_NAMES = {
    "physx": "PhysX",
    "ovphysx": "OVPhysX",
    "newton": "Newton MJWarp",
}


class NewtonViewerRerun(ViewerRerun):
    """Wrapper around Newton's ViewerRerun with rendering pause controls."""

    #: Manager names set by :meth:`RerunVisualizer.add_live_plots`; when non-empty,
    #: ``_get_blueprint`` produces one ``TimeSeriesView`` per manager instead of the
    #: default single view.
    _live_plot_manager_names: list[str]

    def __init__(self, *args, open_browser: bool = False, streaming_view: bool = False, **kwargs):
        """Initialize viewer wrapper and Isaac Lab pause state."""
        self._live_plot_manager_names = []
        self._instance_appearance: dict[str, tuple[np.ndarray | None, np.ndarray | None]] = {}
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
        if self._streaming_view_active:
            view = rrb.Spatial2DView(name="Streaming View", origin="streaming/view")
        else:
            eye_controls = (
                rrb.EyeControls3D(position=self._camera_pose[0], look_target=self._camera_pose[1])
                if self._camera_pose is not None
                else None
            )
            view = rrb.Spatial3DView(name="3D View", origin="/", eye_controls=eye_controls)
        if self._live_plot_manager_names:
            plots = [rrb.TimeSeriesView(name=name, origin=f"/{name}") for name in self._live_plot_manager_names]
            view = rrb.Horizontal(view, rrb.Vertical(*plots), column_shares=[4, 1])
        return rrb.Blueprint(view, rrb.TimePanel(state="hidden"), collapse_panels=True)

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

    def log_instances(
        self,
        name: str,
        mesh: str,
        xforms: wp.array[wp.transform] | None,
        scales: wp.array[wp.vec3] | None,
        colors: wp.array[wp.vec3] | None,
        materials: wp.array[wp.vec4] | None,
        hidden: bool = False,
        opacities: wp.array[wp.float32] | None = None,
    ):
        """Avoid resending an instance mesh when its visible appearance did not change."""
        qualified_name = self._qualify(name)
        previous = self._instance_appearance.get(qualified_name) if qualified_name in self._instances else None
        color = self._first_instance_value(colors)
        opacity = self._first_instance_value(opacities)
        if previous is not None:
            if color is not None and np.array_equal(color, previous[0]):
                colors = None
            if opacity is not None and np.array_equal(opacity, previous[1]):
                opacities = None

        result = super().log_instances(name, mesh, xforms, scales, colors, materials, hidden, opacities)
        if hidden:
            self._instance_appearance.pop(qualified_name, None)
        elif qualified_name in self._instances:
            previous_color, previous_opacity = previous or (None, None)
            self._instance_appearance[qualified_name] = (
                previous_color if color is None else color,
                previous_opacity if opacity is None else opacity,
            )
        return result

    @staticmethod
    def _first_instance_value(values: wp.array | np.ndarray | None) -> np.ndarray | None:
        """Copy the first value that Rerun uses as the batch appearance."""
        if values is None or len(values) == 0:
            return None
        values_np = values.numpy() if isinstance(values, wp.array) else np.asarray(values)
        return np.asarray(values_np[0]).copy()


class RerunVisualizer(BaseVisualizer):
    """Rerun visualizer for Isaac Lab."""

    def __init__(self, cfg: VisualizerCfg):
        """Initialize Rerun visualizer state.

        Args:
            cfg: Rerun visualizer configuration.
        """
        super().__init__(cfg)
        self._viewer: NewtonViewerRerun | None = None
        self._backend_display: str | None = None
        self._step_counter = 0
        self.backend = None
        self._last_camera_pose: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None

    def initialize(self, sim: SimulationContext, *, cameras: list[PerspectiveCameraCfg | Camera]) -> None:
        """Initialize rerun viewer and bind scene data provider.

        Args:
            sim: Simulation owner used to resolve scene data and native resources.
            cameras: Resolved perspective settings and borrowed scene sensors, in display order.
        """
        if self._is_initialized:
            return

        super().initialize(sim, cameras=cameras)
        scene_data_provider = self._sim.get_scene_data_provider()
        num_envs = scene_data_provider.num_envs
        self.newton_cfg = NewtonBackendCfg(physics_cfg=sim.cfg.physics, device=sim.device)
        self.backend = sim.get_or_create_backend(self.newton_cfg)
        self._transform_mapping = scene_data_provider.create_mapping(list(self.backend.model.body_label))

        grpc_port = int(self.cfg.backend.grpc_port)
        web_port = int(self.cfg.backend.web_port)
        bind_address = self.cfg.backend.bind_address or "0.0.0.0"
        viewer_host = "127.0.0.1" if bind_address in ("0.0.0.0", "localhost") else bind_address
        rerun_address = f"rerun+http://{viewer_host}:{grpc_port}/proxy"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            start_server_in_viewer = sock.connect_ex((viewer_host, grpc_port)) != 0
        if start_server_in_viewer:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind((viewer_host, web_port))
                except OSError as error:
                    raise RuntimeError(
                        f"Rerun web port {web_port} is in use. Free the port or choose a different `web_port`."
                    ) from error
        else:
            logger.info("[RerunVisualizer] Reusing existing rerun server at %s.", rerun_address)

        self._viewer = NewtonViewerRerun(
            app_id=self.cfg.backend.app_id,
            address=None if start_server_in_viewer else rerun_address,
            serve_web_viewer=start_server_in_viewer,
            web_port=web_port,
            grpc_port=grpc_port,
            keep_historical_data=self.cfg.backend.keep_historical_data,
            keep_scalar_history=self.cfg.backend.keep_scalar_history or self.cfg.enable_live_plots,
            record_to_rrd=self.cfg.backend.record_to_rrd,
            open_browser=self.cfg.backend.open_browser,
            streaming_view=(self.current_image_view is not None and self.current_image_view.camera is not None),
        )
        self._viewer.marker_groups = sim.vis_marker_registry.get_groups().values()
        if start_server_in_viewer:
            rerun_address = self._viewer._grpc_server_uri
        viewer_url = f"http://{viewer_host}:{web_port}/?url={quote(rerun_address, safe=':/')}"
        print()
        self._log_viewer_url("RerunVisualizer", viewer_url)
        if self.cfg.backend.open_browser and not start_server_in_viewer:
            try:
                if not webbrowser.open_new_tab(viewer_url):
                    logger.info("[RerunVisualizer] Could not auto-open browser tab. Open manually: %s", viewer_url)
            except Exception:
                logger.info("[RerunVisualizer] Could not auto-open browser tab. Open manually: %s", viewer_url)
        self._viewer.set_model(self.backend.model)
        self._viewer.show_particles = self.cfg.backend.show_particles
        self._viewer.set_visible_worlds(self._env_ids)
        # Preserve simulation world positions (env_spacing) rather than adding viewer-side offsets.
        self._viewer.set_world_offsets((0.0, 0.0, 0.0))
        backend = self.physics_backend or "unknown"
        self._backend_display = _BACKEND_DISPLAY_NAMES.get(backend, backend)
        self._apply_camera_pose((self.cfg.eye, self.cfg.lookat))
        self._viewer.up_axis = 2
        self._viewer.scaling = 1.0
        self._viewer._paused = False

        num_visualized_envs = len(self._env_ids) if self._env_ids is not None else num_envs
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
                ("open_browser", self.cfg.backend.open_browser),
                ("show_particles", self.cfg.backend.show_particles),
                ("record_to_rrd", self.cfg.backend.record_to_rrd or "<none>"),
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
            backend, provider = self.backend, self._sim.get_scene_data_provider()
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
                        render_newton_visualization_markers(self._viewer, self._env_ids, num_envs=num_envs)
                self._render_live_plots()
            finally:
                self._viewer.end_frame()

        # Compose outside the viewer frame context; paused viewers keep their last logged image.
        composite = self.render_tiled_rgb_array()
        if composite is not None and not self._viewer.is_paused():
            rr.log("streaming/view", rr.Image(composite))

    def reset(self, soft: bool = False) -> None:
        """Rebind the viewer when a hard reset replaces the shared native model."""
        super().reset(soft)
        if soft or not self._is_initialized or self._is_closed:
            return
        backend = self._sim.get_or_create_backend(self.newton_cfg)
        if backend is self.backend:
            return
        self.backend = backend
        self._transform_mapping = self._sim.get_scene_data_provider().create_mapping(list(backend.model.body_label))
        self._viewer.set_model(backend.model)
        self._viewer.set_visible_worlds(self._env_ids)
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
        if self.current_image_view is not None and self.current_image_view.camera is not None:
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
