# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Viser-based visualizer using Newton's ViewerViser."""

from __future__ import annotations

import contextlib
import io
import logging
import math
import os
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any

import newton
import numpy as np
from isaaclab_newton.physics import NewtonBackendCfg
from newton.viewer import ViewerViser

from isaaclab.scene_data import SceneDataFormat
from isaaclab.sim import SimulationContext
from isaaclab.visualizers import VisualizerCfg
from isaaclab.visualizers.base_visualizer import BaseVisualizer
from isaaclab.visualizers.visualizer_cfg import PerspectiveCameraCfg

from isaaclab_visualizers.newton.newton_visualization_markers import render_newton_visualization_markers
from isaaclab_visualizers.newton_adapter import log_geo_with_expanded_plane_scale

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from isaaclab.sensors.camera import Camera


def _letterbox_16_9(image: np.ndarray) -> np.ndarray:
    """Pad *image* with black bars to 16:9 so Viser doesn't stretch it.

    Args:
        image: ``uint8 (H, W, 3)`` composite frame.

    Returns:
        ``uint8 (H', W', 3)`` image with ``W'/H' == 16/9``.
    """
    h, w = image.shape[:2]
    target_w = max(w, int(h * 16 / 9))
    target_h = max(h, int(w * 9 / 16))
    if target_w == w and target_h == h:
        return image
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - h) // 2
    x0 = (target_w - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = image
    return canvas


def _scalar_base_name(name: str) -> str:
    """Strip a trailing ``[N]`` component index from a scalar name to get the term base name."""
    if name.endswith("]") and "[" in name:
        bracket = name.rfind("[")
        if name[bracket + 1 : -1].isdigit():
            return name[:bracket]
    return name


_BACKEND_DISPLAY_NAMES = {
    "physx": "PhysX",
    "ovphysx": "OVPhysX",
    "newton": "Newton MJWarp",
}


class NewtonViewerViser(ViewerViser):
    """Isaac Lab wrapper for Newton's ViewerViser."""

    def __init__(
        self,
        port: int = 8080,
        bind_address: str = "0.0.0.0",
        label: str | None = None,
        verbose: bool = True,
        share: bool = False,
        record_to_viser: str | None = None,
    ):
        """Initialize Newton-backed viser viewer wrapper.

        Args:
            port: HTTP port for viser server.
            bind_address: Host/interface for the Viser server to bind.
            label: Optional viewer label.
            verbose: Whether to keep verbose startup output enabled.
            share: Whether to enable sharing/tunneling.
            record_to_viser: Optional recording destination.
        """
        import viser
        import viser._client_autobuild as client_autobuild

        self._isaaclab_plane_grid_cache: dict[str, tuple] = {}
        self._per_plot_folders: dict[str, Any] = {}
        self._live_plots_folder: Any = None

        # Use the bundled frontend without rebuilding it at viewer startup.
        if (Path(viser.__file__).resolve().parent / "client" / "build" / "index.html").exists():
            client_autobuild.ensure_client_is_built = lambda: None
        original_viser_server = viser.ViserServer

        def _viser_server_with_bind_address(*args, **kwargs):
            kwargs["host"] = bind_address
            kwargs["verbose"] = verbose
            return original_viser_server(*args, **kwargs)

        with contextlib.ExitStack() as stack:
            viser.ViserServer = _viser_server_with_bind_address
            stack.callback(setattr, viser, "ViserServer", original_viser_server)
            if not verbose:
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            super().__init__(
                port=port,
                label=label,
                verbose=verbose,
                share=share,
                record_to_viser=record_to_viser,
            )

    def clear_model(self) -> None:
        """Clear cached state and remove per-plot GUI folders with the viewer model."""
        self._isaaclab_plane_grid_cache.clear()
        super().clear_model()
        for folder in self._per_plot_folders.values():
            folder.remove()
        self._per_plot_folders.clear()
        # The sidebar's live-plots folder survives model reloads.

    @staticmethod
    def _array_signature(array) -> tuple[tuple[int, ...], bytes] | None:
        """Return a stable signature for small transform/scale arrays."""
        if array is None:
            return None
        array_np = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
        return tuple(int(dim) for dim in array_np.shape), array_np.tobytes()

    def _log_plane_instances(
        self,
        name: str,
        plane_info: dict[str, float | bool],
        xforms,
        scales,
        hidden: bool = False,
    ) -> None:
        """Avoid removing/re-adding unchanged Viser plane grids every frame."""
        if hidden or xforms is None:
            self._isaaclab_plane_grid_cache.pop(name, None)
            return super()._log_plane_instances(name, plane_info, xforms, scales, hidden=hidden)

        xforms_np = self._to_numpy(xforms)
        if xforms_np is None or len(xforms_np) == 0:
            self._isaaclab_plane_grid_cache.pop(name, None)
            return super()._log_plane_instances(name, plane_info, xforms, scales, hidden=hidden)

        scales_np = self._to_numpy(scales) if scales is not None else None
        signature = (
            float(plane_info["width"]),
            float(plane_info["length"]),
            self._array_signature(xforms_np),
            self._array_signature(scales_np),
        )
        if self._isaaclab_plane_grid_cache.get(name) == signature and name in self._plane_handles:
            return None
        self._isaaclab_plane_grid_cache[name] = signature
        return super()._log_plane_instances(name, plane_info, xforms, scales, hidden=hidden)

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

    def _update_scalar_plots(self) -> None:
        """Create one collapsible folder per term, with one multi-series chart per term.

        Components of the same term (e.g. ``joint_pos[0]``, ``joint_pos[1]``) are grouped
        onto a single uPlot chart as separate series, matching the Kit visualizer's per-term
        grouping.  Single-value terms get a chart with one data series.

        Relies on private ViewerViser attributes (_plot_history_size, _scalar_buffers,
        _scalar_dirty, _plot_handles, _plot_folder).  If Newton refactors these internals
        this override should be updated or removed.
        """
        if not self._scalar_dirty:
            return
        try:
            from viser import uplot

            _SERIES_COLORS = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#f97316", "#06b6d4"]

            # Identify which term groups (base names) have at least one dirty component.
            dirty_bases: set[str] = {_scalar_base_name(name) for name in self._scalar_dirty}

            # Collect all known scalars grouped by base term name (insertion order preserved).
            all_groups: dict[str, list[str]] = {}
            for name in self._scalar_buffers:
                base = _scalar_base_name(name)
                all_groups.setdefault(base, []).append(name)

            for base_name, names in all_groups.items():
                if base_name not in dirty_bases:
                    continue

                # Use only the filled portion of the buffer — no NaN padding — so the
                # chart visibly grows over time rather than appearing static at the right edge.
                bufs = [self._scalar_buffers.get(name) for name in names]
                n_actual = max((len(b) for b in bufs if b), default=0)
                if n_actual == 0:
                    continue
                x = np.arange(n_actual, dtype=np.float64)
                ys = [np.array(list(b)[:n_actual], dtype=np.float64) if b else np.full(n_actual, np.nan) for b in bufs]
                data = (x, *ys)

                handle = self._plot_handles.get(names[0])
                if handle is None:
                    folder_label = base_name.rsplit("/", 1)[-1]
                    parent = self._live_plots_folder if self._live_plots_folder is not None else self._server.gui
                    with parent:
                        folder = self._server.gui.add_folder(folder_label, expand_by_default=False)
                    self._per_plot_folders[base_name] = folder
                    if self._plot_folder is None:
                        self._plot_folder = folder

                    series_list = [uplot.Series(label="step", show=False)]
                    for i, name in enumerate(names):
                        suffix = name[len(base_name) :]  # "" for scalar, "[0]" etc. for vector
                        series_list.append(
                            uplot.Series(
                                label=suffix if suffix else folder_label,
                                stroke=_SERIES_COLORS[i % len(_SERIES_COLORS)],
                                width=1,
                            )
                        )
                    with folder:
                        handle = self._server.gui.add_uplot(
                            data=data,
                            series=tuple(series_list),
                            scales={"x": uplot.Scale(time=False)},
                            aspect=1.33,
                        )
                    for name in names:
                        self._plot_handles[name] = handle
                else:
                    handle.data = data
        except Exception:
            pass
        self._scalar_dirty.clear()


class ViserVisualizer(BaseVisualizer):
    """Viser web-based visualizer backed by Newton's ViewerViser."""

    def __init__(self, cfg: VisualizerCfg):
        """Initialize Viser visualizer state.

        Args:
            cfg: Viser visualizer configuration.
        """
        super().__init__(cfg)
        self._viewer: NewtonViewerViser | None = None
        self.backend = None
        self._pending_camera_pose: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
        self._warned_marker_render_failure = False
        self._paused_rendering = False
        self._paused_simulation = False

    def initialize(self, sim: SimulationContext, *, cameras: list[PerspectiveCameraCfg | Camera]) -> None:
        """Initialize viewer resources and bind scene data provider.

        Args:
            sim: Simulation owner used to resolve scene data and native resources.
            cameras: Resolved perspective settings and borrowed scene sensors, in display order.
        """
        if self._is_initialized:
            logger.debug("[ViserVisualizer] initialize() called while already initialized.")
            return

        super().initialize(sim, cameras=cameras)
        scene_data_provider = self._sim.get_scene_data_provider()
        num_envs = scene_data_provider.num_envs
        self.newton_cfg = NewtonBackendCfg(physics_cfg=sim.cfg.physics, device=sim.device)
        self.backend = sim.get_or_create_backend(self.newton_cfg)
        self._transform_mapping = scene_data_provider.create_mapping(list(self.backend.model.body_label))

        self._viewer = NewtonViewerViser(
            port=self.cfg.backend.port,
            bind_address=self.cfg.backend.bind_address,
            label="Isaac Lab",
            verbose=False,
            share=self.cfg.backend.share,
            record_to_viser=self.cfg.backend.record_to_viser,
        )
        self._viewer._server.gui.add_markdown(f"Physics: {_BACKEND_DISPLAY_NAMES[self.physics_backend]}")
        viewer_url = self._viewer._share_url or f"http://{self.cfg.backend.display_address}:{self.cfg.backend.port}"
        if self.cfg.backend.verbose:
            print()
            self._log_viewer_url("ViserVisualizer", viewer_url)
        self._viewer.set_model(self.backend.model)
        self._viewer.show_particles = self.cfg.backend.show_particles
        # set_model() clears model-dependent GUI elements; build the sidebar afterwards.
        self._setup_isaaclab_sidebar(self._viewer._server)
        self._viewer.set_visible_worlds(self._env_ids)
        # Preserve simulation world positions rather than adding viewer-side offsets.
        self._viewer.set_world_offsets((0.0, 0.0, 0.0))
        if self.cfg.backend.open_browser:
            webbrowser.open_new_tab(viewer_url)
        self.set_camera_view(self.cfg.eye, self.cfg.lookat)
        self._viewer.marker_groups = sim.vis_marker_registry.get_groups().values()
        num_visualized_envs = len(self._env_ids) if self._env_ids is not None else num_envs
        self._log_initialization_table(
            logger=logger,
            title="ViserVisualizer Configuration",
            rows=[
                ("eye", self.cfg.eye),
                ("lookat", self.cfg.lookat),
                ("focal_length", self.cfg.focal_length),
                ("num_visualized_envs", num_visualized_envs),
                ("bind_address", self.cfg.backend.bind_address),
                ("display_address", self.cfg.backend.display_address),
                ("port", self.cfg.backend.port),
                ("record_to_viser", self.cfg.backend.record_to_viser or "<none>"),
            ],
        )
        self._is_initialized = True

    def step(self, dt: float) -> None:
        """Advance visualization by one simulation step.

        Args:
            dt: Simulation time-step in seconds.
        """
        if not self._is_initialized:
            return

        if self._pending_camera_pose is not None:
            self.set_camera_view(*self._pending_camera_pose)

        self._sim_time += dt

        # Skip all rendering when no browser clients are connected.
        if not self._viewer._server.get_clients():
            self._render_live_plots()  # still throttled internally; no-ops when no clients
            # No browser clients: skip compositing and pushing entirely.  If a
            # VideoRecorder calls render_tiled_rgb_array() it will compose on demand.
            return

        if self._paused_rendering:
            # Push streaming outside the pause-gate so it updates even when
            # rendering is paused, matching Rerun's behaviour.
            self._push_streaming_frame()
            return

        self._viewer.begin_frame(self._sim_time)
        try:
            # When streaming_view is active, skip the 3D Newton scene so the
            # background streaming composite is the only content visible.
            if not self.cfg.streaming_view:
                backend, provider = self.backend, self._sim.get_scene_data_provider()
                poses = SceneDataFormat.Transform()
                if provider.get_transforms(poses, mapping=self._transform_mapping, count=backend.model.body_count):
                    backend.state_0.body_q = poses.transforms
                if backend.geometry_offsets:
                    provider.get_geometry_points(output=backend.state_0.particle_q, offsets=backend.geometry_offsets)
                self._viewer.log_state(self.backend.state_0)
                if self.cfg.enable_markers:
                    self._render_markers(self.backend.model.num_envs)
            self._render_live_plots()
            self._push_streaming_frame()
        finally:
            self._viewer.end_frame()

    # ------------------------------------------------------------------
    # Streaming view
    # ------------------------------------------------------------------

    def _push_streaming_frame(self) -> None:
        """Compose the streaming frame and push it to connected Viser clients."""
        composite = self.render_tiled_rgb_array()
        if composite is None:
            return
        # Letterbox to 16:9 so the composite isn't stretched when Viser fills
        # the browser canvas.  Black bars are added on whichever axis needs it.
        self._viewer._server.scene.set_background_image(_letterbox_16_9(composite), format="jpeg")

    def _render_markers(self, num_envs: int) -> None:
        """Render marker overlays without letting them interrupt Viser body updates."""
        try:
            render_newton_visualization_markers(self._viewer, self._env_ids, num_envs=num_envs)
        except Exception as exc:
            if not self._warned_marker_render_failure:
                logger.warning("[ViserVisualizer] Marker rendering failed; continuing body updates: %s", exc)
                self._warned_marker_render_failure = True
            else:
                logger.debug("[ViserVisualizer] Marker rendering failed: %s", exc)

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
        self._setup_isaaclab_sidebar(self._viewer._server)
        self._viewer.set_visible_worlds(self._env_ids)
        self._viewer.set_world_offsets((0.0, 0.0, 0.0))

    def close(self) -> None:
        """Close viewer resources and finalize optional recording."""
        if not self._is_initialized:
            return
        try:
            self._viewer.close()
            if self.cfg.backend.record_to_viser:
                if os.path.exists(self.cfg.backend.record_to_viser):
                    size = os.path.getsize(self.cfg.backend.record_to_viser)
                    logger.info(
                        "[ViserVisualizer] Recording saved: %s (%s bytes)", self.cfg.backend.record_to_viser, size
                    )
                else:
                    logger.warning("[ViserVisualizer] Recording file not found: %s", self.cfg.backend.record_to_viser)
        except Exception as exc:
            logger.warning("[ViserVisualizer] Error during close: %s", exc)

        self._viewer = None
        self._is_initialized = False
        self.backend = self._transform_mapping = None
        super().close()
        self._pending_camera_pose = None

    def is_running(self) -> bool:
        """Return whether the visualizer should continue stepping.

        Returns:
            ``True`` while the visualizer is active, otherwise ``False``.
        """
        return self._is_initialized and not self._is_closed and self._viewer.is_running()

    def is_training_paused(self) -> bool:
        """Return whether simulation is paused from viewer controls."""
        return self._paused_simulation

    def is_rendering_paused(self) -> bool:
        """Return whether rendering is paused from viewer controls."""
        return self._paused_rendering

    def supports_markers(self) -> bool:
        """Viser backend supports Isaac Lab markers through Newton viewer primitives."""
        return bool(self.cfg.enable_markers)

    def supports_live_plots(self) -> bool:
        """Viser backend supports live plots via :meth:`newton.Viewer.log_scalar` (uPlot sidebar charts)."""
        return True

    def _render_live_plots(self) -> None:
        """Push manager-term scalars to the Viser viewer's per-term plot folders."""
        if self._viewer is None or not self._live_plot_sources:
            return
        # Skip when no browser clients are connected — nobody is watching the plots.
        if not self._viewer._server.get_clients():
            return
        self._live_plots_step_counter += 1
        if self._live_plots_step_counter % max(1, self.cfg.live_plots_update_interval) != 0:
            return
        for source in self._live_plot_sources:
            for term_name, values in source.collect(self._live_plot_env_idx).items():
                if len(values) == 1:
                    self._viewer.log_scalar(f"{source.manager_name}/{term_name}", values[0])
                else:
                    for i, v in enumerate(values):
                        self._viewer.log_scalar(f"{source.manager_name}/{term_name}[{i}]", v)

    def _setup_isaaclab_sidebar(self, server) -> None:
        """Configure the Viser sidebar as the Isaac Lab panel.

        The panel is renamed to ``Isaac Lab``.  ``Live Plots`` and
        ``Visualization Markers`` are added as top-level collapsed folders
        directly inside the panel alongside the physics backend label.
        """
        viewer = self._viewer
        with contextlib.suppress(Exception):
            server.gui.set_panel_label("Isaac Lab")

            pause_rendering_btn = server.gui.add_button("Pause Rendering", color=None)

            @pause_rendering_btn.on_click
            def _(_):
                self._paused_rendering = not self._paused_rendering
                pause_rendering_btn.label = "Resume Rendering" if self._paused_rendering else "Pause Rendering"
                pause_rendering_btn.color = "orange" if self._paused_rendering else None

            pause_simulation_btn = server.gui.add_button("Pause Simulation", color=None)

            @pause_simulation_btn.on_click
            def _(_):
                self._paused_simulation = not self._paused_simulation
                pause_simulation_btn.label = "Resume Simulation" if self._paused_simulation else "Pause Simulation"
                pause_simulation_btn.color = "orange" if self._paused_simulation else None

            reset_button = server.gui.add_button("Reset Episode")

            @reset_button.on_click
            def _(_):
                self._reset_requested = True

            live_plots_folder = server.gui.add_folder("Live Plots", expand_by_default=False)
            viewer._live_plots_folder = live_plots_folder

            vis_folder = server.gui.add_folder("Visualization Markers", expand_by_default=False)

            _VIZ_FLAGS = [
                ("Joints", "show_joints"),
                ("Contacts", "show_contacts"),
                ("Center of Mass", "show_com"),
                ("Particles", "show_particles"),
                ("Visual", "show_visual"),
                ("Collision", "show_collision"),
                ("Springs", "show_springs"),
                ("Cloth", "show_triangles"),
                ("Inertia Boxes", "show_inertia_boxes"),
            ]
            with vis_folder:
                for label, attr in _VIZ_FLAGS:
                    if not hasattr(viewer, attr):
                        continue
                    cb = server.gui.add_checkbox(label, initial_value=getattr(viewer, attr))

                    @cb.on_update
                    def _(event, _attr=attr):
                        setattr(viewer, _attr, event.target.value)

    def set_camera_view(
        self, eye: tuple[float, float, float] | list[float], target: tuple[float, float, float] | list[float]
    ) -> None:
        """Set every connected client's camera eye/target.

        Args:
            eye: Camera eye position.
            target: Camera look-at target.
        """
        self._pending_camera_pose = (tuple(float(v) for v in eye), tuple(float(v) for v in target))
        if self._viewer is None:
            return
        clients = self._viewer._server.get_clients()
        for client in clients.values():
            client.camera.fov = math.radians(self._focal_length_to_vertical_fov_degrees())
            client.camera.position, client.camera.look_at = self._pending_camera_pose
        if clients:
            self._pending_camera_pose = None
