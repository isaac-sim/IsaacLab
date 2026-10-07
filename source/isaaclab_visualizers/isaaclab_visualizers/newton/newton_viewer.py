# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton presentation window and camera controls, independent of simulation and scene import."""

from __future__ import annotations

import contextlib
import math
import os
import sys

import numpy as np
import torch
import warp as wp

# Pyglet must choose EGL before Newton imports its window class.
if sys.platform not in ("win32", "darwin") and not os.environ.get("DISPLAY"):
    import pyglet

    pyglet.options["headless"] = True
elif sys.platform not in ("win32", "darwin"):
    # Monitor-free X servers use XlibScreen, which pyglet assumes has is_primary.
    with contextlib.suppress(ImportError, AttributeError):
        from pyglet.display import xlib as _pyglet_xlib

        if not hasattr(_pyglet_xlib.XlibScreen, "is_primary"):
            _pyglet_xlib.XlibScreen.is_primary = False
        del _pyglet_xlib

from newton.viewer import ViewerGL
from pyglet.math import Vec3 as PygletVec3

from isaaclab.utils.math import quat_apply, quat_from_matrix, quat_mul
from isaaclab.visualizers.visualizer_cfg import PerspectiveCameraCfg

_BACKEND_DISPLAY_NAMES = {"physx": "PhysX", "ovphysx": "OVPhysX", "newton": "Newton MJWarp"}


def _imgui_optional_checkbox(imgui, label: str, value: bool, available: bool, tip: str) -> bool:
    """Render a checkbox greyed out with a tooltip when *available* is False."""
    if not available:
        imgui.begin_disabled()
    _, new_val = imgui.checkbox(label, value)
    if not available:
        imgui.end_disabled()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(tip)
        return value
    return new_val


# ---------------------------------------------------------------------------
# Newton viewer wrappers (add IsaacLab ImGui controls to Newton's viewers)
# ---------------------------------------------------------------------------


class NewtonViewerGL(ViewerGL):
    """Newton window with Isaac Lab controls and device-image presentation."""

    # Set to False by NewtonGLVisualizer.initialize() when neither native Newton
    # contacts nor a ContactSensor exists in the scene, so the Show Contacts
    # checkbox can be greyed out in the UI.
    _contacts_available: bool = True

    CAMERA_SPEED_BOOST_MULTIPLIER = 2.0
    """Factor applied to :attr:`camera_speed` while the speed-boost modifier is held."""

    def _is_camera_speed_boost_active(self) -> bool:
        """Return whether the camera speed-boost modifier (Left/Right Shift) is held."""
        import pyglet

        return bool(self.is_key_down(pyglet.window.key.LSHIFT) or self.is_key_down(pyglet.window.key.RSHIFT))

    @property
    def camera_speed(self) -> float:
        """Keyboard camera translation speed [m/s], doubled while Shift is held."""
        base_speed = self._camera_speed
        if self._is_camera_speed_boost_active():
            return base_speed * self.CAMERA_SPEED_BOOST_MULTIPLIER
        return base_speed

    @camera_speed.setter
    def camera_speed(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("camera_speed must be finite and nonnegative")
        self._camera_speed = value

    def _patch_scalar_plot_width(self) -> None:
        """Set up ImPlot and suppress Newton's built-in floating Plots window.

        Plots are rendered inline in the left panel by
        :meth:`~NewtonGLVisualizer._live_plots_panel_imgui` instead.
        """
        gui = self.gui

        # Initialise ImPlot context once.  Newton does not use ImPlot itself, so we create
        # and own the context here.  set_imgui_context links it to the active imgui context.
        try:
            from imgui_bundle import implot as _implot

            self._implot_ctx = _implot.create_context()
            _implot.set_imgui_context(gui.ui.imgui.get_current_context())
            self._implot = _implot
        except Exception:
            self._implot = None
            self._implot_ctx = None

        # Replace Newton's floating plots window with a no-op; rendering is in the panel.
        gui._render_scalar_plots = lambda: None

    def is_reset_requested(self) -> bool:
        """Return whether an episode reset was requested without clearing the flag."""
        return self._reset_requested

    def consume_reset_request(self) -> bool:
        """Return whether an episode reset was requested and clear the flag."""
        requested = self._reset_requested
        self._reset_requested = False
        return requested

    def _patch_viewer_panel(self) -> None:
        """Replace Newton's left panel with an IsaacLab-oriented layout.

        New section order:

        1. **Isaac Lab** (open) — physics backend, model info, training controls.
        2. **Live Plots** (closed) — injected when :meth:`~NewtonGLVisualizer.add_live_plots`
           is called.
        3. **Model Overlays** and **Visualization Markers** (closed) — debug visibility controls.
        4. **Rendering Options** (open) — VSync and renderer-specific options.
        5. **Wind** (closed) — only shown when ``viewer.wind`` is set.
        6. **Controls** (closed) — camera keyboard reference.
        7. **Selection API** (closed) — Newton's selection panel.

        The top-level Newton ``Pause / Step`` row is suppressed; pause/resume is
        handled by the IsaacLab training controls inside **Isaac Lab**.
        """
        import newton as nt

        gui = self.gui

        def _render_left_panel(_g=gui):
            if not _g.is_available:
                return

            viewer = _g._viewer
            imgui = _g.ui.imgui
            io = _g.ui.io
            s = _g.ui.dpi_scale
            nav_highlight_color = _g.ui.get_theme_color(imgui.Col_.nav_cursor, (1.0, 1.0, 1.0, 1.0))

            imgui.set_next_window_pos(imgui.ImVec2(10 * s, 10 * s), imgui.Cond_.first_use_ever)
            imgui.set_next_window_size(
                imgui.ImVec2(363 * s, io.display_size[1] - 20 * s),
                imgui.Cond_.first_use_ever,
            )
            panel_h = io.display_size[1] - 20 * s
            imgui.set_next_window_size_constraints(
                imgui.ImVec2(160 * s, panel_h),
                imgui.ImVec2(io.display_size[0], panel_h),
            )

            if not imgui.begin(f"Newton Viewer v{nt.__version__}"):
                imgui.end()
                return

            imgui.separator()

            # Layers panel callback (ViewerGL built-in, only shown with >1 layer).
            for callback in _g._ui_callbacks.get("panel", []):
                callback(imgui)

            # --- Simulation -------------------------------------------------
            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Simulation"):
                imgui.separator()
                imgui.text(f"Physics: {viewer._backend_display}")
                if viewer.model is not None:
                    axis_names = ["X", "Y", "Z"]
                    imgui.text(f"Up Axis: {axis_names[viewer.model.up_axis]}")
                    gravity = viewer._metadata["gravity"]
                    imgui.text(f"Gravity: ({gravity[0]:.2f}, {gravity[1]:.2f}, {gravity[2]:.2f})")
                imgui.separator()
                for callback in _g._ui_callbacks.get("side", []):
                    callback(imgui)

            # --- Live Plots -------------------------------------------------
            live_plots_cb = viewer._live_plots_callback
            if live_plots_cb is not None:
                live_plots_cb(imgui)

            # --- Model overlays ---------------------------------------------
            if viewer.model is not None and viewer._main_image_name is None:
                imgui.set_next_item_open(False, imgui.Cond_.appearing)
                if imgui.collapsing_header("Model Overlays"):
                    imgui.separator()
                    renderer = viewer.renderer
                    _c, viewer.show_joints = imgui.checkbox("Show Joints", viewer.show_joints)
                    if viewer.show_joints:
                        _, renderer.joint_scale = imgui.slider_float("Joint Scale", renderer.joint_scale, 0.25, 5.0)
                    _contacts_available = viewer._contacts_available
                    viewer.show_contacts = _imgui_optional_checkbox(
                        imgui,
                        "Show Contacts",
                        viewer.show_contacts,
                        _contacts_available,
                        "No contact sensors in this environment",
                    )
                    if viewer.show_contacts and _contacts_available:
                        _, renderer.arrow_length_scale = imgui.slider_float(
                            "Contact Length", renderer.arrow_length_scale, 0.25, 5.0
                        )
                        _, renderer.arrow_scale = imgui.slider_float("Contact Width", renderer.arrow_scale, 0.25, 5.0)
                    _model = viewer.model
                    _has_particles = _model.particle_count > 0
                    _has_springs = _model.spring_count > 0
                    _has_cloth = _model.tri_count > 0
                    viewer.show_particles = _imgui_optional_checkbox(
                        imgui,
                        "Show Particles",
                        viewer.show_particles,
                        _has_particles,
                        "No particle bodies in this environment",
                    )
                    viewer.show_springs = _imgui_optional_checkbox(
                        imgui,
                        "Show Springs",
                        viewer.show_springs,
                        _has_springs,
                        "No spring constraints in this environment",
                    )
                    _c, viewer.show_com = imgui.checkbox("Show Center of Mass", viewer.show_com)
                    if viewer.show_com:
                        _, renderer.com_scale = imgui.slider_float("COM Scale", renderer.com_scale, 0.25, 5.0)
                    viewer.show_triangles = _imgui_optional_checkbox(
                        imgui,
                        "Show Cloth",
                        viewer.show_triangles,
                        _has_cloth,
                        "No cloth/triangle meshes in this environment",
                    )
                    _c, viewer.show_collision = imgui.checkbox("Show Collision", viewer.show_collision)
                    _c, renderer.draw_edges = imgui.checkbox("Show Edges", renderer.draw_edges)
                    _sdf_labels = ["Off", "Margin", "Margin + Gap"]
                    _, new_sdf_idx = imgui.combo("Gap + Margin", int(viewer.sdf_margin_mode), _sdf_labels)
                    viewer.sdf_margin_mode = viewer.SDFMarginMode(new_sdf_idx)
                    if viewer.sdf_margin_mode != viewer.SDFMarginMode.OFF:
                        _, renderer.wireframe_line_width = imgui.slider_float(
                            "Wireframe Width (px)", renderer.wireframe_line_width, 0.5, 5.0
                        )
                    _c, viewer.show_visual = imgui.checkbox("Show Visual", viewer.show_visual)
                    _c, viewer.show_inertia_boxes = imgui.checkbox("Show Inertia Boxes", viewer.show_inertia_boxes)
            if viewer.marker_groups:
                imgui.set_next_item_open(False, imgui.Cond_.appearing)
                if imgui.collapsing_header("Visualization Markers"):
                    for marker in viewer.marker_groups:
                        name = marker.cfg.prim_path.rsplit("/", 1)[-1].replace("_", " ")
                        changed, visible = imgui.checkbox(f"Show {name}##{id(marker)}", marker.is_visible())
                        if changed:
                            marker.set_visibility(visible)

            # --- Rendering Options ------------------------------------------
            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Rendering Options"):
                imgui.separator()
                _c, viewer.vsync = imgui.checkbox("VSync", viewer.vsync)
                if viewer.model is not None and viewer._main_image_name is None:
                    for callback in _g._ui_callbacks.get("rendering", []):
                        callback(imgui)

            # --- Wind -------------------------------------------------------
            wind = getattr(viewer, "wind", None)
            if wind is not None:
                imgui.set_next_item_open(False, imgui.Cond_.once)
                if imgui.collapsing_header("Wind"):
                    imgui.separator()
                    changed, wind.amplitude = imgui.slider_float("Wind Amplitude", wind.amplitude, -2.0, 2.0, "%.2f")
                    changed, wind.period = imgui.slider_float("Wind Period", wind.period, 1.0, 30.0, "%.2f")
                    changed, wind.frequency = imgui.slider_float("Wind Frequency", wind.frequency, 0.1, 5.0, "%.2f")
                    direction = [wind.direction[0], wind.direction[1], wind.direction[2]]
                    changed, direction = imgui.slider_float3("Wind Direction", direction, -1.0, 1.0, "%.2f")
                    if changed:
                        wind.direction = direction

            # --- Controls ---------------------------------------------------
            imgui.set_next_item_open(False, imgui.Cond_.appearing)
            if imgui.collapsing_header("Controls"):
                imgui.separator()
                _g._render_camera_info()
                imgui.separator()
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*nav_highlight_color))
                imgui.text("Controls:")
                imgui.pop_style_color()
                imgui.text("WASD - Move camera")
                imgui.text("Shift + WASD - Move camera 2x speed")
                imgui.text("QE - Pan up/down")
                imgui.text("Left Click - Look around")
                if viewer.picking_enabled and viewer._main_image_name is None:
                    imgui.text("Right Click - Pick and drag objects")
                imgui.text("Middle Click - Orbit")
                imgui.text("Shift + Middle Click - Pan")
                imgui.text("Ctrl + Middle Click - Dolly")
                imgui.text("Scroll - Dolly")
                imgui.text("Ctrl + Scroll - FOV zoom")
                imgui.text("Space - Pause/Resume Rendering")
                imgui.text(". - Step one frame (when paused)")
                imgui.text("H - Toggle UI")
                imgui.text("F - Frame camera around model")

            # --- Selection API ----------------------------------------------
            _g._render_selection_panel()

            imgui.end()

        gui._render_left_panel = _render_left_panel

    def _render_training_controls(self, imgui):
        """Render Isaac Lab training control widgets inside the Isaac Lab panel section."""
        pause_label = "Resume Simulation" if self._paused_training else "Pause Simulation"
        if imgui.button(pause_label):
            self._paused_training = not self._paused_training

        rendering_label = "Resume Rendering" if self._paused else "Pause Rendering"
        if imgui.button(rendering_label):
            self._paused = not self._paused

        if imgui.button("Reset Episode"):
            self._reset_requested = True

        imgui.text("Visualizer Update Frequency")
        current_frequency = self._update_frequency
        changed, new_frequency = imgui.slider_int(
            "##VisualizerUpdateFreq", current_frequency, 1, 20, f"Every {current_frequency} frames"
        )
        if changed:
            self._update_frequency = new_frequency

        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Controls visualizer update frequency\nlower values -> more responsive visualizer but slower"
                " training\nhigher values -> less responsive visualizer but faster training"
            )

    def __init__(self, *args, metadata: dict | None = None, update_frequency: int = 1, **kwargs):
        """Initialize Newton viewer wrapper state.

        Args:
            *args: Positional arguments forwarded to ``ViewerGL``.
            metadata: Optional metadata shown in viewer panels.
            update_frequency: Viewer refresh cadence in simulation frames.
            **kwargs: Keyword arguments forwarded to ``ViewerGL``.
        """
        super().__init__(*args, **kwargs)
        self._paused_training = False
        self._reset_requested = False
        self._metadata = metadata or {}
        self._update_frequency = update_frequency
        self.particle_color: tuple[float, float, float] | None = None
        self._particle_color_buffer: wp.array | None = None
        self._particle_color_buffer_count = 0
        self._particle_color_buffer_value: tuple[float, float, float] | None = None
        self._mpm_particle_flags_cache_key: tuple[int, int, int] | None = None
        self._mpm_particles_all_active = False
        self._live_plots_callback = None
        self.marker_groups = ()
        backend = self._metadata.get("physics_backend", "Unknown")
        self._backend_display = _BACKEND_DISPLAY_NAMES.get(backend, backend)

        if self.gui is not None:
            self._patch_scalar_plot_width()
            self._patch_viewer_panel()

        self.register_ui_callback(self._render_training_controls, position="side")
        self._close_requested = False

    def is_training_paused(self) -> bool:
        """Return whether simulation is paused by viewer controls."""
        return self._paused_training

    def is_rendering_paused(self) -> bool:
        """Return whether rendering is paused by viewer controls.

        Mirrors ``self._paused`` directly since the Newton viewer's Space key handler toggles it
        in-place, outside the Isaac Lab "Pause Rendering" button.
        """
        return self._paused

    def request_close(self) -> None:
        """Close the window once the current frame ends."""
        self._close_requested = True

    def end_frame(self) -> None:
        """Finish the frame, then close the window if :meth:`request_close` was called."""
        super().end_frame()
        if self._close_requested:
            self.renderer.close()

    def on_key_press(self, symbol, modifiers):
        """Forward key presses unless UI is currently capturing input."""
        if self.ui.is_capturing():
            return
        super().on_key_press(symbol, modifiers)

    def on_mouse_press(self, x, y, button, modifiers):
        """Image coordinates must not pick objects in the hidden perspective scene."""
        if self._main_image_name is None:
            super().on_mouse_press(x, y, button, modifiers)

    def _particle_color_array(self, count: int) -> wp.array:
        """Return a cached Warp color array for Newton's particle point batch."""
        color = tuple(self.particle_color)
        if (
            self._particle_color_buffer is None
            or self._particle_color_buffer_count != count
            or self._particle_color_buffer_value != color
        ):
            self._particle_color_buffer = wp.full(
                shape=count,
                value=wp.vec3(*color),
                dtype=wp.vec3,
                device=self.device,
            )
            self._particle_color_buffer_count = count
            self._particle_color_buffer_value = color
        return self._particle_color_buffer

    def _particle_color_update_array(self, name: str, count: int) -> wp.array | None:
        """Return particle colors only when Newton needs the GL color buffer refreshed."""
        obj = self.objects.get(name)
        capacity = obj.num_instances if obj is not None else 0
        if obj is None or count > capacity or self._particle_color_buffer_value != tuple(self.particle_color):
            return self._particle_color_array(max(count, capacity))
        return None

    def log_points(self, name, points, radii=None, colors=None, hidden=False):
        """Apply configured model-particle appearance while preserving Newton's point logging.

        The configured particle color only applies to Newton's canonical
        ``/model/particles`` point batch. User-defined point clouds retain the
        colors provided by their own ``log_points`` calls.
        """
        if name != "/model/particles" or points is None or self.particle_color is None:
            return super().log_points(name, points, radii, colors, hidden)

        colors = self._particle_color_update_array(name, len(points))
        return super().log_points(name, points, radii, colors, hidden)

    def _all_mpm_particles_active(self) -> bool:
        """Return whether an MPM model's static particle flags are all active."""
        model = self.model
        if model is None or getattr(model, "mpm", None) is None or not model.particle_count:
            return False
        if model.particle_flags is None:
            return False

        cache_key = (id(model), id(model.particle_flags), int(model.particle_count))
        if self._mpm_particle_flags_cache_key != cache_key:
            import newton as nt

            flags = model.particle_flags.numpy()[: model.particle_count]
            self._mpm_particles_all_active = bool(((flags & int(nt.ParticleFlags.ACTIVE)) != 0).all())
            self._mpm_particle_flags_cache_key = cache_key
        return self._mpm_particles_all_active

    def _log_particles(self, state):
        """Log MPM particles without per-frame active-flag compaction when all particles are active.

        Newton's base implementation stream-compacts active particles every
        frame, which costs two device-to-host reads per render. MPM particle
        flags are static, so when they are all active the compaction is skipped
        and ``state.particle_q`` is logged directly.
        """
        if not self._all_mpm_particles_active():
            super()._log_particles(state)
            return

        colors = None
        if self.model_changed and self.particle_color is None:
            colors = wp.full(shape=len(state.particle_q), value=wp.vec3(0.7, 0.6, 0.4), device=self.device)

        self.log_points(
            name="/model/particles",
            points=state.particle_q,
            radii=self.model.particle_radius,
            colors=colors,
            hidden=not self.show_particles,
        )


class _NewtonCameraControls:
    """Camera selection and navigation shared by the GL scene and RTX image windows."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._camera_index = 0
        self._navigation_view = None

    def _draw_streaming_view_controls(self, imgui) -> None:
        """Choose one declared display source without reading the other sensors."""
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if not imgui.collapsing_header("Camera View"):
            return
        labels = [
            f"Perspective {i + 1}" if isinstance(camera, PerspectiveCameraCfg) else camera.cfg.prim_path
            for i, camera in enumerate(self._camera_choices)
        ]
        changed, index = imgui.combo("Camera", self._camera_index, labels)
        if changed:
            self._select_camera(index)
        if imgui.is_item_hovered():
            imgui.set_tooltip(labels[self._camera_index])

    def _select_camera(self, index: int) -> None:
        """Bind the selected source; all scene cameras retain their normal lazy-data lifecycle."""
        camera = self._camera_choices[index]
        self._camera_sensor = None if isinstance(camera, PerspectiveCameraCfg) else camera
        self._streaming_frame.timestamp = -1.0
        if isinstance(camera, PerspectiveCameraCfg):
            self.set_camera_view(camera.eye, camera.lookat)
            self.cfg.focal_length = camera.focal_length
            self._viewer.camera.fov = self._focal_length_to_vertical_fov_degrees()
        self._camera_index = index
        if self._viewer.picking is not None:
            # Release an existing drag without replacing any graph-captured picking buffers.
            self._viewer.picking.release()

        self._navigation_view = self._viewer.camera.get_view_matrix().reshape(4, 4).T.copy()

    def _navigate_scene_camera(self) -> None:
        camera, viewer = self._camera_sensor, self._viewer
        if camera is None or viewer is None:
            return
        width, height = viewer.renderer.window.get_framebuffer_size()
        self._streaming_aspect = width / max(height, 1)
        if self._runtime_headless or viewer.is_paused():
            return
        current = viewer.camera.get_view_matrix().reshape(4, 4).T
        previous, self._navigation_view = self._navigation_view, current.copy()
        if previous is None or np.array_equal(previous, current):
            return
        delta = previous @ np.linalg.inv(current)
        positions, orientations = camera.get_world_poses(convention="opengl")
        delta = torch.as_tensor(delta, dtype=positions.dtype, device=positions.device)
        translation = quat_apply(orientations, delta[:3, 3].expand_as(positions))
        rotation = quat_from_matrix(delta[:3, :3]).expand_as(orientations)
        camera.set_world_poses(positions + translation, quat_mul(orientations, rotation), convention="opengl")
        self._streaming_frame.timestamp = -1.0

    def set_camera_view(
        self, eye: tuple[float, float, float] | list[float], target: tuple[float, float, float] | list[float]
    ) -> None:
        """Set the configured and active camera eye and target positions [m]."""
        self.cfg.eye, self.cfg.lookat = tuple(float(v) for v in eye), tuple(float(v) for v in target)
        if self._viewer is not None:
            self._viewer.camera.pos = PygletVec3(*self.cfg.eye)
            self._viewer.camera.look_at(self.cfg.lookat)
