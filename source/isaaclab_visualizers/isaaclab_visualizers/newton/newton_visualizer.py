# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton GL and RTX visualization of the authored scene and camera images."""

from __future__ import annotations

import contextlib
import logging
import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

import newton
from isaaclab_newton.physics import NewtonBackendCfg, NewtonManager
from newton.viewer import ViewerGL, ViewerRTX
from pyglet.math import Vec3 as PygletVec3

from isaaclab.scene_data import SceneDataFormat
from isaaclab.sensors.camera import Camera
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_apply, quat_from_matrix, quat_mul
from isaaclab.visualizers.base_visualizer import BaseVisualizer
from isaaclab.visualizers.visualizer_cfg import PerspectiveCameraCfg, WindowCfg

from isaaclab_visualizers.desktop_entry import write_desktop_entry
from isaaclab_visualizers.newton.newton_visualization_markers import render_newton_visualization_markers

from .newton_visualizer_cfg import NewtonGLVisualizerCfg

logger = logging.getLogger(__name__)


CONTACT_ARROW_PATH = "/contacts"
"""Viewer path used for native and synthesized contact arrows."""

CONTACT_ARROW_COLOR = (0.0, 1.0, 0.0)
"""Color used by Newton's native contact visualization."""

CONTACT_ARROW_LENGTH = 0.1
"""Length of synthesized contact arrows in meters."""


@dataclass(frozen=True)
class _MeshSubmission:
    """Mesh data staged for the next Newton viewer frame."""

    name: str
    points: wp.array
    indices: wp.array
    normals: wp.array | None
    uvs: wp.array | None
    texture: np.ndarray | str | None
    hidden: bool
    backface_culling: bool
    color: tuple[float, float, float] | None
    roughness: float | None
    metallic: float | None
    dynamic: bool
    opacity: float | None


_NEWTON_ICON_DIR = Path(newton.__file__).parent / "_src" / "viewer" / "gl"


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


class NewtonViewerUI:
    """Isaac Lab controls shared by Newton's native GL and RTX viewers."""

    _contacts_available = True
    CAMERA_SPEED_BOOST_MULTIPLIER = 2.0

    @property
    def camera_speed(self) -> float:
        """Keyboard camera translation speed [m/s], doubled while Shift is held."""
        from pyglet.window import key

        boosted = self.is_key_down(key.LSHIFT) or self.is_key_down(key.RSHIFT)
        return self._camera_speed * (self.CAMERA_SPEED_BOOST_MULTIPLIER if boosted else 1.0)

    @camera_speed.setter
    def camera_speed(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("camera_speed must be finite and nonnegative")
        self._camera_speed = value

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
        5. **Controls** (closed) — camera keyboard reference.
        6. **Selection API** (closed) — Newton's selection panel.

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
            if isinstance(viewer, ViewerGL) and viewer.model is not None and viewer._main_image_name is None:
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
                for callback in _g._ui_callbacks.get("rendering", []):
                    callback(imgui)

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
                if isinstance(viewer, ViewerGL) and viewer.picking_enabled and viewer._main_image_name is None:
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

        _, self.window_cfg.fps = imgui.slider_float("Window FPS", self.window_cfg.fps, 1.0, 120.0, "%.0f FPS")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Maximum window updates per second; independent of simulation speed.")

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

    def __init__(self, *args, window_cfg: WindowCfg, metadata: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._paused_training = False
        self._reset_requested = False
        self._metadata = metadata or {}
        self.window_cfg = window_cfg
        self._live_plots_callback = None
        self.marker_groups = ()
        self._close_requested = False
        backend = self._metadata.get("physics_backend", "Unknown")
        self._backend_display = _BACKEND_DISPLAY_NAMES.get(backend, backend)
        if self.gui is not None:
            self._patch_viewer_panel()
        self.register_ui_callback(self._render_training_controls, position="side")


class NewtonViewerRTX(NewtonViewerUI, ViewerRTX):
    """Newton RTX window borrowing the simulation's authored stage."""

    def _init_window(self) -> None:
        super()._init_window()
        self._patch_viewer_panel()

    def end_frame(self) -> None:
        super().end_frame()
        if self._close_requested:
            self.close()

    def get_frame(self) -> np.ndarray:
        """Explicitly download the latest perspective image for recording."""
        return np.ascontiguousarray(self._capture_screenshot_pixels()[..., :3])


class NewtonViewerGL(NewtonViewerUI, ViewerGL):
    """Newton GL window with Isaac Lab controls and device-image presentation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.particle_color: tuple[float, float, float] | None = None
        self._particle_color_buffer: wp.array | None = None
        self._particle_color_buffer_value: tuple[float, float, float] | None = None
        self._mpm_particle_flags_cache_key: tuple[int, int, int] | None = None
        self._mpm_particles_all_active = False
        self._implot = self._implot_ctx = None
        if self.gui is not None:
            from imgui_bundle import implot

            self._implot = implot
            self._implot_ctx = implot.create_context()
            implot.set_imgui_context(self.gui.ui.imgui.get_current_context())
            # Live plots are drawn in the sidebar, not Newton's floating window.
            self.gui._render_scalar_plots = lambda: None

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

    def log_points(self, name, points, radii=None, colors=None, hidden=False):
        """Apply configured model-particle appearance while preserving Newton's point logging.

        The configured particle color only applies to Newton's canonical
        ``/model/particles`` point batch. User-defined point clouds retain the
        colors provided by their own ``log_points`` calls.
        """
        if name != "/model/particles" or points is None or self.particle_color is None:
            return super().log_points(name, points, radii, colors, hidden)

        color = tuple(self.particle_color)
        obj = self.objects.get(name)
        capacity = obj.num_instances if obj is not None else 0
        colors = None
        if obj is None or len(points) > capacity or self._particle_color_buffer_value != color:
            count = max(len(points), capacity)
            if (
                self._particle_color_buffer is None
                or len(self._particle_color_buffer) != count
                or self._particle_color_buffer_value != color
            ):
                self._particle_color_buffer = wp.full(count, wp.vec3(*color), dtype=wp.vec3, device=self.device)
                self._particle_color_buffer_value = color
            colors = self._particle_color_buffer
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


class NewtonVisualizerBase(BaseVisualizer):
    """Own Newton resource bindings, frame cadence, and camera selection for both renderers."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._viewer = None
        self.backend = None
        self._runtime_headless = cfg.headless
        self._last_present_time = -math.inf
        self._camera_index = 0
        self._navigation_view = None

    def initialize(self, sim: SimulationContext, *, cameras: list[PerspectiveCameraCfg | Camera]) -> None:
        """Bind scene data and the Newton model shared with the simulation."""
        super().initialize(sim, cameras=cameras)
        self.newton_cfg = NewtonBackendCfg(physics_cfg=sim.cfg.physics, device=sim.device)
        self.backend = self._sim.get_or_create_backend(self.newton_cfg)
        self._transform_mapping = self._sim.get_scene_data_provider().create_mapping(
            list(self.backend.model.body_label)
        )
        self._runtime_headless = self.cfg.headless or (
            sys.platform not in ("win32", "darwin") and not os.environ.get("DISPLAY")
        )

    def _update_state(self) -> newton.State:
        """Bind current scene arrays without reading any camera sensors."""
        backend, provider = self.backend, self._sim.get_scene_data_provider()
        state = backend.state_0
        poses = SceneDataFormat.Transform()
        if provider.get_transforms(poses, mapping=self._transform_mapping, count=backend.model.body_count):
            state.body_q = poses.transforms
        if backend.geometry_offsets:
            provider.get_geometry_points(output=state.particle_q, offsets=backend.geometry_offsets)
        return state

    def _draw_streaming_view_controls(self, imgui) -> None:
        """Choose one declared display source without reading the other sensors."""
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if not imgui.collapsing_header("Camera View"):
            return
        labels = [
            f"Perspective {i + 1}" if isinstance(camera, PerspectiveCameraCfg) else camera.cfg.prim_path
            for i, camera in enumerate(self._cameras)
        ]
        changed, index = imgui.combo("Camera", self._camera_index, labels)
        if changed:
            self._select_camera(index)
        if imgui.is_item_hovered():
            imgui.set_tooltip(labels[self._camera_index])

    def _select_camera(self, index: int) -> None:
        """Bind the selected source; all scene cameras retain their normal lazy-data lifecycle."""
        camera = self._cameras[index]
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
        if camera is None or viewer is None or self._runtime_headless:
            return
        if isinstance(viewer, NewtonViewerGL):
            width, height = viewer.renderer.window.get_framebuffer_size()
            self._streaming_aspect = width / max(height, 1)
        if viewer.is_paused():
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
            camera = self._viewer.camera
            self._viewer.set_camera(camera.pos, camera.pitch, camera.yaw)

    def step(self, dt: float) -> None:
        """Advance the display at the configured cadence; headless consumers capture on demand."""
        if not self._is_initialized or self._is_closed:
            return
        self._sim_time += dt
        if self._runtime_headless:
            return
        now = time.monotonic()
        if now - self._last_present_time < 1.0 / self.cfg.window.fps:
            return
        self._last_present_time = now
        self._render_frame()

    def reset(self, soft: bool = False) -> None:
        """Rebind the native viewer only when a hard reset replaces the simulation model."""
        super().reset(soft)
        if soft or not self._is_initialized or self._is_closed:
            return
        backend = self._sim.get_or_create_backend(self.newton_cfg)
        if backend is self.backend:
            return
        self.backend = backend
        self._transform_mapping = self._sim.get_scene_data_provider().create_mapping(list(backend.model.body_label))
        self._viewer.set_model(backend.model)
        self._viewer.register_ui_callback(self._viewer._render_training_controls, position="side")
        self._viewer.register_ui_callback(self._draw_streaming_view_controls, position="side")

    def supports_markers(self) -> bool:
        """Return whether visualization markers are enabled for this viewer."""
        return self.cfg.enable_markers

    def is_running(self) -> bool:
        """Return whether the viewer is open."""
        return self._viewer is not None and self._viewer.is_running()

    def is_training_paused(self) -> bool:
        """Return whether the window requested a simulation pause."""
        return self._viewer is not None and self._viewer.is_training_paused()

    def is_rendering_paused(self) -> bool:
        """Return whether the window paused scene updates."""
        return self._viewer is not None and self._viewer.is_rendering_paused()

    def is_reset_requested(self) -> bool:
        """Return whether an episode reset is pending."""
        return self._viewer is not None and self._viewer.is_reset_requested()

    def consume_reset_request(self) -> bool:
        """Consume an episode reset request from the window."""
        return self._viewer is not None and self._viewer.consume_reset_request()

    def is_key_down(self, key: str) -> bool:
        """Return whether a window key is held."""
        return self._viewer is not None and self._viewer.is_key_down(key)

    def _log_streaming_image(self) -> None:
        """Present sensor pixels directly; paused rendering retains the last displayed image."""
        composite = self._streaming_frame.data
        if composite is None or self._streaming_frame.timestamp < 0 or not self._viewer.is_paused():
            composite = self.render_tiled_rgba_array()
        if composite is None:
            raise ValueError("Scene-camera display requires at least one selected camera frame.")
        self._viewer.log_image("Streaming View", composite, fullscreen=True)


class NewtonGLVisualizer(NewtonVisualizerBase):
    """Newton model rendering, picking, and contact overlays used by the GL visualizer."""

    @property
    def visual_material_writer(self):
        """Return the shared Newton model color-writer factory."""
        return self.backend.create_visual_material_writer

    class _ViewerPickingBinding:
        """Stable Newton-manager callback for viewer picking.

        CUDA graphs record picking arrays by address, so closing the window
        neutralizes and retains them until the captured graph is gone.
        """

        def __init__(self) -> None:
            self._viewer: NewtonViewerGL | None = None
            self._retained_picking = None

        def bind(self, viewer: NewtonViewerGL) -> None:
            """Bind picking to the current viewer model."""
            self._viewer = viewer
            self._retained_picking = None

        def apply(self, state: newton.State) -> None:
            """Apply picking while the viewer is active."""
            if self._viewer is None:
                # Host callbacks do not run during graph replay, so reaching
                # this branch means captured inputs are no longer needed.
                self._retained_picking = None
                return
            self._viewer.apply_forces(state)

        def deactivate(self) -> None:
            """Make captured picking inert while preserving its inputs."""
            viewer = self._viewer
            if viewer is None:
                return

            picking = viewer.picking
            if picking is not None:
                viewer.picking_enabled = False
                picking.release()

            self._retained_picking = picking
            self._viewer = None

    def __init__(self, cfg: NewtonGLVisualizerCfg):
        """Initialize the GL scene renderer and window.

        Args:
            cfg: Newton visualizer configuration.
        """
        super().__init__(cfg)
        self._viewer_picking_binding = self._ViewerPickingBinding()
        self._picking_enabled = False
        self._pending_mesh_submissions: dict[str, _MeshSubmission] = {}

    # ------------------------------------------------------------------
    # GL lifecycle
    # ------------------------------------------------------------------

    def initialize(self, sim: SimulationContext, *, cameras: list[PerspectiveCameraCfg | Camera]) -> None:
        """Initialize viewer resources and bind scene data provider.

        Args:
            sim: Simulation owner used to resolve scene data and native resources.
            cameras: Resolved perspective settings and borrowed scene sensors, in display order.
        """

        if self._is_initialized:
            logger.debug("[%s] initialize() called while already initialized.", type(self).__name__)
            return

        super().initialize(sim, cameras=cameras)
        scene_data_provider = self._sim.get_scene_data_provider()
        newton_backend_active = self.physics_backend == "newton"
        picking_supported = newton_backend_active and sim.physics_manager._supports_rigid_body_force_input
        num_envs = scene_data_provider.num_envs
        if self._runtime_headless and not self.cfg.headless:
            logger.warning(
                "[NewtonVisualizer] No display found (DISPLAY is unset); the Newton viewer runs"
                " headless via EGL and no window will open. Run from a session with a display (or set"
                " DISPLAY, e.g. 'export DISPLAY=:0') to see the viewer."
            )

        self._picking_enabled = self.cfg.enable_picking and picking_supported and not self._runtime_headless
        if not self._runtime_headless:
            # pyglet sets WM_CLASS from the window caption, which ViewerGL defaults to "Newton".
            write_desktop_entry("isaaclab-newton-gl-viewer", "Newton", "Newton", _NEWTON_ICON_DIR / "icon_64.png")
        self._viewer = NewtonViewerGL(
            width=self.cfg.window.size[0],
            height=self.cfg.window.size[1],
            headless=self._runtime_headless,
            metadata={"num_envs": num_envs, "physics_backend": self.physics_backend, "gravity": sim.cfg.gravity},
            window_cfg=self.cfg.window,
        )
        self._viewer.marker_groups = sim.vis_marker_registry.get_groups().values()
        self._viewer.set_model(self.backend.model)
        if self._picking_enabled:
            # Keep Newton's public force path scoped to picking for this integration.
            self._viewer.wind = None
        self._viewer.set_visible_worlds(self._env_ids)
        self._viewer.set_world_offsets(self.cfg.world_spacing)
        self._viewer._paused = False
        self._viewer.picking_enabled = self._picking_enabled
        self._configure_viewer()

        self._setup_streaming_view(
            num_envs,
            visible_env_ids=self._env_ids,
            target_aspect=self.cfg.window.size[0] / self.cfg.window.size[1],
        )

        self._viewer.register_ui_callback(self._draw_streaming_view_controls, position="side")
        self._select_camera(self._camera_index)

        num_visualized_envs = len(self._env_ids) if self._env_ids is not None else num_envs
        self._log_initialization_table(
            logger=logger,
            title=f"{type(self).__name__} Configuration",
            rows=[
                ("eye", self.cfg.eye),
                ("lookat", self.cfg.lookat),
                ("focal_length", self.cfg.focal_length),
                ("background_color", self.cfg.background_color),
                ("streaming_view", self.cfg.streaming_view),
                ("streaming_gt_types", list(self.cfg.streaming_gt_types)),
                ("num_visualized_envs", num_visualized_envs),
                ("headless", self.cfg.headless),
                ("show_particles", self.cfg.show_particles),
                ("enable_picking", self._picking_enabled),
            ],
        )
        if self._picking_enabled:
            self._viewer_picking_binding.bind(self._viewer)
            NewtonManager.register_state_force_callback(self._viewer_picking_binding.apply)
        if self.cfg.enable_picking and not picking_supported:
            logger.info(
                "[NewtonVisualizer] Object dragging is disabled because the active physics solver does not support"
                " rigid-body force input."
            )
        self._is_initialized = True
        # Inform the viewer whether contact data is available so the UI can grey
        # out "Show Contacts" when neither native Newton contacts nor a ContactSensor
        # exists in the scene.
        self._viewer._contacts_available = newton_backend_active or bool(scene_data_provider.get_contact_sensors())

    def _render_frame(self) -> None:
        """Render one GL frame for the window or an on-demand capture."""
        viewer = self._viewer
        viewer.begin_frame(self._sim_time)
        try:
            self._navigate_scene_camera()
            if self._camera_sensor is not None:
                self._log_streaming_image()
            else:
                if not viewer.is_paused():
                    state = self._update_state()
                    if state.body_q is None or len(state.body_q):
                        viewer.log_state(state)
                        contacts = NewtonManager.get_contacts()
                        if contacts is not None:
                            viewer.log_contacts(contacts, state)
                        else:
                            self._log_scene_contact_sensor_arrows()
                        if self.cfg.enable_markers:
                            render_newton_visualization_markers(
                                viewer, self._env_ids, num_envs=self.backend.model.num_envs
                            )
                self._log_pending_meshes()
            if not viewer.is_paused():
                self._render_live_plots()
        finally:
            viewer.end_frame()
            if not viewer.is_running():
                self._viewer_picking_binding.deactivate()

    def reset(self, soft: bool = False) -> None:
        """Restore GL overlays and picking after the shared model binding changes."""
        previous = self.backend
        super().reset(soft)
        if self.backend is previous:
            return
        if self._picking_enabled:
            self._viewer.wind = None
        self._viewer.set_visible_worlds(self._env_ids)
        self._viewer.set_world_offsets(self.cfg.world_spacing)
        self._configure_viewer()
        self._viewer.picking_enabled = self._picking_enabled
        if self._picking_enabled:
            self._viewer_picking_binding.bind(self._viewer)

    def close(self) -> None:
        """Release picking and scene references without destroying Kit's shared GL context."""
        if self._is_closed:
            return
        try:
            if self._picking_enabled:
                # Captured graphs retain neutral picking inputs until their last replay.
                self._viewer_picking_binding.deactivate()
        finally:
            self._viewer = self.backend = self._transform_mapping = None
            self._pending_mesh_submissions.clear()
            super().close()

    def log_mesh(
        self,
        name: str,
        points: wp.array[wp.vec3],
        indices: wp.array[wp.int32] | wp.array[wp.uint32],
        normals: wp.array[wp.vec3] | None = None,
        uvs: wp.array[wp.vec2] | None = None,
        texture: np.ndarray | str | None = None,
        hidden: bool = False,
        backface_culling: bool = True,
        color: tuple[float, float, float] | None = None,
        roughness: float | None = None,
        metallic: float | None = None,
        dynamic: bool = False,
        opacity: float | None = None,
    ) -> None:
        """Stage a mesh registration or update for the next Newton viewer frame.

        Newton viewers require geometry updates between ``begin_frame()`` and
        ``end_frame()``. Staging here keeps that lifecycle internal to the
        visualizer and lets callers submit meshes before :meth:`step`.

        Args:
            name: Unique viewer path for the mesh.
            points: Vertex positions [m].
            indices: Flattened triangle vertex indices.
            normals: Optional per-vertex normals.
            uvs: Optional per-vertex texture coordinates.
            texture: Optional texture path or image.
            hidden: Whether to hide the mesh.
            backface_culling: Whether to cull back-facing triangles.
            color: Optional RGB base color in ``[0, 1]``.
            roughness: Optional surface roughness in ``[0, 1]``.
            metallic: Optional surface metallic value in ``[0, 1]``.
            dynamic: Whether the mesh topology may change between updates.
            opacity: Optional surface opacity in ``[0, 1]``.

        Raises:
            RuntimeError: If the Newton viewer has not been initialized.
        """
        if not self._is_initialized or self._viewer is None:
            raise RuntimeError("Newton visualizer must be initialized before logging meshes.")
        self._pending_mesh_submissions[name] = _MeshSubmission(
            name=name,
            points=points,
            indices=indices,
            normals=normals,
            uvs=uvs,
            texture=texture,
            hidden=hidden,
            backface_culling=backface_culling,
            color=color,
            roughness=roughness,
            metallic=metallic,
            dynamic=dynamic,
            opacity=opacity,
        )

    def _log_pending_meshes(self) -> None:
        """Publish staged meshes inside the active viewer frame."""
        if self._viewer is None or not self._pending_mesh_submissions:
            return

        submissions = tuple(self._pending_mesh_submissions.values())
        self._pending_mesh_submissions.clear()
        for mesh in submissions:
            self._viewer.log_mesh(
                mesh.name,
                mesh.points,
                mesh.indices,
                normals=mesh.normals,
                uvs=mesh.uvs,
                texture=mesh.texture,
                hidden=mesh.hidden,
                backface_culling=mesh.backface_culling,
                color=mesh.color,
                roughness=mesh.roughness,
                metallic=mesh.metallic,
                dynamic=mesh.dynamic,
                opacity=mesh.opacity,
            )

    # ------------------------------------------------------------------
    # Shared internals
    # ------------------------------------------------------------------

    def _log_scene_contact_sensor_arrows(self) -> None:
        """Draw filtered contact points, or body origins when only net forces are available."""
        starts, ends = [], []
        env_ids = slice(None) if self._env_ids is None else self._env_ids
        if self._viewer.show_contacts:
            for sensor in self._sim.get_scene_data_provider().get_contact_sensors().values():
                data = sensor.data
                forces = data.net_normal_forces_w.torch
                if data.contact_pos_w is not None and data.normal_force_matrix_w is not None:
                    positions = data.contact_pos_w.torch[env_ids]
                    contact_forces = data.normal_force_matrix_w.torch[env_ids]
                    active = torch.linalg.norm(contact_forces, dim=-1) > sensor.cfg.force_threshold
                    active &= torch.isfinite(positions).all(dim=-1)
                    if torch.any(active):
                        positions = positions[active]
                        directions = torch.nn.functional.normalize(contact_forces[active], dim=-1)
                        starts.append(positions)
                        ends.append(positions + directions * CONTACT_ARROW_LENGTH)
                        continue

                if data.pos_w is not None:
                    positions = data.pos_w.torch
                elif self.physics_backend in ("physx", "isaacsim_physx"):
                    # PhysX exposes body-major poses even when pose tracking is disabled.
                    poses = wp.to_torch(sensor.body_physx_view.get_transforms())
                    positions = poses.view(forces.shape[1], forces.shape[0], 7).transpose(0, 1)[..., :3]
                else:
                    continue
                forces, positions = forces[env_ids], positions[env_ids]
                active = torch.linalg.norm(forces, dim=-1) > sensor.cfg.force_threshold
                positions = positions[active]
                if len(positions):
                    starts.append(positions)
                    directions = torch.nn.functional.normalize(forces[active], dim=-1)
                    ends.append(positions + directions * CONTACT_ARROW_LENGTH)

        if starts:
            starts = torch.cat(starts).to(dtype=torch.float32, device=str(self._viewer.device))
            ends = torch.cat(ends).to(dtype=torch.float32, device=str(self._viewer.device))
            self._viewer.log_arrows(
                CONTACT_ARROW_PATH,
                wp.from_torch(starts, dtype=wp.vec3),
                wp.from_torch(ends, dtype=wp.vec3),
                CONTACT_ARROW_COLOR,
            )
        else:
            self._viewer.log_arrows(CONTACT_ARROW_PATH, None, None, None)

    def register_ui_callback(self, callback: Callable[[Any], None], position: str = "side") -> None:
        """Add a panel to the viewer's ImGui interface.

        Args:
            callback: Callable invoked with the ``imgui`` module every UI frame.
            position: Newton viewer UI slot, such as ``"side"`` or ``"panel"``.
        """
        if self._viewer is not None:
            self._viewer.register_ui_callback(callback, position=position)

    def request_close(self) -> None:
        """Close the viewer window once the current frame ends.

        Safe to call from a UI callback, where closing immediately would destroy the GL context
        mid-frame.
        """
        if self._viewer is not None:
            self._viewer.request_close()

    def supports_live_plots(self) -> bool:
        """Newton GL supports live scalar/array plots via the ImGui sidebar."""
        return True

    def add_live_plots(
        self,
        managers: dict,
        scalars: dict | None = None,
        term_names: dict[str, list[str]] | None = None,
        env_idx: int = 0,
    ) -> None:
        """Register manager terms and scalars for the live-plot sidebar.

        Calls the base implementation to populate :attr:`_live_plot_sources`, then registers
        the Live Plots collapsing section in the Newton viewer sidebar.

        Args:
            managers: Mapping of manager name to manager instance.
            scalars: Optional mapping of group name to a dict of ``{term_name: callable}``.
            term_names: Optional per-manager allowlists of term names to include.
            env_idx: Environment index to sample each step.  Defaults to ``0``.
        """
        super().add_live_plots(managers, scalars=scalars, term_names=term_names, env_idx=env_idx)
        if not self._live_plot_sources or self._viewer is None:
            return
        self._viewer._live_plots_callback = self._live_plots_panel_imgui

    def _live_plots_panel_imgui(self, imgui) -> None:
        """Render logged scalars and arrays in the Newton GL sidebar."""
        viewer = self._viewer
        plots, implot = viewer._plot_logger, viewer._implot
        if not plots._scalar_buffers and not plots._array_buffers:
            return
        imgui.set_next_item_open(False, imgui.Cond_.appearing)
        if not imgui.collapsing_header("Live Plots"):
            return
        imgui.separator()
        scale = viewer.gui.ui.dpi_scale

        groups: dict[str, list[str]] = {}
        for name in plots._scalar_buffers:
            base, bracket, component = name.rpartition("[")
            if not bracket or not component.endswith("]") or not component[:-1].isdigit():
                base = name
            groups.setdefault(base, []).append(name)

        for base_name in sorted(groups, key=lambda name: not name.startswith("episode/")):
            term_label = base_name.rsplit("/", 1)[-1]
            if not imgui.collapsing_header(term_label):
                continue
            if implot.begin_plot(f"##{base_name}", imgui.ImVec2(-1, 180 * scale)):
                auto = implot.AxisFlags_.auto_fit.value
                implot.setup_axes("", "", auto, auto)
                implot.setup_finish()
                for name in groups[base_name]:
                    array = plots._scalar_arrays[name]
                    if array is None:
                        history = plots._scalar_buffers[name]
                        array = np.full(plots._plot_history_size, np.nan, dtype=np.float32)
                        array[len(array) - len(history) :] = history
                        plots._scalar_arrays[name] = array
                    implot.plot_line(name[len(base_name) :] or term_label, array)
                implot.end_plot()

        panel_width = imgui.get_content_region_avail().x
        for name, array in plots._array_buffers.items():
            if imgui.collapsing_header(name):
                plots._render_array_heatmap(imgui, name, array, panel_width - 20 * scale, dpi_scale=scale)

    def _render_live_plots(self) -> None:
        """Push manager-term scalars to the Newton viewer's built-in plot panel."""
        if self._viewer is None or not self._live_plot_sources:
            return
        if self._runtime_headless:
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

    def _configure_viewer(self) -> None:
        """Apply model overlays and GL appearance after binding a model."""
        self._viewer.show_joints = self.cfg.show_joints
        self._viewer.show_contacts = self.cfg.show_contacts
        self._viewer.show_collision = self.cfg.show_collision
        self._viewer.show_springs = self.cfg.show_springs
        self._viewer.show_inertia_boxes = self.cfg.show_inertia_boxes
        self._viewer.show_com = self.cfg.show_com
        self._viewer.show_particles = self.cfg.show_particles
        self._viewer.up_axis = 2  # Z-up
        self._viewer.scaling = 1.0
        self._viewer.particle_color = self.cfg.particle_color
        self._viewer.renderer.draw_shadows = self.cfg.enable_shadows
        self._viewer.renderer.draw_wireframe = self.cfg.enable_wireframe
        # Accept list/tuple/array-like config colors; provide a stable tuple for nanobind conversion.
        if self.cfg.background_color is None:
            self._viewer.renderer.draw_sky = self.cfg.enable_sky
            upper_color = self.cfg.sky_upper_color
            lower_color = self.cfg.sky_lower_color
        else:
            self._viewer.renderer.draw_sky = False
            upper_color = lower_color = self.cfg.background_color
        self._viewer.renderer.sky_upper = tuple(upper_color)
        self._viewer.renderer.sky_lower = tuple(lower_color)
        self._viewer.renderer._light_color = tuple(self.cfg.light_color)

    def render_rgb_array(self) -> np.ndarray:
        """Return the latest RGB pixels, rendering on demand for a headless viewer."""
        if self._viewer is None:
            raise RuntimeError("NewtonGLVisualizer must be initialized before capturing an RGB frame.")
        if self._runtime_headless and not self._viewer.is_paused():
            self._render_frame()
        return self._viewer.get_frame().numpy()


class NewtonRTXVisualizer(NewtonVisualizerBase):
    """Use Newton's native RTX camera, GPU presentation, and markers on a borrowed scene.

    The simulation owns the stage and sensors. Newton owns the perspective render product and window.
    Resizing the window scales the displayed image without resizing either source's buffers.
    """

    def initialize(self, sim: SimulationContext, *, cameras: list[PerspectiveCameraCfg | Camera]) -> None:
        """Attach to the populated scene and bind the model used to publish body poses."""
        from isaaclab_ov.stage import OvstageBackendCfg

        if self._is_initialized:
            return
        if self.physics_backend in ("physx", "isaacsim_physx"):
            raise RuntimeError(
                "Newton RTX is kitless and cannot share a process with Kit physics; use Newton or OVPhysX."
            )
        super().initialize(sim, cameras=cameras)
        scene_data_provider = self._sim.get_scene_data_provider()
        scene = self._sim.get_or_create_backend(OvstageBackendCfg(viewer_id=id(self)))
        scene.populate(self._sim.stage, sim.get_clone_plan())
        cfg = self.cfg
        settings = dict(cfg.render_settings)
        if cfg.background_color is not None:
            settings["omni:rtx:background:source:type"] = ("Token", "color")
            settings["omni:rtx:background:source:color"] = ("Color3f", cfg.background_color)
        self._viewer = NewtonViewerRTX(
            width=cfg.window.size[0],
            height=cfg.window.size[1],
            headless=self._runtime_headless,
            async_rendering=not self._runtime_headless,
            ovstage=scene.stage,
            render_settings=settings,
            up_axis="Z",
            metadata={
                "num_envs": scene_data_provider.num_envs,
                "physics_backend": self.physics_backend,
                "gravity": sim.cfg.gravity,
            },
            window_cfg=cfg.window,
        )
        self._viewer.set_model(self.backend.model)
        self._viewer.marker_groups = sim.vis_marker_registry.get_groups().values()
        self._viewer.picking_enabled = False
        self._setup_streaming_view(
            scene_data_provider.num_envs,
            visible_env_ids=self._env_ids,
            target_aspect=cfg.window.size[0] / cfg.window.size[1],
        )
        self._viewer.register_ui_callback(self._draw_streaming_view_controls, position="side")
        self._select_camera(self._camera_index)
        self._is_initialized = True

    def _render_frame(self) -> None:
        """Update controls and publish GPU poses or sensor pixels to Newton once per frame."""
        viewer = self._viewer
        viewer.begin_frame(self._sim_time)
        try:
            self._navigate_scene_camera()
            if self._camera_sensor is not None:
                self._log_streaming_image()
            else:
                if not viewer.is_paused():
                    viewer.log_state(self._update_state())
                if self.cfg.enable_markers:
                    render_newton_visualization_markers(viewer, self._env_ids, num_envs=self.backend.model.num_envs)
        finally:
            viewer.end_frame()

    def reset(self, soft: bool = False) -> None:
        """Restore RTX camera selection after Newton replaces its camera during model binding."""
        previous = self.backend
        super().reset(soft)
        if self.backend is not previous:
            self._viewer.picking_enabled = False
            self._select_camera(self._camera_index)

    def render_rgb_array(self) -> np.ndarray | None:
        """Explicitly download the selected sensor or perspective image for recording."""
        if self._viewer is None:
            return None
        if self._camera_sensor is not None:
            return self.render_tiled_rgb_array()
        if self._runtime_headless:
            self._render_frame()
        return self._viewer.get_frame()

    def close(self) -> None:
        """Close Newton's viewer before the simulation releases the borrowed scene."""
        if self._is_closed:
            return
        try:
            if self._viewer is not None:
                self._viewer.close()
        finally:
            self._viewer = self.backend = None
            super().close()
