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
import webbrowser
from collections.abc import Callable
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
from isaaclab_newton.physics import NewtonBackendCfg
from newton.viewer import ViewerGL
from pyglet.math import Vec3 as PygletVec3

from isaaclab.scene_data import SceneDataFormat
from isaaclab.sensors.camera import Camera
from isaaclab.sim import SimulationContext
from isaaclab.utils.backend_utils import FactoryBase
from isaaclab.utils.math import quat_apply, quat_from_matrix, quat_mul
from isaaclab.visualizers.base_visualizer import BaseVisualizer
from isaaclab.visualizers.policy_ghost import PolicyGhostController
from isaaclab.visualizers.visualizer_cfg import PerspectiveCameraCfg, WindowCfg

from isaaclab_visualizers.desktop_entry import write_desktop_entry
from isaaclab_visualizers.newton.newton_visualization_markers import render_newton_visualization_markers

from .newton_policy_ghosts import NewtonPolicyGhostPanel, draw_policy_ghost_launcher
from .newton_recording import NewtonRecordingPanel
from .newton_visualizer_cfg import NewtonGLVisualizerCfg

logger = logging.getLogger(__name__)


CONTACT_ARROW_PATH = "/contacts"
"""Viewer path used for native and synthesized contact arrows."""

CONTACT_ARROW_COLOR = (0.0, 1.0, 0.0)
"""Color used by Newton's native contact visualization."""

CONTACT_ARROW_LENGTH = 0.1
"""Length of synthesized contact arrows in meters."""


VISUALIZER_DOCS_URL = "https://isaac-sim.github.io/IsaacLab/main/source/concepts/visualization.html"
"""Published documentation of the visualizers, opened from the viewer's HUD."""

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
# Newton GL window with Isaac Lab controls
# ---------------------------------------------------------------------------


class NewtonViewerGL(ViewerGL):
    """Newton GL window with Isaac Lab controls and device-image presentation."""

    _contacts_available = True
    CAMERA_SPEED_BOOST_MULTIPLIER = 2.0
    _owner = None
    _policy_ghost_panel = None
    _policy_ghost_restart = None
    _recording_panel = None
    image_matches_camera = False
    """Whether the fullscreen image is rendered from :attr:`camera`, so picking rays and window fill are valid."""
    _view_controls = None
    _run_info: dict = {}
    _steps_per_second = 0.0
    _real_time_factor = 0.0
    _rate_sample = (0.0, 0)

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
        self.particle_color: tuple[float, float, float] | None = None
        self._particle_color_buffer: wp.array | None = None
        self._particle_color_buffer_value: tuple[float, float, float] | None = None
        self._mpm_particle_flags_cache_key: tuple[int, int, int] | None = None
        self._mpm_particles_all_active = False
        self._fullscreen_image_active = False
        draw_texture = getattr(self.renderer, "render_texture", None)

        def render_texture(texture_id, texture_width=0, texture_height=0, **kwargs):
            screen = self.renderer._screen_width / max(self.renderer._screen_height, 1)
            if self.image_matches_camera and texture_height and abs(texture_width / texture_height / screen - 1) < 0.1:
                texture_width = texture_height = 0  # Stretch a nearly matching aspect instead of drawing black bars.
            draw_texture(texture_id, texture_width, texture_height, **kwargs)

        if draw_texture is not None:
            self.renderer.render_texture = render_texture
        self._implot = self._implot_ctx = None
        if self.gui is not None:
            from imgui_bundle import implot

            self._patch_viewer_panel()
            # Keep camera keys active while ImGui captures mouse input for its controls.
            self.gui.is_capturing = self.gui.should_ignore_keyboard_input
            self._implot = implot
            self._implot_ctx = implot.create_context()
            implot.set_imgui_context(self.gui.ui.imgui.get_current_context())
            # Live plots are drawn in the sidebar, not Newton's floating window.
            self.gui._render_scalar_plots = lambda: None
        self.register_ui_callback(self._render_training_controls, position="side")

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

        An always-visible header (task and run, physics backend and device, environment counts, step and simulation
        time, simulation rate and real-time factor, renderer, status badges, and a documentation link) sits above the
        sections, which are ordered by how often they are used:

        1. **Playback** (open) — pause, reset, and window-rate controls, and any ``"side"`` UI callbacks.
        2. **View** (open) — renderer (OpenGL or RTX), camera, VSync, and renderer-specific options.
        3. **Scene** (open) — visible environments, visualization markers, and model overlays.
        4. **Policy Ghosts** (open) — added by :meth:`~NewtonGLVisualizer.attach_policy_ghosts`.
        5. **Recording** (closed) — output settings, video, ``.viser`` scene, and simulation-state checkpoint.
        6. **Live Plots** (closed) — injected when :meth:`~NewtonGLVisualizer.add_live_plots` is called.
        7. **Controls** and **Selection API** (closed) — camera keyboard reference and Newton's selection panel.

        The top-level Newton ``Pause / Step`` row is suppressed; pause/resume is handled by the Isaac Lab controls
        inside **Playback**.
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

            viewer._draw_header(imgui)

            # --- Playback ---------------------------------------------------
            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Playback"):
                imgui.separator()
                for callback in _g._ui_callbacks.get("side", []):
                    callback(imgui)

            # --- View -------------------------------------------------------
            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("View"):
                imgui.separator()
                if viewer._view_controls is not None:
                    viewer._view_controls(imgui)
                _c, viewer.vsync = imgui.checkbox("VSync", viewer.vsync)
                if _g._ui_callbacks.get("rendering") and imgui.tree_node("Appearance"):
                    for callback in _g._ui_callbacks["rendering"]:
                        callback(imgui)
                    imgui.tree_pop()

            # --- Scene ------------------------------------------------------
            viewer._draw_scene_controls(imgui)

            # --- Policy Ghosts ----------------------------------------------
            if viewer._policy_ghost_panel is not None:
                viewer._policy_ghost_panel.draw(imgui)
            elif viewer._policy_ghost_restart is not None:
                draw_policy_ghost_launcher(imgui, viewer._policy_ghost_restart)

            # --- Recording --------------------------------------------------
            if viewer._recording_panel is not None:
                viewer._recording_panel.draw(imgui)

            # --- Live Plots -------------------------------------------------
            live_plots_cb = viewer._live_plots_callback
            if live_plots_cb is not None:
                live_plots_cb(imgui)

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
                if viewer.picking_enabled and (not viewer._fullscreen_image_active or viewer.image_matches_camera):
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

    def _has_environment_controls(self) -> bool:
        """Return whether more than one environment can be shown, so the Environments section is useful."""
        owner = self._owner
        return owner is not None and owner.visible_env_limit > 1

    def _render_environment_controls(self, imgui) -> None:
        """Render the visible-environment count slider."""
        owner = self._owner
        imgui.text("Visible Envs (per slot)" if owner._policy_ghost_panel is not None else "Visible Envs")
        changed, new_count = imgui.slider_int("##VisibleEnvs", owner.visible_env_count, 1, owner.visible_env_limit)
        if changed:
            owner.set_visible_env_count(new_count)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Number of environments drawn in the viewer; simulation is unaffected")

    def _draw_scene_controls(self, imgui) -> None:
        """Draw what can be shown in the scene: environments, visualization markers, and model overlays."""
        has_environments = self._has_environment_controls()
        if not (has_environments or self.marker_groups or self.model is not None):
            return
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if not imgui.collapsing_header("Scene"):
            return
        imgui.separator()
        if has_environments:
            self._render_environment_controls(imgui)
        if self.marker_groups:
            imgui.separator_text("Markers")
            for marker in self.marker_groups:
                name = marker.cfg.prim_path.rsplit("/", 1)[-1].replace("_", " ")
                changed, visible = imgui.checkbox(f"Show {name}##{id(marker)}", marker.is_visible())
                if changed:
                    marker.set_visibility(visible)
        if self.model is not None:
            imgui.separator_text("Model Overlays")
            if self._fullscreen_image_active or self.image_matches_camera:
                imgui.text_disabled("OpenGL only")
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Switch View > Renderer to OpenGL to draw joints, contacts, and other overlays")
            elif imgui.tree_node("Overlay options"):
                self._draw_model_overlays(imgui)
                imgui.tree_pop()

    def _draw_model_overlays(self, imgui) -> None:
        """Draw Newton's per-overlay toggles."""
        renderer = self.renderer
        _c, self.show_joints = imgui.checkbox("Show Joints", self.show_joints)
        if self.show_joints:
            _, renderer.joint_scale = imgui.slider_float("Joint Scale", renderer.joint_scale, 0.25, 5.0)
        _contacts_available = self._contacts_available
        self.show_contacts = _imgui_optional_checkbox(
            imgui,
            "Show Contacts",
            self.show_contacts,
            _contacts_available,
            "No contact sensors in this environment",
        )
        if self.show_contacts and _contacts_available:
            _, renderer.arrow_length_scale = imgui.slider_float(
                "Contact Length", renderer.arrow_length_scale, 0.25, 5.0
            )
            _, renderer.arrow_scale = imgui.slider_float("Contact Width", renderer.arrow_scale, 0.25, 5.0)
        _model = self.model
        _has_particles = _model.particle_count > 0
        _has_springs = _model.spring_count > 0
        _has_cloth = _model.tri_count > 0
        self.show_particles = _imgui_optional_checkbox(
            imgui,
            "Show Particles",
            self.show_particles,
            _has_particles,
            "No particle bodies in this environment",
        )
        self.show_springs = _imgui_optional_checkbox(
            imgui,
            "Show Springs",
            self.show_springs,
            _has_springs,
            "No spring constraints in this environment",
        )
        _c, self.show_com = imgui.checkbox("Show Center of Mass", self.show_com)
        if self.show_com:
            _, renderer.com_scale = imgui.slider_float("COM Scale", renderer.com_scale, 0.25, 5.0)
        self.show_triangles = _imgui_optional_checkbox(
            imgui,
            "Show Cloth",
            self.show_triangles,
            _has_cloth,
            "No cloth/triangle meshes in this environment",
        )
        _c, self.show_collision = imgui.checkbox("Show Collision", self.show_collision)
        _c, renderer.draw_edges = imgui.checkbox("Show Edges", renderer.draw_edges)
        _sdf_labels = ["Off", "Margin", "Margin + Gap"]
        _, new_sdf_idx = imgui.combo("Gap + Margin", int(self.sdf_margin_mode), _sdf_labels)
        self.sdf_margin_mode = self.SDFMarginMode(new_sdf_idx)
        if self.sdf_margin_mode != self.SDFMarginMode.OFF:
            _, renderer.wireframe_line_width = imgui.slider_float(
                "Wireframe Width (px)", renderer.wireframe_line_width, 0.5, 5.0
            )
        _c, self.show_visual = imgui.checkbox("Show Visual", self.show_visual)
        _c, self.show_inertia_boxes = imgui.checkbox("Show Inertia Boxes", self.show_inertia_boxes)

    def _update_rates(self, steps: int, dt: float) -> None:
        """Refresh the simulation rate and real-time factor from the step counter, about twice a second."""
        now = time.perf_counter()
        wall, last_steps = self._rate_sample
        if wall == 0.0:
            self._rate_sample = (now, steps)
        elif now - wall >= 0.5:
            self._steps_per_second = max(0, steps - last_steps) / (now - wall)
            self._real_time_factor = self._steps_per_second * dt
            self._rate_sample = (now, steps)

    def set_run_info(self, task: str | None = None, mode: str | None = None, run: str | None = None) -> None:
        """Name the run in the HUD header.

        Args:
            task: Task name.
            mode: How the run uses the task, such as ``"play"`` or ``"train"``.
            run: Name of the run's log directory.
        """
        self._run_info = {key: value for key, value in (("task", task), ("mode", mode), ("run", run)) if value}

    def _draw_header(self, imgui) -> None:
        """Draw the always-visible summary: run, physics, scale, progress, rates, and status badges."""
        owner = self._owner
        info = self._run_info
        if "task" in info:
            imgui.text_wrapped(f"Task: {info['task']}")
        if "run" in info or "mode" in info:
            imgui.text(f"Run: {' · '.join(info[key] for key in ('mode', 'run') if key in info)}")
        device = getattr(getattr(owner, "_sim", None), "device", None)
        imgui.text(f"Physics: {self._backend_display}" + (f" ({device})" if device else ""))
        if self.model is not None:
            axis_names = ["X", "Y", "Z"]
            gravity = self._metadata["gravity"]
            imgui.text(
                f"Up: {axis_names[self.model.up_axis]}   "
                f"Gravity: ({gravity[0]:.2f}, {gravity[1]:.2f}, {gravity[2]:.2f})"
            )
        if owner is not None and owner._sim is not None:
            sim = owner._sim
            steps, dt = sim.get_physics_step_count(), sim.get_physics_dt()
            self._update_rates(steps, dt)
            total = self._metadata.get("num_envs")
            shown = owner.visible_env_count if owner.visible_env_limit > 1 else total
            imgui.text(f"Envs: {total}" + (f" ({shown} shown)" if shown != total else ""))
            imgui.text(f"Step: {steps:,}   Sim time: {steps * dt:.1f} s")
            imgui.text(f"Sim: {self._steps_per_second:.0f} steps/s ({self._real_time_factor:.2f}x real time)")
            imgui.text(f"Window: {imgui.get_io().framerate:.0f} FPS")
            imgui.text(f"Renderer: {'RTX' if owner.rtx_rendering else 'OpenGL'}")
        badges = []
        if self._recording_panel is not None and self._recording_panel.recording_video:
            panel = self._recording_panel
            badges.append((f"REC video {panel._video_frames / max(panel._video_rate, 1):.0f}s", (1.0, 0.25, 0.25, 1.0)))
        if self._recording_panel is not None and self._recording_panel.recording_scene:
            badges.append((f"REC scene {self._recording_panel._scene_frames}f", (1.0, 0.25, 0.25, 1.0)))
        if self._paused_training or self._paused:
            badges.append(("PAUSED", (1.0, 0.8, 0.2, 1.0)))
        if self._policy_ghost_panel is not None:
            ghosts = sum(slot.active for slot in self._policy_ghost_panel.layout.slots[1:])
            badges.append((f"{ghosts} ghost{'s' * (ghosts != 1)}", (0.5, 0.8, 1.0, 1.0)))
        for index, (text, color) in enumerate(badges):
            if index:
                imgui.same_line()
            imgui.text_colored(imgui.ImVec4(*color), f"[{text}]")
        if imgui.text_link("Visualizer docs"):
            webbrowser.open(VISUALIZER_DOCS_URL)
        if imgui.is_item_hovered():
            imgui.set_tooltip(VISUALIZER_DOCS_URL)

    def _render_training_controls(self, imgui):
        """Render the Isaac Lab playback controls inside the Playback section."""
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

    def begin_frame(self, time: float) -> None:
        """Update mesh VAOs in their owning context because VAOs are not shared between viewers."""
        self.renderer.window.switch_to()
        super().begin_frame(time)

    def is_running(self) -> bool:
        """Track this window, independently of another window closing Pyglet's shared event loop."""
        return self.renderer.window.context is not None and (
            not self._headless or self.num_frames is None or self._frame_count < self.num_frames
        )

    def get_frame(self, target_image: wp.array | None = None, render_ui: bool = False) -> wp.array:
        """Reuse device capture storage while the native framebuffer size remains unchanged."""
        shape = (self.renderer._screen_height, self.renderer._screen_width, 3)
        if target_image is not None and target_image.shape != shape:
            target_image = None
        return super().get_frame(output=target_image, render_ui=render_ui)

    def end_frame(self) -> None:
        """Finish the frame, then close the window if :meth:`request_close` was called."""
        try:
            super().end_frame()
        finally:
            self._fullscreen_image_active = False
        if self._close_requested:
            self.close()

    def close(self) -> None:
        """Release context-local GL resources before closing their owning window."""
        self.renderer.window.switch_to()
        super().close()

    def log_image(self, name: str, image: wp.array | np.ndarray, *, fullscreen: bool = False) -> None:
        """Track whether the current frame replaces the perspective scene with an image."""
        super().log_image(name, image, fullscreen=fullscreen)
        if fullscreen:
            self._fullscreen_image_active = True

    def on_mouse_press(self, x, y, button, modifiers):
        """Image coordinates must not pick objects in the hidden perspective scene, unless it is the image's camera."""
        if self._fullscreen_image_active and not self.image_matches_camera:
            return
        if self._fullscreen_image_active and self.picking_enabled and self._owner is not None:
            self._last_state = (
                self._owner._update_state()
            )  # An image frame does not log state; cast against fresh poses.
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
        self._last_render_step = -1
        self._pending_mesh_submissions: dict[str, dict[str, Any]] = {}

    def initialize(self, sim: SimulationContext, *, cameras: list[PerspectiveCameraCfg | Camera]) -> None:
        """Bind camera sources and the display mode shared by both windows."""
        super().initialize(sim, cameras=cameras)
        self._runtime_headless = self.cfg.headless or (
            sys.platform not in ("win32", "darwin") and not os.environ.get("DISPLAY")
        )
        if not self._runtime_headless:
            # pyglet sets WM_CLASS from the window caption, which ViewerGL defaults to "Newton".
            write_desktop_entry("isaaclab-newton-gl-viewer", "Newton", "Newton", _NEWTON_ICON_DIR / "icon_64.png")
        with wp.ScopedDevice(sim.device):
            self._viewer = NewtonViewerGL(
                width=self.cfg.window.size[0], height=self.cfg.window.size[1], headless=self._runtime_headless,
                metadata={
                    "num_envs": sim.get_scene_data_provider().num_envs,
                    "physics_backend": self.physics_backend,
                    "gravity": sim.cfg.gravity,
                },
                window_cfg=self.cfg.window,
            )  # fmt: skip
        self._viewer.marker_groups = sim.vis_marker_registry.get_groups().values()
        self._viewer._view_controls = self._draw_streaming_view_controls

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
        self._pending_mesh_submissions[name] = dict(
            points=points, indices=indices, normals=normals, uvs=uvs, texture=texture,
            hidden=hidden, backface_culling=backface_culling, color=color,
            roughness=roughness, metallic=metallic, dynamic=dynamic, opacity=opacity,
        )  # fmt: skip

    def _draw_streaming_view_controls(self, imgui) -> None:
        """Choose one declared display source without reading the other sensors."""
        rtx_index = getattr(self, "rtx_view_index", None)
        if rtx_index is not None:
            rtx_on = self.rtx_rendering
            imgui.text("Renderer")
            imgui.same_line()
            if imgui.radio_button("OpenGL", not rtx_on) and rtx_on:
                self.set_rtx_rendering(False)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Fast rasterizer with model overlays, markers, and Policy Ghosts")
            imgui.same_line()
            if imgui.radio_button("RTX", rtx_on) and not rtx_on:
                self.set_rtx_rendering(True)
            if imgui.is_item_hovered():
                imgui.set_tooltip("OVRTX path-traced rendering of the scene; slower, without model overlays")
        labels = [
            f"Perspective {i + 1}"
            if view.camera is None
            else "RTX viewer camera"
            if i == rtx_index
            else view.cfg.source
            for i, view in enumerate(self._image_views)
        ]
        changed, index = imgui.combo("Camera", self._image_views.index(self.current_image_view), labels)
        if changed:
            self._select_camera(index)
        if imgui.is_item_hovered():
            imgui.set_tooltip(labels[self._image_views.index(self.current_image_view)])

    def _select_camera(self, index: int, *, keep_pose: bool = False) -> None:
        """Bind the selected source; all scene cameras retain their normal lazy-data lifecycle.

        Args:
            index: Image view to display.
            keep_pose: Continue from the viewer's current camera instead of the source's configured pose.
        """
        self.current_image_view = self._image_views[index]
        camera = self.current_image_view.cfg.source
        self.current_image_view.invalidate()
        if isinstance(camera, PerspectiveCameraCfg):
            if self.current_image_view.render is not None and self.current_image_view.render != self._capture_frame:
                raise ValueError("A perspective image view must have exactly one rendering owner.")
            self.current_image_view.render = self._capture_frame
            if not keep_pose:
                self.set_camera_view(camera.eye, camera.lookat)
                self.cfg.focal_length = camera.focal_length
                self._viewer.camera.fov = self._focal_length_to_vertical_fov_degrees()
        elif keep_pose:
            self._place_scene_camera_at_viewer(self.current_image_view.camera)
        else:
            sensor = self.current_image_view.camera
            intrinsics = sensor._data.intrinsic_matrices.torch[0]
            self._viewer.camera.fov = math.degrees(2 * math.atan(sensor.cfg.height / (2 * float(intrinsics[1, 1]))))
        # None makes the first navigation step copy the viewer's field of view to a scene camera.
        self._navigation_fov = (
            None if keep_pose and not isinstance(camera, PerspectiveCameraCfg) else self._viewer.camera.fov
        )
        if self._viewer.picking is not None:
            # Release an existing drag without replacing any graph-captured picking buffers.
            self._viewer.picking.release()

        self._navigation_view = self._viewer.camera.get_view_matrix().reshape(4, 4).T.copy()

    def _place_scene_camera_at_viewer(self, camera: Camera) -> None:
        """Move a scene camera to the viewer camera's current world pose."""
        world_from_camera = np.linalg.inv(self._viewer.camera.get_view_matrix().reshape(4, 4).T)
        positions, orientations = camera.get_world_poses(convention="opengl")
        position = torch.as_tensor(world_from_camera[:3, 3], dtype=positions.dtype, device=positions.device)
        rotation = quat_from_matrix(
            torch.as_tensor(world_from_camera[:3, :3], dtype=positions.dtype, device=positions.device)
        )
        camera.set_world_poses(position.expand_as(positions), rotation.expand_as(orientations), convention="opengl")

    def _navigate_scene_camera(self) -> None:
        camera, viewer = self.current_image_view.camera, self._viewer
        if viewer.is_paused():
            return
        if viewer.camera.fov != self._navigation_fov:
            height, width = camera.image_shape
            focal = height / (2 * math.tan(math.radians(viewer.camera.fov) / 2))
            intrinsics = torch.tensor(
                [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], device=camera.device
            ).expand(camera.num_instances, -1, -1)
            camera.set_intrinsic_matrices(intrinsics)
            self._navigation_fov = viewer.camera.fov
            self.current_image_view.invalidate()
        current = viewer.camera.get_view_matrix().reshape(4, 4).T
        previous, self._navigation_view = self._navigation_view, current.copy()
        if np.array_equal(previous, current):
            return
        delta = previous @ np.linalg.inv(current)
        positions, orientations = camera.get_world_poses(convention="opengl")
        delta = torch.as_tensor(delta, dtype=positions.dtype, device=positions.device)
        translation = quat_apply(orientations, delta[:3, 3].expand_as(positions))
        rotation = quat_from_matrix(delta[:3, :3]).expand_as(orientations)
        camera.set_world_poses(positions + translation, quat_mul(orientations, rotation), convention="opengl")
        self.current_image_view.invalidate()

    def set_camera_view(
        self, eye: tuple[float, float, float] | list[float], target: tuple[float, float, float] | list[float]
    ) -> None:
        """Set the configured and active camera eye and target positions [m]."""
        self.cfg.eye, self.cfg.lookat = tuple(float(v) for v in eye), tuple(float(v) for v in target)
        self._last_render_step = -1
        if self.current_image_view is not None:
            self.current_image_view.invalidate()
            camera = self.current_image_view.camera
            if camera is not None:
                eyes = torch.tensor([self.cfg.eye], device=camera.device).expand(camera.num_instances, -1)
                targets = torch.tensor([self.cfg.lookat], device=camera.device).expand(camera.num_instances, -1)
                camera.set_world_poses_from_view(eyes, targets)
        if self._viewer is not None:
            self._viewer.camera.pos = PygletVec3(*self.cfg.eye)
            self._viewer.camera.look_at(self.cfg.lookat)
            camera = self._viewer.camera
            self._viewer.set_camera(camera.pos, camera.pitch, camera.yaw)
            self._navigation_view = camera.get_view_matrix().reshape(4, 4).T.copy()

    def step(self, dt: float) -> None:
        """Advance the display at the configured cadence; headless consumers capture on demand."""
        if not self._is_initialized or self._is_closed:
            return
        self._sim_time += dt
        if self._runtime_headless:
            self.current_image_view.invalidate()
            self._last_render_step = -1
            return
        now = time.monotonic()
        if now - self._last_present_time < 1.0 / self.cfg.window.fps:
            return
        self._last_present_time = now
        self._render_frame()

    def reset(self, soft: bool = False) -> None:
        """Invalidate the displayed frame after a simulation reset."""
        super().reset(soft)
        self._last_render_step = -1

    def close(self) -> None:
        """Detach perspective producers before releasing the visualizer's borrowed views."""
        self._pending_mesh_submissions.clear()
        for view in self._image_views:
            if view.render == self._capture_frame:
                view.render = None
                view.invalidate()
        super().close()

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
        composite = self.current_image_view.frame.data
        if composite is None or not self._viewer.is_paused():
            composite = self.current_image_view.read(self._sim.get_physics_step_count())
        if composite is not None:
            self._viewer.log_image("Streaming View", composite, fullscreen=True)

    def _capture_frame(self, output: wp.array | None) -> wp.array:
        """Read the frame already displayed, or render once when capture is the only consumer."""
        if self._last_render_step != self._sim.get_physics_step_count() or (
            self.current_image_view.camera is not None and self.current_image_view.frame.timestamp < 0
        ):
            if not self._sim.is_rendering:
                self._sim.forward()
                if self.supports_markers():
                    self._sim.vis_marker_registry.dispatch_callbacks()
            self._render_frame()
        if self.current_image_view.camera is None:
            return self._viewer.get_frame(output)
        return self.current_image_view.frame.data

    def render_rgb_array(self) -> np.ndarray:
        """Download the selected shared view for legacy capture callers."""
        if self.current_image_view is None:
            raise RuntimeError("The visualizer must be initialized before capturing an image.")
        if self.current_image_view.camera is not None:
            self._capture_frame(None)
        frame = self.current_image_view.frame
        frame_id = (
            frame.timestamp
            if self._viewer.is_paused() and frame.data is not None
            else self._sim.get_physics_step_count()
        )
        return self.current_image_view.read_rgb(frame_id)


class NewtonGLVisualizer(NewtonVisualizerBase):
    """Newton model rendering, picking, and contact overlays used by the GL visualizer."""

    _recording: NewtonRecordingPanel | None = None

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
        self._cfg_visible_env_ids: list[int] | None = None
        self._visible_env_count: int | None = None
        self._num_envs = 0
        self._policy_ghost_panel: NewtonPolicyGhostPanel | None = None

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
        self.newton_cfg = NewtonBackendCfg(physics_cfg=sim.cfg.physics, device=sim.device)
        self.backend = sim.get_or_create_backend(self.newton_cfg)
        self._transform_mapping = sim.get_scene_data_provider().create_mapping(list(self.backend.model.body_label))
        scene_data_provider = self._sim.get_scene_data_provider()
        newton_backend_active = self.physics_backend == "newton"
        picking_supported = newton_backend_active and sim.backend_manager._supports_rigid_body_force_input
        num_envs = scene_data_provider.num_envs
        if self._runtime_headless and not self.cfg.headless:
            logger.warning(
                "[NewtonVisualizer] No display found (DISPLAY is unset); the Newton viewer runs"
                " headless via EGL and no window will open. Run from a session with a display (or set"
                " DISPLAY, e.g. 'export DISPLAY=:0') to see the viewer."
            )

        self._picking_enabled = self.cfg.enable_picking and picking_supported and not self._runtime_headless
        self._viewer.set_model(self.backend.model)
        if self._picking_enabled:
            # Keep Newton's public force path scoped to picking for this integration.
            self._viewer.wind = None
        self._cfg_visible_env_ids = self._env_ids
        self._num_envs = num_envs
        self._viewer._owner = self
        self._recording = self._viewer._recording_panel = NewtonRecordingPanel(self)
        self._apply_visible_worlds()
        self._viewer._paused = False
        self._viewer.picking_enabled = self._picking_enabled
        self._configure_viewer()

        self._select_camera(0)
        if self.cfg.enable_markers and self.rtx_view_index is not None:
            # OVRTX draws markers as stage prims; the viewer camera's renderer shares the registry's groups.
            renderer_cfg = self._image_views[self.rtx_view_index].camera.cfg.renderer_cfg
            sim.get_or_create_backend(renderer_cfg).marker_groups = sim.vis_marker_registry.get_groups().values()

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
            sim.backend_manager.register_state_force_callback(self._viewer_picking_binding.apply)
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
        if self._policy_ghost_panel is not None:
            self._policy_ghost_panel.apply_tint(self.backend.model)
            if self._policy_ghost_panel.consume_slots_changed():
                self._apply_visible_worlds()
            self._policy_ghost_panel.apply_materials(viewer)
        viewer.begin_frame(self._sim_time)
        try:
            viewer.image_matches_camera = self.rtx_rendering
            if self.current_image_view.camera is not None:
                self._navigate_scene_camera()
                self._log_streaming_image()
            elif not viewer.is_paused():
                state = self._update_state()
                if state.body_q is None or len(state.body_q):
                    viewer.log_state(state)
                    contacts = self._sim.backend_manager.get_contacts() if self.physics_backend == "newton" else None
                    if contacts is not None:
                        viewer.log_contacts(contacts, state)
                    else:
                        self._log_scene_contact_sensor_arrows()
                    if self.cfg.enable_markers:
                        render_newton_visualization_markers(viewer, self._env_ids, num_envs=self.backend.model.num_envs)
            for name, arguments in self._pending_mesh_submissions.items():
                viewer.log_mesh(name, **arguments)
            self._pending_mesh_submissions.clear()
            if not viewer.is_paused():
                self._render_live_plots()
        finally:
            viewer.end_frame()
            if not viewer.is_running():
                self._viewer_picking_binding.deactivate()
        if self._recording is not None and not viewer.is_paused():
            self._recording.capture()
        self._last_render_step = self._sim.get_physics_step_count()
        if self.current_image_view.camera is None:
            self.current_image_view.invalidate()

    def _update_state(self):
        """Publish current scene state while the GL frame is active."""
        state = self.backend.state_0
        provider = self._sim.get_scene_data_provider()
        poses = SceneDataFormat.Transform()
        if provider.get_transforms(poses, mapping=self._transform_mapping, count=self.backend.model.body_count):
            state.body_q = poses.transforms
        if self.backend.geometry_offsets:
            provider.get_geometry_points(output=state.particle_q, offsets=self.backend.geometry_offsets)
        return state

    @property
    def rtx_view_index(self) -> int | None:
        """Index of the OVRTX viewer camera among the image views, or None when the scene has none."""
        for index, view in enumerate(self._image_views):
            if view.camera is not None and getattr(view.camera.cfg, "prim_path", "").startswith(
                "/World/viewer_camera_"
            ):
                return index
        return None

    @property
    def rtx_rendering(self) -> bool:
        """Whether the main view is currently rendered by OVRTX."""
        index = self.rtx_view_index
        return index is not None and self.current_image_view is self._image_views[index]

    def set_rtx_rendering(self, enabled: bool) -> None:
        """Render the main camera with OVRTX or with the Newton OpenGL rasterizer, keeping the camera pose.

        Args:
            enabled: ``True`` for the OVRTX viewer camera, ``False`` for the Newton GL perspective view.
        """
        rtx_index = self.rtx_view_index
        if rtx_index is None:
            raise RuntimeError("The scene has no OVRTX viewer camera; see NewtonGLVisualizerCfg.rtx_camera.")
        perspective = next(i for i, view in enumerate(self._image_views) if view.camera is None)
        self._select_camera(rtx_index if enabled else perspective, keep_pose=True)

    def set_run_info(self, task: str | None = None, mode: str | None = None, run: str | None = None) -> None:
        """Name the run in the HUD header.

        Args:
            task: Task name.
            mode: How the run uses the task, such as ``"play"`` or ``"train"``.
            run: Name of the run's log directory.
        """
        if self._viewer is not None:
            self._viewer.set_run_info(task, mode, run)

    def attach_policy_ghost_restart(self, restart: Callable[[bool], None]) -> None:
        """Offer an Enable Policy Ghosts button in the HUD of a run that started without the feature.

        Args:
            restart: Callback that restarts playback with Policy Ghosts enabled (``True``) or disabled (``False``).
        """
        if self._viewer is not None:
            self._viewer._policy_ghost_restart = restart

    def attach_policy_ghosts(
        self, controller: PolicyGhostController, restart: Callable[[bool], None] | None = None
    ) -> None:
        """Show the Policy Ghosts HUD section for a playback loop that drives several checkpoints.

        Only environments of slots with a checkpoint are visible, and each ghost slot is tinted and overlaid on
        the reference slot. The visible-environment slider then counts environments per slot.

        Args:
            controller: Playback-loop side of Policy Ghosts.
            restart: Callback that restarts playback with Policy Ghosts enabled (``True``) or disabled (``False``).
                When given, the HUD shows a Disable button.
        """
        self._policy_ghost_panel = NewtonPolicyGhostPanel(controller, restart)
        if self._viewer is not None:
            self._viewer._policy_ghost_panel = self._policy_ghost_panel
            self._apply_visible_worlds()

    @property
    def visible_env_limit(self) -> int:
        """Largest value of the visible-environment count: environments per slot with Policy Ghosts, else all."""
        if self._policy_ghost_panel is not None:
            return self._policy_ghost_panel.layout.slice_size
        return len(self._cfg_visible_env_ids) if self._cfg_visible_env_ids is not None else self._num_envs

    @property
    def visible_env_count(self) -> int:
        """Number of environments currently shown (per slot with Policy Ghosts)."""
        if self._visible_env_count is not None:
            return min(self._visible_env_count, self.visible_env_limit)
        return self.visible_env_limit

    def set_visible_env_count(self, count: int) -> None:
        """Show only the first ``count`` environments (per slot with Policy Ghosts).

        Args:
            count: Number of environments to show, clamped to ``[1, visible_env_limit]``.
        """
        self._visible_env_count = max(1, min(int(count), self.visible_env_limit))
        self._apply_visible_worlds()

    def _select_visible_env_ids(self) -> list[int] | None:
        """Resolve the environments to show from the cfg, the HUD count, and the Policy Ghosts slots."""
        panel = self._policy_ghost_panel
        if panel is not None:
            return panel.layout.visible_env_ids(self.visible_env_count)
        if self._visible_env_count is None:
            return self._cfg_visible_env_ids
        ids = self._cfg_visible_env_ids if self._cfg_visible_env_ids is not None else range(self._num_envs)
        return list(ids)[: self._visible_env_count]

    def _drop_shape_instancers(self) -> None:
        """Free the GL shape instancers so they are re-allocated for the new visible worlds.

        Newton keeps an instancer per shape batch by name and reuses it with its original capacity, so showing
        more worlds than were visible when it was allocated fails every frame with a capacity error.
        """
        objects = getattr(self._viewer, "objects", None)
        if objects is None or not hasattr(self._viewer, "_qualify"):
            return
        prefix = self._viewer._qualify("/model/shapes/")
        for key in [k for k in objects if k.startswith(prefix) and hasattr(objects[k], "num_instances")]:
            del objects[key]

    def _apply_visible_worlds(self) -> None:
        """Apply the visible environments and, with Policy Ghosts, overlay each ghost on the reference."""
        if self._viewer is None:
            return
        self._env_ids = self._select_visible_env_ids()
        self._drop_shape_instancers()
        self._viewer.set_visible_worlds(self._env_ids)
        self._viewer.set_world_offsets(self.cfg.world_spacing)
        if self._policy_ghost_panel is not None and getattr(self._viewer, "world_offsets", None) is not None:
            offsets = self._viewer.world_offsets.numpy() + self._policy_ghost_panel.controller.env_origin_offsets()
            self._viewer.world_offsets = wp.array(offsets, dtype=wp.vec3, device=self._viewer.world_offsets.device)
        self._apply_rtx_visible_envs()

    def _apply_rtx_visible_envs(self) -> None:
        """Hide the environments outside the visible set in the OVRTX viewer camera's scene."""
        index = self.rtx_view_index
        if index is None:
            return
        renderer = self._sim.get_or_create_backend(self._image_views[index].camera.cfg.renderer_cfg)
        num_envs = len(renderer._env_paths)
        if num_envs:
            visible = np.ones(num_envs, dtype=bool) if self._env_ids is None else np.zeros(num_envs, dtype=bool)
            if self._env_ids is not None:
                visible[np.asarray(list(self._env_ids), dtype=int)] = True
            renderer.set_environment_visibility(visible)
            self._image_views[index].invalidate()

    def reset(self, soft: bool = False) -> None:
        """Restore GL overlays and picking after the shared model binding changes."""
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
        self._viewer._view_controls = self._draw_streaming_view_controls
        if self._picking_enabled:
            self._viewer.wind = None
        self._apply_visible_worlds()
        self._configure_viewer()
        self._viewer.picking_enabled = self._picking_enabled
        if self._picking_enabled:
            self._viewer_picking_binding.bind(self._viewer)

    def close(self) -> None:
        """Release picking and scene references without destroying Kit's shared GL context."""
        if self._is_closed:
            return
        try:
            if self._recording is not None:
                self._recording.close()
            if self._picking_enabled:
                # Captured graphs retain neutral picking inputs until their last replay.
                self._viewer_picking_binding.deactivate()
        finally:
            self._viewer = self.backend = self._transform_mapping = None
            super().close()

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


class NewtonRTXVisualizer(NewtonVisualizerBase):
    """Display simulation-owned RTX images in a Newton GL window.

    Scene cameras and the perspective product share the configured OVRTX renderer. Window resizing
    only scales presentation; sensor and perspective output storage keeps its initial resolution.
    """

    def initialize(self, sim: SimulationContext, *, cameras: list[PerspectiveCameraCfg | Camera]) -> None:
        """Bind existing scene cameras to the window; cameras retain renderer ownership."""
        if self._is_initialized:
            return
        if FactoryBase._get_backend(sim=sim) == "physx":
            raise RuntimeError(
                "Newton RTX is kitless and cannot share a process with Kit physics; use Newton or OVPhysX."
            )
        if any(isinstance(camera, PerspectiveCameraCfg) for camera in cameras) or (
            self.cfg.view is not None and isinstance(self.cfg.view.source, PerspectiveCameraCfg)
        ):
            raise ValueError(
                "RTX perspective cameras must be declared before scene construction. Pass the environment config "
                "to launch_simulation, or use instantiate(camera_cfg) within ReplicateSession and select it with "
                "SceneCameraCfg."
            )
        super().initialize(sim, cameras=cameras)
        self._viewer.renderer.set_title("Isaac Lab RTX")
        self._viewer.picking_enabled = False
        self._select_camera(0)
        self._renderers = tuple(dict.fromkeys(
            sim.get_or_create_backend(view.camera.cfg.renderer_cfg) for view in self._image_views
            if view.camera.cfg.renderer_cfg.renderer_type == "ovrtx"
        ))  # fmt: skip
        if self.cfg.enable_markers:
            for renderer in self._renderers:
                renderer.marker_groups = sim.vis_marker_registry.get_groups().values()
        self._marker_revision = -1
        self._is_initialized = True

    def log_mesh(self, name: str, *args, **kwargs) -> None:
        """Publish geometry to the shared scene, including cameras captured without a window."""
        if not self._is_initialized:
            raise RuntimeError("The visualizer must be initialized before logging a mesh.")
        for renderer in self._renderers:
            renderer.log_mesh(f"Viewer_{id(self)}/{name.lstrip('/')}", *args, **kwargs)
        for view in self._image_views:
            view.camera.update(0.0)
            view.invalidate()

    def _render_frame(self) -> None:
        """Update navigation and scene overlays, then display the selected camera's GPU image."""
        self._viewer.begin_frame(self._sim_time)
        try:
            if not self._viewer.is_paused():
                self._navigate_scene_camera()
                if self.cfg.enable_markers and self._marker_revision != self._sim.vis_marker_registry.revision:
                    for view in self._image_views:
                        view.camera.update(0.0)
                        view.invalidate()
                    self._marker_revision = self._sim.vis_marker_registry.revision
            self._log_streaming_image()
            self._last_render_step = self._sim.get_physics_step_count()
        finally:
            self._viewer.end_frame()

    def close(self) -> None:
        """Close the window; scene cameras retain their render products and shared renderer."""
        if self._is_closed:
            return
        try:
            if self._viewer is not None:
                self._viewer.close()
        finally:
            self._viewer = None
            super().close()
