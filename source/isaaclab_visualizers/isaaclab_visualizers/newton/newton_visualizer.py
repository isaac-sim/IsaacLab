# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton GL and RTX visualization of the authored scene and camera images."""

from __future__ import annotations

import contextlib
import inspect
import logging
import math
import os
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

from isaaclab.visualizers import ImageViewCfg, VisualizerCfg

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
from isaaclab.utils.images import display_channels, sensor_key_for_gt_type
from isaaclab.utils.math import quat_apply, quat_from_matrix, quat_mul
from isaaclab.visualizers.base_visualizer import _VISUALIZER_MAX_TILES as MAX_TILED_ENVS
from isaaclab.visualizers.base_visualizer import BaseVisualizer
from isaaclab.visualizers.visualizer_cfg import PerspectiveCameraCfg, WindowCfg

from isaaclab_visualizers.desktop_entry import write_desktop_entry
from isaaclab_visualizers.lidar_points import frame_visualizer_closeup, scene_has_lidar, sync_lidar_hits
from isaaclab_visualizers.newton.newton_overlay_meshes import OverlayMeshPublisher
from isaaclab_visualizers.newton.newton_visualization_markers import render_newton_visualization_markers

from .newton_key_event_source import NewtonKeyEventSource
from .newton_rtx_appearance import SKY_TEXTURES, RtxAppearance

logger = logging.getLogger(__name__)


CONTACT_ARROW_PATH = "/contacts"
"""Viewer path used for native and synthesized contact arrows."""

CONTACT_ARROW_COLOR = (0.0, 1.0, 0.0)
"""Color used by Newton's native contact visualization."""

CONTACT_ARROW_LENGTH = 0.1
"""Length of synthesized contact arrows in meters."""


_NEWTON_ICON_DIR = Path(newton.__file__).parent / "_src" / "viewer" / "gl"


_MARKER_TIPS = {
    "velocity_goal": "Commanded base velocity",
    "velocity_current": "Current base velocity",
    "goal_pose": "Commanded goal pose",
    "pose_goal": "Commanded goal pose",
    "body_pose": "Current pose of the commanded body",
    "current_pose": "Current pose of the commanded body",
    "goal_marker": "Goal the task asks the object to reach",
    "Goal": "Goal position",
    "ee_goal": "Goal pose of the end effector",
    "ee_current": "Current pose of the end effector",
    "SuccessMarkers": "Marks environments that succeeded",
    "Policy_Heightmap": "Height scan the policy observes",
    "TerrainOrigin": "Origin of each terrain tile",
    "DeformableTarget": "Target of the deformable object",
    "RayCaster": "Points hit by the ray-caster sensor",
    "ContactSensor": "Contact points reported by a contact sensor",
    "FrameTransformer": "Frames tracked by a frame transformer sensor",
    "EndEffectorFrameTransformer": "End-effector frame tracked by a frame transformer sensor",
    "CabinetFrameTransformer": "Cabinet frames tracked by a frame transformer sensor",
    "HandJointMarkers": "Hand joint positions from the hand tracker",
}
"""What each marker group shows, by the last component of its prim path."""


def hud_note(imgui: Any, text: str) -> None:
    """Draw a short muted-blue note that the user should know about the control above or below it."""
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.55, 0.7, 0.95, 0.85))
    imgui.text_wrapped(text)
    imgui.pop_style_color()


def hud_warning(imgui: Any, text: str) -> None:
    """Draw a prominent warning line, for something the user is waiting on."""
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 0.78, 0.25, 1.0))
    imgui.text_wrapped(text)
    imgui.pop_style_color()


def hover_tooltip(imgui: Any, text: str) -> None:
    """Show ``text`` while the last item is hovered, including when it is disabled."""
    if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
        imgui.set_tooltip(text)


def marker_tooltip(prim_path: str) -> str:
    """Return a short description of the marker group at ``prim_path``, or a generic line when unknown."""
    return _MARKER_TIPS.get(prim_path.rsplit("/", 1)[-1], "Show or hide this marker group")


def term_summary(manager, term: str) -> str:
    """Return the first docstring line of a manager term's function, or an empty string when it has none.

    Args:
        manager: Environment manager that owns the term.
        term: Name of the term within the manager.
    """
    for getter in ("get_term_cfg", "get_term"):
        try:
            found = getattr(manager, getter)(term)
        except Exception:  # Not every manager has the accessor, and some take a group as well.
            continue
        target = getattr(found, "func", found)
        doc = inspect.getdoc(target) or ""
        line = next((line.strip() for line in doc.splitlines() if line.strip()), "")
        if line and not line.startswith(("Initialize self", "Initialise self")):
            return line if len(line) <= 120 else line[:117] + "..."
    return ""


DEFAULT_TILED_ENVS = 16
"""Environments a multi-environment scene camera tiles when it is first shown."""

_BACKEND_DISPLAY_NAMES = {"physx": "PhysX", "ovphysx": "OVPhysX", "newton": "Newton MJWarp"}


def _imgui_optional_checkbox(imgui, label: str, value: bool, available: bool, tip: str, hint: str = "") -> bool:
    """Render a checkbox greyed out with the reason as tooltip when *available* is False, else with *hint*."""
    if not available:
        imgui.begin_disabled()
    _, new_val = imgui.checkbox(label, value)
    if not available:
        imgui.end_disabled()
        hover_tooltip(imgui, tip)
        return value
    if hint:
        hover_tooltip(imgui, hint)
    return new_val


# ---------------------------------------------------------------------------
# Newton GL window with Isaac Lab controls
# ---------------------------------------------------------------------------


class NewtonViewerGL(ViewerGL):
    """Newton GL window with Isaac Lab controls and device-image presentation."""

    _contacts_available = True
    _lidar_available = False
    show_lidar_hits = False
    """Whether lidar hits are copied onto the host and drawn in the OpenGL view."""
    CAMERA_SPEED_BOOST_MULTIPLIER = 2.0
    camera_keys_suspended = False
    """Set while keyboard camera movement is disabled (see :meth:`NewtonVisualizerBase.set_camera_keys_enabled`)."""
    keyboard_frame_hook: Callable[[], object] | None = None
    """Called after every frame while a :class:`NewtonKeyEventSource` listens, to track ImGui keyboard ownership."""
    _owner = None
    _show_stats = False
    """Whether the performance and model statistics overlay in the top-right corner is open."""
    image_matches_camera = False
    """Whether the fullscreen image is rendered from :attr:`camera`, so picking rays and window fill are valid."""
    _overlay_capture = None
    """Publisher that also receives the line and arrow overlays, while the OVRTX view draws them."""
    _view_controls = None

    PICKING_ARROW_SCALE = 4.0
    """Screen-space size of the arrow drawn while dragging a body in the OpenGL view, relative to a contact arrow."""
    PICKING_ARROW_WIDTH = 0.025
    """Width of the same arrow in the OVRTX view, where it is a mesh [m]. OpenGL sizes it with the scale instead."""

    def log_lines(self, name, starts, ends, colors, width=0.01, hidden=False):
        """Draw line segments, and hand them to the overlay publisher when one is capturing.

        Newton draws the drag from the picked point to the cursor target as a thin line; it is drawn as an arrow here.
        """
        if name == "picking_line":
            self.log_arrows("picking_arrow", starts, ends, colors, width=self.PICKING_ARROW_WIDTH, hidden=hidden)
            return
        if self._overlay_capture is not None:
            self._overlay_capture.record(name, starts, ends, colors, width, hidden, arrow=False)
        super().log_lines(name, starts, ends, colors, width=width, hidden=hidden)

    def _enlarge_picking_arrow(self) -> None:
        """Draw the picking arrow in a second pass at :attr:`PICKING_ARROW_SCALE`, leaving the contact arrows alone."""
        renderer = self.renderer
        draw_arrows = getattr(renderer, "_render_arrows", None)
        if draw_arrows is None:
            return

        def render_arrows(arrows):
            picking = {key: arrow for key, arrow in arrows.items() if key.endswith("picking_arrow")}
            draw_arrows({key: arrow for key, arrow in arrows.items() if key not in picking})
            if picking:
                scale = renderer.arrow_scale
                renderer.arrow_scale = max(scale, self.PICKING_ARROW_SCALE)
                try:
                    draw_arrows(picking)
                finally:
                    renderer.arrow_scale = scale

        renderer._render_arrows = render_arrows

    def log_arrows(self, name, starts, ends, colors, width=0.01, hidden=False):
        """Draw arrows, and hand them to the overlay publisher when one is capturing."""
        if self._overlay_capture is not None:
            self._overlay_capture.record(name, starts, ends, colors, width, hidden, arrow=True)
        super().log_arrows(name, starts, ends, colors, width=width, hidden=hidden)

    def __init__(self, *args, window_cfg: WindowCfg, metadata: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self.renderer, "set_title"):
            self.renderer.set_title("Newton Visualizer")  # Newton's default caption is "Newton Viewer".
        self._enlarge_picking_arrow()
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

        Section order:

        1. **Simulation** (open) — physics backend, model info, playback and environment controls.
        2. **Live Plots** (closed) — injected when :meth:`~NewtonGLVisualizer.add_live_plots`
           is called.
        3. **Model Overlays** and **Visualization Markers** (closed) — debug visibility controls.
        4. **Rendering Options** (open) — VSync, the renderer and camera source, and renderer-specific options.
        5. **Controls** (closed) — camera keyboard reference.
        6. **Selection API** (closed) — Newton's selection panel.

        The top-level Newton ``Pause / Step`` row is suppressed; pause/resume is
        handled by the IsaacLab controls inside **Simulation**.
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
                if viewer._has_environment_controls():
                    viewer._render_environment_controls(imgui)

            # --- Live Plots -------------------------------------------------
            live_plots_cb = viewer._live_plots_callback
            if live_plots_cb is not None:
                live_plots_cb(imgui)

            # --- Model overlays ---------------------------------------------
            if viewer.model is not None:
                imgui.set_next_item_open(False, imgui.Cond_.appearing)
                if imgui.collapsing_header("Model Overlays"):
                    imgui.separator()
                    if viewer._fullscreen_image_active and not viewer.image_matches_camera:
                        imgui.text_disabled("OpenGL only")
                        hover_tooltip(
                            imgui, "Switch the Renderer to OpenGL to draw joints, contacts, and other overlays"
                        )
                    elif viewer.image_matches_camera:
                        viewer._draw_model_overlays(imgui, rtx=True)
                        imgui.text_disabled("Collision, visual, edges, and wireframes: OpenGL only")
                        hud_note(imgui, "RTX: also changes what RTX sensor cameras see.")
                    else:
                        viewer._draw_model_overlays(imgui)
            if viewer.marker_groups:
                imgui.set_next_item_open(False, imgui.Cond_.appearing)
                if imgui.collapsing_header("Visualization Markers"):
                    viewer._draw_marker_controls(imgui)

            # --- Rendering Options ------------------------------------------
            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Rendering Options"):
                imgui.separator()
                _c, viewer.vsync = imgui.checkbox("VSync", viewer.vsync)
                if viewer._view_controls is not None:
                    # these controls draw the rendering callbacks in their Lighting & Appearance group
                    viewer._view_controls(imgui)
                else:
                    for callback in _g._ui_callbacks.get("rendering", []):
                        callback(imgui)

            # --- Controls ---------------------------------------------------
            imgui.set_next_item_open(False, imgui.Cond_.appearing)
            if imgui.collapsing_header("Controls"):
                imgui.separator()
                picking = viewer.picking_enabled and (
                    not viewer._fullscreen_image_active or viewer.image_matches_camera
                )
                sections = [
                    (
                        "Camera",
                        [
                            "WASD - Move",
                            "Shift + WASD - Move 2x faster",
                            "QE - Up / down",
                            "Left Click - Look around",
                            "Middle Click - Orbit",
                            "Shift + Middle Click - Pan",
                            "Ctrl + Middle Click - Dolly",
                            "Scroll - Dolly",
                            "Ctrl + Scroll - FOV zoom",
                            "F - Frame the model",
                        ],
                    ),
                    (
                        "Objects",
                        ["Right Click - Pick and drag"] if picking else [],
                    ),
                    (
                        "Playback",
                        [
                            "Space - Pause / resume rendering",
                            ". - Step a frame (rendering paused)",
                        ],
                    ),
                    ("Window", ["H - Show / hide the UI"]),
                ]
                for title, lines in sections:
                    if not lines:
                        continue
                    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*nav_highlight_color))
                    imgui.text(title)
                    imgui.pop_style_color()
                    for line in lines:
                        imgui.text(line)
                    imgui.spacing()

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
        imgui.text("Visible Envs")
        changed, new_count = imgui.slider_int("##VisibleEnvs", owner.visible_env_count, 1, owner.visible_env_limit)
        if changed:
            owner.set_visible_env_count(new_count)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Number of environments drawn in the viewer; simulation is unaffected")
        if getattr(owner, "rtx_rendering", False):
            hud_note(imgui, "RTX: also changes what RTX sensor cameras see.")

    def _draw_marker_controls(self, imgui) -> None:
        """Draw one toggle per visualization marker group the task registers."""
        groups = list(self.marker_groups)
        if (
            getattr(self._owner, "rtx_view_index", None) is not None
        ):  # OVRTX draws the markers as prims of the shared scene
            hud_note(imgui, "RTX: markers also show up in RTX sensor cameras.")
        if not groups:
            imgui.text_disabled("This task registers no visualization markers")
            hover_tooltip(
                imgui, "Markers such as velocity-command arrows appear when a task enables debug visualization"
            )
            return
        for marker in groups:
            name = marker.cfg.prim_path.rsplit("/", 1)[-1].replace("_", " ")
            changed, visible = imgui.checkbox(f"Show {name}##{id(marker)}", marker.is_visible())
            hover_tooltip(imgui, marker_tooltip(marker.cfg.prim_path))
            if changed:
                marker.set_visibility(visible)
        if len(groups) > 1:
            if imgui.button("Show All##markers"):
                for marker in groups:
                    marker.set_visibility(True)
            hover_tooltip(imgui, "Show every marker group")
            imgui.same_line()
            if imgui.button("Hide All##markers"):
                for marker in groups:
                    marker.set_visibility(False)
            hover_tooltip(imgui, "Hide every marker group")

    def _draw_model_overlays(self, imgui, rtx: bool = False) -> None:
        """Draw Newton's per-overlay toggles; the OVRTX view draws only the line and arrow overlays."""
        renderer = self.renderer

        def tip(text: str) -> None:
            hover_tooltip(imgui, text)

        _c, self.show_joints = imgui.checkbox("Show Joints", self.show_joints)
        tip("Draw the model's joints")
        if self.show_joints:
            _, renderer.joint_scale = imgui.slider_float("Joint Scale", renderer.joint_scale, 0.25, 5.0)
            tip("Size of the joint drawings")
        _contacts_available = self._contacts_available
        self.show_contacts = _imgui_optional_checkbox(
            imgui,
            "Show Contacts",
            self.show_contacts,
            _contacts_available,
            "No contact sensors in this environment",
            "Draw contacts as arrows",
        )
        if self.show_contacts and _contacts_available:
            _, renderer.arrow_length_scale = imgui.slider_float(
                "Contact Length", renderer.arrow_length_scale, 0.25, 5.0
            )
            tip("Length of the contact arrows")
            _, renderer.arrow_scale = imgui.slider_float("Contact Width", renderer.arrow_scale, 0.25, 5.0)
            tip("Width of the contact arrows")
        _lidar_available = (not rtx) and self._lidar_available
        self.show_lidar_hits = _imgui_optional_checkbox(
            imgui,
            "Show Lidar Hits",
            self.show_lidar_hits,
            _lidar_available,
            "OpenGL only" if rtx else "No lidar sensors in this environment",
            "Copy and draw lidar hits",
        )
        _model = self.model
        _has_particles = _model.particle_count > 0
        _has_springs = _model.spring_count > 0
        if rtx:
            self.show_springs = _imgui_optional_checkbox(
                imgui,
                "Show Springs",
                self.show_springs,
                _has_springs,
                "No spring constraints in this environment",
                "Draw spring constraints",
            )
            _c, self.show_com = imgui.checkbox("Show Center of Mass", self.show_com)
            tip("Draw each body's center of mass")
            _c, self.show_inertia_boxes = imgui.checkbox("Show Inertia Boxes", self.show_inertia_boxes)
            tip("Draw each body's inertia as a box")
            return
        _has_cloth = _model.tri_count > 0
        self.show_particles = _imgui_optional_checkbox(
            imgui,
            "Show Particles",
            self.show_particles,
            _has_particles,
            "No particle bodies in this environment",
            "Draw simulation particles",
        )
        self.show_springs = _imgui_optional_checkbox(
            imgui,
            "Show Springs",
            self.show_springs,
            _has_springs,
            "No spring constraints in this environment",
            "Draw spring constraints",
        )
        _c, self.show_com = imgui.checkbox("Show Center of Mass", self.show_com)
        tip("Draw each body's center of mass")
        if self.show_com:
            _, renderer.com_scale = imgui.slider_float("COM Scale", renderer.com_scale, 0.25, 5.0)
            tip("Size of the center-of-mass markers")
        self.show_triangles = _imgui_optional_checkbox(
            imgui,
            "Show Cloth",
            self.show_triangles,
            _has_cloth,
            "No cloth/triangle meshes in this environment",
            "Draw cloth and triangle meshes",
        )
        _c, self.show_collision = imgui.checkbox("Show Collision", self.show_collision)
        tip("Draw collision shapes")
        _c, renderer.draw_edges = imgui.checkbox("Show Edges", renderer.draw_edges)
        tip("Draw mesh edges")
        _sdf_labels = ["Off", "Margin", "Margin + Gap"]
        _, new_sdf_idx = imgui.combo("Gap + Margin", int(self.sdf_margin_mode), _sdf_labels)
        tip("Draw the collision margin, and optionally the gap, around shapes")
        self.sdf_margin_mode = self.SDFMarginMode(new_sdf_idx)
        if self.sdf_margin_mode != self.SDFMarginMode.OFF:
            _, renderer.wireframe_line_width = imgui.slider_float(
                "Wireframe Width (px)", renderer.wireframe_line_width, 0.5, 5.0
            )
            tip("Line width of the margin and gap drawings")
        _c, self.show_visual = imgui.checkbox("Show Visual", self.show_visual)
        tip("Draw the visual meshes")
        _c, self.show_inertia_boxes = imgui.checkbox("Show Inertia Boxes", self.show_inertia_boxes)
        tip("Draw each body's inertia as a box")

    def _render_training_controls(self, imgui):
        """Render the Isaac Lab playback controls inside the Simulation section."""
        pause_label = "Resume Simulation" if self._paused_training else "Pause Simulation"
        if imgui.button(pause_label):
            self._paused_training = not self._paused_training
        hover_tooltip(
            imgui,
            "Stop or resume stepping the simulation. The window keeps updating",
        )

        rendering_label = "Resume Rendering" if self._paused else "Pause Rendering"
        if imgui.button(rendering_label):
            self._paused = not self._paused
        hover_tooltip(imgui, "Freeze or resume the window image (Space). The simulation keeps running")

        if imgui.button("Reset Episode"):
            self._reset_requested = True
        hover_tooltip(imgui, "Reset every environment on the next step")

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
        if self.keyboard_frame_hook is not None:
            self.keyboard_frame_hook()
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

    def _update_camera(self, dt: float):
        """Move the camera from held keys, unless keyboard camera movement is disabled."""
        if self.camera_keys_suspended:
            return
        super()._update_camera(dt)

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
        if not self.show_particles or not self._all_mpm_particles_active():
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


def _is_ovrtx(renderer_cfg) -> bool:
    """Whether a camera's renderer config selects the OVRTX renderer."""
    return getattr(renderer_cfg, "renderer_type", None) == "ovrtx"


class NewtonVisualizerBase(BaseVisualizer):
    """Own Newton resource bindings, frame cadence, and camera selection for both renderers."""

    _overlay_publisher: OverlayMeshPublisher | None = None
    _rtx_camera_path: str | None = None
    """Prim path of the viewer camera launch declared for this window; None accepts any viewer camera."""
    _rtx_arrows_pending = False
    _rtx_overlay_flags: tuple | None = None
    _rtx_visibility_limited = False
    _rtx_switch: str | None = None
    """``"requested"`` once the HUD asked for RTX and showed the wait message, ``"loading"`` until its first frame."""
    _borrowing_view = False
    """Whether an off-screen capture has swapped the displayed view, so the window controls must not act on it."""
    _rtx_appearance: RtxAppearance | None = None

    # The OpenGL window overrides these; the standalone RTX window has no OpenGL view to switch to.
    rtx_view_index: int | None = None
    rtx_rendering: bool = False
    sensor_view_selected: bool = False
    _sensor_ui_callbacks: tuple[Callable[[Any, Camera], None], ...] = ()
    """Callbacks that draw a scene sensor's own controls in the Camera Sensor section; see
    :meth:`register_sensor_ui_callback`."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._viewer = None
        self.backend = None
        self._runtime_headless = cfg.headless
        self._last_present_time = -math.inf
        self._last_render_step = -1
        self._pending_mesh_submissions: dict[str, dict[str, Any]] = {}
        self._camera_keys_enabled = True
        self._overlay_publisher = None
        self._pose_started = False
        self._marker_revision = -1
        self._perspective_pose: tuple | None = None
        """Last pose and focal length of the Perspective view, kept while another source is shown."""
        self._sensor_tiles: dict[object, int] = {}
        """Number of environments each multi-environment scene camera offers, by image view."""
        self._rtx_appearance: RtxAppearance | None = None
        self._rtx_appearance_failed = False
        self._sky_custom = ""
        self._sky_custom_open = False

    def initialize(self, sim: SimulationContext, *, cameras: list[PerspectiveCameraCfg | Camera]) -> None:
        """Bind camera sources and the display mode shared by both windows."""
        super().initialize(sim, cameras=cameras)
        # each perspective view renders itself, so a recorder can capture one the window does not show
        for view in self._image_views:
            if view.camera is not None:
                continue
            if view.render is not None:
                raise ValueError("A perspective image view must have exactly one rendering owner.")
            view.render = partial(self._capture_view, view)
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
        self._viewer.camera_keys_suspended = not self._camera_keys_enabled
        if not self._runtime_headless:
            # keyboard input is optional: a failure here must not stop the visualizer from starting
            try:
                self._key_event_source = NewtonKeyEventSource(self._viewer)
            except Exception:
                logger.exception("[%s] Keyboard input is unavailable", type(self).__name__)

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
        renderers = self._rtx_renderers
        for renderer in renderers:
            # The OVRTX viewer camera draws its own scene, so the mesh is published there as well.
            renderer.log_mesh(
                f"Viewer_{id(self)}/{name.lstrip('/')}", points, indices, normals, uvs, texture, hidden,
                backface_culling, color, roughness, metallic, dynamic, opacity,
            )  # fmt: skip
        view = self.current_image_view
        camera = getattr(view, "camera", None)
        if renderers and camera is not None and (self.rtx_rendering or _is_ovrtx(camera.cfg.renderer_cfg)):
            # The cached RTX image predates the mesh, which matters while paused or between renders.
            view.camera._invalidate_buffers()
            view.invalidate()

    @property
    def _rtx_renderers(self) -> tuple:
        """OVRTX renderers of the viewer camera and of the scene camera the window displays; empty when neither is."""
        if self._sim is None:
            return ()
        index = getattr(self, "rtx_view_index", None)
        views = [] if index is None else [self._image_views[index]]
        current = getattr(self, "current_image_view", None)
        if current is not None and all(current is not view for view in views):
            views.append(current)
        renderers = {}
        for view in views:
            camera = getattr(view, "camera", None)
            cfg = None if camera is None else camera.cfg.renderer_cfg
            if cfg is not None and (index is not None and view is self._image_views[index] or _is_ovrtx(cfg)):
                renderer = self._sim.get_or_create_backend(cfg)
                renderers[id(renderer)] = renderer
        return tuple(renderers.values())

    def _draw_rtx_unavailable(self, imgui) -> None:
        """Show the Renderer choice greyed out, since without the OVRTX viewer camera it cannot be made at runtime."""
        imgui.begin_disabled()
        imgui.text("Renderer")
        imgui.same_line()
        imgui.radio_button("OpenGL", True)
        imgui.same_line()
        imgui.radio_button("RTX", False)
        imgui.end_disabled()
        hover_tooltip(
            imgui, "RTX adds an OVRTX camera, which costs some startup time and a little speed, so it is opt-in"
        )
        hud_note(imgui, "Launch with --viz newton_rtx to enable the RTX toggle.")

    def _draw_streaming_view_controls(self, imgui) -> None:
        """Choose one declared display source without reading the other sensors."""
        if self._borrowing_view:  # an off-screen capture swapped the view; the window controls must not act on it
            return
        rtx_index = getattr(self, "rtx_view_index", None)
        if rtx_index is not None:
            rtx_on = self.rtx_rendering
            sensor = self.sensor_view_selected
            if sensor:  # The sensor draws its own image, so the choice would only leave the sensor view.
                imgui.begin_disabled()
            imgui.text("Renderer")
            imgui.same_line()
            if imgui.radio_button("OpenGL", not rtx_on and not sensor) and rtx_on:
                self.set_rtx_rendering(False)
            if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                imgui.set_tooltip(
                    "Fast rasterizer with visual overlays and markers"
                    if not sensor
                    else "A scene sensor is rendered by its own renderer. This applies to the free camera"
                )
            imgui.same_line()
            if imgui.radio_button("RTX", rtx_on) and not rtx_on and self._rtx_switch is None:
                # The first RTX frame blocks the window, so show the message for a frame before switching.
                self._rtx_switch = "requested"
            if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                imgui.set_tooltip(
                    "OVRTX path-traced rendering of the scene; slower than OpenGL"
                    if not sensor
                    else "A scene sensor is rendered by its own renderer. This applies to the free camera"
                )
            if sensor:
                imgui.end_disabled()
            if self._rtx_switch is not None:
                hud_warning(imgui, "Switching to RTX. The first frame can take a while and the window may not respond.")
        elif isinstance(self, NewtonGLVisualizer):
            self._draw_rtx_unavailable(imgui)
        self._draw_appearance(imgui)
        self._draw_focal_length(imgui)
        # The OVRTX camera is chosen with the Renderer toggle, so the list holds only the other sources.
        indices = sorted(
            (i for i in range(len(self._image_views)) if i != rtx_index),
            key=lambda i: self._source_rank(self._image_views[i]),
        )
        labels = self._view_labels(indices)
        current = self._image_views.index(self.current_image_view)
        if current == rtx_index:  # RTX renders the free perspective source.
            current = next(
                (i for i in indices if self._image_views[i].camera is None),
                indices[0],  # No free perspective view: show the first source
            )
        position = indices.index(current)
        changed, chosen = imgui.combo("Camera Source", position, labels)
        if changed:
            self._choose_camera_source(indices[chosen])
        if imgui.is_item_hovered():
            imgui.set_tooltip(labels[position])
        if self.sensor_view_selected:
            self._draw_sensor_controls(imgui, self.current_image_view)
        self._draw_camera_info(imgui)

    def _draw_camera_info(self, imgui) -> None:
        """Draw the navigation camera's position, field of view, and orientation."""
        camera = getattr(self._viewer, "camera", None)
        if camera is None or self.sensor_view_selected:  # A sensor draws its own image, not this camera's view.
            return
        imgui.separator()
        pos = camera.pos
        imgui.text_disabled(f"Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        hover_tooltip(imgui, "World position of the camera [m]")
        imgui.text_disabled(f"Rotation: (pitch {camera.pitch:.1f}, yaw {camera.yaw:.1f}) deg")
        hover_tooltip(imgui, "Look direction: pitch up or down, yaw around the vertical axis")
        imgui.text_disabled(f"FOV: {camera.fov:.1f} deg")
        hover_tooltip(imgui, "Vertical field of view; the Focal Length slider changes it")

    def _rtx_appearance_controller(self) -> RtxAppearance | None:
        """Create the OVRTX appearance controller on first use.

        Returns:
            The controller. None until the viewer camera has a rendered frame, or if the controller cannot be created.
        """
        if self._rtx_appearance is None and not self._rtx_appearance_failed:
            view = self._rtx_view()
            try:
                camera = view.camera
                renderer = self._sim.get_or_create_backend(camera.cfg.renderer_cfg)
                if camera._render_data in renderer.backend.unrendered:
                    # The viewer camera shares the OVRTX engine. Until a frame renders, a failed frame can move it to a
                    # new render product, and the controller would keep writing to the old one without an error.
                    return None
                # The viewer camera has one view, so it has one render product. The unpacking fails if it has more,
                # because the controls would change only the first one.
                (product,) = camera._render_data.products
                self._rtx_appearance = RtxAppearance(
                    self._sim.stage,
                    renderer,
                    product.path,
                    camera.cfg.background_color,
                )
            except Exception:
                logger.exception("[NewtonVisualizer] The OVRTX appearance controls are unavailable.")
                self._rtx_appearance_failed = True
        return self._rtx_appearance

    def _draw_appearance(self, imgui) -> None:
        """Draw the lighting and appearance options of the active renderer, right below the Camera Source selector."""
        rtx_rendering = getattr(self, "rtx_rendering", None)
        if rtx_rendering is None:  # the standalone RTX window has no OpenGL rendering callbacks or toggle
            return
        if rtx_rendering:
            self._draw_rtx_appearance(imgui)
            hud_note(imgui, "RTX: lighting also changes what RTX sensor cameras see.")  # shown with the section closed
            return
        callbacks = self._viewer.gui._ui_callbacks.get("rendering") if self._viewer.gui is not None else None
        if callbacks and imgui.tree_node("Lighting & Appearance###appearance"):
            for callback in callbacks:
                callback(imgui)
            imgui.tree_pop()

    def _draw_rtx_appearance(self, imgui) -> None:
        """Draw the Appearance options the OVRTX view supports: its lights, one group per type."""
        imgui.set_next_item_open(False, imgui.Cond_.appearing)
        if not imgui.tree_node("Lighting & Appearance###rtx_appearance"):
            return
        appearance = self._rtx_appearance_controller()
        if appearance is None or not appearance.available:
            imgui.text_disabled("Not available for this renderer")
            imgui.tree_pop()
            return
        edited = appearance.poll()

        def tip(text: str) -> None:
            hover_tooltip(imgui, text)

        def light_changed(changed: bool) -> bool:
            return appearance.apply_lights() if changed else False

        def intensity(label: str, value: float, high: float) -> tuple[bool, float]:
            return imgui.slider_float(label, value, 0.0, high, "%.0f", imgui.SliderFlags_.logarithmic)

        imgui.separator_text("Lights")
        if appearance.lights:
            changed, appearance.scene_enabled = imgui.checkbox("Scene Lights", appearance.scene_enabled)
            tip("The lights the scene authored")
            edited |= light_changed(changed)
            if appearance.scene_enabled:
                imgui.indent()
                changed, appearance.scene_scale = imgui.slider_float(
                    "Intensity##scene", appearance.scene_scale, 0.0, 3.0, "%.2fx"
                )
                tip("Intensity relative to what the scene authored")
                edited |= light_changed(changed)
                changed, tint = imgui.color_edit3("Color##scene", list(appearance.scene_tint))
                tip("Tints the scene's lights; white keeps the colors the scene authored")
                if changed:
                    appearance.scene_tint = tuple(float(c) for c in tint)
                edited |= light_changed(changed)
                imgui.unindent()
        else:
            imgui.text_disabled("This scene has no lights of its own")

        changed, appearance.dome_enabled = imgui.checkbox("Dome Light", appearance.dome_enabled)
        tip("Soft light from a sky all around the scene, and the sky behind it. Off by default")
        edited |= light_changed(changed)
        if appearance.dome_enabled:
            imgui.indent()
            changed, appearance.dome_intensity = intensity("Intensity##dome", appearance.dome_intensity, 10000.0)
            tip("Brightness of the dome light")
            edited |= light_changed(changed)
            changed, color = imgui.color_edit3("Color##dome", list(appearance.dome_color))
            tip("Color of the dome light; with a sky texture it tints the texture")
            if changed:
                appearance.dome_color = tuple(float(c) for c in color)
            edited |= light_changed(changed)
            edited |= self._draw_sky_texture(imgui, appearance)
            imgui.unindent()

        changed, appearance.sun_enabled = imgui.checkbox("Distant Light", appearance.sun_enabled)
        tip("Parallel light from one direction, like the sun, casting sharp shadows. Off by default")
        edited |= light_changed(changed)
        if appearance.sun_enabled:
            imgui.indent()
            changed, appearance.sun_intensity = intensity("Intensity##sun", appearance.sun_intensity, 20000.0)
            tip("Brightness of the distant light")
            edited |= light_changed(changed)
            changed, color = imgui.color_edit3("Color##sun", list(appearance.sun_color))
            tip("Color of the distant light")
            if changed:
                appearance.sun_color = tuple(float(c) for c in color)
            edited |= light_changed(changed)
            changed, appearance.sun_angle = imgui.slider_float("Angle", appearance.sun_angle, 0.0, 10.0, "%.1f deg")
            tip("Apparent size of the light; larger values give softer shadow edges")
            edited |= light_changed(changed)
            changed, appearance.sun_elevation = imgui.slider_float(
                "Elevation", appearance.sun_elevation, 5.0, 90.0, "%.0f deg"
            )
            tip("Height of the light above the horizon; 90 is straight down")
            edited |= light_changed(changed)
            changed, appearance.sun_azimuth = imgui.slider_float(
                "Azimuth", appearance.sun_azimuth, 0.0, 360.0, "%.0f deg"
            )
            tip("Direction the light comes from, around the vertical axis")
            edited |= light_changed(changed)
            imgui.unindent()

        if imgui.button("Reset##rtx_appearance"):
            edited |= appearance.reset()
        tip("Return to the scene's own lights and background")
        if appearance.error:
            failed = appearance.error.startswith("Could")
            imgui.push_style_color(
                imgui.Col_.text, imgui.ImVec4(1.0, 0.35, 0.35, 1.0) if failed else imgui.ImVec4(0.8, 0.8, 0.8, 1.0)
            )
            imgui.text_wrapped(appearance.error)
            imgui.pop_style_color()
        if edited:
            self._rtx_view().invalidate()
            self._last_render_step = -1
        imgui.tree_pop()

    def _draw_sky_texture(self, imgui, appearance: RtxAppearance) -> bool:
        """Choose the dome light's sky texture from the ones tasks use, or a file; returns whether it changed."""
        names = list(SKY_TEXTURES)
        current = appearance.dome_texture
        custom = len(names) + 1
        if self._sky_custom_open:  # Stays selected until a listed sky is chosen, since the file is set only on Load.
            index = custom
        elif not current:
            index = 0
        elif current in SKY_TEXTURES.values():
            index = names.index(next(n for n in names if SKY_TEXTURES[n] == current)) + 1
        else:
            index = custom
        changed, chosen = imgui.combo("Sky", index, ["None", *names, "Custom File..."])
        hover_tooltip(imgui, "Wrap a panoramic HDR image around the scene as its sky")
        edited = False
        if changed:
            self._sky_custom_open = chosen == custom
        if changed and chosen <= len(names):
            appearance.use_sky(SKY_TEXTURES[names[chosen - 1]] if chosen else "")
            edited = True
        if chosen == custom:
            _c, self._sky_custom = imgui.input_text("File##sky", self._sky_custom)
            hover_tooltip(imgui, "Path or URL of a panoramic .hdr or .exr image")
            if imgui.button("Load##sky") and self._sky_custom:
                appearance.use_sky(self._sky_custom)
        status = appearance.texture_status
        if status:  # Yellow while the sky downloads, red when it could not be loaded.
            failed = status.startswith("Could")
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.35, 0.35, 1.0) if failed else imgui.ImVec4(1.0, 0.8, 0.2, 1.0), status
            )
        return edited

    def _source_rank(self, view) -> int:
        """Order the Camera Source list: the free Perspective, then scene sensors."""
        return 0 if view.camera is None else 1

    def register_sensor_ui_callback(self, callback: Callable[[Any, Camera], None]) -> None:
        """Draw a scene sensor's own controls in the Camera Sensor section, below the Camera Source selector.

        While a scene sensor is the camera source, ``callback`` is invoked every UI frame with the ``imgui`` module and
        that sensor, after the section's channel and environment controls. It draws only for the sensors it handles.
        """
        self._sensor_ui_callbacks = (*self._sensor_ui_callbacks, callback)

    def _draw_sensor_controls(self, imgui, view) -> None:
        """Choose which channel of a sensor is shown and which environments, then draw the sensor's own controls."""
        imgui.separator_text("Camera Sensor")
        self._draw_sensor_channels(imgui, view)
        self._draw_sensor_envs(imgui, view)
        for callback in self._sensor_ui_callbacks:
            callback(imgui, view.camera)

    def _draw_sensor_envs(self, imgui, view) -> None:
        """Choose which environment of a sensor is shown, or tile several environments."""
        camera = view.camera
        offered = self._sensor_tiles.get(view)
        if offered is None:  # A single-instance sensor has no environment to choose.
            return
        shown = len(view.cfg.envs)
        limit = min(offered, MAX_TILED_ENVS)
        changed, count = imgui.input_int("Tiled Envs", shown)
        hover_tooltip(
            imgui, f"How many environments are shown side by side, from environment 0 (up to {limit}). 1 shows one."
        )
        if changed:
            count = max(1, min(count, limit))
            view.cfg.envs = tuple(range(count)) if count > 1 else (view.cfg.envs[0] if shown == 1 else 0,)
            self._invalidate_sensor_view(view)
        if len(view.cfg.envs) > 1:
            imgui.begin_disabled()
        env = view.cfg.envs[0] if view.cfg.envs else 0
        changed, new_env = imgui.input_int("Environment", env)
        hover_tooltip(imgui, "Environment whose camera is shown when Tiled Envs is 1")
        if len(view.cfg.envs) > 1:
            imgui.end_disabled()
        if changed and len(view.cfg.envs) == 1 and 0 <= new_env < camera.num_instances:
            view.cfg.envs = (new_env,)
            self._invalidate_sensor_view(view)

    _CHANNEL_LABELS = {"rgb": "RGB", "depth": "Depth", "normals": "Normals", "segmentation": "Segmentation"}

    def _draw_sensor_channels(self, imgui, view) -> None:
        """Choose which output of the sensor is shown, with a color scale range for depth."""
        available = display_channels(view.camera.cfg.data_types)
        if not available:
            return
        keys = frozenset(view.camera.cfg.data_types)
        options = []
        for channel in available:
            key = sensor_key_for_gt_type(channel, keys)
            # A color image that is not the plain render, such as albedo, is named by its output.
            options.append(
                (key if channel == "rgb" and key not in ("rgb", "rgba") else self._CHANNEL_LABELS[channel], (channel,))
            )
        if len(available) > 1:
            options.append(("All", available))
        current = tuple(view.cfg.channels)
        index = next((i for i, (_, channels) in enumerate(options) if channels == current), -1)
        if index < 0:
            options.append(("Custom", current))
            index = len(options) - 1
        changed, chosen = imgui.combo("Channel", index, [label for label, _ in options])
        hover_tooltip(imgui, "Which output of the sensor is shown: its color image, depth, normals, or segmentation")
        if changed:
            self.set_sensor_channels(view, options[chosen][1])
        if "depth" in view.cfg.channels:
            near, far = view.cfg.depth_range
            changed, (new_near, new_far) = imgui.input_float2("Depth Range (m)", [near, far])
            hover_tooltip(imgui, "Distances mapped to the ends of the depth color scale")
            if changed and 0.0 <= new_near < new_far:
                view.cfg.depth_range = (float(new_near), float(new_far))
                self._invalidate_sensor_view(view)

    def set_sensor_channels(self, view, channels: tuple[str, ...]) -> None:
        """Show ``channels`` of a sensor view, which the sensor must provide.

        Args:
            view: Image view of a scene camera.
            channels: Display channels, such as ``("depth",)``, shown side by side.
        """
        view.cfg.channels = tuple(channels)
        self._invalidate_sensor_view(view)

    def _invalidate_sensor_view(self, view) -> None:
        view.invalidate()
        self._last_render_step = -1

    def _draw_focal_length(self, imgui) -> None:
        """Draw the focal length of the free camera; sensors have fixed intrinsics."""
        view = self.current_image_view
        rtx_index = getattr(self, "rtx_view_index", None)
        rtx_view = self._image_views[rtx_index] if rtx_index is not None else None
        if view.camera is not None and view is not rtx_view:
            return
        source = self.cfg
        changed, focal = imgui.slider_float("Focal Length", float(source.focal_length), 5.0, 64.0, "%.0f mm")
        hover_tooltip(imgui, "Lens focal length of the camera; smaller values see a wider area")
        if changed:
            source.focal_length = focal
            self._viewer.camera.fov = self._focal_length_to_vertical_fov_degrees(focal)
            view.invalidate()
            self._last_render_step = -1

    def _view_labels(self, indices: list[int] | None = None) -> list[str]:
        """Name the listed image views: a free view is the Perspective."""
        views = self._image_views if indices is None else [self._image_views[i] for i in indices]
        return [
            "Perspective" if view.camera is None else self._sensor_label(view.cfg.source, view.camera.cfg.data_types)
            for view in views
        ]

    _OUTPUT_NAMES = {
        "rgb": "RGB",
        "rgba": "RGB",
        "depth": "depth",
        "distance_to_image_plane": "depth",
        "distance_to_camera": "distance",
        "normals": "normals",
        "semantic_segmentation": "semantic seg",
        "instance_segmentation": "instance seg",
        "instance_id_segmentation_fast": "instance seg",
        "albedo": "albedo",
        "simple_shading_constant_diffuse": "shaded",
        "simple_shading_diffuse_mdl": "shaded",
        "simple_shading_full_mdl": "shaded",
    }
    """Short names of the outputs a scene camera lists in the Camera Source dropdown."""

    @staticmethod
    def _sensor_label(prim_path: str, data_types=()) -> str:
        """Name a scene camera by its own prim name, without the per-environment path pattern.

        The outputs follow in parentheses, such as ``Camera (depth sensor)`` or ``Camera (RGB + depth sensor)``.
        """
        names = NewtonVisualizerBase._OUTPUT_NAMES
        outputs = list(dict.fromkeys(names[key] for key in data_types if key in names))
        kind = " + ".join(outputs[:3]) + " " if outputs else ""
        return f"{prim_path.rstrip('/').rsplit('/', 1)[-1]} ({kind}sensor)"

    def _choose_camera_source(self, index: int) -> None:
        """Switch the camera source; a free source keeps the OVRTX renderer when it is on."""
        view = self._image_views[index]
        if self.rtx_rendering and view.camera is None:
            return  # the OVRTX camera already follows the Perspective
        self._select_camera(index)

    def _remember_perspective_pose(self) -> None:
        """Save the Perspective's pose before another source takes over the viewer camera."""
        current = self.current_image_view
        if current is None or self._viewer is None:
            return
        # The viewer camera holds the Perspective's pose while it is shown, and while the OVRTX camera follows it.
        if current.camera is None or (self.rtx_rendering and any(v.camera is None for v in self._image_views)):
            camera = self._viewer.camera
            pos = camera.pos
            self._perspective_pose = (
                (float(pos[0]), float(pos[1]), float(pos[2])),
                camera.pitch,
                camera.yaw,
                self.cfg.focal_length,
            )

    def _restore_perspective_pose(self) -> bool:
        """Put the viewer camera back where the Perspective was left; False when it was never left."""
        if self._perspective_pose is None:
            return False
        pos, pitch, yaw, focal = self._perspective_pose
        self._viewer.set_camera(PygletVec3(*pos), pitch, yaw)
        self.cfg.focal_length = focal
        self._viewer.camera.fov = self._focal_length_to_vertical_fov_degrees(focal)
        return True

    def _release_rtx_scene(self) -> None:
        """Give the scene cameras that share the OVRTX renderer back the overlays and lights the viewer added."""
        if self._overlay_publisher is not None:
            self._overlay_publisher.hide_all(self._rtx_renderers)
        if self._rtx_appearance is not None and self._rtx_appearance.available:
            self._rtx_appearance.suspend()

    def _select_camera(self, index: int, *, keep_pose: bool = False) -> None:
        """Bind the selected source; all scene cameras retain their normal lazy-data lifecycle.

        Args:
            index: Image view to display.
            keep_pose: Continue from the viewer's current camera instead of the source's configured pose.
        """
        if getattr(self, "_pose_started", True):  # the viewer camera holds no pose of its own before the first source
            self._remember_perspective_pose()
        leaving_rtx = self.rtx_rendering and index != self.rtx_view_index
        if leaving_rtx:
            self._release_rtx_scene()
        self.current_image_view = self._image_views[index]
        if leaving_rtx:
            self._apply_rtx_visible_envs()  # restore every environment to the sensors that share the renderer
        camera = self.current_image_view.cfg.source
        self.current_image_view.invalidate()
        if isinstance(camera, PerspectiveCameraCfg):
            if not keep_pose and not self._restore_perspective_pose():
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
        self._pose_started = True

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

    def _capture_view(self, view, output: wp.array | None) -> wp.array:
        """Capture a perspective *view*, borrowing the viewer camera when the window shows another one."""
        current = self.current_image_view
        # a free view records what an RTX window shows
        if view is current or self.rtx_rendering:
            return self._capture_frame(output)
        camera = self._viewer.camera
        saved = (PygletVec3(camera.pos.x, camera.pos.y, camera.pos.z), camera.pitch, camera.yaw, camera.fov)
        try:
            self._borrowing_view = True
            self.current_image_view = view
            source = view.cfg.source  # a free view renders from its declared pose and optics
            camera.pos = PygletVec3(*source.eye)
            camera.look_at(source.lookat)
            self._viewer.set_camera(camera.pos, camera.pitch, camera.yaw)
            camera.fov = self._focal_length_to_vertical_fov_degrees(source.focal_length)
            self._last_render_step = -1
            return self._capture_frame(output)
        finally:
            self._borrowing_view = False
            self.current_image_view = current
            camera.fov = saved[3]
            self._viewer.set_camera(*saved[:3])
            self._last_render_step = -1

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
        self._perspective_pose = None
        self._sensor_tiles.clear()
        if self._rtx_appearance is not None:
            if self._rtx_appearance.available:
                self._rtx_appearance.suspend()
            self._rtx_appearance.close()
            self._rtx_appearance = None
        for view in self._image_views:
            if view.camera is None and view.render is not None:  # the perspective views bound in initialize
                view.render = None
                view.invalidate()
        super().close()

    def set_camera_keys_enabled(self, enabled: bool) -> None:
        """Enable or disable the window's WASD/QE and arrow-key camera movement.

        Mouse camera control and the viewer's other hotkeys keep working. The setting also applies to a
        window opened later by :meth:`initialize`.
        """
        self._camera_keys_enabled = enabled
        if self._viewer is not None:
            self._viewer.camera_keys_suspended = not enabled

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

    def __init__(self, cfg: VisualizerCfg):
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
        self._logged_lidar_names: list[str] = []

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

        self._rtx_camera_path = getattr(self.cfg.backend, "rtx_camera_path", None)
        super().initialize(sim, cameras=cameras)
        if self.rtx_view_index is None:
            self._add_viewer_camera(sim)
        self.newton_cfg = NewtonBackendCfg(physics_cfg=sim.cfg.physics, device=sim.device)
        self.backend = sim.get_or_create_backend(self.newton_cfg)
        self._transform_mapping = sim.get_scene_data_provider().create_mapping(list(self.backend.model.body_label))
        scene_data_provider = self._sim.get_scene_data_provider()
        newton_backend_active = self.physics_backend == "newton"
        physics_backend = sim.backend_manager.backend if newton_backend_active else None
        picking_supported = (
            physics_backend is not None
            and physics_backend.solver_ops is not None
            and physics_backend.solver_ops.applies_body_forces()
        )
        num_envs = scene_data_provider.num_envs
        if self._runtime_headless and not self.cfg.headless:
            logger.warning(
                "[NewtonVisualizer] No display found (DISPLAY is unset); the Newton viewer runs"
                " headless via EGL and no window will open. Run from a session with a display (or set"
                " DISPLAY, e.g. 'export DISPLAY=:0') to see the viewer."
            )

        self._picking_enabled = self.cfg.backend.enable_picking and picking_supported and not self._runtime_headless
        self._viewer.set_model(self.backend.model)
        self._geometry_change_version = self.backend.geometry_change_version
        if self._picking_enabled:
            # Keep Newton's public force path scoped to picking for this integration.
            self._viewer.wind = None
        self._cfg_visible_env_ids = self._env_ids
        self._num_envs = num_envs
        self._viewer._owner = self
        self._apply_visible_worlds()
        self._viewer._paused = False
        self._viewer.picking_enabled = self._picking_enabled
        self._configure_viewer()

        for view in self._image_views:
            camera = view.camera
            if camera is not None and camera.num_instances > 1 and view is not self._rtx_view():
                # Tile the first environments unless the view chose its own; the HUD changes how many.
                self._sensor_tiles[view] = camera.num_instances
                if self.cfg.view is None and not isinstance(self.cfg.streaming_envs, list) and view.cfg.envs == (0,):
                    view.cfg.envs = tuple(range(min(DEFAULT_TILED_ENVS, camera.num_instances)))
        frame_visualizer_closeup(self, cameras)
        self._select_camera(0)
        if self.cfg.enable_markers and self.rtx_view_index is not None:
            # OVRTX draws markers as stage prims; the viewer camera's renderer shares the registry's groups.
            renderer_cfg = self._image_views[self.rtx_view_index].camera.cfg.renderer_cfg
            sim.get_or_create_backend(renderer_cfg).marker_groups = sim.vis_marker_registry.get_groups().values()
        if self.rtx_view_index is not None:
            self._overlay_publisher = OverlayMeshPublisher(f"Viewer_{id(self)}")
        if self.cfg.backend.initial_renderer == "rtx":
            if self.rtx_view_index is None:
                raise RuntimeError(
                    "NewtonGLBackendCfg.initial_renderer='rtx' needs the OVRTX viewer camera. Launch declares it for"
                    " a run with an InteractiveSceneCfg (see NewtonGLBackendCfg.rtx_camera); a script that builds its"
                    " scene itself must add an OVRTX Camera whose prim_path starts with '/World/viewer_camera_'."
                )
            self.set_rtx_rendering(True)

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
                ("show_particles", self.cfg.backend.show_particles),
                ("enable_picking", self._picking_enabled),
            ],
        )
        if self._picking_enabled:
            self._viewer_picking_binding.bind(self._viewer)
            sim.backend_manager.register_state_force_callback(self._viewer_picking_binding.apply)
        if self.cfg.backend.enable_picking and not picking_supported:
            logger.info(
                "[NewtonVisualizer] Object dragging is disabled because the active physics solver does not support"
                " rigid-body force input."
            )
        self._is_initialized = True
        # Inform the viewer whether contact data is available so the UI can grey
        # out "Show Contacts" when neither native Newton contacts nor a ContactSensor
        # exists in the scene.
        self._viewer._contacts_available = newton_backend_active or bool(scene_data_provider.get_contact_sensors())
        self._viewer._lidar_available = scene_has_lidar(sim)

    def _log_lidar_hits(self, viewer: Any) -> None:
        """Log every visualized environment's lidar hits into the Newton GL scene."""
        self._logged_lidar_names = sync_lidar_hits(
            self._sim, self._env_ids, viewer, viewer.show_lidar_hits, self._logged_lidar_names
        )

    def _render_frame(self) -> None:
        """Render one GL frame for the window or an on-demand capture."""
        viewer = self._viewer
        appearance = self._rtx_appearance
        if self._rtx_switch == "requested":  # the previous frame showed the message
            self._rtx_switch = "loading"
            try:
                self.set_rtx_rendering(True)
            except BaseException:
                self._rtx_switch = None
                raise
        try:
            if appearance is not None and self.rtx_rendering and appearance.available and appearance.poll():
                self.current_image_view.invalidate()  # a sky finished downloading, whether or not its panel is open
            viewer.begin_frame(self._sim_time)
        except BaseException:
            if self._rtx_switch == "loading":  # the switch failed before the first RTX frame
                self._rtx_switch = None
            raise
        try:
            viewer.image_matches_camera = self.rtx_rendering
            if self.current_image_view.camera is not None:
                self._navigate_scene_camera()
                self._refresh_rtx_markers()
                self._refresh_paused_rtx_view(self._log_rtx_overlays())
                self._log_streaming_image()
            elif not viewer.is_paused():
                state = self._update_state()
                if state.body_q is None or len(state.body_q):
                    if self._geometry_change_version != self.backend.geometry_change_version:
                        viewer.notify_model_changed(newton.ModelFlags.SHAPE_TYPE_AND_SOURCE)
                        self._geometry_change_version = self.backend.geometry_change_version
                    viewer.log_state(state)
                    contacts = self.backend.contacts if self.physics_backend == "newton" else None
                    if contacts is not None:
                        viewer.log_contacts(contacts, state)
                    else:
                        self._log_scene_contact_sensor_arrows()
                    if self.cfg.enable_markers:
                        render_newton_visualization_markers(viewer, self._env_ids, num_envs=self.backend.model.num_envs)
                self._log_lidar_hits(viewer)
            for name, arguments in self._pending_mesh_submissions.items():
                viewer.log_mesh(name, **arguments)
            self._pending_mesh_submissions.clear()
            if not viewer.is_paused():
                self._render_live_plots()
        finally:
            viewer.end_frame()
            if self._rtx_switch == "loading":  # the first RTX frame arrived, or the switch failed
                self._rtx_switch = None
            if not viewer.is_running():
                self._viewer_picking_binding.deactivate()
        self._last_render_step = self._sim.get_physics_step_count()
        if self.current_image_view.camera is None:
            self.current_image_view.invalidate()

    def _log_rtx_overlays(self) -> bool:
        """Draw the line and arrow overlays the HUD enables, and the drag arrow, as meshes in OVRTX.

        Returns:
            Whether the drag arrow is being drawn, which an OVRTX view that is not stepping must render again.
        """
        publisher, viewer = self._overlay_publisher, self._viewer
        if publisher is None or not self.rtx_rendering:
            return False
        renderers = self._rtx_renderers
        overlays = any(
            getattr(viewer, flag)
            for flag in ("show_joints", "show_contacts", "show_springs", "show_com", "show_inertia_boxes")
        )
        dragging = viewer.picking_enabled and viewer.picking is not None and viewer.picking.is_picking()
        flags = tuple(
            getattr(viewer, flag)
            for flag in ("show_joints", "show_contacts", "show_springs", "show_com", "show_inertia_boxes")
        )
        changed = flags != self._rtx_overlay_flags  # a paused view must render again to show the change
        self._rtx_overlay_flags = flags
        if not overlays and not dragging:
            publisher.hide_all(renderers)
            return changed
        if viewer.is_paused():
            return False
        state = self._update_state()
        if state.body_q is not None and not len(state.body_q):
            return False
        viewer._overlay_capture = publisher
        # Contact disks are instanced meshes that only the OpenGL view draws, and they dominate the contact cost.
        disks = viewer.show_contact_disks
        viewer.show_contact_disks = False
        try:
            if overlays:
                # Only the overlays are needed, not the per-shape instances that log_state also updates for OpenGL.
                getattr(viewer, "_log_non_shape_state", viewer.log_state)(state)
                contacts = self._sim.backend_manager.get_contacts() if self.physics_backend == "newton" else None
                if contacts is not None:
                    viewer.log_contacts(contacts, state)
                else:
                    self._log_scene_contact_sensor_arrows()
            else:
                viewer._render_picking_line(state)
        finally:
            viewer._overlay_capture = None
            viewer.show_contact_disks = disks
        publisher.publish(renderers)
        return bool(dragging) or changed

    def _refresh_rtx_markers(self) -> None:
        """Render the OVRTX cameras again when a marker group is shown, hidden, or moved, which OVRTX draws as prims."""
        if not self.cfg.enable_markers:
            return
        revision = self._sim.vis_marker_registry.revision
        if self._marker_revision == revision:
            return
        self._marker_revision = revision
        for view in self._image_views:
            if view.camera is not None:
                view.camera._invalidate_buffers()
                view.invalidate()

    def _refresh_paused_rtx_view(self, arrows_changed: bool) -> None:
        """Render the OVRTX camera again while the simulation is paused, which does not render it by itself.

        Args:
            arrows_changed: Whether the drag arrow is being drawn, so the image changes every frame.
        """
        view = self.current_image_view
        if not self.rtx_rendering or not self.is_training_paused():
            self._rtx_arrows_pending = False
            return
        if not (arrows_changed or self._rtx_arrows_pending or view.frame.timestamp < 0):
            return
        view.camera._is_outdated.fill_(True)
        view.camera.update(0.0, force_recompute=True)
        view.invalidate()
        self._rtx_arrows_pending = arrows_changed  # One more render shows the arrows gone.

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

    def _rtx_view(self):
        """The OVRTX viewer camera's image view, or None when the scene has none."""
        index = self.rtx_view_index
        return None if index is None else self._image_views[index]

    @property
    def rtx_view_index(self) -> int | None:
        """Index of this window's OVRTX viewer camera among the image views, or None when the scene has none."""
        for index, view in enumerate(self._image_views):
            if view.camera is not None and self._is_viewer_camera(getattr(view.camera.cfg, "prim_path", "")):
                return index
        return None

    def _is_viewer_camera(self, prim_path: str) -> bool:
        """Whether *prim_path* is the viewer camera launch declared for this window, or any viewer camera if none."""
        declared = self._rtx_camera_path
        return prim_path == declared if declared else prim_path.startswith("/World/viewer_camera_")

    @property
    def sensor_view_selected(self) -> bool:
        """Whether the main view shows a scene sensor, which is rendered by its own renderer."""
        view = self.current_image_view
        return view is not None and view.camera is not None and not self.rtx_rendering

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
            raise RuntimeError("The scene has no OVRTX viewer camera; see NewtonGLBackendCfg.rtx_camera.")
        free = [i for i, view in enumerate(self._image_views) if view.camera is None]
        # While the RTX view is shown its camera joins the scene's shared render, so reading it does not submit every
        # camera product a second time; otherwise it renders only on demand.
        self._image_views[rtx_index].camera.cfg.render_on_demand = not enabled
        if enabled:
            self._select_camera(rtx_index, keep_pose=True)
            self._apply_rtx_visible_envs()
            if self._rtx_appearance is not None and self._rtx_appearance.available:
                self._rtx_appearance.apply_lights()
        else:
            target = free[0] if free else next(i for i in range(len(self._image_views)) if i != rtx_index)
            # Only a perspective view continues from the viewer's pose; a scene sensor keeps the task's own placement.
            self._select_camera(target, keep_pose=self._image_views[target].camera is None)

    @property
    def visible_env_limit(self) -> int:
        """Largest value of the visible-environment count: the environments the cfg allows, else all."""
        return len(self._cfg_visible_env_ids) if self._cfg_visible_env_ids is not None else self._num_envs

    @property
    def visible_env_count(self) -> int:
        """Number of environments currently shown."""
        if self._visible_env_count is not None:
            return min(self._visible_env_count, self.visible_env_limit)
        return self.visible_env_limit

    def set_visible_env_count(self, count: int) -> None:
        """Show only the first ``count`` environments.

        Args:
            count: Number of environments to show, clamped to ``[1, visible_env_limit]``.
        """
        self._visible_env_count = max(1, min(int(count), self.visible_env_limit))
        self._apply_visible_worlds()

    def _select_visible_env_ids(self) -> list[int] | None:
        """Resolve the environments to show from the cfg and the HUD count."""
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

    def _add_viewer_camera(self, sim: SimulationContext) -> None:
        """Add the OVRTX viewer camera launch declared for this window when its explicit view or cameras omit it."""
        sensors = sim.get_scene_data_provider().get_camera_sensors()
        camera = next((sensor for sensor in sensors.values() if self._is_viewer_camera(sensor.cfg.prim_path)), None)
        if camera is None:
            return
        # Launch declares this camera with RGBA only, whatever channels the window's scene cameras show.
        view_cfg = ImageViewCfg(source=camera.cfg.prim_path, envs=(0,), channels=("rgb",))
        view = sim.get_or_create_backend(view_cfg, camera=camera)
        view.aspect = self.cfg.window.size[0] / self.cfg.window.size[1]
        self._image_views.append(view)

    def _apply_visible_worlds(self) -> None:
        """Apply the visible environments."""
        if self._viewer is None:
            return
        self._env_ids = self._select_visible_env_ids()
        self._drop_shape_instancers()
        self._viewer.set_visible_worlds(self._env_ids)
        self._viewer.set_world_offsets(self.cfg.backend.world_spacing)
        self._apply_rtx_visible_envs()

    def _apply_rtx_visible_envs(self) -> None:
        """Hide the environments outside the visible set in the OVRTX viewer camera's scene."""
        index = self.rtx_view_index
        if index is None:
            return
        renderer = self._sim.get_or_create_backend(self._image_views[index].camera.cfg.renderer_cfg)
        num_envs = len(renderer._env_paths)
        if num_envs:
            # The renderer is shared with scene cameras, so the limit applies only while the RTX view is shown, and
            # OpenGL leaves the renderer alone unless an earlier limit has to be lifted.
            limited = self._env_ids is not None and self.rtx_rendering
            if not limited and not self._rtx_visibility_limited:
                return
            visible = np.zeros(num_envs, dtype=bool) if limited else np.ones(num_envs, dtype=bool)
            if limited:
                visible[np.asarray(list(self._env_ids), dtype=int)] = True
            renderer.set_environment_visibility(visible)
            self._rtx_visibility_limited = limited
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
        self._viewer.set_model(self.backend.model)
        self._geometry_change_version = self.backend.geometry_change_version
        self._viewer.register_ui_callback(self._viewer._render_training_controls, position="side")
        self._viewer._view_controls = self._draw_streaming_view_controls
        if self._picking_enabled:
            self._viewer.wind = None
        self._apply_visible_worlds()
        self._configure_viewer()
        self._viewer.picking_enabled = self._picking_enabled
        if self._picking_enabled:
            self._viewer_picking_binding.bind(self._viewer)
            self._sim.backend_manager.register_state_force_callback(self._viewer_picking_binding.apply)

    def close(self) -> None:
        """Release picking and scene references without destroying Kit's shared GL context."""
        if self._is_closed:
            return
        self._close_key_event_source()
        try:
            if self._rtx_visibility_limited:  # the shared renderer must not keep hiding environments from sensors
                try:
                    self._env_ids = None
                    self._apply_rtx_visible_envs()
                except Exception:
                    logger.debug("[NewtonVisualizer] The environment visibility could not be restored.", exc_info=True)
            if self._overlay_publisher is not None:
                try:
                    self._overlay_publisher.hide_all(self._rtx_renderers)
                except Exception:
                    logger.debug("[NewtonVisualizer] The overlay meshes could not be hidden.", exc_info=True)
            if self._picking_enabled:
                # Captured graphs retain neutral picking inputs until their last replay.
                self._viewer_picking_binding.deactivate()
            if sys.platform == "win32" and self._viewer is not None and not self._runtime_headless:
                window = self._viewer.renderer.window
                if window.context is not None:
                    # Release focus while pyglet's weak event handlers still have live targets.
                    window.set_visible(False)
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
        expanded = imgui.collapsing_header("Live Plots")
        hover_tooltip(imgui, "Plot task values such as rewards and observations while it runs")
        if not expanded:
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
            shown = imgui.collapsing_header(term_label)
            hover_tooltip(imgui, self._live_plot_tooltip(base_name))
            if not shown:
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
            shown = imgui.collapsing_header(name)
            hover_tooltip(imgui, self._live_plot_tooltip(name))
            if shown:
                plots._render_array_heatmap(imgui, name, array, panel_width - 20 * scale, dpi_scale=scale)

    def _live_plot_tooltip(self, name: str) -> str:
        """Describe a plotted ``<manager>/<term>`` from the term's docstring, or name where it comes from."""
        manager_name, _, term = name.partition("/")
        source = next((source for source in self._live_plot_sources if source.manager_name == manager_name), None)
        if source is None or not term:
            return name
        summary = term_summary(getattr(source, "_manager", None), term)
        return summary or f"{term} from the {manager_name.replace('_', ' ')}"

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
        self._viewer.show_joints = self.cfg.backend.show_joints
        self._viewer.show_contacts = self.cfg.backend.show_contacts
        self._viewer.show_collision = self.cfg.backend.show_collision
        self._viewer.show_springs = self.cfg.backend.show_springs
        self._viewer.show_inertia_boxes = self.cfg.backend.show_inertia_boxes
        self._viewer.show_com = self.cfg.backend.show_com
        self._viewer.show_particles = self.cfg.backend.show_particles
        self._viewer.show_lidar_hits = self.cfg.backend.show_lidar_hits
        self._viewer.up_axis = 2  # Z-up
        self._viewer.scaling = 1.0
        self._viewer.particle_color = self.cfg.backend.particle_color
        self._viewer.renderer.draw_shadows = self.cfg.backend.enable_shadows
        self._viewer.renderer.draw_wireframe = self.cfg.backend.enable_wireframe
        # Accept list/tuple/array-like config colors; provide a stable tuple for nanobind conversion.
        if self.cfg.background_color is None:
            self._viewer.renderer.draw_sky = self.cfg.backend.enable_sky
            upper_color = self.cfg.backend.sky_upper_color
            lower_color = self.cfg.backend.sky_lower_color
        else:
            self._viewer.renderer.draw_sky = False
            upper_color = lower_color = self.cfg.background_color
        self._viewer.renderer.sky_upper = tuple(upper_color)
        self._viewer.renderer.sky_lower = tuple(lower_color)
        self._viewer.renderer._light_color = tuple(self.cfg.backend.light_color)
