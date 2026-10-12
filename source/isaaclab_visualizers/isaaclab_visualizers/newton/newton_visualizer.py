# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton Visualizer implementation — shared base plus GL and RTX backends."""

from __future__ import annotations

import contextlib
import logging
import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np  # noqa: F401 — used in type hints and colorization helpers
import torch
import warp as wp

# On Linux without a display, set pyglet's headless option BEFORE importing newton.viewer
# so ViewerGL resolves to an EGL HeadlessWindow at class-definition time.  Only apply on
# headless Linux; on macOS/Windows or when DISPLAY is set the flag is left unset so
# interactive windows open normally.
if __import__("sys").platform not in ("win32", "darwin") and not __import__("os").environ.get("DISPLAY"):
    import pyglet as _pyglet_headless_init

    _pyglet_headless_init.options["headless"] = True
    del _pyglet_headless_init
elif sys.platform not in ("win32", "darwin"):
    # A monitor-free X server (e.g. a GPU-backed virtual display) makes pyglet fall back to a plain
    # XlibScreen, which lacks the ``is_primary`` its default-screen lookup reads, so no window opens.
    # Defaulting to non-primary makes pyglet use its first screen.
    with contextlib.suppress(ImportError, AttributeError):  # best effort; skip if pyglet's Xlib layout differs
        from pyglet.display import xlib as _pyglet_xlib

        if not hasattr(_pyglet_xlib.XlibScreen, "is_primary"):
            _pyglet_xlib.XlibScreen.is_primary = False
        del _pyglet_xlib

import newton
from isaaclab_newton.physics import NewtonBackendCfg, NewtonManager
from newton.viewer import ViewerGL, ViewerRTX
from pyglet.math import Vec3 as PygletVec3

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
    evict_visualizer_camera,
    find_camera_by_prim_path,
    prim_world_positions,
    remove_generated_prims,
    resolve_streaming_envs,
)
from isaaclab.scene_data import SceneDataFormat
from isaaclab.sim import SimulationContext
from isaaclab.visualizers.base_visualizer import BaseVisualizer

from isaaclab_visualizers.desktop_entry import write_desktop_entry
from isaaclab_visualizers.newton.newton_visualization_markers import render_newton_visualization_markers
from isaaclab_visualizers.newton_adapter import resolve_visible_env_indices

from .newton_visualizer_cfg import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg, NewtonVisualizerCfg

logger = logging.getLogger(__name__)


def _newton_scalar_base_name(name: str) -> str:
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


def _apply_newton_icon(window) -> None:
    """Set Newton's bundled icon on a ``ViewerRTX`` window, which unlike ``ViewerGL`` never sets one.

    TODO(Newton): remove once ``ViewerRTX`` sets its own icon
    (https://github.com/newton-physics/newton/issues/4321).
    """
    import pyglet

    try:
        images = [pyglet.image.load(str(_NEWTON_ICON_DIR / f"icon_{size}.png")) for size in (16, 32, 64)]
    except (OSError, pyglet.util.DecodeException) as error:
        logger.warning("Could not load Newton's bundled icon for the RTX viewer window: %s", error)
        return
    # Headless/EGL windows may not support icons.
    with contextlib.suppress(Exception):
        window.set_icon(*images)


if TYPE_CHECKING:
    from newton import State

    from isaaclab.scene_data import SceneDataProvider


def _imgui_optional_checkbox(imgui, label: str, value: bool, available: bool, tip: str) -> bool:
    """Render a checkbox greyed out with a tooltip when *available* is False."""
    if not available:
        imgui.begin_disabled()
    _, new_val = imgui.checkbox(label, value)
    if not available:
        imgui.end_disabled()
        try:
            if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                imgui.set_tooltip(tip)
        except Exception:
            pass
        return value
    return new_val


def _eye_lookat_to_pitch_yaw(
    eye: tuple[float, float, float],
    lookat: tuple[float, float, float],
) -> tuple[float, float]:
    """Convert an eye/lookat camera specification to pitch and yaw angles in degrees.

    Uses a Z-up coordinate convention: pitch is elevation above the XY plane,
    yaw is the counter-clockwise angle from the +X axis.

    Args:
        eye: Camera position in world space.
        lookat: Camera look-at target in world space.

    Returns:
        Tuple of ``(pitch_degrees, yaw_degrees)``.
    """
    dx = lookat[0] - eye[0]
    dy = lookat[1] - eye[1]
    dz = lookat[2] - eye[2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-8:
        return 0.0, 0.0
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, dz / length))))
    yaw = math.degrees(math.atan2(dy, dx))
    return pitch, yaw


# ---------------------------------------------------------------------------
# Newton viewer wrappers (add IsaacLab ImGui controls to Newton's viewers)
# ---------------------------------------------------------------------------


class _NewtonViewerUIMixin:
    """Mixin providing shared Isaac Lab UI for Newton viewer wrappers.

    Both :class:`NewtonViewerRTX` and :class:`NewtonViewerGL` inherit from this
    mixin to share panel-patching helpers and training-controls widgets without
    duplicating code.
    """

    # Set to False by NewtonVisualizer.initialize() when neither native Newton
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

    def _register_isaaclab_ui_callbacks(self) -> None:
        """Register model-dependent Isaac Lab viewer controls."""
        self.register_ui_callback(self._render_training_controls, position="side")

    def _patch_scalar_plot_width(self) -> None:
        """Set up ImPlot and suppress Newton's built-in floating Plots window.

        Plots are rendered inline in the left panel by
        :meth:`~NewtonVisualizer._live_plots_panel_imgui` instead.
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

    def _patch_image_logger(self) -> None:
        """Patch the image logger for streaming view integration.

        Suppresses Newton's built-in ``draw_controls`` sidebar section
        (which labels itself "Logged Images (N)"), since :meth:`_draw_streaming_view_controls`
        provides the selection UI. Also overrides the initial window size to
        75 % of the available viewport area so the first-open panel is large.

        When no ``_image_logger`` attribute is present (e.g. on the RTX backend),
        this method returns immediately without making any changes.
        """
        import types

        image_logger = getattr(self, "_image_logger", None)
        if image_logger is None:
            return

        # Suppress Newton's own "Logged Images" sidebar section.
        image_logger.draw_controls = lambda: None

        # Override draw() to open the floating panel sized to the composite aspect ratio.
        _orig_draw = type(image_logger).draw
        _viewer_ref = self  # capture for closure — used to read composite dimensions

        def _draw_large(self_logger: object) -> None:
            # Use our own flag (not Newton's entry.window_initialized) so Newton cannot
            # preempt our sizing by marking the window as initialized via the placeholder.
            if getattr(_viewer_ref, "_streaming_panel_needs_sizing", False):
                comp_w = getattr(_viewer_ref, "_streaming_composite_w", 0)
                comp_h = getattr(_viewer_ref, "_streaming_composite_h", 0)
                if comp_w > 0 and comp_h > 0:
                    from imgui_bundle import imgui as _imgui

                    vp = _imgui.get_main_viewport()
                    sidebar_w = float(self_logger._sidebar_width_px)
                    margin = 20.0
                    avail_w = max(320.0, vp.work_size.x - sidebar_w - 2.0 * margin)
                    avail_h = max(240.0, vp.work_size.y - 2.0 * margin)
                    title_h = 40.0
                    composite_wh = comp_w / comp_h

                    if avail_w / composite_wh + title_h <= avail_h:
                        w = avail_w
                        h = avail_w / composite_wh + title_h
                    else:
                        h = avail_h
                        w = (avail_h - title_h) * composite_wh

                    x = sidebar_w + margin + (avail_w - w) * 0.5
                    y = margin + (avail_h - h) * 0.5
                    # Cond_.always overrides whatever size Newton or imgui.ini gave the window.
                    _imgui.set_next_window_pos(_imgui.ImVec2(float(x), float(y)), _imgui.Cond_.always)
                    _imgui.set_next_window_size(_imgui.ImVec2(float(w), float(h)), _imgui.Cond_.always)
                    # Force the window uncollapsed — imgui.ini may have saved a collapsed state.
                    _imgui.set_next_window_collapsed(False, _imgui.Cond_.always)
                    _viewer_ref._streaming_panel_needs_sizing = False
            return _orig_draw(self_logger)

        image_logger.draw = types.MethodType(_draw_large, image_logger)

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
        2. **Live Plots** (closed) — injected when :meth:`~NewtonVisualizer.add_live_plots`
           is called.
        3. **Visualization Markers** (open) — Newton's debug overlays, renamed.
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
                imgui.text(f"Physics: {getattr(viewer, '_backend_display', 'Unknown')}")
                if viewer.model is not None:
                    axis_names = ["X", "Y", "Z"]
                    imgui.text(f"Up Axis: {axis_names[viewer.model.up_axis]}")
                    gravity = viewer.model.gravity.numpy()[0]
                    imgui.text(f"Gravity: ({gravity[0]:.2f}, {gravity[1]:.2f}, {gravity[2]:.2f})")
                imgui.separator()
                for callback in _g._ui_callbacks.get("side", []):
                    callback(imgui)

            # --- Streaming View ---------------------------------------------
            viewer._draw_streaming_view_controls()

            # --- Live Plots -------------------------------------------------
            live_plots_cb = getattr(viewer, "_live_plots_callback", None)
            if live_plots_cb is not None:
                live_plots_cb(imgui)

            # --- Visualization Markers (GL only; RTX log_mesh crashes on empty USD path) --
            if isinstance(viewer, NewtonViewerGL) and viewer.model is not None and not viewer._fullscreen_image_active:
                imgui.set_next_item_open(False, imgui.Cond_.appearing)
                if imgui.collapsing_header("Visualization Markers"):
                    imgui.separator()
                    renderer = getattr(viewer, "renderer", None)
                    _c, viewer.show_joints = imgui.checkbox("Show Joints", viewer.show_joints)
                    if viewer.show_joints and renderer is not None and hasattr(renderer, "joint_scale"):
                        _, renderer.joint_scale = imgui.slider_float("Joint Scale", renderer.joint_scale, 0.25, 5.0)
                    _contacts_available = viewer._contacts_available
                    viewer.show_contacts = _imgui_optional_checkbox(
                        imgui,
                        "Show Contacts",
                        viewer.show_contacts,
                        _contacts_available,
                        "No contact sensors in this environment",
                    )
                    if viewer.show_contacts and _contacts_available and renderer is not None:
                        if hasattr(renderer, "arrow_length_scale"):
                            _, renderer.arrow_length_scale = imgui.slider_float(
                                "Contact Length", renderer.arrow_length_scale, 0.25, 5.0
                            )
                        if hasattr(renderer, "arrow_scale"):
                            _, renderer.arrow_scale = imgui.slider_float(
                                "Contact Width", renderer.arrow_scale, 0.25, 5.0
                            )
                    _model = viewer.model
                    _has_particles = _model is not None and int(getattr(_model, "particle_count", 0)) > 0
                    _has_springs = _model is not None and int(getattr(_model, "spring_count", 0)) > 0
                    _has_cloth = _model is not None and int(getattr(_model, "tri_count", 0)) > 0
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
                    if viewer.show_com and renderer is not None and hasattr(renderer, "com_scale"):
                        _, renderer.com_scale = imgui.slider_float("COM Scale", renderer.com_scale, 0.25, 5.0)
                    viewer.show_triangles = _imgui_optional_checkbox(
                        imgui,
                        "Show Cloth",
                        viewer.show_triangles,
                        _has_cloth,
                        "No cloth/triangle meshes in this environment",
                    )
                    _c, viewer.show_collision = imgui.checkbox("Show Collision", viewer.show_collision)
                    if renderer is not None and hasattr(renderer, "draw_edges"):
                        _c, renderer.draw_edges = imgui.checkbox("Show Edges", renderer.draw_edges)
                    sdf_margin_mode = getattr(viewer, "sdf_margin_mode", None)
                    SDFMarginMode = getattr(type(viewer), "SDFMarginMode", None)
                    if sdf_margin_mode is not None and SDFMarginMode is not None:
                        _sdf_labels = ["Off", "Margin", "Margin + Gap"]
                        _, new_sdf_idx = imgui.combo("Gap + Margin", int(sdf_margin_mode), _sdf_labels)
                        viewer.sdf_margin_mode = SDFMarginMode(new_sdf_idx)
                        if viewer.sdf_margin_mode != SDFMarginMode.OFF and renderer is not None:
                            _, renderer.wireframe_line_width = imgui.slider_float(
                                "Wireframe Width (px)", renderer.wireframe_line_width, 0.5, 5.0
                            )
                    _c, viewer.show_visual = imgui.checkbox("Show Visual", viewer.show_visual)
                    _c, viewer.show_inertia_boxes = imgui.checkbox("Show Inertia Boxes", viewer.show_inertia_boxes)
                    from isaaclab.sim import SimulationContext

                    sim = SimulationContext.instance()
                    marker_groups = () if sim is None else sim.vis_marker_registry.get_groups().values()
                    for marker in marker_groups:
                        name = marker.cfg.prim_path.rsplit("/", 1)[-1].replace("_", " ")
                        changed, visible = imgui.checkbox(f"Show {name}##{marker.group_id}", marker.is_visible())
                        if changed:
                            marker.set_visibility(visible)

            # --- Rendering Options ------------------------------------------
            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Rendering Options"):
                imgui.separator()
                _c, viewer.vsync = imgui.checkbox("VSync", viewer.vsync)
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
                if (
                    isinstance(viewer, NewtonViewerGL)
                    and viewer.picking_enabled
                    and not viewer._fullscreen_image_active
                ):
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

        # Pause/Resume Rendering is not exposed on RTX: stopping end_frame() would
        # freeze the imgui, and OVRTX naturally converges when the scene is paused.
        # ``self._paused`` (not a separately tracked flag) is the single source of truth here
        # because the Newton viewer's own Space key handler toggles it directly, bypassing this
        # button; mirroring a separate flag would let the button state drift out of sync with Space.
        if not isinstance(self, NewtonViewerRTX):
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

    def _draw_streaming_view_controls(self) -> None:
        """Render streaming image panel selector in the HUD sidebar.

        On the RTX backend (no ``_image_logger``), renders a small disabled note
        instead of the image-panel combo.
        """
        image_logger = getattr(self, "_image_logger", None)
        if image_logger is None:
            # RTX backend: no floating image panel (ViewerRTX has no image_logger).
            # Streaming composite is pushed to external sinks (Rerun, Viser).
            # Show nothing here — the streaming is silently active.
            return

        if not image_logger._images:
            return

        imgui = self.ui.imgui
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if not imgui.collapsing_header("Streaming View"):
            return

        names = list(image_logger._images.keys())
        # Display "Open" as the action label regardless of the underlying image key.
        display_items = ["Hide"] + ["Open" for _ in names]
        if image_logger._selected is not None and image_logger._selected in names:
            current = names.index(image_logger._selected) + 1
        else:
            current = 0

        imgui.text("Toggle")
        changed, new_idx = imgui.combo("##streaming_view", current, display_items)
        if changed:
            new_selected = None if new_idx == 0 else names[new_idx - 1]
            image_logger._selected = new_selected
            # Signal the image-logger draw hook to resize to the composite aspect ratio.
            if new_selected is not None:
                entry = image_logger._images.get(new_selected)
                if entry is not None:
                    entry.window_initialized = False
                # Set our flag so _draw_large applies correct aspect-ratio sizing.
                viewer = getattr(self, "_viewer", None) or self
                viewer._streaming_panel_needs_sizing = True

    def _coerce_color3(self, color) -> tuple[float, float, float]:
        """Normalize color values from imgui/renderer into an RGB tuple."""
        if hasattr(color, "x") and hasattr(color, "y") and hasattr(color, "z"):
            return (float(color.x), float(color.y), float(color.z))
        return (float(color[0]), float(color[1]), float(color[2]))

    def _color_edit3_compat(self, imgui, label: str, color):
        """Handle imgui.color_edit3 API differences between bindings.

        Some require vector-like objects, others require a Sequence[float].
        This method tries both approaches, caching the one that works to avoid
        repeated exceptions.

        .. note::
            This is a compatibility workaround; it can be removed once the
            imgui_bundle binding API stabilises.
        """
        color_tuple = self._coerce_color3(color)
        sequence_color = [color_tuple[0], color_tuple[1], color_tuple[2]]
        if self._color_edit3_prefers_sequence is not True:
            try:
                imvec4 = imgui.ImVec4(sequence_color[0], sequence_color[1], sequence_color[2], 1.0)
                changed, edited = imgui.color_edit3(label, imvec4)
                self._color_edit3_prefers_sequence = False
                return changed, self._coerce_color3(edited)
            except Exception:
                self._color_edit3_prefers_sequence = True

        try:
            changed, edited = imgui.color_edit3(label, sequence_color)
            return changed, self._coerce_color3(edited)
        except Exception as exc:
            logger.debug("[NewtonGLVisualizer] color_edit3 failed for '%s': %s", label, exc)
            return False, color_tuple


class NewtonViewerRTX(_NewtonViewerUIMixin, ViewerRTX):
    """Wrapper around Newton's ViewerRTX with Isaac Lab training/rendering pause controls.

    The ImGui sidebar (training pause, rendering pause, update-frequency slider,
    physics backend label) is registered via ``register_ui_callback``, which
    ``ViewerRTX`` supports with the same interface as ``ViewerGL``.

    .. note::
        ``log_scalar`` and ``log_array`` are inherited as no-ops from ``ViewerUSD``
        on ``ViewerRTX``, so live scalar/array plots will not appear in the RTX viewer.
        This is a Newton-side limitation; no changes here are needed when it is fixed.

    .. note::
        Newton's ``ViewerRTX`` requires all layers to be activated before the first
        ``end_frame()`` call. Isaac Lab visualization markers use named paths
        (``log_instances`` / ``log_lines``) rather than the Newton layer system, so
        no existing Isaac Lab use case is affected by this constraint.
    """

    def __init__(
        self,
        *args,
        metadata: dict | None = None,
        update_frequency: int = 1,
        background_color: tuple[float, float, float] | None = None,
        render_settings: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Initialize Newton RTX viewer wrapper state.

        Args:
            *args: Positional arguments forwarded to ``ViewerRTX``.
            metadata: Optional metadata shown in viewer panels.
            update_frequency: Viewer refresh cadence in simulation frames.
            background_color: Optional solid background color RGB [0, 1].
            render_settings: Extra RTX attributes to author on the render product. See
                :attr:`~isaaclab_visualizers.newton.NewtonRTXVisualizerCfg.render_settings`.
            **kwargs: Keyword arguments forwarded to ``ViewerRTX``.
        """
        # Assigned before super().__init__(): ViewerRTX reaches
        # _add_camera_lights_and_render_product() during initialization, and the override reads
        # these values. The render settings are copied so a caller cannot mutate them afterwards.
        self._background_color = (
            tuple(float(value) for value in background_color) if background_color is not None else None
        )
        self._render_settings = dict(render_settings or {})

        super().__init__(*args, **kwargs)
        self._ovstage_hierarchy_holder = None
        self._paused_training = False
        self._paused_rendering = False
        self._reset_requested = False
        self._metadata = metadata or {}
        self._update_frequency = update_frequency
        self._color_edit3_prefers_sequence: bool | None = None
        self.particle_color: tuple[float, float, float] | None = None

        from isaaclab.utils.backend_utils import FactoryBase

        backend = FactoryBase._get_backend()
        self._backend_display = _BACKEND_DISPLAY_NAMES.get(backend, backend)

        # UI patches must be deferred: ViewerRTX creates self.gui lazily in
        # _init_window() (called from _init_ovrtx() on the first end_frame()).
        # _patch_viewer_panel() sets gui._render_left_panel, which requires gui to
        # exist.  Register the training controls now (they are buffered by ViewerRTX until
        # the GUI is available); the panel patch is applied in _init_window() below.
        self.register_ui_callback(self._render_training_controls, position="side")

        # TODO: Move this into Newton's ViewerRTX, which should create its ovstage.Stage with an explicit
        # hierarchy computation model. With the default, OVStage 0.2 renders stale transforms. The model
        # is process-wide and set by the first stage, so hold one for the viewer's lifetime. Acquired
        # last so a failed construction cannot pin it.
        with contextlib.suppress(ImportError):
            from isaaclab_ov.stage import create_ovstage

            self._ovstage_hierarchy_holder = create_ovstage("isaaclab.newton_rtx_hierarchy_model")

    def close(self) -> None:
        """Close the viewer, then release the stage held for the OVStage hierarchy model."""
        try:
            super().close()
        finally:
            holder, self._ovstage_hierarchy_holder = self._ovstage_hierarchy_holder, None
            if holder is not None:
                holder.destroy()

    def log_points(self, name, points, radii=None, colors=None, hidden=False):
        """Apply the configured color to Newton's canonical particle batch."""
        if name == "/model/particles" and points is not None and self.particle_color is not None:
            colors = self.particle_color
        return super().log_points(name, points, radii, colors, hidden)

    def _add_camera_lights_and_render_product(self) -> None:
        """Author the configured RTX attributes onto the render product.

        Runs here because ``_init_ovrtx`` exports the stage right after this call and the renderer
        reads that export, so later edits are ignored.
        """
        super()._add_camera_lights_and_render_product()
        if self._background_color is None and not self._render_settings:
            return
        from pxr import Gf, Sdf

        prim = self.stage.GetPrimAtPath(self._render_product_path)
        if self._background_color is not None:
            prim.CreateAttribute("omni:rtx:background:source:type", Sdf.ValueTypeNames.Token).Set("color")
            prim.CreateAttribute("omni:rtx:background:source:color", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*self._background_color)
            )
        for name, (type_name, value) in self._render_settings.items():
            value_type = getattr(Sdf.ValueTypeNames, type_name, None)
            if value_type is None:
                raise ValueError(f"Render setting {name!r} names unknown USD type {type_name!r}.")
            prim.CreateAttribute(name, value_type).Set(value)

    _ovrtx_render_products: dict | None = None

    @property
    def _render_products(self):
        return self._ovrtx_render_products

    @_render_products.setter
    def _render_products(self, products) -> None:
        # TODO: Workaround until Newton's ViewerRTX accepts both keys. ovrtx 0.5 keys render vars by
        # prim path ("/Render/Vars/LdrColor"), but ViewerRTX looks up "LdrColor" and silently skips
        # the blit and capture otherwise, leaving the window black.
        if products:
            for product in products.values():
                for frame in product.frames:
                    if "/Render/Vars/LdrColor" in frame.render_vars:
                        frame.render_vars.setdefault("LdrColor", frame.render_vars["/Render/Vars/LdrColor"])
        self._ovrtx_render_products = products

    def get_frame(self) -> np.ndarray:
        """Return the latest OVRTX LDR framebuffer as contiguous RGB pixels."""
        # TODO: Use Newton's public RGB capture API when one becomes available.
        return np.ascontiguousarray(self._capture_screenshot_pixels()[..., :3])

    def _capture_screenshot_pixels(self) -> np.ndarray:
        """Normalize the first asynchronous result before Newton reads its color output."""
        if self._render_products is None and self._render_result is not None:
            self._render_products = self._render_result.wait().fetch()
        return super()._capture_screenshot_pixels()

    def _init_window(self) -> None:
        """Create the viewer window and immediately apply Isaac Lab UI patches."""
        super()._init_window()
        _apply_newton_icon(self._window)
        # Disable imgui's automatic ini file I/O — the file would be written to the
        # current working directory (often the repo root), polluting it with
        # session-specific UI state and causing hard-to-diagnose bugs when a stale
        # Collapsed=1 entry prevents the streaming panel from ever opening.
        with contextlib.suppress(Exception):
            from imgui_bundle import imgui as _imgui

            _imgui.get_io().ini_filename = None
        with contextlib.suppress(AttributeError):
            self._patch_scalar_plot_width()
        with contextlib.suppress(AttributeError):
            self._patch_viewer_panel()
        with contextlib.suppress(AttributeError):
            self._patch_image_logger()

    def is_training_paused(self) -> bool:
        """Return whether simulation is paused by viewer controls."""
        return self._paused_training

    def is_rendering_paused(self) -> bool:
        """Return whether rendering is paused by viewer controls."""
        return self._paused_rendering


class NewtonViewerGL(_NewtonViewerUIMixin, ViewerGL):
    """Wrapper around Newton's ViewerGL with training/rendering pause controls."""

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
        self._color_edit3_prefers_sequence: bool | None = None
        self.particle_color: tuple[float, float, float] | None = None
        self._particle_color_buffer: wp.array | None = None
        self._particle_color_buffer_count = 0
        self._particle_color_buffer_value: tuple[float, float, float] | None = None
        self._mpm_particle_flags_cache_key: tuple[int, int, int] | None = None
        self._mpm_particles_all_active = False
        self._live_plots_callback = None
        self._fullscreen_image_active = False

        from isaaclab.utils.backend_utils import FactoryBase

        backend = FactoryBase._get_backend()
        self._backend_display = _BACKEND_DISPLAY_NAMES.get(backend, backend)

        if self.gui is not None:
            # Keep camera keys active while ImGui captures mouse input for its controls.
            self.gui.is_capturing = self.gui.should_ignore_keyboard_input

        with contextlib.suppress(AttributeError):
            self._patch_scalar_plot_width()
            self._patch_viewer_panel()
            self._patch_image_logger()

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
        try:
            super().end_frame()
        finally:
            self._fullscreen_image_active = False
        if self._close_requested:
            self.renderer.close()

    def log_image(self, name: str, image: wp.array | np.ndarray, *, fullscreen: bool = False) -> None:
        """Track whether the current frame replaces the perspective scene with an image."""
        super().log_image(name, image, fullscreen=fullscreen)
        if fullscreen:
            self._fullscreen_image_active = True

    def on_mouse_press(self, x, y, button, modifiers):
        """Image coordinates must not pick objects in the hidden perspective scene."""
        if not self._fullscreen_image_active:
            super().on_mouse_press(x, y, button, modifiers)

    def _render_ui(self):
        """Render the Newton viewer UI."""
        return super()._render_ui()

    def _particle_color_array(self, count: int) -> wp.array:
        """Return a cached Warp color array for Newton's particle point batch."""
        color = self._coerce_color3(self.particle_color)
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
        if (
            obj is None
            or count > capacity
            or self._particle_color_buffer_value != self._coerce_color3(self.particle_color)
        ):
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

    def _prime_image_logger_window_layout(self) -> None:
        """Snap the streaming image panel to a good default position when first shown.

        Newton's ImageLogger opens new windows at a fixed per-tile size; this
        primes the position and size once per selection so the panel fills the
        available viewer space.
        """
        image_logger = self._image_logger
        if image_logger is None:
            return
        selected = image_logger._selected
        if selected is None:
            return
        entry = image_logger._images.get(selected)
        if entry is None or entry.window_initialized:
            return

        imgui = self.ui.imgui
        viewport = imgui.get_main_viewport()
        sidebar_width = float(image_logger._sidebar_width_px)
        margin = 20.0
        available_w = max(320.0, viewport.work_size.x - sidebar_width - 2.0 * margin)
        available_h = max(240.0, viewport.work_size.y - 2.0 * margin)

        n_tiles = max(1, int(entry.n))
        tile_aspect = float(entry.tile_aspect)
        cols = max(1, math.ceil(math.sqrt(n_tiles)))
        rows = math.ceil(n_tiles / cols)
        grid_aspect = (rows * tile_aspect) / cols
        title_and_padding_h = 40.0

        window_w = available_w
        window_h = min(available_h, max(240.0, window_w * grid_aspect + title_and_padding_h))
        pos_x = sidebar_width + margin
        pos_y = margin

        imgui.set_next_window_pos(imgui.ImVec2(float(pos_x), float(pos_y)), imgui.Cond_.always)
        imgui.set_next_window_size(imgui.ImVec2(float(window_w), float(window_h)), imgui.Cond_.always)
        entry.window_initialized = True


# ---------------------------------------------------------------------------
# Shared base visualizer
# ---------------------------------------------------------------------------


class NewtonVisualizer(BaseVisualizer):
    """Internal base class for Newton visualizer backends.

    Implements the shared ``initialize / step / close`` lifecycle and all
    contact-sensor helpers. Subclasses override the hook methods that differ
    between GL and RTX:

    - :meth:`_create_viewer` — instantiate the correct Newton viewer class.
    - :meth:`_apply_viewer_post_init` — apply backend-specific post-init settings.
    - :meth:`_apply_camera_pose` — set camera position with the backend's API.
    - :meth:`_apply_camera_focal_length` — set or defer FOV.
    - :meth:`_pump_paused` — keep the event loop alive while simulation is paused.
    - :meth:`_pre_step` — per-frame hook before the render block (e.g. deferred FOV).
    - :meth:`render_rgb_array` — capture and return the current frame.
    - :meth:`_log_streaming_image` — push the composited streaming frame into the viewer image panel.
    - :meth:`_uses_streaming_view` — whether the streaming view is active.

    Do not instantiate this class directly; use :class:`NewtonGLVisualizer` or
    :class:`NewtonRTXVisualizer`.
    """

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
            self._viewer: NewtonViewerGL | NewtonViewerRTX | None = None
            self._retained_picking = None

        def bind(self, viewer: NewtonViewerGL | NewtonViewerRTX) -> None:
            """Bind picking to the current viewer model."""
            self._viewer = viewer
            self._retained_picking = None

        def apply(self, state: State) -> None:
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

            picking = getattr(viewer, "picking", None)
            if picking is not None:
                viewer.picking_enabled = False
                picking.release()

            self._retained_picking = picking
            self._viewer = None

    def __init__(self, cfg: NewtonVisualizerCfg):
        """Initialize shared Newton visualizer state.

        Args:
            cfg: Newton visualizer configuration.
        """
        super().__init__(cfg)
        self.cfg: NewtonVisualizerCfg = cfg
        self._viewer: NewtonViewerGL | NewtonViewerRTX | None = None
        self._sim_time = 0.0
        self._step_counter = 0
        self._runtime_headless: bool = False
        self.backend = None
        self._last_camera_pose: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
        self._headless_no_viewer = False
        self._resolved_visible_env_ids: list[int] | None = None
        self._camera_sensor = None
        self._camera_sensor_indices: list[int] = []
        self._camera_env_indices: list[int] = []
        self._camera_is_owned = False
        self._generated_camera_prim_paths: list[str] = []
        self._viewer_picking_binding = self._ViewerPickingBinding()
        self._picking_enabled = False
        self._streaming_camera_key: tuple | None = None
        self._live_plots_manager_visible: dict[str, bool] = {}
        self._last_streaming_composite: np.ndarray | None = None
        self._composite_step: int = -1
        self._scene_cameras: dict = {}
        self._scene_camera_names: list[str] = []
        self._active_camera_idx: int = 0
        self._pending_mesh_submissions: dict[str, _MeshSubmission] = {}

    def initialize(self, scene_data_provider: SceneDataProvider) -> None:
        """Initialize viewer resources and bind scene data provider."""
        if self._is_initialized:
            logger.debug("[%s] initialize() called while already initialized.", type(self).__name__)
            return

        scene_data_provider = self._set_scene_data_provider(scene_data_provider)
        if isinstance(self, NewtonRTXVisualizer) and self.physics_backend in ("physx", "isaacsim_physx"):
            raise RuntimeError(
                f"[{type(self).__name__}] Newton RTX (OVRTX) cannot be used with physics backend"
                f" {self.physics_backend!r}. It is a kitless renderer and cannot run in the same process as Kit"
                " (isaacsim_physx). Use `presets=newton_mjwarp,ovrtx` or `presets=ovphysx,ovrtx` with"
                " `--viz newton_rtx`, or switch to `--viz newton_gl`, `--viz viser`, `--viz rerun`, or"
                " `--viz kit` with a Kit-compatible physics backend."
            )
        newton_backend_active = self.physics_backend == "newton"
        sim = SimulationContext.instance()
        physics_manager = sim.physics_manager
        picking_supported = newton_backend_active and bool(
            getattr(physics_manager, "_supports_rigid_body_force_input", False)
        )
        num_envs = scene_data_provider.num_envs
        metadata = {"num_envs": num_envs}
        self._env_ids = self._compute_visualized_env_ids()
        self._resolved_visible_env_ids = resolve_visible_env_indices(self._env_ids, self.cfg.max_visible_envs, num_envs)
        self.newton_cfg = NewtonBackendCfg(physics_cfg=sim.cfg.physics, device=sim.device)
        self.backend = sim.get_or_create_backend(self.newton_cfg)
        self._transform_mapping = scene_data_provider.create_mapping(list(self.backend.model.body_label))

        runtime_headless = self.cfg.headless or (
            sys.platform not in ("win32", "darwin") and not os.environ.get("DISPLAY")
        )
        if runtime_headless and not self.cfg.headless:
            print(
                "[WARNING] [NewtonVisualizer] No display found (DISPLAY is unset); the Newton viewer runs"
                " headless via EGL and no window will open. Run from a session with a display (or set"
                " DISPLAY, e.g. 'export DISPLAY=:0') to see the viewer."
            )
        self._runtime_headless = runtime_headless
        if runtime_headless:
            import pyglet

            pyglet.options["headless"] = True

        self._picking_enabled = self.cfg.enable_picking and picking_supported and not runtime_headless
        self._viewer = self._create_viewer(runtime_headless, metadata)

        if self._viewer is not None:
            self._viewer.set_model(self.backend.model)
            if self._picking_enabled:
                self._viewer.wind = None
            self._viewer.set_visible_worlds(self._resolved_visible_env_ids)
            self._viewer.set_world_offsets(self.cfg.world_spacing)
            self._apply_camera_focal_length()
            initial_pose = self._resolve_initial_camera_pose()
            self._apply_camera_pose(initial_pose)
            self._viewer._paused = False
            self._apply_model_visualization_options()
            self._viewer.picking_enabled = self._picking_enabled
            self._apply_viewer_post_init()

        self._setup_streaming_view(num_envs)
        num_visualized_envs = (
            len(self._resolved_visible_env_ids) if self._resolved_visible_env_ids is not None else num_envs
        )
        try:
            current_eye = tuple(float(x) for x in self._viewer.camera.pos) if self._viewer is not None else self.cfg.eye
        except AttributeError:
            current_eye = self.cfg.eye
        self._log_initialization_table(
            logger=logger,
            title=f"{type(self).__name__} Configuration",
            rows=[
                ("eye", current_eye),
                ("lookat", self._last_camera_pose[1] if self._last_camera_pose else self.cfg.lookat),
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
        if self._viewer is not None and self._picking_enabled:
            self._viewer_picking_binding.bind(self._viewer)
            NewtonManager.register_state_force_callback(self._viewer_picking_binding.apply)
        if self._viewer is not None and self.cfg.enable_picking and not picking_supported:
            logger.info(
                "[NewtonVisualizer] Object dragging is disabled because the active physics solver does not support"
                " rigid-body force input."
            )
        self._is_initialized = True
        if self._viewer is not None:
            contact_sensors = self._scene_data_provider.get_contact_sensors() if self._scene_data_provider else {}
            self._viewer._contacts_available = newton_backend_active or bool(contact_sensors)

    def _apply_model_visualization_options(self) -> None:
        """Apply configured options reset by Newton model changes."""
        if self._viewer is None:
            return
        self._viewer.show_joints = self.cfg.show_joints
        self._viewer.show_contacts = self.cfg.show_contacts
        self._viewer.show_collision = self.cfg.show_collision
        self._viewer.show_springs = self.cfg.show_springs
        self._viewer.show_inertia_boxes = self.cfg.show_inertia_boxes
        self._viewer.show_com = self.cfg.show_com
        self._viewer.show_particles = self.cfg.show_particles

    def step(self, dt: float) -> None:
        """Advance visualization by one simulation step."""
        if not self._is_initialized or self._is_closed:
            return
        self._sim_time += dt
        self._step_counter += 1
        if self._runtime_headless or self._viewer is None:
            return
        if self._step_counter % self._viewer._update_frequency != 0:
            return
        self._pre_step()
        num_envs = self.backend.model.num_envs
        try:
            if not self._viewer.is_paused():
                backend, provider = self.backend, self._scene_data_provider
                poses = SceneDataFormat.Transform()
                if provider.get_transforms(poses, mapping=self._transform_mapping, count=backend.model.body_count):
                    backend.state_0.body_q = poses.transforms
                if backend.geometry_offsets:
                    provider.get_geometry_points(output=backend.state_0.particle_q, offsets=backend.geometry_offsets)
                self._viewer.begin_frame(self._sim_time)
                try:
                    state = backend.state_0
                    if state.body_q is not None and state.body_q.shape[0] == 0:
                        self._log_pending_meshes()
                        return
                    self._viewer.log_state(state)
                    contacts = NewtonManager.get_contacts()
                    if contacts is not None:
                        self._viewer.log_contacts(contacts, state)
                    else:
                        self._log_scene_contact_sensor_arrows(num_envs)
                    self._render_markers(num_envs)
                    self._log_streaming_image()
                    self._render_live_plots()
                    self._log_pending_meshes()
                finally:
                    self._viewer.end_frame()
                    if not self._viewer.is_running():
                        self._viewer_picking_binding.deactivate()
            else:
                self._pump_paused()
                if not self._viewer.is_running():
                    self._viewer_picking_binding.deactivate()
        except Exception:
            logger.exception("[%s] Viewer update failed.", type(self).__name__)
            if getattr(self, "_disable_viewer_on_step_exception", False):
                logger.error(
                    "[%s] Permanently disabling viewer after unrecoverable initialization failure.",
                    type(self).__name__,
                )
                try:
                    self._release_viewer()
                except Exception:
                    logger.exception("[%s] Viewer teardown failed.", type(self).__name__)

    def is_reset_requested(self) -> bool:
        """Return whether an episode reset was requested via the viewer UI."""
        return self._viewer is not None and self._viewer.is_reset_requested()

    def consume_reset_request(self) -> bool:
        """Return whether an episode reset was requested and clear the flag."""
        return self._viewer is not None and self._viewer.consume_reset_request()

    def reset(self, soft: bool = False) -> None:
        """Rebind viewer resources after a hard Newton model reset."""
        if soft or not self._is_initialized or self._is_closed:
            return
        sim = SimulationContext.instance()
        backend = sim.get_or_create_backend(self.newton_cfg)
        if backend is self.backend:
            return
        self.backend = backend
        self._transform_mapping = self._scene_data_provider.create_mapping(list(backend.model.body_label))
        if self._viewer is not None:
            self._viewer.set_model(self.backend.model)
            if self._picking_enabled:
                self._viewer.wind = None
            self._viewer._register_isaaclab_ui_callbacks()
            self._viewer.set_visible_worlds(self._resolved_visible_env_ids)
            self._viewer.set_world_offsets(self.cfg.world_spacing)
            self._apply_model_visualization_options()
            self._viewer.picking_enabled = self._picking_enabled
            if self._picking_enabled:
                self._viewer_picking_binding.bind(self._viewer)

    def _release_viewer(self) -> None:
        """Release the viewer this visualizer owns and drop the reference to it."""
        viewer = self._viewer
        if viewer is None:
            return
        try:
            if self._picking_enabled:
                self._viewer_picking_binding.deactivate()
            if isinstance(viewer, NewtonViewerRTX):
                viewer.close()
        finally:
            self._viewer = None

    def close(self) -> None:
        """Release viewer resources."""
        if self._is_closed:
            return
        try:
            self._release_viewer()
        finally:
            self._pending_mesh_submissions.clear()
            if self._camera_sensor is not None and self._camera_is_owned:
                evict_visualizer_camera(self._streaming_camera_key)
                remove_generated_prims(self._generated_camera_prim_paths)
            self._camera_sensor = None
            self.backend = self._scene_data_provider = self._transform_mapping = None
            self._is_closed = True

    def is_running(self) -> bool:
        """Return whether the visualizer should continue stepping."""
        if not self._is_initialized or self._is_closed:
            return False
        if self._headless_no_viewer and self._viewer is None:
            return True
        if self._viewer is None:
            return False
        return self._viewer.is_running()

    def supports_markers(self) -> bool:
        return bool(self.cfg.enable_markers)

    def supports_live_plots(self) -> bool:
        return False

    def is_training_paused(self) -> bool:
        return bool(self._is_initialized and self._viewer is not None and self._viewer.is_training_paused())

    def is_rendering_paused(self) -> bool:
        return bool(self._is_initialized and self._viewer is not None and self._viewer.is_rendering_paused())

    def is_key_down(self, key: str) -> bool:
        if not self._is_initialized or self._viewer is None:
            return False
        return bool(self._viewer.is_key_down(key))

    def set_camera_view(self, eye, target) -> None:
        eye_t = tuple(float(v) for v in eye)
        target_t = tuple(float(v) for v in target)
        self.cfg.eye = eye_t
        self.cfg.lookat = target_t
        self._apply_camera_pose((eye_t, target_t))

    def log_mesh(
        self,
        name,
        points,
        indices,
        normals=None,
        uvs=None,
        texture=None,
        hidden=False,
        backface_culling=True,
        color=None,
        roughness=None,
        metallic=None,
        dynamic=False,
        opacity=None,
    ) -> None:
        if not self._is_initialized or self._viewer is None:
            raise RuntimeError("Newton visualizer must be initialized before logging meshes.")
        self._pending_mesh_submissions[name] = _MeshSubmission(
            name,
            points,
            indices,
            normals,
            uvs,
            texture,
            hidden,
            backface_culling,
            color,
            roughness,
            metallic,
            dynamic,
            opacity,
        )

    def _log_pending_meshes(self) -> None:
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

    def _create_viewer(self, runtime_headless: bool, metadata: dict):
        raise NotImplementedError

    def _apply_viewer_post_init(self) -> None:
        pass

    def _apply_camera_pose(self, pose) -> None:
        raise NotImplementedError

    def _apply_camera_focal_length(self) -> None:
        raise NotImplementedError

    def _pump_paused(self) -> None:
        self._viewer.begin_frame(self._sim_time)
        try:
            self._log_pending_meshes()
        finally:
            self._viewer.end_frame()

    def _pre_step(self) -> None:
        pass

    def render_rgb_array(self):
        raise NotImplementedError

    def _render_markers(self, num_envs: int) -> None:
        if self.cfg.enable_markers:
            render_newton_visualization_markers(
                self._viewer,
                self._resolved_visible_env_ids,
                num_envs=num_envs,
                sanitize_group_ids=isinstance(self._viewer, NewtonViewerRTX),
            )

    def _render_headless_frame(self) -> None:
        if not self._runtime_headless or self._viewer.is_paused():
            return
        backend, provider = self.backend, self._scene_data_provider
        poses = SceneDataFormat.Transform()
        if provider.get_transforms(poses, mapping=self._transform_mapping, count=backend.model.body_count):
            backend.state_0.body_q = poses.transforms
        if backend.geometry_offsets:
            provider.get_geometry_points(output=backend.state_0.particle_q, offsets=backend.geometry_offsets)
        self._pre_step()
        self._viewer.begin_frame(self._sim_time)
        try:
            self._viewer.log_state(backend.state_0)
            self._render_markers(backend.model.num_envs)
            self._log_pending_meshes()
        finally:
            self._viewer.end_frame()

    def _log_streaming_image(self) -> None:
        pass

    def _uses_streaming_view(self) -> bool:
        return bool(self.cfg.streaming_view)

    def render_tiled_rgb_array(self):
        return self._build_streaming_composite()

    def _build_streaming_composite(self):
        if self._camera_sensor is None:
            return self._last_streaming_composite
        if self._composite_step == self._step_counter:
            return self._last_streaming_composite
        if self._camera_is_owned:
            self._update_owned_camera_poses()
            self._camera_sensor.update(dt=0.0, force_recompute=True)
        available = frozenset(self._camera_sensor.data.output.keys())
        gt_types = []
        for gt in self.cfg.streaming_gt_types:
            if gt not in SUPPORTED_GT_TYPES:
                continue
            try:
                sensor_key_for_gt_type(gt, available)
                gt_types.append(gt)
            except (ValueError, KeyError):
                pass
        if not gt_types:
            try:
                sensor_key_for_gt_type("rgb", available)
                gt_types = ["rgb"]
            except (ValueError, KeyError):
                return None
        frames = []
        for env_idx in self._camera_sensor_indices:
            for gt in gt_types:
                key = sensor_key_for_gt_type(gt, available)
                raw = camera_gt_batch(self._camera_sensor, [env_idx], key)[0]
                frames.append(
                    CameraFrameColorizer.colorize(
                        raw, gt, depth_min=self.cfg.streaming_depth_min, depth_max=self.cfg.streaming_depth_max
                    )
                )
        target_aspect = self.cfg.window_width / self.cfg.window_height if self.cfg.window_height > 0 else 1.0
        composite = compose_streaming_grid(
            frames, len(self._camera_sensor_indices), len(gt_types), target_aspect=target_aspect
        )
        self._last_streaming_composite = composite
        self._composite_step = self._step_counter
        return composite

    def _resolve_initial_camera_pose(self):
        return self._resolve_cfg_camera_pose(type(self).__name__)

    def _setup_streaming_view(self, num_envs: int) -> None:
        if not self._uses_streaming_view():
            return
        gt_types = list(self.cfg.streaming_gt_types)
        for gt in gt_types:
            if gt not in SUPPORTED_GT_TYPES:
                raise ValueError(f"Unsupported streaming GT type: {gt!r}")
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
            return
        if self.cfg.streaming_cam_target_prim_path is None:
            cameras = self._scene_data_provider.get_camera_sensors()
            if cameras:
                self._camera_sensor = next(iter(cameras.values()))
                self._camera_sensor_indices = env_ids
            return
        count = max(1, len(env_ids))
        tile_w, tile_h = compute_tile_resolution(
            self.cfg.window_width, self.cfg.window_height, count, n_gt=len(gt_types)
        )
        result = create_visualizer_camera(
            num_envs=num_envs,
            width=tile_w,
            height=tile_h,
            renderer_cfg=self.cfg.streaming_cam_renderer_cfg,
            data_types=sensor_keys_for_gt_types(gt_types),
            target_prim_path=self.cfg.streaming_cam_target_prim_path,
            eye=self.cfg.streaming_cam_eye,
            streaming_envs=tuple(int(i) for i in env_ids),
        )
        self._camera_sensor, self._generated_camera_prim_paths, self._camera_is_owned, self._streaming_camera_key = (
            result
        )
        self._camera_sensor_indices = env_ids
        self._update_owned_camera_poses()

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

    def _log_scene_contact_sensor_arrows(self, num_envs: int) -> None:
        if self._viewer is None:
            return
        if not self._viewer.show_contacts:
            self._viewer.log_arrows(CONTACT_ARROW_PATH, None, None, None)
            return
        contact_sensors = self._scene_data_provider.get_contact_sensors() if self._scene_data_provider else {}
        starts, ends = [], []
        for sensor in contact_sensors.values():
            sensor_starts, sensor_ends = self._contact_sensor_arrow_tensors(sensor, num_envs)
            if sensor_starts is not None and sensor_ends is not None:
                starts.append(sensor_starts)
                ends.append(sensor_ends)
        if not starts:
            self._viewer.log_arrows(CONTACT_ARROW_PATH, None, None, None)
            return
        starts_t = torch.cat(starts).detach().to(dtype=torch.float32, device="cpu").contiguous()
        ends_t = torch.cat(ends).detach().to(dtype=torch.float32, device="cpu").contiguous()
        self._viewer.log_arrows(
            CONTACT_ARROW_PATH,
            wp.array(starts_t.numpy(), dtype=wp.vec3, device=self._viewer.device),
            wp.array(ends_t.numpy(), dtype=wp.vec3, device=self._viewer.device),
            CONTACT_ARROW_COLOR,
        )

    def _contact_sensor_arrow_tensors(self, sensor, num_envs):
        try:
            data = sensor.data
            proxy = data.net_normal_forces_w
            net_forces = proxy.torch if proxy is not None else None
        except (AttributeError, NotImplementedError, RuntimeError):
            return None, None
        if net_forces is None or net_forces.numel() == 0:
            return None, None
        net_forces = self._filter_visible_env_tensor(net_forces, num_envs)
        threshold = getattr(getattr(sensor, "cfg", None), "force_threshold", 0.0) or 0.0
        try:
            contact_pos = data.contact_pos_w
            force_matrix = data.normal_force_matrix_w
        except (AttributeError, NotImplementedError):
            contact_pos = force_matrix = None
        if contact_pos is not None and force_matrix is not None:
            positions = self._filter_visible_env_tensor(contact_pos.torch, num_envs)
            forces = self._filter_visible_env_tensor(force_matrix.torch, num_envs)
            active = (torch.linalg.norm(forces, dim=-1) > threshold) & torch.isfinite(positions).all(dim=-1)
            if torch.any(active):
                starts = positions[active]
                return starts, starts + torch.nn.functional.normalize(forces[active], dim=-1) * CONTACT_ARROW_LENGTH
        origins = self._contact_sensor_origin_positions(sensor, data, net_forces)
        if origins is None:
            return None, None
        origins = self._filter_visible_env_tensor(origins, num_envs)
        active = torch.linalg.norm(net_forces, dim=-1) > threshold
        if not torch.any(active):
            return None, None
        starts = origins[active]
        return starts, starts + torch.nn.functional.normalize(net_forces[active], dim=-1) * CONTACT_ARROW_LENGTH

    def _contact_sensor_origin_positions(self, sensor, data, net_forces):
        try:
            pos_w = data.pos_w
        except (AttributeError, NotImplementedError):
            pos_w = None
        if pos_w is not None:
            return pos_w.torch
        view = getattr(sensor, "body_physx_view", None)
        if view is None:
            return None
        try:
            pose = view.get_transforms()
        except RuntimeError:
            return None
        num_envs, num_bodies = net_forces.shape[:2]
        return wp.to_torch(pose).view(num_bodies, num_envs, 7).transpose(0, 1)[..., :3]

    def _filter_visible_env_tensor(self, tensor, num_envs):
        if self._resolved_visible_env_ids is None or tensor.ndim == 0 or tensor.shape[0] != num_envs:
            return tensor
        ids = torch.as_tensor(self._resolved_visible_env_ids, dtype=torch.long, device=tensor.device)
        return tensor.index_select(0, ids)


class NewtonGLVisualizer(NewtonVisualizer):
    """Newton OpenGL rasterizer visualizer for Isaac Lab."""

    def __init__(self, cfg: NewtonGLVisualizerCfg):
        super().__init__(cfg)
        self.cfg = cfg
        self._scene_camera_map = {}
        self._streaming_camera_choices = []
        self._streaming_camera_selection = 0
        self._custom_camera = None
        self._custom_camera_indices = []

    def initialize(self, scene_data_provider: SceneDataProvider) -> None:
        super().initialize(scene_data_provider)
        if self._is_initialized:
            self._build_streaming_camera_dropdown()

    def _build_streaming_camera_dropdown(self) -> None:
        self._scene_camera_map = self._scene_data_provider.get_camera_sensors()
        choices = list(self._scene_camera_map.keys())
        current_idx = 0
        if self._camera_is_owned and self._camera_sensor is not None:
            self._custom_camera = self._camera_sensor
            self._custom_camera_indices = list(self._camera_sensor_indices or [])
            choices = ["Custom"] + choices
        else:
            for i, cam in enumerate(self._scene_camera_map.values()):
                if cam is self._camera_sensor:
                    current_idx = i
                    break
        self._streaming_camera_choices = choices
        self._streaming_camera_selection = current_idx
        if choices and self._viewer is not None:
            self._patch_streaming_camera_controls()

    def _patch_streaming_camera_controls(self) -> None:
        import types

        viewer = self._viewer
        visualizer = self

        def _patched(self_viewer):
            image_logger = getattr(self_viewer, "_image_logger", None)
            if image_logger is None or not image_logger._images:
                return
            imgui = self_viewer.ui.imgui
            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if not imgui.collapsing_header("Streaming View"):
                return
            names = list(image_logger._images)
            current = names.index(image_logger._selected) + 1 if image_logger._selected in names else 0
            changed, new_idx = imgui.combo("##streaming_view", current, ["Hide"] + ["Open" for _ in names])
            if changed:
                image_logger._selected = None if new_idx == 0 else names[new_idx - 1]
                if image_logger._selected is not None:
                    self_viewer._streaming_panel_needs_sizing = True
            if visualizer._streaming_camera_choices:
                imgui.separator()
                imgui.text("Source Camera")
                changed, index = imgui.combo(
                    "##streaming_cam_src",
                    visualizer._streaming_camera_selection,
                    visualizer._streaming_camera_choices,
                )
                if changed:
                    visualizer._switch_streaming_camera(index)

        viewer._draw_streaming_view_controls = types.MethodType(_patched, viewer)

    def _switch_streaming_camera(self, new_idx: int) -> None:
        if new_idx == self._streaming_camera_selection:
            return
        self._streaming_camera_selection = new_idx
        choice = self._streaming_camera_choices[new_idx]
        if choice == "Custom" and self._custom_camera is not None:
            self._camera_sensor = self._custom_camera
            self._camera_sensor_indices = list(self._custom_camera_indices)
            self._camera_is_owned = True
        elif choice in self._scene_camera_map:
            cam = self._scene_camera_map[choice]
            self._camera_sensor = cam
            count = getattr(getattr(cam, "_view", None), "count", None) or len(self._camera_sensor_indices)
            self._camera_sensor_indices = resolve_streaming_envs(
                count,
                self.cfg.streaming_envs,
                max_tiles=VISUALIZER_TILED_CAMERA_MAX_TILES,
                sample_from=self._resolved_visible_env_ids,
            )
            self._camera_is_owned = False
        self._last_streaming_composite = None
        self._composite_step = -1

    def _create_viewer(self, runtime_headless, metadata):
        if not runtime_headless:
            write_desktop_entry("isaaclab-newton-gl-viewer", "Newton", "Newton", _NEWTON_ICON_DIR / "icon_64.png")
        return NewtonViewerGL(
            width=self.cfg.window_width,
            height=self.cfg.window_height,
            headless=runtime_headless,
            metadata=metadata,
            update_frequency=self.cfg.update_frequency,
        )

    def register_ui_callback(self, callback: Callable[[Any], None], position: str = "side") -> None:
        if self._viewer is not None:
            self._viewer.register_ui_callback(callback, position=position)

    def request_close(self) -> None:
        if self._viewer is not None:
            self._viewer.request_close()

    def supports_live_plots(self) -> bool:
        return True

    def add_live_plots(self, managers, scalars=None, term_names=None, env_idx=0) -> None:
        super().add_live_plots(managers, scalars=scalars, term_names=term_names, env_idx=env_idx)
        if self._live_plot_sources and self._viewer is not None:
            self._live_plots_manager_visible = {source.manager_name: True for source in self._live_plot_sources}
            self._viewer._live_plots_callback = self._live_plots_panel_imgui

    def _live_plots_panel_imgui(self, imgui) -> None:
        if not self._live_plot_sources or self._viewer is None:
            return
        viewer = self._viewer
        scalar_buffers = getattr(viewer, "_scalar_buffers", None)
        array_buffers = getattr(viewer, "_array_buffers", None)
        if not scalar_buffers and not array_buffers:
            return
        implot = getattr(viewer, "_implot", None)
        if not hasattr(viewer, "_scalar_arrays"):
            viewer._scalar_arrays = {}
        groups = {}
        for name in scalar_buffers or {}:
            groups.setdefault(_newton_scalar_base_name(name), []).append(name)
        imgui.set_next_item_open(False, imgui.Cond_.appearing)
        if not imgui.collapsing_header("Live Plots"):
            return
        for base, names in groups.items():
            if not imgui.collapsing_header(base.rsplit("/", 1)[-1]):
                continue
            if implot is not None and implot.begin_plot(f"##{base}", imgui.ImVec2(-1, 180)):
                auto = implot.AxisFlags_.auto_fit.value
                implot.setup_axes("", "", auto, auto)
                implot.setup_finish()
                for name in names:
                    history = scalar_buffers[name]
                    array = viewer._scalar_arrays.get(name)
                    if array is None:
                        array = np.full(getattr(viewer, "_plot_history_size", 250), np.nan, dtype=np.float32)
                        array[len(array) - len(history) :] = history
                        viewer._scalar_arrays[name] = array
                    implot.plot_line(name[len(base) :] or base, array)
                implot.end_plot()

    def _render_live_plots(self) -> None:
        if self._viewer is None or not self._live_plot_sources or self._runtime_headless:
            return
        self._live_plots_step_counter += 1
        if self._live_plots_step_counter % max(1, getattr(self.cfg, "live_plots_update_interval", 10)):
            return
        for source in self._live_plot_sources:
            if not self._live_plots_manager_visible.get(source.manager_name, True):
                continue
            for name, values in source.collect(self._live_plot_env_idx).items():
                for i, value in enumerate(values):
                    suffix = "" if len(values) == 1 else f"[{i}]"
                    self._viewer.log_scalar(f"{source.manager_name}/{name}{suffix}", value)

    def _apply_viewer_post_init(self) -> None:
        viewer = self._viewer
        viewer.up_axis = 2
        viewer.scaling = 1.0
        viewer.particle_color = self.cfg.particle_color
        viewer.renderer.draw_shadows = self.cfg.enable_shadows
        viewer.renderer.draw_wireframe = self.cfg.enable_wireframe
        if self.cfg.background_color is None:
            viewer.renderer.draw_sky = self.cfg.enable_sky
            upper, lower = self.cfg.sky_upper_color, self.cfg.sky_lower_color
        else:
            viewer.renderer.draw_sky = False
            upper = lower = self.cfg.background_color
        viewer.renderer.sky_upper = viewer._coerce_color3(upper)
        viewer.renderer.sky_lower = viewer._coerce_color3(lower)
        viewer.renderer._light_color = viewer._coerce_color3(self.cfg.light_color)

    def _apply_camera_pose(self, pose) -> None:
        if self._viewer is None:
            return
        position, target = pose
        self._viewer.camera.pos = PygletVec3(*position)
        self._viewer.camera.look_at(target)
        self._last_camera_pose = pose

    def _apply_camera_focal_length(self) -> None:
        if self._viewer is not None:
            self._viewer.camera.fov = self._focal_length_to_vertical_fov_degrees()

    def render_rgb_array(self) -> np.ndarray:
        if self._viewer is None:
            raise RuntimeError("NewtonGLVisualizer must be initialized before capturing an RGB frame.")
        self._render_headless_frame()
        return self._viewer.get_frame().numpy()

    def _log_streaming_image(self) -> None:
        if self._viewer is None or self._camera_sensor is None:
            return
        key = "Streaming View"
        image_logger = getattr(self._viewer, "_image_logger", None)
        if image_logger is None:
            return
        if key not in image_logger._images:
            self._viewer.log_image(key, wp.zeros((1, 1, 3), dtype=wp.uint8))
            image_logger._selected = None
            return
        if image_logger._selected is None:
            return
        composite = self._build_streaming_composite()
        if composite is None:
            return
        height, width = composite.shape[:2]
        if getattr(self._viewer, "_streaming_composite_w", 0) <= 1:
            self._viewer._streaming_panel_needs_sizing = True
        self._viewer._streaming_composite_h = height
        self._viewer._streaming_composite_w = width
        self._viewer.log_image(key, wp.from_torch(torch.from_numpy(composite).contiguous()))


class NewtonRTXVisualizer(NewtonVisualizer):
    """Newton OVRTX path-tracer visualizer for Isaac Lab."""

    def __init__(self, cfg: NewtonRTXVisualizerCfg):
        super().__init__(cfg)
        self.cfg = cfg
        self._rtx_fov_pending = False
        self._disable_viewer_on_step_exception = True

    def _create_viewer(self, runtime_headless, metadata):
        if not runtime_headless:
            write_desktop_entry(
                "isaaclab-newton-rtx-viewer",
                "Newton RTX Viewer",
                "Newton RTX Viewer",
                _NEWTON_ICON_DIR / "icon_64.png",
            )
        return NewtonViewerRTX(
            width=self.cfg.window_width,
            height=self.cfg.window_height,
            headless=runtime_headless,
            up_axis="Z",
            metadata=metadata,
            update_frequency=self.cfg.update_frequency,
            environment=self.cfg.rtx_environment,
            background_color=self.cfg.background_color,
            render_settings=self.cfg.render_settings,
        )

    def _apply_viewer_post_init(self) -> None:
        self._viewer.particle_color = self.cfg.particle_color

    def _apply_camera_pose(self, pose) -> None:
        if self._viewer is None:
            return
        position, target = pose
        pitch, yaw = _eye_lookat_to_pitch_yaw(position, target)
        self._viewer.set_camera(PygletVec3(*position), pitch, yaw)
        self._last_camera_pose = pose

    def _apply_camera_focal_length(self) -> None:
        self._rtx_fov_pending = True

    def _apply_rtx_fov_if_pending(self) -> None:
        if not self._rtx_fov_pending or self._viewer is None:
            return
        try:
            self._viewer.camera.fov = self._focal_length_to_vertical_fov_degrees()
            self._rtx_fov_pending = False
        except AttributeError:
            pass

    def _pre_step(self) -> None:
        self._apply_rtx_fov_if_pending()

    def render_rgb_array(self) -> np.ndarray | None:
        if self._viewer is None:
            return None
        self._render_headless_frame()
        return self._viewer.get_frame()
