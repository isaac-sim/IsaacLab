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
from typing import TYPE_CHECKING

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

from isaaclab_newton.physics import NewtonManager
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
from isaaclab.visualizers.base_visualizer import BaseVisualizer

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
            if viewer.model is not None and not isinstance(viewer, NewtonViewerRTX):
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
                imgui.text("Right Click - Pick and drag objects")
                imgui.text("Middle Click - Orbit")
                imgui.text("Shift + Middle Click - Pan")
                imgui.text("Ctrl + Middle Click - Dolly")
                imgui.text("Scroll - Dolly")
                imgui.text("Ctrl + Scroll - FOV zoom")
                imgui.text("Space - Pause/Resume")
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
        if not isinstance(self, NewtonViewerRTX):
            rendering_label = "Resume Rendering" if self._paused_rendering else "Pause Rendering"
            if imgui.button(rendering_label):
                self._paused_rendering = not self._paused_rendering
                self._paused = self._paused_rendering

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

    def __init__(self, *args, metadata: dict | None = None, update_frequency: int = 1, **kwargs):
        """Initialize Newton RTX viewer wrapper state.

        Args:
            *args: Positional arguments forwarded to ``ViewerRTX``.
            metadata: Optional metadata shown in viewer panels.
            update_frequency: Viewer refresh cadence in simulation frames.
            **kwargs: Keyword arguments forwarded to ``ViewerRTX``.
        """
        # Patch environment so OVRTX's CRenderApiLibLoader can find libovrtx.dylib.so.
        # libovrtx-dynamic.so's built-in RPATH uses paths from the original deploy layout
        # which don't match the pip install layout. LD_LIBRARY_PATH (read by glibc at each
        # dlopen call) and OMNI_USD_PLUGINS_BASE_PATH (read by CRenderApiLibLoader) redirect
        # the search to the correct location.
        if sys.platform.startswith("linux"):
            import importlib.util as _ilu
            import pathlib as _pl

            _spec = _ilu.find_spec("ovrtx")
            if _spec is not None:
                _bin = _pl.Path(_spec.origin).parent / "bin"
                _extra = os.pathsep.join([str(_bin / "plugins" / "rtx"), str(_bin / "plugins"), str(_bin)])
                _ld = os.environ.get("LD_LIBRARY_PATH", "")
                if str(_bin / "plugins" / "rtx") not in _ld:
                    os.environ["LD_LIBRARY_PATH"] = _extra + (os.pathsep + _ld if _ld else "")
                os.environ.setdefault("OMNI_USD_PLUGINS_BASE_PATH", str(_bin))

        super().__init__(*args, **kwargs)
        self._paused_training = False
        self._paused_rendering = False
        self._reset_requested = False
        self._metadata = metadata or {}
        self._update_frequency = update_frequency
        self._color_edit3_prefers_sequence: bool | None = None

        from isaaclab.utils.backend_utils import FactoryBase

        backend = FactoryBase._get_backend()
        self._backend_display = _BACKEND_DISPLAY_NAMES.get(backend, backend)

        # UI patches must be deferred: ViewerRTX creates self.gui lazily in
        # _init_window() (called from _init_ovrtx() on the first end_frame()).
        # _patch_viewer_panel() sets gui._render_left_panel, which requires gui to
        # exist.  Register the training controls now (they are buffered by ViewerRTX until
        # the GUI is available); the panel patch is applied in _init_window() below.
        self.register_ui_callback(self._render_training_controls, position="side")

    def get_frame(self) -> np.ndarray:
        """Return the latest OVRTX LDR framebuffer as contiguous RGB pixels."""
        # TODO: Use Newton's public RGB capture API when one becomes available.
        return np.ascontiguousarray(self._capture_screenshot_pixels()[..., :3])

    def _init_window(self) -> None:
        """Create the viewer window and immediately apply Isaac Lab UI patches."""
        super()._init_window()
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
        self._paused_rendering = False
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

        from isaaclab.utils.backend_utils import FactoryBase

        backend = FactoryBase._get_backend()
        self._backend_display = _BACKEND_DISPLAY_NAMES.get(backend, backend)

        with contextlib.suppress(AttributeError):
            self._patch_scalar_plot_width()
            self._patch_viewer_panel()
            self._patch_image_logger()

        self.register_ui_callback(self._render_training_controls, position="side")

    def is_training_paused(self) -> bool:
        """Return whether simulation is paused by viewer controls."""
        return self._paused_training

    def is_rendering_paused(self) -> bool:
        """Return whether rendering is paused by viewer controls."""
        return self._paused_rendering

    def on_key_press(self, symbol, modifiers):
        """Forward key presses unless UI is currently capturing input."""
        if self.ui.is_capturing():
            return
        super().on_key_press(symbol, modifiers)

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
        self._model = None
        self._state = None
        self._update_frequency = cfg.update_frequency
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

    # ------------------------------------------------------------------
    # Shared lifecycle
    # ------------------------------------------------------------------

    def initialize(self, scene_data_provider: SceneDataProvider) -> None:
        """Initialize viewer resources and bind scene data provider.

        Args:
            scene_data_provider: Scene data provider used to fetch model/state data.
        """

        from isaaclab.sim import SimulationContext

        if self._is_initialized:
            logger.debug("[%s] initialize() called while already initialized.", type(self).__name__)
            return

        scene_data_provider = self._set_scene_data_provider(scene_data_provider)
        newton_backend_active = self.physics_backend == "newton"
        physics_manager = SimulationContext.instance().physics_manager
        picking_supported = newton_backend_active and bool(
            getattr(physics_manager, "_supports_rigid_body_force_input", False)
        )
        num_envs = scene_data_provider.num_envs
        metadata = {"num_envs": num_envs}
        self._env_ids = self._compute_visualized_env_ids()
        self._resolved_visible_env_ids = resolve_visible_env_indices(self._env_ids, self.cfg.max_visible_envs, num_envs)
        self._model = NewtonManager.get_model()
        self._state = (
            NewtonManager.get_state_0() if newton_backend_active else NewtonManager.get_state(self._scene_data_provider)
        )

        runtime_headless = self.cfg.headless or (
            sys.platform not in ("win32", "darwin") and not os.environ.get("DISPLAY")
        )
        if runtime_headless and not self.cfg.headless:
            # print() instead of logger.warning(): the kitless launch path does not
            # install a logging handler, so this user-facing notice would be swallowed.
            print(
                "[WARNING] [NewtonVisualizer] No display found (DISPLAY is unset); the Newton viewer runs"
                " headless via EGL and no window will open. Run from a session with a display (or set"
                " DISPLAY, e.g. 'export DISPLAY=:0') to see the viewer."
            )
        self._runtime_headless = runtime_headless

        # Use pyglet's EGL headless backend when requested or when no Linux X display is available.
        # NOTE: this call is only effective when ``DISPLAY`` is unset on Linux.  When a display
        # is present, ``from newton.viewer import ViewerGL, ViewerRTX`` at module-import time
        # already initialised pyglet (and resolved the ``Window`` class), so setting
        # ``pyglet.options["headless"]`` here is a no-op.  In that situation ``cfg.headless=True``
        # has no effect and a real windowed viewer is created.  To guarantee headless behaviour
        # when a display is present, unset DISPLAY before importing this module.
        if runtime_headless:
            import pyglet

            pyglet.options["headless"] = True

        self._picking_enabled = self.cfg.enable_picking and picking_supported and not runtime_headless
        self._viewer = self._create_viewer(runtime_headless, metadata)

        if self._viewer is not None:
            self._viewer.set_model(self._model)
            if self._picking_enabled:
                # Keep Newton's public force path scoped to picking for this integration.
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
        # Inform the viewer whether contact data is available so the UI can grey
        # out "Show Contacts" when neither native Newton contacts nor a ContactSensor
        # exists in the scene.
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
        """Advance visualization by one simulation step.

        Args:
            dt: Simulation time-step in seconds.
        """
        if not self._is_initialized or self._is_closed:
            return

        self._sim_time += dt
        self._step_counter += 1

        # Headless mode renders on demand via render_rgb_array(). Keep the latest
        # physics state available without paying the per-step render cost.
        if self._runtime_headless:
            self._state = NewtonManager.get_state(self._scene_data_provider)
            return

        if self._viewer is None:
            self._state = NewtonManager.get_state(self._scene_data_provider)
            return

        update_frequency = self._viewer._update_frequency if self._viewer else self._update_frequency
        if self._step_counter % update_frequency != 0:
            return

        self._pre_step()
        num_envs = NewtonManager.get_num_envs()

        try:
            if not self._viewer.is_paused():
                self._state = NewtonManager.get_state(self._scene_data_provider)
                self._viewer.begin_frame(self._sim_time)
                try:
                    if self._state is not None:
                        body_q = getattr(self._state, "body_q", None)
                        if hasattr(body_q, "shape") and body_q.shape[0] == 0:
                            return
                        self._viewer.log_state(self._state)
                        contacts = NewtonManager.get_contacts()
                        if contacts is not None:
                            self._viewer.log_contacts(contacts, self._state)
                        else:
                            self._log_scene_contact_sensor_arrows(num_envs)
                        if self.cfg.enable_markers and not isinstance(self._viewer, NewtonViewerRTX):
                            # ViewerRTX uses a USD stage whose prim paths are not set up
                            # for the debug mesh overlays that markers require; skip for RTX.
                            render_newton_visualization_markers(
                                self._viewer, self._resolved_visible_env_ids, num_envs=num_envs
                            )
                        self._log_streaming_image()
                        self._render_live_plots()
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
            # Subclasses that cannot recover from a viewer failure (e.g. RTX when OVRTX is
            # unavailable) set _disable_viewer_on_step_exception = True so the viewer is
            # permanently disabled after the first failure instead of spamming the log.
            if getattr(self, "_disable_viewer_on_step_exception", False):
                logger.error(
                    "[%s] Permanently disabling viewer after unrecoverable initialization failure.",
                    type(self).__name__,
                )
                self._viewer = None

    def is_reset_requested(self) -> bool:
        """Return whether an episode reset was requested via the viewer UI."""
        if self._viewer is not None:
            return self._viewer.is_reset_requested()
        return False

    def consume_reset_request(self) -> bool:
        """Return whether an episode reset was requested and clear the flag."""
        if self._viewer is not None:
            return self._viewer.consume_reset_request()
        return False

    def reset(self, soft: bool = False) -> None:
        """Rebind viewer resources after a hard Newton model reset."""
        if soft or not self._picking_enabled or not self._is_initialized or self._is_closed:
            return

        model = NewtonManager.get_model()
        if model is self._model:
            return
        self._model = model
        self._state = NewtonManager.get_state_0()
        if self._viewer is not None:
            self._viewer.set_model(self._model)
            if self._picking_enabled:
                self._viewer.wind = None
            self._viewer._register_isaaclab_ui_callbacks()
            self._viewer.set_visible_worlds(self._resolved_visible_env_ids)
            self._viewer.set_world_offsets(self.cfg.world_spacing)
            self._apply_model_visualization_options()
            self._viewer.picking_enabled = self._picking_enabled
            if self._picking_enabled:
                self._viewer_picking_binding.bind(self._viewer)

    def close(self) -> None:
        """Release viewer resources."""
        if self._is_closed:
            return
        if self._picking_enabled:
            # Keep the stable callback registered: captured graphs replay its
            # now-neutral device inputs without retaining the viewer.
            self._viewer_picking_binding.deactivate()
        if self._viewer is not None:
            self._viewer = None
        if self._camera_sensor is not None and self._camera_is_owned:
            evict_visualizer_camera(self._streaming_camera_key)
            remove_generated_prims(self._generated_camera_prim_paths)
        self._camera_sensor = None
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
        """Newton viewers support Isaac Lab markers through viewer-side meshes and lines."""
        return bool(self.cfg.enable_markers)

    def supports_live_plots(self) -> bool:
        """Newton RTX viewers do not provide live-plot panels; GL viewers do."""
        return False

    def is_training_paused(self) -> bool:
        """Return whether training is paused from viewer controls."""
        if not self._is_initialized or self._viewer is None:
            return False
        return self._viewer.is_training_paused()

    def is_rendering_paused(self) -> bool:
        """Return whether rendering is paused from viewer controls."""
        if not self._is_initialized or self._viewer is None:
            return False
        return self._viewer.is_rendering_paused()

    def set_camera_view(
        self, eye: tuple[float, float, float] | list[float], target: tuple[float, float, float] | list[float]
    ) -> None:
        """Set active viewer camera eye/target.

        Args:
            eye: Camera eye position.
            target: Camera look-at target.
        """
        eye_t = (float(eye[0]), float(eye[1]), float(eye[2]))
        target_t = (float(target[0]), float(target[1]), float(target[2]))
        self.cfg.eye = eye_t
        self.cfg.lookat = target_t
        self._apply_camera_pose((eye_t, target_t))

    # ------------------------------------------------------------------
    # Hook methods — override in subclasses
    # ------------------------------------------------------------------

    def _create_viewer(self, runtime_headless: bool, metadata: dict) -> NewtonViewerGL | NewtonViewerRTX | None:
        """Create and return the backend viewer instance.

        Args:
            runtime_headless: Whether to run without a display.
            metadata: Metadata dict passed to the viewer constructor.
        """
        raise NotImplementedError

    def _apply_viewer_post_init(self) -> None:
        """Apply backend-specific settings after the viewer is constructed."""

    def _apply_camera_pose(
        self,
        pose: tuple[tuple[float, float, float], tuple[float, float, float]],
    ) -> None:
        """Apply camera eye/target pose to the viewer.

        Args:
            pose: ``(eye, lookat)`` tuple.
        """
        raise NotImplementedError

    def _apply_camera_focal_length(self) -> None:
        """Apply cfg focal length to the viewer camera."""
        raise NotImplementedError

    def _pump_paused(self) -> None:
        """Keep the event loop alive while simulation is paused without advancing state."""
        raise NotImplementedError

    def _pre_step(self) -> None:
        """Per-frame hook called before the render block. No-op by default."""

    def render_rgb_array(self) -> np.ndarray | None:
        """Return the latest RGB frame as a uint8 array with shape ``(H, W, 3)``."""
        raise NotImplementedError

    def _log_streaming_image(self) -> None:
        """Push the composited streaming frame into the viewer image panel."""

    def _uses_streaming_view(self) -> bool:
        """Return whether the streaming camera view is active."""
        return bool(self.cfg.streaming_view)

    # ------------------------------------------------------------------
    # Shared internals
    # ------------------------------------------------------------------

    def _resolve_initial_camera_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Resolve initial camera pose from config or USD camera path."""
        return self._resolve_cfg_camera_pose(type(self).__name__)

    def _resolve_streaming_renderer_cfg(self):
        """Return the renderer cfg for the auto-created streaming camera.

        Uses :attr:`~isaaclab.visualizers.VisualizerCfg.streaming_cam_renderer`
        when set, otherwise falls back to :class:`~isaaclab_newton.renderers.NewtonWarpRendererCfg`.
        """
        from isaaclab_newton.renderers import NewtonWarpRendererCfg

        renderer_name = self.cfg.streaming_cam_renderer
        if renderer_name is None or renderer_name == "newton_warp":
            return NewtonWarpRendererCfg()
        if renderer_name == "ovrtx":
            from isaaclab_ov.renderers import OVRTXRendererCfg

            return OVRTXRendererCfg()
        raise ValueError(
            f"[{type(self).__name__}] streaming_cam_renderer={renderer_name!r} is not supported. "
            "Valid values for Newton visualizers: 'newton_warp', 'ovrtx', None."
        )

    def _setup_streaming_view(self, num_envs: int) -> None:
        """Resolve or create the camera sensor for the streaming view."""
        if not self._uses_streaming_view():
            return

        gt_types = list(self.cfg.streaming_gt_types)
        for gt in gt_types:
            if gt not in SUPPORTED_GT_TYPES:
                raise ValueError(
                    f"[{type(self).__name__}] streaming_gt_types contains unsupported type {gt!r}. "
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
            logger.debug(
                "[%s] streaming_sensor_prim_path uses existing camera sensor; streaming_cam_* fields are ignored.",
                type(self).__name__,
            )
            cameras = self._scene_data_provider.get_camera_sensors()
            self._camera_sensor = find_camera_by_prim_path(cameras, self.cfg.streaming_sensor_prim_path, env_ids)
            self._camera_sensor_indices = env_ids
            return

        # When streaming_cam_target_prim_path is None, try to adopt the first scene camera
        # rather than creating a new one with a hardcoded prim path.
        if self.cfg.streaming_cam_target_prim_path is None:
            cameras = self._scene_data_provider.get_camera_sensors()
            if cameras:
                first_name, first_cam = next(iter(cameras.items()))
                logger.debug(
                    "[%s] streaming_cam_target_prim_path is None; adopting scene camera %r.",
                    type(self).__name__,
                    first_name,
                )
                self._camera_sensor = first_cam
                self._camera_sensor_indices = env_ids
                return
            logger.debug(
                "[%s] streaming_cam_target_prim_path is None and no scene cameras found; "
                "streaming view will be empty. Add a TiledCamera sensor or set "
                "streaming_cam_target_prim_path to enable the streaming panel.",
                type(self).__name__,
            )
            return

        renderer_cfg = self._resolve_streaming_renderer_cfg()
        count = max(1, len(env_ids))
        tile_w, tile_h = compute_tile_resolution(
            self.cfg.window_width, self.cfg.window_height, count, n_gt=len(gt_types)
        )
        try:
            result = create_visualizer_camera(
                num_envs=num_envs,
                width=tile_w,
                height=tile_h,
                renderer_cfg=renderer_cfg,
                data_types=sensor_keys_for_gt_types(gt_types),
                target_prim_path=self.cfg.streaming_cam_target_prim_path,
                eye=self.cfg.streaming_cam_eye,
                streaming_envs=tuple(int(i) for i in env_ids),
            )
        except Exception:
            logger.warning(
                "[%s] Streaming view disabled: could not auto-create a camera sensor. "
                "Add a TiledCamera to the scene config or set streaming_sensor_prim_path "
                "to point to an existing camera.",
                type(self).__name__,
            )
            return
        self._camera_sensor, self._generated_camera_prim_paths, self._camera_is_owned, self._streaming_camera_key = (
            result
        )
        self._camera_sensor_indices = env_ids
        self._update_owned_camera_poses()

    def _update_owned_camera_poses(self) -> None:
        """Update generated camera poses from env origins or follow prims."""
        if self._camera_sensor is None or not self._camera_is_owned:
            return
        target_positions = prim_world_positions(
            self._scene_data_provider.get_usd_stage(),
            self.cfg.streaming_cam_target_prim_path,
            self._camera_env_indices,
            scene=self._scene_data_provider.get_interactive_scene(),
        )
        apply_camera_target_positions(
            self._camera_sensor, target_positions, self.cfg.streaming_cam_eye, self._camera_env_indices
        )

    def _log_scene_contact_sensor_arrows(self, num_envs: int) -> None:
        """Render contact sensor data as Newton-style arrows when native contacts are unavailable."""
        if self._viewer is None:
            return
        if not self._viewer.show_contacts:
            self._viewer.log_arrows(CONTACT_ARROW_PATH, None, None, None)
            return
        contact_sensors = (
            self._scene_data_provider.get_contact_sensors() if self._scene_data_provider is not None else {}
        )
        if not contact_sensors:
            self._viewer.log_arrows(CONTACT_ARROW_PATH, None, None, None)
            return

        starts: list[torch.Tensor] = []
        ends: list[torch.Tensor] = []
        for sensor in contact_sensors.values():
            sensor_starts, sensor_ends = self._contact_sensor_arrow_tensors(sensor, num_envs)
            if sensor_starts is not None and sensor_ends is not None:
                starts.append(sensor_starts)
                ends.append(sensor_ends)

        if not starts:
            self._viewer.log_arrows(CONTACT_ARROW_PATH, None, None, None)
            return

        starts_t = torch.cat(starts, dim=0).detach().to(dtype=torch.float32, device="cpu").contiguous()
        ends_t = torch.cat(ends, dim=0).detach().to(dtype=torch.float32, device="cpu").contiguous()
        self._viewer.log_arrows(
            CONTACT_ARROW_PATH,
            wp.array(starts_t.numpy(), dtype=wp.vec3, device=self._viewer.device),
            wp.array(ends_t.numpy(), dtype=wp.vec3, device=self._viewer.device),
            CONTACT_ARROW_COLOR,
        )

    def _contact_sensor_arrow_tensors(self, sensor, num_envs: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Build Newton-style arrow starts/ends from an Isaac Lab contact sensor."""
        try:
            data = sensor.data
            net_forces_proxy = data.net_forces_w
            net_forces = net_forces_proxy.torch if net_forces_proxy is not None else None
        except (AttributeError, NotImplementedError, RuntimeError):
            return None, None

        if net_forces is None or net_forces.numel() == 0:
            return None, None
        net_forces = self._filter_visible_env_tensor(net_forces, num_envs)

        force_threshold = getattr(getattr(sensor, "cfg", None), "force_threshold", None)
        if force_threshold is None:
            force_threshold = 0.0

        try:
            contact_pos = getattr(data, "contact_pos_w", None)
            force_matrix = getattr(data, "force_matrix_w", None)
        except NotImplementedError:
            contact_pos = None
            force_matrix = None
        if contact_pos is not None and force_matrix is not None:
            contact_pos_t = self._filter_visible_env_tensor(contact_pos.torch, num_envs)
            force_matrix_t = self._filter_visible_env_tensor(force_matrix.torch, num_envs)
            if contact_pos_t.numel() != 0 and force_matrix_t.numel() != 0:
                force_norm = torch.linalg.norm(force_matrix_t, dim=-1)
                finite_pos = torch.isfinite(contact_pos_t).all(dim=-1)
                active = (force_norm > force_threshold) & finite_pos
                if torch.any(active):
                    starts = contact_pos_t[active]
                    directions = torch.nn.functional.normalize(force_matrix_t[active], dim=-1)
                    return starts, starts + directions * CONTACT_ARROW_LENGTH

        origins = self._contact_sensor_origin_positions(sensor, data, net_forces)
        if origins is None:
            return None, None
        origins = self._filter_visible_env_tensor(origins, num_envs)

        force_norm = torch.linalg.norm(net_forces, dim=-1)
        active = force_norm > force_threshold
        if not torch.any(active):
            return None, None

        starts = origins[active]
        directions = torch.nn.functional.normalize(net_forces[active], dim=-1)
        return starts, starts + directions * CONTACT_ARROW_LENGTH

    def _contact_sensor_origin_positions(self, sensor, data, net_forces: torch.Tensor) -> torch.Tensor | None:
        """Return per-sensor origins for contact arrow starts."""
        try:
            pos_w = getattr(data, "pos_w", None)
        except NotImplementedError:
            pos_w = None
        if pos_w is not None:
            return pos_w.torch

        body_physx_view = getattr(sensor, "body_physx_view", None)
        if body_physx_view is None:
            return None
        try:
            pose = body_physx_view.get_transforms()
        except RuntimeError:
            return None
        num_envs, num_bodies = net_forces.shape[0], net_forces.shape[1]
        return wp.to_torch(pose).view(num_bodies, num_envs, 7).transpose(0, 1)[..., :3]

    def _filter_visible_env_tensor(self, tensor: torch.Tensor, num_envs: int) -> torch.Tensor:
        """Apply Newton visualizer visible-world filtering to a sensor tensor."""
        if self._resolved_visible_env_ids is None or tensor.ndim == 0 or tensor.shape[0] != num_envs:
            return tensor
        ids = torch.as_tensor(self._resolved_visible_env_ids, dtype=torch.long, device=tensor.device)
        return tensor.index_select(0, ids)


# ---------------------------------------------------------------------------
# GL backend
# ---------------------------------------------------------------------------


class NewtonGLVisualizer(NewtonVisualizer):
    """Newton OpenGL rasterizer visualizer for Isaac Lab.

    Wraps :class:`NewtonViewerGL` — fast local window with the full Isaac Lab
    feature set: streaming camera panel, particle color override, live scalar and array
    plots (via Newton's ImGui sidebar), and :meth:`render_rgb_array` support.

    Use :class:`NewtonGLVisualizerCfg` (factory type ``"newton"``) to select this backend.
    """

    def __init__(self, cfg: NewtonGLVisualizerCfg):
        """Initialize Newton GL visualizer.

        Args:
            cfg: GL visualizer configuration.
        """
        super().__init__(cfg)
        self.cfg: NewtonGLVisualizerCfg = cfg

        # Camera-selector dropdown state — populated in _build_streaming_camera_dropdown().
        # _scene_camera_map  : sensor name → Camera, from get_camera_sensors() at init time.
        # _streaming_camera_choices : display names for the combo (may include "Custom").
        # _streaming_camera_selection : current combo index.
        # _custom_camera / _custom_camera_indices : snapshot of the owned camera created by
        #   _setup_streaming_view() so the user can return to it after switching cameras.
        self._scene_camera_map: dict = {}
        self._streaming_camera_choices: list[str] = []
        self._streaming_camera_selection: int = 0
        self._custom_camera: object | None = None
        self._custom_camera_indices: list[int] = []

    def initialize(self, scene_data_provider: SceneDataProvider) -> None:
        """Initialize the GL visualizer and build the streaming camera dropdown.

        Args:
            scene_data_provider: Provider for scene data and camera sensors.
        """
        super().initialize(scene_data_provider)
        if self._is_initialized:
            self._build_streaming_camera_dropdown()

    def _build_streaming_camera_dropdown(self) -> None:
        """Populate the camera-selector combo from scene sensors captured at init.

        Adds a ``"Custom"`` entry at the top of the list only when
        ``_setup_streaming_view()`` created an owned camera (i.e. the user
        configured :attr:`~isaaclab.visualizers.VisualizerCfg.streaming_cam_target_prim_path`
        or a similar auto-follow option).  If the current camera already comes
        from a scene sensor, it appears in the list naturally by name.

        Patches :meth:`~_NewtonViewerUIMixin._draw_streaming_view_controls` on
        the viewer instance to inject the combo into the existing sidebar section.
        """
        self._scene_camera_map = self._scene_data_provider.get_camera_sensors()
        choices: list[str] = list(self._scene_camera_map.keys())
        current_idx = 0

        if self._camera_is_owned and self._camera_sensor is not None:
            # Owned camera = user configured something explicit (custom target prim or
            # auto-created).  Expose it as "Custom" so the user can return to it.
            self._custom_camera = self._camera_sensor
            self._custom_camera_indices = list(self._camera_sensor_indices or [])
            choices = ["Custom"] + choices
            current_idx = 0
        else:
            # Scene camera: find which entry matches the current sensor.
            for i, cam in enumerate(self._scene_camera_map.values()):
                if cam is self._camera_sensor:
                    current_idx = i
                    break

        self._streaming_camera_choices = choices
        self._streaming_camera_selection = current_idx

        if choices and self._viewer is not None:
            self._patch_streaming_camera_controls()

    def _patch_streaming_camera_controls(self) -> None:
        """Inject the source-camera combo into the viewer's streaming sidebar section.

        Monkey-patches :meth:`_draw_streaming_view_controls` on the *viewer
        instance* (not the class) so that the closure captures ``self``
        (the visualizer) without modifying any Newton viewer code.
        """
        import types

        viewer = self._viewer
        _orig = type(viewer)._draw_streaming_view_controls
        _vis = self  # closure reference to the visualizer

        def _patched(self_viewer):
            # Re-implement the whole Streaming View accordion so Source Camera
            # can be rendered inside it (calling _orig first would close the
            # accordion before we could inject additional content).
            image_logger = getattr(self_viewer, "_image_logger", None)
            if image_logger is None or not image_logger._images:
                return

            imgui = self_viewer.ui.imgui
            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if not imgui.collapsing_header("Streaming View"):
                return

            # Open / Hide image panel combo.
            names = list(image_logger._images.keys())
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
                if new_selected is not None:
                    entry = image_logger._images.get(new_selected)
                    if entry is not None:
                        entry.window_initialized = False
                    # Signal _draw_large to apply aspect-ratio sizing.
                    self_viewer._streaming_panel_needs_sizing = True

            # Source Camera selector — inside the accordion, below Open/Hide.
            if _vis._streaming_camera_choices:
                imgui.separator()
                imgui.text("Source Camera")
                changed, new_cam_idx = imgui.combo(
                    "##streaming_cam_src",
                    _vis._streaming_camera_selection,
                    _vis._streaming_camera_choices,
                )
                if changed:
                    _vis._switch_streaming_camera(new_cam_idx)
                if imgui.is_item_hovered():
                    choice = _vis._streaming_camera_choices[_vis._streaming_camera_selection]
                    cam = _vis._scene_camera_map.get(choice)
                    if cam is not None:
                        prim = getattr(getattr(cam, "cfg", None), "prim_path", None)
                        if prim:
                            imgui.set_tooltip(prim)

        viewer._draw_streaming_view_controls = types.MethodType(_patched, viewer)

    def _switch_streaming_camera(self, new_idx: int) -> None:
        """Switch the active streaming camera to the combo selection at *new_idx*.

        Reassigns :attr:`_camera_sensor` and :attr:`_camera_sensor_indices` and
        clears :attr:`_last_streaming_composite` so the panel refreshes
        immediately on the next step.

        Args:
            new_idx: Index into :attr:`_streaming_camera_choices`.
        """
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
            # Resolve env indices for the new camera's env count.
            n_envs = getattr(getattr(cam, "_view", None), "count", None) or len(self._camera_sensor_indices)
            self._camera_sensor_indices = resolve_streaming_envs(
                n_envs,
                self.cfg.streaming_envs,
                max_tiles=VISUALIZER_TILED_CAMERA_MAX_TILES,
                sample_from=self._resolved_visible_env_ids,
            )
            self._camera_is_owned = False

        # Invalidate the cached composite and clear the panel's window_initialized flag
        # so _draw_large re-sizes it to the new camera's grid aspect ratio on next open.
        self._last_streaming_composite = None
        self._composite_step = -1
        if self._viewer is not None:
            self._viewer._streaming_composite_h = 0
            self._viewer._streaming_composite_w = 0
            image_logger = getattr(self._viewer, "_image_logger", None)
            if image_logger is not None:
                entry = image_logger._images.get("Streaming View")
                if entry is not None:
                    entry.window_initialized = False

    def _create_viewer(self, runtime_headless: bool, metadata: dict) -> NewtonViewerGL:
        return NewtonViewerGL(
            width=self.cfg.window_width,
            height=self.cfg.window_height,
            headless=runtime_headless,
            metadata=metadata,
            update_frequency=self.cfg.update_frequency,
        )

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
        """Register managers for live plotting and add per-manager sidebar toggles.

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
        self._live_plots_manager_visible = {source.manager_name: True for source in self._live_plot_sources}
        self._viewer._live_plots_callback = self._live_plots_panel_imgui

    def _live_plots_panel_imgui(self, imgui) -> None:
        """Render a Live Plots collapsing section in the Newton GL sidebar."""
        if not self._live_plot_sources or self._viewer is None:
            return
        viewer = self._viewer
        scalar_buffers = getattr(viewer, "_scalar_buffers", None)
        array_buffers = getattr(viewer, "_array_buffers", None)
        if not scalar_buffers and not array_buffers:
            return

        _ip = getattr(viewer, "_implot", None)
        if not hasattr(viewer, "_scalar_arrays"):
            viewer._scalar_arrays = {}
        scalar_arrays = viewer._scalar_arrays
        n = getattr(viewer, "_plot_history_size", 250)
        s = viewer.gui.ui.dpi_scale
        plot_h = 180 * s

        groups: dict[str, list[str]] = {}
        for name in scalar_buffers or {}:
            base = _newton_scalar_base_name(name)
            groups.setdefault(base, []).append(name)

        episode_keys = [k for k in groups if k.startswith("episode/")]
        other_keys = [k for k in groups if not k.startswith("episode/")]
        groups = {k: groups[k] for k in episode_keys + other_keys}

        imgui.set_next_item_open(False, imgui.Cond_.appearing)
        if not imgui.collapsing_header("Live Plots"):
            return
        imgui.separator()

        for base_name, names in groups.items():
            term_label = base_name.rsplit("/", 1)[-1]
            if not imgui.collapsing_header(term_label):
                continue
            for name in names:
                buf = scalar_buffers.get(name, [])
                arr = scalar_arrays.get(name)
                if arr is None:
                    arr = np.full(n, np.nan, dtype=np.float32)
                    arr[n - len(buf) :] = np.array(buf, dtype=np.float32)
                    scalar_arrays[name] = arr
            if _ip is not None and _ip.begin_plot(f"##{base_name}", imgui.ImVec2(-1, plot_h)):
                _auto = _ip.AxisFlags_.auto_fit.value
                _ip.setup_axes("", "", _auto, _auto)
                _ip.setup_finish()
                for name in names:
                    arr = scalar_arrays.get(name)
                    if arr is not None:
                        suffix = name[len(base_name) :]
                        label = suffix if suffix else term_label
                        _ip.plot_line(label, arr)
                _ip.end_plot()
            else:
                graph_size = imgui.ImVec2(-1, 80 * s)
                for name in names:
                    arr = scalar_arrays.get(name)
                    if arr is not None:
                        buf = scalar_buffers.get(name, [])
                        overlay = f"{buf[-1]:.4g}" if buf else ""
                        imgui.plot_lines(f"##{name}", arr, graph_size=graph_size, overlay_text=overlay)

        render_heatmap = getattr(viewer, "_render_array_heatmap", None)
        if render_heatmap is not None:
            panel_width = imgui.get_content_region_avail().x
            for name, array in (array_buffers or {}).items():
                if imgui.collapsing_header(name):
                    render_heatmap(name, array, panel_width - 20.0 * s, dpi_scale=s)

    def _render_live_plots(self) -> None:
        """Push manager-term scalars to the Newton viewer's built-in plot panel."""
        if self._viewer is None or not self._live_plot_sources:
            return
        if getattr(self, "_runtime_headless", False):
            return
        self._live_plots_step_counter += 1
        if self._live_plots_step_counter % max(1, getattr(self.cfg, "live_plots_update_interval", 10)) != 0:
            return
        for source in self._live_plot_sources:
            if not self._live_plots_manager_visible.get(source.manager_name, True):
                continue
            for term_name, values in source.collect(self._live_plot_env_idx).items():
                if len(values) == 1:
                    self._viewer.log_scalar(f"{source.manager_name}/{term_name}", values[0])
                else:
                    for i, v in enumerate(values):
                        self._viewer.log_scalar(f"{source.manager_name}/{term_name}[{i}]", v)

    def _apply_viewer_post_init(self) -> None:
        """Apply GL-specific renderer settings after viewer construction."""
        self._viewer.up_axis = 2  # Z-up
        self._viewer.scaling = 1.0
        self._viewer.particle_color = self.cfg.particle_color
        self._viewer.renderer.draw_shadows = self.cfg.enable_shadows
        self._viewer.renderer.draw_sky = self.cfg.enable_sky
        self._viewer.renderer.draw_wireframe = self.cfg.enable_wireframe
        # Accept list/tuple/array-like config colors; provide a stable tuple for nanobind conversion.
        self._viewer.renderer.sky_upper = self._viewer._coerce_color3(self.cfg.sky_upper_color)
        self._viewer.renderer.sky_lower = self._viewer._coerce_color3(self.cfg.sky_lower_color)
        self._viewer.renderer._light_color = self._viewer._coerce_color3(self.cfg.light_color)

    def _apply_camera_pose(
        self,
        pose: tuple[tuple[float, float, float], tuple[float, float, float]],
    ) -> None:
        if self._viewer is None:
            return
        cam_pos, cam_target = pose
        # Match Newton's Camera native pos type: PygletVec3, not wp.vec3.
        self._viewer.camera.pos = PygletVec3(*cam_pos)
        self._viewer.camera.look_at(cam_target)
        self._last_camera_pose = (cam_pos, cam_target)

    def _apply_camera_focal_length(self) -> None:
        if self._viewer is None:
            return
        self._viewer.camera.fov = self._focal_length_to_vertical_fov_degrees()

    def _pump_paused(self) -> None:
        self._viewer._update()

    def render_rgb_array(self) -> np.ndarray:
        """Return the latest RGB frame rendered by the Newton GL viewer.

        In headless mode, a full render cycle is executed on demand using the
        state captured during the most recent :meth:`step` call.

        Returns:
            The latest viewer framebuffer as a uint8 array with shape ``(H, W, 3)``.

        Raises:
            RuntimeError: If the visualizer has not been initialized.
        """
        if self._viewer is None:
            raise RuntimeError("NewtonGLVisualizer must be initialized before capturing an RGB frame.")
        if self._runtime_headless and self._state is not None and not self._viewer.is_paused():
            self._pre_step()
            self._viewer.begin_frame(self._sim_time)
            try:
                self._viewer.log_state(self._state)
            finally:
                self._viewer.end_frame()
        return self._viewer.get_frame().numpy()

    def render_tiled_rgb_array(self) -> np.ndarray | None:
        """Return the last composited streaming frame (all GT types side-by-side).

        Returns the full multi-GT composite produced by the streaming camera panel —
        including depth (turbo colormap), segmentation, and normals when configured via
        :attr:`~isaaclab.visualizers.VisualizerCfg.streaming_gt_types`.

        When the streaming panel is hidden (headless training or panel closed by the
        user), this method builds the composite on demand so that :class:`VideoRecorder`
        and similar consumers always receive a valid frame.

        Returns:
            ``uint8 (H, W, 3)`` composite array, or ``None`` if no camera sensor has
            been configured or no usable GT output is available.
        """
        return self._build_streaming_composite()

    def _build_streaming_composite(self) -> np.ndarray | None:
        """Build (or return the cached) streaming composite for the current step.

        The composite is built at most once per visualizer step.  A step-counter
        comparison is used so repeated calls within the same step (e.g. from both
        :meth:`_log_streaming_image` and :meth:`render_tiled_rgb_array`) share the
        same result without redundant camera work.

        Returns:
            ``uint8 (H, W, 3)`` composite array, or ``None`` if no camera sensor has
            been configured or no usable GT output is available on this step.
        """
        if self._camera_sensor is None:
            return self._last_streaming_composite

        # Return the cached composite when it was already built this step.
        if self._composite_step == self._step_counter:
            return self._last_streaming_composite

        if self._camera_is_owned:
            self._update_owned_camera_poses()
            self._camera_sensor.update(dt=0.0, force_recompute=True)

        available = frozenset(self._camera_sensor.data.output.keys())

        # Filter configured GT types to those actually available on this camera.
        # Scene cameras may not produce every GT type; unrecognised keys are silently
        # dropped so switching cameras never raises.  Fallback to "rgb" when nothing
        # from the configured list is available.
        gt_types: list[str] = []
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
                return None  # camera has no usable output at all

        frames: list[np.ndarray] = []
        for env_idx in self._camera_sensor_indices:
            for gt in gt_types:
                key = sensor_key_for_gt_type(gt, available)
                raw = camera_gt_batch(self._camera_sensor, [env_idx], key)[0]
                frame = CameraFrameColorizer.colorize(
                    raw,
                    gt,
                    depth_min=self.cfg.streaming_depth_min,
                    depth_max=self.cfg.streaming_depth_max,
                )
                frames.append(frame)

        n_envs = len(self._camera_sensor_indices)
        target_aspect = self.cfg.window_width / self.cfg.window_height if self.cfg.window_height > 0 else 1.0
        composite = compose_streaming_grid(frames, n_envs, len(gt_types), target_aspect=target_aspect)
        self._last_streaming_composite = composite
        self._composite_step = self._step_counter
        return composite

    def _log_streaming_image(self) -> None:
        """Fetch GT frames, colorize, composite, and push to Newton's image panel.

        Skips all camera rendering work when the streaming panel is hidden (no image key
        selected in the sidebar combo).  The panel key is registered with a 1×1 placeholder
        on the first call so the combo always appears in the sidebar, but no GPU/CPU
        rendering is performed until the user opens the panel.

        When the panel is visible the composite is built via :meth:`_build_streaming_composite`
        (which caches by step counter) and pushed to the image logger.
        """
        if self._viewer is None or self._camera_sensor is None:
            return

        _PANEL_KEY = "Streaming View"
        image_logger = getattr(self._viewer, "_image_logger", None)
        if image_logger is None:
            return

        # First call: register the panel key in the image logger so the sidebar combo
        # appears.  Use a 1×1 black placeholder — no camera work needed yet.
        # Clear _selected so the panel starts hidden; the user opens it via the
        # "Streaming View" → "Open" combo in the Newton sidebar.
        if image_logger is not None and _PANEL_KEY not in getattr(image_logger, "_images", {}):
            placeholder = wp.zeros((1, 1, 3), dtype=wp.uint8)
            self._viewer.log_image(_PANEL_KEY, placeholder)
            if hasattr(image_logger, "_selected"):
                image_logger._selected = None
            return

        # When the panel is hidden (selected=None), skip all camera rendering.
        # Work resumes the next step after the user selects "Open" in the combo.
        # render_tiled_rgb_array() calls _build_streaming_composite() directly for
        # headless VideoRecorder use-cases, bypassing this guard.
        if image_logger is not None and image_logger._selected is None:
            return

        composite = self._build_streaming_composite()
        if composite is None:
            return

        # Store actual dimensions so _draw_large can size the panel correctly.
        new_h, new_w = composite.shape[:2]
        prev_w = getattr(self._viewer, "_streaming_composite_w", 0)
        prev_h = getattr(self._viewer, "_streaming_composite_h", 0)
        self._viewer._streaming_composite_h = new_h
        self._viewer._streaming_composite_w = new_w
        # Trigger a panel resize whenever the composite first arrives (prev dims were
        # 0 or the 1×1 placeholder) so the window expands from the initial title-bar
        # state to fit the real frame.
        if prev_w <= 1 or prev_h <= 1:
            self._viewer._streaming_panel_needs_sizing = True
        composite_t = torch.from_numpy(composite).contiguous()
        self._viewer.log_image(_PANEL_KEY, wp.from_torch(composite_t))


# ---------------------------------------------------------------------------
# RTX backend
# ---------------------------------------------------------------------------


class NewtonRTXVisualizer(NewtonVisualizer):
    """Newton OVRTX path-tracer visualizer for Isaac Lab.

    Wraps :class:`NewtonViewerRTX` — photorealistic rendering using the same
    ``begin_frame / log_state / end_frame`` step interface as the GL backend.
    The ImGui sidebar (training pause, rendering pause, update-frequency slider,
    physics backend label) is fully supported via ``register_ui_callback``.

    Use :class:`NewtonRTXVisualizerCfg` (factory type ``"newton_rtx"``) to select this backend.

    ``render_rgb_array()`` reads back the path-traced LDR render product directly.
    The tiled camera panel remains disabled because ``ViewerRTX.log_image`` has no
    display sink.

    .. note::
        RTX render quality settings (fps, lighting environment, denoiser, etc.)
        use ``ViewerRTX`` defaults. These will be exposed in a future revision
        consistently with other RTX-capable renderers.
    """

    def __init__(self, cfg: NewtonRTXVisualizerCfg):
        """Initialize Newton RTX visualizer.

        Args:
            cfg: RTX visualizer configuration.
        """
        super().__init__(cfg)
        self.cfg: NewtonRTXVisualizerCfg = cfg
        self._rtx_fov_pending = False
        # OVRTX loads lazily on first begin_frame(); disable permanently on first failure
        # so a missing/broken OVRTX install doesn't spam the log every step.
        self._disable_viewer_on_step_exception = True

    def _create_viewer(self, runtime_headless: bool, metadata: dict) -> NewtonViewerRTX:
        return NewtonViewerRTX(
            width=self.cfg.window_width,
            height=self.cfg.window_height,
            headless=runtime_headless,
            up_axis="Z",
            metadata=metadata,
            update_frequency=self.cfg.update_frequency,
            environment=self.cfg.rtx_environment,
        )

    def _apply_camera_pose(
        self,
        pose: tuple[tuple[float, float, float], tuple[float, float, float]],
    ) -> None:
        if self._viewer is None:
            return
        cam_pos, cam_target = pose
        # ViewerRTX uses set_camera(pos, pitch, yaw) rather than direct camera attribute access.
        pitch, yaw = _eye_lookat_to_pitch_yaw(cam_pos, cam_target)
        self._viewer.set_camera(PygletVec3(*cam_pos), pitch, yaw)
        self._last_camera_pose = (cam_pos, cam_target)

    def _apply_camera_focal_length(self) -> None:
        """Queue FOV for deferred application; ViewerRTX creates its camera on first end_frame()."""
        self._rtx_fov_pending = True

    def _apply_rtx_fov_if_pending(self) -> None:
        """Apply the deferred camera FOV once ViewerRTX's camera object is available."""
        if not self._rtx_fov_pending or self._viewer is None:
            return
        try:
            self._viewer.camera.fov = self._focal_length_to_vertical_fov_degrees()
            self._rtx_fov_pending = False
        except AttributeError:
            pass  # camera not yet created by ViewerRTX; retry next frame

    def _pre_step(self) -> None:
        self._apply_rtx_fov_if_pending()

    def _pump_paused(self) -> None:
        # Both begin_frame/end_frame are required to close the imgui frame each tick.
        # log_state() is skipped so the scene is frozen; the path-tracer accumulates
        # samples on it, producing a progressively cleaner image while paused.
        # Note: full RTX render cost is incurred every tick even while paused.
        self._viewer.begin_frame(self._sim_time)
        self._viewer.end_frame()

    def _uses_streaming_view(self) -> bool:
        # Newton RTX has no display sink for the composited frame (ViewerRTX.log_image
        # is a no-op). Return False until a sink is available so no camera is created
        # and no per-frame colorization work is performed.
        if self.cfg.streaming_view:
            import logging

            logging.getLogger(__name__).warning(
                "streaming_view is not yet supported for NewtonRTXVisualizer (no display sink). "
                "Use NewtonGLVisualizerCfg or pair with a RerunVisualizerCfg/ViserVisualizerCfg "
                "for streaming camera output."
            )
        return False

    def render_rgb_array(self) -> np.ndarray | None:
        """Return the latest RGB frame rendered by the Newton RTX viewer.

        In headless mode, render the state captured during the latest simulation step
        before reading back the path-traced LDR framebuffer.

        Returns:
            The latest viewer framebuffer as a uint8 array with shape ``(H, W, 3)``,
            or ``None`` when the viewer is unavailable.
        """
        if self._viewer is None:
            return None
        if self._runtime_headless and self._state is not None and not self._viewer.is_paused():
            self._pre_step()
            self._viewer.begin_frame(self._sim_time)
            try:
                self._viewer.log_state(self._state)
            finally:
                self._viewer.end_frame()
        return self._viewer.get_frame()
