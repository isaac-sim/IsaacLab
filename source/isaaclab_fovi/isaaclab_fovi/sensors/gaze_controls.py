# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gaze sliders for foveated cameras in the side panel of Newton viewer windows."""

from __future__ import annotations

import weakref
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar

from isaaclab.sim import SimulationContext

if TYPE_CHECKING:
    from isaaclab.visualizers import BaseVisualizer

    from .foveated_camera import FoveatedCamera

GAZE_LIMIT_DEG = 90.0
"""Largest tilt or pan [deg] the sliders reach."""


class ViewerGazeControls:
    """Gaze and foveation controls for one foveated camera, in the side panel of each Newton viewer window that shows
    it.

    A window draws the controls while its camera view shows the camera, so a camera the viewer cannot show, or another
    camera chosen in the window's camera menu, has none. The tilt and pan sliders show the first environment's gaze
    every frame, so they follow whatever sets it, such as a gaze action or a drag in the viewer. Moving one sets that
    gaze in every environment; a gaze action that sets the gaze every step overrides them. The foveation slider steps
    through :attr:`cmf_a_values` and the camera's starting CMF constant where :meth:`FoveatedCamera.set_cmf_a` can
    change the retina, and otherwise shows the CMF constant and why it is fixed. A change in any window redraws the
    camera view of every window.
    """

    cmf_a_values: ClassVar[tuple[float, ...]] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    """CMF constants [deg] the foveation slider offers every camera, with its starting one. Assign another tuple to
    change them; the sliders read it as they draw."""

    def __init__(self, camera: FoveatedCamera):
        """Create the sliders for ``camera``, which they hold weakly: once it is gone, they draw nothing.

        The viewer cannot unregister a UI callback, so each window keeps a callback that then returns at once.
        """
        self._camera = weakref.ref(camera)
        self._label = camera.cfg.prim_path.rsplit("/", 1)[-1]
        self._start_cmf_a = float(camera.cfg.cmf_a_deg)
        """The camera's starting CMF constant [deg], which stays a stop of the foveation slider after a switch."""
        self._windows: list[BaseVisualizer] = []
        """The Newton viewer windows that draw the controls while they show the camera."""

    def attach(self, sim: SimulationContext) -> None:
        """Add the controls to every Newton viewer window of ``sim`` that does not have them yet."""
        for visualizer in sim.visualizers:
            if visualizer.cfg.backend.visualizer_type == "newton" and all(
                window is not visualizer for window in self._windows
            ):
                self._windows.append(visualizer)
                visualizer.register_ui_callback(partial(self._render, visualizer), position="side")

    def _render(self, window: BaseVisualizer, imgui: Any) -> None:
        """Draw the controls in the side panel of ``window`` while its camera view shows the camera."""
        camera = self._camera()
        if camera is None or not camera.is_initialized:
            return
        view = window.current_image_view
        if view is None or view.camera is not camera:
            return
        # Each camera's widgets share labels, so they need their own ID scope; a header does not open one.
        imgui.push_id(f"isaaclab_fovi.viewer_gaze_controls.{id(self)}")
        try:
            if imgui.collapsing_header(f"Foveated camera: {self._label}"):
                self._draw_controls(imgui, camera)
        finally:
            imgui.pop_id()

    def _draw_controls(self, imgui: Any, camera: FoveatedCamera) -> None:
        """Draw the gaze sliders, the center button and the foveation control."""
        tilt, pan = camera.gaze_angles.torch[0].tolist()
        tilt_changed, tilt = imgui.slider_float("Gaze tilt", tilt, -GAZE_LIMIT_DEG, GAZE_LIMIT_DEG, "%.1f deg")
        pan_changed, pan = imgui.slider_float("Gaze pan", pan, -GAZE_LIMIT_DEG, GAZE_LIMIT_DEG, "%.1f deg")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Positive tilt looks up, positive pan looks left. A gaze action overrides these.")
        center = imgui.button("Center gaze")
        if tilt_changed or pan_changed or center:
            # The camera's stored angles are writable; set_gaze_angles then publishes them.
            gaze = camera.gaze_angles.torch
            gaze[:] = gaze.new_tensor((0.0, 0.0) if center else (tilt, pan))
            camera.set_gaze_angles(gaze)
            self._redraw()
        self._draw_foveation(imgui, camera)

    def _draw_foveation(self, imgui: Any, camera: FoveatedCamera) -> None:
        """Draw the CMF constant: a slider over :attr:`cmf_a_values`, or the fixed value and why it is fixed."""
        from .foveated_camera import _runtime_cmf_a_limitation  # The camera module imports this one.

        current = float(camera.cfg.cmf_a_deg)
        if not camera.supports_runtime_cmf_a:
            imgui.text(f"Foveation: cmf_a {current:g} deg (fixed)")
            if imgui.is_item_hovered():
                imgui.set_tooltip(f"Fixed because {_runtime_cmf_a_limitation(camera)}.")
            return
        values = sorted({float(value) for value in self.cmf_a_values} | {self._start_cmf_a, current})
        index = values.index(current)
        changed, index = imgui.slider_int("Foveation", index, 0, len(values) - 1, f"cmf_a {current:g} deg")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Smaller cmf_a concentrates the samples at the fixation; larger spreads them evenly.")
        if changed and values[index] != current:
            camera.set_cmf_a(values[index])
            self._redraw()

    def _redraw(self) -> None:
        """Redraw the camera view of every window, which shows the change even while paused."""
        for window in self._windows:
            view = getattr(window, "current_image_view", None)
            if view is not None:
                view.invalidate()


def add_viewer_gaze_controls(camera: FoveatedCamera) -> None:
    """Add ``camera``'s controls to the Newton viewer windows of every simulation reset while it exists; a window
    draws them while it shows the camera."""
    controls = ViewerGazeControls(camera)
    name = f"isaaclab_fovi.viewer_gaze_controls.{id(controls)}"
    SimulationContext.add_reset_callback(name, controls.attach)
    weakref.finalize(camera, SimulationContext.remove_reset_callback, name)
