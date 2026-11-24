# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gaze sliders for foveated cameras in the side panel of Newton viewer windows."""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.sim import SimulationContext

if TYPE_CHECKING:
    from isaaclab.visualizers import BaseVisualizer

    from .foveated_camera import FoveatedCamera

GAZE_LIMIT_DEG = 90.0
"""Largest tilt or pan [deg] the sliders reach."""


class ViewerGazeControls:
    """Gaze and foveation controls for one foveated camera in each Newton viewer window.

    The tilt and pan sliders show the first environment's gaze every frame, so they follow whatever sets it, such as a
    gaze action or a drag in the viewer. Moving one sets that gaze in every environment; a gaze action that sets the
    gaze every step overrides them. The foveation slider steps through :attr:`FoveatedCameraCfg.viewer_cmf_a_values`
    where :meth:`FoveatedCamera.set_cmf_a` can change the retina, and otherwise shows the CMF constant and why it is
    fixed.
    """

    def __init__(self, camera: FoveatedCamera):
        """Create the sliders for ``camera``, which they hold weakly: once it is gone, they draw nothing."""
        self._camera = weakref.ref(camera)
        self._label = camera.cfg.prim_path.rsplit("/", 1)[-1]
        self._attached: set[int] = set()
        self._angles: torch.Tensor | None = None

    def attach(self, sim: SimulationContext) -> None:
        """Add the sliders to every Newton viewer window of ``sim`` that does not show them yet."""
        for visualizer in sim.visualizers:
            if visualizer.cfg.backend.visualizer_type == "newton" and id(visualizer) not in self._attached:
                self._attached.add(id(visualizer))
                visualizer.register_ui_callback(self._render_callback(visualizer), position="side")

    def _render_callback(self, visualizer: BaseVisualizer) -> Callable[[Any], None]:
        """Return the ImGui callback that draws the sliders in one window."""

        def render(imgui: Any) -> None:
            camera = self._camera()
            if camera is None or not camera.is_initialized:
                return
            if not imgui.collapsing_header(f"Foveated camera: {self._label}##{id(self)}"):
                return
            tilt, pan = camera.gaze_angles.torch[0].tolist()
            tilt_changed, tilt = imgui.slider_float("Gaze tilt", tilt, -GAZE_LIMIT_DEG, GAZE_LIMIT_DEG, "%.1f deg")
            pan_changed, pan = imgui.slider_float("Gaze pan", pan, -GAZE_LIMIT_DEG, GAZE_LIMIT_DEG, "%.1f deg")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Positive tilt looks up, positive pan looks left. A gaze action overrides these.")
            center = imgui.button("Center gaze")
            if tilt_changed or pan_changed or center:
                self._set_gaze(camera, (0.0, 0.0) if center else (tilt, pan), visualizer)
            self._draw_foveation(imgui, camera, visualizer)

        return render

    def _draw_foveation(self, imgui: Any, camera: FoveatedCamera, visualizer: BaseVisualizer) -> None:
        """Draw the CMF constant: a slider over the configured values, or the fixed value and why it is fixed."""
        from .foveated_camera import runtime_cmf_a_limitation  # The camera module imports this one.

        current = float(camera.cfg.cmf_a_deg)
        if not camera.supports_runtime_cmf_a:
            imgui.text(f"Foveation: cmf_a {current:g} deg (fixed)")
            if imgui.is_item_hovered():
                imgui.set_tooltip(f"Fixed because {runtime_cmf_a_limitation(camera)}.")
            return
        values = sorted({float(value) for value in camera.cfg.viewer_cmf_a_values} | {current})
        index = values.index(current)
        changed, index = imgui.slider_int("Foveation", index, 0, len(values) - 1, f"cmf_a {current:g} deg")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Smaller cmf_a concentrates the samples at the fixation; larger spreads them evenly.")
        if changed and values[index] != current:
            camera.set_cmf_a(values[index])
            view = getattr(visualizer, "current_image_view", None)
            if view is not None:
                view.invalidate()

    def _set_gaze(self, camera: FoveatedCamera, angles: tuple[float, float], visualizer: BaseVisualizer) -> None:
        """Set ``angles`` [deg] in every environment and redraw the window's camera view, even while paused."""
        gaze = camera.gaze_angles.torch
        if self._angles is None or self._angles.shape != gaze.shape:
            self._angles = torch.empty_like(gaze)
        self._angles[:] = torch.tensor(angles, device=gaze.device)
        camera.set_gaze_angles(self._angles)
        view = getattr(visualizer, "current_image_view", None)
        if view is not None:
            view.invalidate()


def add_viewer_gaze_controls(camera: FoveatedCamera) -> None:
    """Show ``camera``'s gaze sliders in the Newton viewer windows of every simulation reset while it exists."""
    controls = ViewerGazeControls(camera)
    name = f"isaaclab_fovi.viewer_gaze_controls.{id(controls)}"
    SimulationContext.add_reset_callback(name, controls.attach)
    weakref.finalize(camera, SimulationContext.remove_reset_callback, name)
