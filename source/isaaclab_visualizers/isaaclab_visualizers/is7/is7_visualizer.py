# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualizer that displays an Isaac Lab simulation in the Isaac Sim 7 viewer."""

from __future__ import annotations

import contextlib
import logging
import math
import time
from typing import TYPE_CHECKING

from pxr import Usd

from isaaclab.sim import SimulationContext
from isaaclab.visualizers.base_visualizer import BaseVisualizer
from isaaclab.visualizers.visualizer_cfg import PerspectiveCameraCfg

from .is7_stage import IS7StageBridge
from .is7_visualizer_cfg import IS7VisualizerCfg

if TYPE_CHECKING:
    from isaaclab.sensors.camera import Camera

logger = logging.getLogger(__name__)

_CAMERA_PATH = "/IsaacLabViewer/Camera"
_PRODUCT_PATH = "/IsaacLabViewer/Product"
_RENDER_VAR_PATH = "/IsaacLabViewer/Color"


def _orbit_from_eye_lookat(eye, lookat) -> dict[str, object]:
    """Convert an eye and look-at point to the viewer's orbit camera (Z-up)."""
    offset = [e - t for e, t in zip(eye, lookat)]
    distance = math.sqrt(sum(axis * axis for axis in offset)) or 1.0
    return {
        "target": tuple(float(axis) for axis in lookat),
        "distance": distance,
        "azimuth_deg": math.degrees(math.atan2(offset[1], offset[0])),
        "elevation_deg": math.degrees(math.asin(max(-1.0, min(1.0, offset[2] / distance)))),
    }


class IS7Visualizer(BaseVisualizer):
    """Isaac Sim 7 viewer over an OVStage that mirrors Isaac Lab's simulated poses.

    Lifecycle per step: ``viewer.update()`` applies camera input, the bridge publishes world poses
    into the stage, and the OVGL renderer draws the frame. Physics and rendering backends stay
    Isaac Lab's; the viewer only reads the stage.
    """

    def __init__(self, cfg: IS7VisualizerCfg):
        """Initialize visualizer state.

        Args:
            cfg: Isaac Sim 7 viewer configuration.
        """
        super().__init__(cfg)
        self.cfg: IS7VisualizerCfg = cfg
        self._resources = contextlib.ExitStack()
        self._context = None
        self._renderer = None
        self._viewport = None
        self._viewer = None
        self._bridge: IS7StageBridge | None = None
        self._running = False
        self._step_count = 0
        self.timings = {"update": 0.0, "publish": 0.0, "render": 0.0, "steps": 0, "renders": 0}
        """Accumulated wall time [s] per phase, for benchmarking the connection."""

    def initialize(self, sim: SimulationContext, *, cameras: list[PerspectiveCameraCfg | Camera]) -> None:
        """Export the scene to Isaac Sim 7, start the viewer, and bind pose publication.

        Args:
            sim: Simulation owner that provides the authored stage and scene data.
            cameras: Resolved perspective settings and borrowed scene sensors, in display order.
        """
        if self._is_initialized:
            logger.debug("[IS7Visualizer] initialize() called while already initialized.")
            return
        super().initialize(sim, cameras=cameras)

        from isaacsim.application.viewer import Camera as OrbitCamera
        from isaacsim.application.viewer import Viewer
        from isaacsim.context.simulation import make_simulation_context
        from isaacsim.rendering.ovgl import StageRenderer
        from isaacsim.viewports.adapters import Viewport, define_viewport

        layer = sim.stage.Flatten()
        if self.cfg.deinstance:
            expanded = Usd.Stage.Open(layer)
            for prim in expanded.Traverse():
                if prim.IsInstance():
                    prim.SetInstanceable(False)
            layer = expanded.Flatten()
        usda = layer.ExportToString()
        context = make_simulation_context()
        context.create_from_string(usda)
        self._context = context
        self._resources.callback(context.close)
        width, height = self.cfg.resolution
        define_viewport(
            context,
            camera_path=_CAMERA_PATH,
            render_product_path=_PRODUCT_PATH,
            render_var_path=_RENDER_VAR_PATH,
            resolution=(width, height),
        )
        context.finalize()

        self._renderer = self._resources.enter_context(StageRenderer(context, product=_PRODUCT_PATH))
        self._viewport = self._resources.enter_context(Viewport(context, product=_PRODUCT_PATH, camera=_CAMERA_PATH))
        self._viewer = self._resources.enter_context(
            Viewer(
                self._viewport,
                title=self.cfg.title,
                camera=OrbitCamera(**_orbit_from_eye_lookat(self.cfg.eye, self.cfg.lookat)),
                desktop=self.cfg.ui == "desktop",
            )
        )
        # The bridge borrows the context's stage, so it must be released before the context closes.
        self._bridge = IS7StageBridge(
            context.stage_pointer, sim.stage, sim.get_scene_data_provider(), device=sim.device
        )
        self._resources.callback(self._bridge.close)

        self._bridge.publish()
        self._renderer.render()
        if self.cfg.ui == "browser":
            self._log_viewer_url("IS7Visualizer", self._viewer.url)
        self._log_initialization_table(
            logger=logger,
            title="IS7Visualizer Configuration",
            rows=[
                ("ui", self.cfg.ui),
                ("resolution", self.cfg.resolution),
                ("bodies", len(self._bridge.body_paths)),
                ("render_every", self.cfg.render_every),
            ],
        )
        self._running = True
        self._is_initialized = True

    def step(self, dt: float) -> None:
        """Publish poses and draw one frame.

        Args:
            dt: Simulation time-step in seconds.
        """
        if not self._is_initialized or not self._running:
            return
        self._sim_time += dt
        t0 = time.perf_counter()
        if not self._viewer.update():
            self._running = False
            return
        t1 = time.perf_counter()
        self._bridge.publish()
        t2 = time.perf_counter()
        self._step_count += 1
        rendered = False
        if self._step_count % self.cfg.render_every == 0 and (self.cfg.ui == "headless" or self._viewport.visible):
            self._renderer.render()
            rendered = True
        t3 = time.perf_counter()
        self.timings["update"] += t1 - t0
        self.timings["publish"] += t2 - t1
        self.timings["render"] += t3 - t2
        self.timings["steps"] += 1
        self.timings["renders"] += rendered

    def capture(self):
        """Return the viewer's latest rendered frame, for tests and benchmarks."""
        return self._viewer.capture()

    def is_running(self) -> bool:
        """Return whether the viewer is still open."""
        return self._running

    def close(self) -> None:
        """Release the viewer, renderer, stage bridge, and the Isaac Sim 7 context in dependency order."""
        if self._is_closed:
            return
        try:
            self._resources.close()
        finally:
            self._running = False
            self._bridge = self._viewer = self._viewport = self._renderer = self._context = None
            super().close()
