# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualizer that displays an Isaac Lab simulation in the Isaac Sim 7 viewer application."""

from __future__ import annotations

import contextlib
import logging
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pxr import Usd, UsdGeom

from isaaclab.scene_data import SceneDataFormat
from isaaclab.sim import SimulationContext
from isaaclab.visualizers.base_visualizer import BaseVisualizer
from isaaclab.visualizers.visualizer_cfg import PerspectiveCameraCfg

from .is7_stage import authored_scales
from .is7_visualizer_cfg import IS7VisualizerCfg

if TYPE_CHECKING:
    from isaaclab.sensors.camera import Camera

logger = logging.getLogger(__name__)

_POSE_ROOT = "/IsaacLabPoses"
"""Pseudo prim that keys Isaac Lab's queued pose writes, so a newer write replaces an unapplied one."""


class IS7Visualizer(BaseVisualizer):
    """Isaac Sim 7 viewer application driven by Isaac Lab's simulation.

    The Isaac Sim 7 host owns the stage, renderer and UI (stage tree, properties, content browser, picking and
    gizmos). Isaac Lab keeps its own loop and physics. Each step it queues the world poses of its bodies into the
    host's command queue, which applies them before the renderer's next frame.
    """

    def __init__(self, cfg: IS7VisualizerCfg):
        """Initialize visualizer state.

        Args:
            cfg: Isaac Sim 7 viewer configuration.
        """
        super().__init__(cfg)
        self.cfg: IS7VisualizerCfg = cfg
        self._resources = contextlib.ExitStack()
        self._host = None
        self._desktop = None
        self._provider = None
        self._paths: list[str] = []
        self._scales = None
        self._pose_output = SceneDataFormat.TransposedMatrix44d()
        self._timestamp = -1
        self._armed_world: int | None = None
        self._generation = -1
        self._running = False
        self.timings = {"publish": 0.0, "steps": 0}
        """Accumulated wall time [s] of pose publication, for benchmarking the connection."""

    @property
    def url(self) -> str:
        """Address of the host's HTTP service."""
        return self._host.url

    @property
    def client_url(self) -> str:
        """Address that opens the viewer in a browser, including the launch token."""
        return f"{self._host.url}/index.html?entry=index.js&token={self._host.token}"

    def initialize(self, sim: SimulationContext, *, cameras: list[PerspectiveCameraCfg | Camera]) -> None:
        """Export the scene, start the Isaac Sim 7 host and bind pose publication.

        Args:
            sim: Simulation owner that provides the authored stage and scene data.
            cameras: Resolved perspective settings and borrowed scene sensors, in display order.
        """
        if self._is_initialized:
            logger.debug("[IS7Visualizer] initialize() called while already initialized.")
            return
        super().initialize(sim, cameras=cameras)

        from sim_viewer.cli import client_roots_for_backend
        from sim_viewer.scene import CAMERA_SCOPE, RENDER_SCOPE, dynamic_viewport_spec, read_scene_metadata

        from isaacsim.application.host import AppHost
        from isaacsim.foundation.orchestration import RationalRate
        from isaacsim.viewports.presentation import FrameEncoder

        from .is7_world import make_physics_free_world_opener

        scene = self._export_scene(sim.stage)
        metadata = read_scene_metadata(scene)
        specs: list = []
        rate = RationalRate(int(self.cfg.viewport_hz))
        client_roots = client_roots_for_backend("web")
        if self.cfg.app_client_root is not None:
            client_roots = [("/app/", Path(self.cfg.app_client_root) / "renderer"), *client_roots[1:]]
        self._host = AppHost(
            open_world=make_physics_free_world_opener(specs, rate, renderer="ovgl"),
            scene=str(scene),
            viewports=specs,
            viewport_factory=lambda index: dynamic_viewport_spec(index, tuple(self.cfg.resolution)),
            encoder=FrameEncoder.create("auto"),
            presentation=self.cfg.presentation,
            renderer_name="ovgl",
            hidden_roots=(CAMERA_SCOPE, RENDER_SCOPE),
            world_up=metadata.up_vector,
            host="127.0.0.1",
            port=self.cfg.port,
            client_roots=client_roots,
            content_writes=True,
            stage_writes=True,
        )
        self._resources.callback(self._host.stop)
        self._host.start()

        self._provider = sim.get_scene_data_provider()
        self._paths = list(self._provider.backend.transform_paths)
        self._scales = authored_scales(sim.stage, self._paths)
        self._publish_poses()
        if self.cfg.ui == "browser":
            self._log_viewer_url("IS7Visualizer", self.client_url)
        elif self.cfg.ui == "desktop":
            self._open_desktop()
        self._log_initialization_table(
            logger=logger,
            title="IS7Visualizer Configuration",
            rows=[("ui", self.cfg.ui), ("bodies", len(self._paths)), ("viewport_hz", self.cfg.viewport_hz)],
        )
        self._running = True
        self._is_initialized = True

    def _open_desktop(self) -> None:
        """Open the Electron window attached to the host that this process runs."""
        from isaacsim.application.client import start_desktop

        if self.cfg.app_client_root is None:
            raise ValueError("IS7VisualizerCfg.app_client_root is required for ui='desktop'.")
        self._desktop = start_desktop(
            Path(self.cfg.app_client_root) / "main.cjs",
            environment={"SIM_VIEWER_ATTACH_URL": self._host.url, "SIM_VIEWER_ATTACH_TOKEN": self._host.token},
        )
        self._resources.callback(self._desktop.close)

    def _export_scene(self, stage: Usd.Stage) -> Path:
        """Write the authored stage to a temporary USD file that the host opens as its scene.

        Isaac Lab authors some containers without a type, which the viewer's stage tree omits, so they become
        ``Xform`` prims. Instanceable prims are expanded while the expanded scene stays within
        :attr:`IS7VisualizerCfg.deinstance_max_points`. The ``Flattened_Prototype_*`` specs stay because the
        expanded prims still reference them.
        """
        exported = Usd.Stage.Open(stage.Flatten())
        deinstance = self._expanded_points(exported) <= self.cfg.deinstance_max_points
        for prim in exported.Traverse():
            if not prim.GetTypeName():
                prim.SetTypeName("Xform")
            if deinstance and prim.IsInstance():
                prim.SetInstanceable(False)
        logger.info("[IS7Visualizer] %s instanced prims in the exported scene.", "Expanded" if deinstance else "Kept")
        directory = Path(self._resources.enter_context(tempfile.TemporaryDirectory(prefix="isaaclab_is7_")))
        scene = directory / "scene.usdc"
        exported.Export(str(scene))
        return scene

    @staticmethod
    def _expanded_points(stage: Usd.Stage) -> int:
        """Count mesh points as if every instance owned a copy of its prototype's geometry."""
        total = 0
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            if prim.IsA(UsdGeom.Mesh):
                points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
                total += 0 if points is None else len(points)
        return total

    def _publish_poses(self) -> bool:
        """Queue the latest simulation world transforms; return whether new ones were queued.

        The host replaces its world when the user stops or reloads the scene. Poses are skipped while that
        happens and queued again, with the world-space flag, once the new world is open.
        """
        from isaacsim.application.host.impl.transport import ReloadingError

        backend = self._provider.backend
        generation = self._host.simulation.progress.generation
        if generation != self._generation:
            self._armed_world = None
            self._timestamp = -1
        if backend.transforms_timestamp == self._timestamp and self._armed_world is not None:
            return False
        if not self._provider.get_transforms(self._pose_output, allow_passthrough=True, scales=self._scales):
            return False
        # The host reads these on its own thread, so hand it a copy that the next step cannot overwrite.
        matrices = self._pose_output.matrices.numpy().copy()
        paths = self._paths
        try:
            with self._host.simulation.reading(wait=False) as world:
                if self._armed_world != id(world):
                    # Bodies carry world matrices, so the host must not compose them with their parents.
                    ones = np.ones(len(paths), dtype=np.bool_)
                    world.commands.submit(
                        f"{_POSE_ROOT}/reset",
                        "omni:resetXformStack",
                        lambda view: view.write(paths, "omni:resetXformStack", ones),
                    )
                    self._armed_world = id(world)
                world.commands.submit(_POSE_ROOT, "omni:xform", lambda view: view.write(paths, "omni:xform", matrices))
        except ReloadingError:
            return False
        self._generation = generation
        self._timestamp = backend.transforms_timestamp
        return True

    def step(self, dt: float) -> None:
        """Queue this step's poses for the host.

        Args:
            dt: Simulation time-step in seconds.
        """
        if not self._is_initialized or not self._running:
            return
        self._sim_time += dt
        start = time.perf_counter()
        self._publish_poses()
        self.timings["publish"] += time.perf_counter() - start
        self.timings["steps"] += 1

    def is_running(self) -> bool:
        """Return whether the viewer is still open; closing the desktop window ends it."""
        if self._desktop is not None and self._desktop.returncode is not None:
            self._running = False
        return self._running

    def close(self) -> None:
        """Stop the host and remove the exported scene."""
        if self._is_closed:
            return
        try:
            self._resources.close()
        finally:
            self._running = False
            self._host = self._desktop = self._provider = None
            super().close()
