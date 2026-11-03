# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualizer that displays an Isaac Lab simulation in the Isaac Sim 7 viewer application."""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from pxr import Sdf, Usd, UsdGeom

from isaaclab.scene_data import SceneDataFormat
from isaaclab.sim import SimulationContext
from isaaclab.visualizers.base_visualizer import BaseVisualizer
from isaaclab.visualizers.visualizer_cfg import PerspectiveCameraCfg, VisualizerCfg

from .is7_shared_stage import Replication
from .is7_stage import authored_scales
from .is7_visualizer_cfg import IS7BackendCfg

if TYPE_CHECKING:
    from isaaclab.sensors.camera import Camera

logger = logging.getLogger(__name__)

_ENVS_ROOT = "/World/envs"
"""Parent of Isaac Lab's cloned environments."""

_POSE_ROOT = "/IsaacLabPoses"
"""Pseudo prim that keys Isaac Lab's queued pose writes, so a newer write replaces an unapplied one."""


class IS7Visualizer(BaseVisualizer):
    """Isaac Sim 7 viewer application driven by Isaac Lab's simulation.

    The Isaac Sim 7 host owns the stage, renderer and UI (stage tree, properties, content browser, picking and
    gizmos). Isaac Lab keeps its own loop and physics. Each step it queues the world poses of its bodies into the
    host's command queue, which applies them before the renderer's next frame.
    """

    def __init__(self, cfg: VisualizerCfg):
        """Initialize visualizer state.

        Args:
            cfg: Shared visualizer settings whose ``backend`` is an :class:`IS7BackendCfg`.
        """
        super().__init__(cfg)
        self.backend_cfg: IS7BackendCfg = cfg.backend
        if self.backend_cfg.app_client_root is None:
            self.backend_cfg.app_client_root = os.environ.get("ISAACLAB_IS7_APP_CLIENT_ROOT")
        self._resources = contextlib.ExitStack()
        self._host = None
        self._desktop = None
        self._provider = None
        self._paths: list[str] = []
        self._slot = None
        self._shared = None
        self._shared_armed = False
        self._markers = None
        self._next_render = 0.0
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

        self._provider = sim.get_scene_data_provider()
        self._paths = list(self._provider.backend.transform_paths)
        self._scales = authored_scales(sim.stage, self._paths)
        started = time.perf_counter()
        replication = None
        if self.backend_cfg.stage_mode == "shared":
            scene, replication = self._export_prototypes(sim)
        else:
            scene = self._export_scene(sim.stage)
        self.timings["export"] = time.perf_counter() - started
        metadata = read_scene_metadata(scene)
        specs: list = []
        rate = RationalRate(int(self.cfg.window.fps))
        client_roots = client_roots_for_backend("web")
        if self.backend_cfg.app_client_root is not None:
            client_roots = [("/app/", Path(self.backend_cfg.app_client_root) / "renderer"), *client_roots[1:]]

        def extra_participants(wake):
            return ()

        if self.backend_cfg.stage_mode == "export" and self.backend_cfg.pose_transport == "participant":
            from sim_viewer.scene import PHYSICS_POSITION

            from isaacsim.application.host import NativeParticipant

            from .is7_poses import PoseEngine, PoseSlot

            self._slot = PoseSlot(self._paths)
            engine = PoseEngine(self._slot)
            self._resources.callback(engine.close)

            def extra_participants(wake):
                return [NativeParticipant("isaaclab-poses", engine.participant(rate), PHYSICS_POSITION)]

        open_world = make_physics_free_world_opener(specs, rate, renderer="ovgl", extra_participants=extra_participants)
        renderer_name = "ovgl"
        if self.backend_cfg.stage_mode == "shared":
            from .is7_shared_world import SharedWorldOpener

            device = sim.device
            from .is7_shared_markers import SharedMarkers

            self._markers = SharedMarkers()
            self._shared = open_world = SharedWorldOpener(
                scene,
                specs,
                rate,
                device=int(device.split(":")[1]) if ":" in str(device) else 0,
                replication=replication,
            )
            self._resources.callback(self._shared.close)
            self.timings["shared_stage"] = time.perf_counter() - started - self.timings["export"]
            renderer_name = "ovrtx"
        self._host = AppHost(
            open_world=open_world,
            scene=str(scene),
            viewports=specs,
            viewport_factory=lambda index: dynamic_viewport_spec(index, tuple(self.cfg.window.size)),
            encoder=FrameEncoder.create("auto"),
            presentation=self.backend_cfg.presentation,
            renderer_name=renderer_name,
            hidden_roots=(CAMERA_SCOPE, RENDER_SCOPE),
            world_up=metadata.up_vector,
            host="127.0.0.1",
            port=self.backend_cfg.port,
            client_roots=client_roots,
            content_writes=True,
            stage_writes=True,
        )
        self._resources.callback(self._host.stop)
        if self._shared is not None:
            self._shared.on_frame = self._host.simulation.request_redraw
        if self._slot is not None:
            # A stopped host renders only on request, so each pose set that Isaac Lab hands over requests a frame.
            self._slot.wake = self._host.simulation.request_redraw
        if self._shared is None:
            # The participant publishes the first poses when the host opens its world, so hand them over first.
            self._publish_poses()
        self._host.start()
        if not self.cfg.headless and self.backend_cfg.ui == "browser":
            self._log_viewer_url("IS7Visualizer", self.client_url)
        elif not self.cfg.headless:
            self._open_desktop()
        self._log_initialization_table(
            logger=logger,
            title="IS7Visualizer Configuration",
            rows=[
                ("ui", "headless" if self.cfg.headless else self.backend_cfg.ui),
                ("bodies", len(self._paths)),
                ("fps", self.cfg.window.fps),
            ],
        )
        self._running = True
        self._is_initialized = True

    def _open_desktop(self) -> None:
        """Open the Electron window attached to the host that this process runs."""
        from isaacsim.application.client import start_desktop

        if self.backend_cfg.app_client_root is None:
            raise ValueError("IS7BackendCfg.app_client_root is required for ui='desktop'.")
        self._desktop = start_desktop(
            Path(self.backend_cfg.app_client_root) / "main.cjs",
            environment={"SIM_VIEWER_ATTACH_URL": self._host.url, "SIM_VIEWER_ATTACH_TOKEN": self._host.token},
        )
        self._resources.callback(self._desktop.close)

    def _export_prototypes(self, sim: SimulationContext) -> tuple[Path, Replication | None]:
        """Export each asset once; the environments are cloned and placed in the stage.

        This is Isaac Lab's OVStage population path. With a clone plan, the copies follow it, so environments may hold
        different assets. Without one, every environment receives a copy of the first environment's children.

        Returns:
            The exported scene and the replication that completes it, or ``None`` when the scene is exported whole.
        """
        from isaaclab_ov.stage import export_stage_to_string

        stage = sim.stage
        plan = sim.get_clone_plan()
        if plan is not None and len(plan.topology.world_prototype_layout) > 1:
            from isaaclab_ov.cloner.replicate import _prepare_scene_copies

            prepared = SimpleNamespace()
            _prepare_scene_copies(plan, tuple(range(plan.topology.num_asset_prototypes)), [prepared])
            replication = Replication(prepared.clone_copies, prepared.clone_env_paths, prepared.clone_positions)
            usd = export_stage_to_string(
                stage,
                prepared.clone_env_paths,
                source_paths=tuple(source for source, _ in prepared.clone_copies),
                keep_env_roots=False,
                env_paths=prepared.population_env_paths,
                preserve_references=True,
            )
        else:
            envs = stage.GetPrimAtPath(_ENVS_ROOT)
            env_paths = [child.GetPath().pathString for child in envs.GetChildren()] if envs else []
            if len(env_paths) < 2:
                return self._export_scene(stage), None
            source = env_paths[0]
            children = [child.GetName() for child in stage.GetPrimAtPath(source).GetChildren()]
            copies = [(f"{source}/{name}", [f"{env}/{name}" for env in env_paths[1:]]) for name in children]
            # Environment roots keep their authored transforms, so the copies land in place.
            replication = Replication(copies, env_paths, None)
            usd = export_stage_to_string(
                stage, env_paths, tuple(source for source, _ in copies), keep_env_roots=True, preserve_references=True
            )
        layer = Sdf.Layer.CreateAnonymous(".usda")
        layer.ImportFromString(usd)
        exported = Usd.Stage.Open(layer)
        for prim in exported.Traverse():
            if not prim.GetTypeName():
                prim.SetTypeName("Xform")
        directory = Path(self._resources.enter_context(tempfile.TemporaryDirectory(prefix="isaaclab_is7_")))
        scene = directory / "scene.usdc"
        exported.Export(str(scene))
        return scene, replication

    def _export_scene(self, stage: Usd.Stage) -> Path:
        """Write the authored stage to a temporary USD file that the host opens as its scene.

        Isaac Lab authors some containers without a type, which the viewer's stage tree omits, so they become
        ``Xform`` prims. Instanceable prims are expanded while the expanded scene stays within
        :attr:`IS7BackendCfg.deinstance_max_points`. The ``Flattened_Prototype_*`` specs stay because the
        expanded prims still reference them.
        """
        exported = Usd.Stage.Open(stage.Flatten())
        deinstance = self._expanded_points(exported) <= self.backend_cfg.deinstance_max_points
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

        With the ``"participant"`` transport, the poses go to the slot that the host's pose engine reads. With
        ``"commands"``, they go through the host's command queue; the host replaces its world when the user stops or
        reloads the scene, so poses are skipped while that happens and queued again, with the world-space flag, once
        the new world is open.
        """
        backend = self._provider.backend
        if self._slot is not None:
            if backend.transforms_timestamp == self._timestamp:
                return False
            if not self._provider.get_transforms(self._pose_output, allow_passthrough=True, scales=self._scales):
                return False
            # The host reads the slot on its own thread, so hand it a copy that the next step cannot overwrite.
            self._slot.publish(self._pose_output.matrices.numpy().reshape(-1, 16).copy())
            self._timestamp = backend.transforms_timestamp
            return True

        from isaacsim.application.host.impl.transport import ReloadingError

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

    def _step_shared(self, dt: float) -> None:
        """Write the latest poses into the shared stage, seal it and render, at most ``window.fps`` times a second.

        Poses only matter when a frame renders, so steps between frames write nothing.
        """
        now = time.monotonic()
        if now < self._next_render:
            return
        self._next_render = now + 1.0 / self.cfg.window.fps
        if not self._provider.get_transforms(self._pose_output, allow_passthrough=True, scales=self._scales):
            return
        from isaaclab_ov.stage import xform_tensor_from_warp

        matrices = self._pose_output.matrices
        stream = wp.get_device(matrices.device).stream.cuda_stream or 1
        paths = self._paths
        time_ns = int(self._sim_time * 1e9)

        first = not self._shared_armed
        self._shared_armed = True

        markers = self._markers if self.supports_markers() else None
        registry = self._sim.vis_marker_registry if markers is not None else None

        def write_state(stage):
            if markers is not None:
                markers.sync(stage, registry)
            if first:
                # Bodies carry world matrices: their local transform is their world transform.
                stage.write(paths, "omni:resetXformStack", np.ones(len(paths), dtype=np.bool_))
            # The stage's hierarchy derives world matrices from ``omni:xform`` when it seals, so poses go there. The
            # matrices stay on the device; ovstage copies them in on the producing stream.
            stage.write_device(paths, "omni:xform", xform_tensor_from_warp(matrices), stream)

        self._shared.step(time_ns, int(dt * 1e9), write_state)
        self.timings["renders"] = self._shared.renders

    def supports_markers(self) -> bool:
        """Return whether visualization markers are drawn: in the shared stage, when markers are enabled."""
        return self.cfg.enable_markers and self.backend_cfg.stage_mode == "shared"

    def step(self, dt: float) -> None:
        """Queue this step's poses for the host.

        Args:
            dt: Simulation time-step in seconds.
        """
        if not self._is_initialized or not self._running:
            return
        self._sim_time += dt
        start = time.perf_counter()
        if self._shared is not None:
            self._step_shared(dt)
        else:
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
            self._host = self._desktop = self._provider = self._slot = None
            super().close()
