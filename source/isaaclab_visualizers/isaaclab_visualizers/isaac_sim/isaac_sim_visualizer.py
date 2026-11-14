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

from pxr import Sdf, Usd

from isaaclab.scene_data import SceneDataFormat
from isaaclab.sim import SimulationContext
from isaaclab.visualizers.base_visualizer import BaseVisualizer
from isaaclab.visualizers.visualizer_cfg import PerspectiveCameraCfg, VisualizerCfg

from .isaac_sim_shared_stage import Replication
from .isaac_sim_stage import authored_scales
from .isaac_sim_visualizer_cfg import IsaacSimBackendCfg

if TYPE_CHECKING:
    from isaaclab.sensors.camera import Camera

logger = logging.getLogger(__name__)

_ENVS_ROOT = "/World/envs"
"""Parent of Isaac Lab's cloned environments."""


class IsaacSimVisualizer(BaseVisualizer):
    """Isaac Sim 7 viewer application on an OVStage shared with Isaac Lab.

    The host's stage context creates and registers one OVStage; Isaac Lab borrows it and owns its ordinals. Each
    rendered frame, Isaac Lab writes body poses and visualization markers, applies the host's commands, seals the
    ordinal and renders the viewer's products with OVRTX. The host reads the same stage for its stage tree and property
    panes and streams the frames to its client. Isaac Lab keeps its own loop and physics.
    """

    def __init__(self, cfg: VisualizerCfg):
        """Initialize visualizer state.

        Args:
            cfg: Shared visualizer settings whose ``backend`` is an :class:`IsaacSimBackendCfg`.
        """
        super().__init__(cfg)
        self.backend_cfg: IsaacSimBackendCfg = cfg.backend
        if self.backend_cfg.app_client_root is None:
            self.backend_cfg.app_client_root = os.environ.get("ISAACLAB_ISAAC_SIM_APP_CLIENT_ROOT")
        self._resources = contextlib.ExitStack()
        self._host = None
        self._desktop = None
        self._provider = None
        self._shared = None
        self._markers = None
        self._paths: list[str] = []
        self._scales = None
        self._pose_output = SceneDataFormat.TransposedMatrix44d()
        self._armed = False
        self._next_render = 0.0
        self._running = False
        self.timings = {"publish": 0.0, "steps": 0}
        """Accumulated wall time [s] of the viewer's work in :meth:`step`, for benchmarking."""

    @property
    def client_url(self) -> str:
        """Address that opens the viewer in a browser, including the launch token."""
        return f"{self._host.url}/index.html?entry=index.js&token={self._host.token}"

    def initialize(self, sim: SimulationContext, *, cameras: list[PerspectiveCameraCfg | Camera]) -> None:
        """Populate the shared stage, start the Isaac Sim 7 host and open its client.

        Args:
            sim: Simulation owner that provides the authored stage and scene data.
            cameras: Resolved perspective settings and borrowed scene sensors, in display order.
        """
        if self._is_initialized:
            logger.debug("[IsaacSimVisualizer] initialize() called while already initialized.")
            return
        if not self.cfg.headless and self.backend_cfg.ui == "desktop" and self.backend_cfg.app_client_root is None:
            raise ValueError(
                "IsaacSimBackendCfg.app_client_root (or ISAACLAB_ISAAC_SIM_APP_CLIENT_ROOT) is required for"
                " ui='desktop'."
            )
        super().initialize(sim, cameras=cameras)

        from .isaac_sim_shared_markers import SharedMarkers

        self._provider = sim.get_scene_data_provider()
        self._paths = list(self._provider.backend.transform_paths)
        self._scales = authored_scales(sim.stage, self._paths)
        started = time.perf_counter()
        scene, replication = self._export_prototypes(sim)
        self.timings["export"] = time.perf_counter() - started
        self._markers = SharedMarkers()
        device = str(sim.device)
        device_index = int(device.split(":")[1]) if ":" in device else 0
        self._start_host(scene, replication, device_index)
        self._log_initialization_table(
            logger=logger,
            title="IsaacSimVisualizer Configuration",
            rows=[
                ("ui", "headless" if self.cfg.headless else self.backend_cfg.ui),
                ("bodies", len(self._paths)),
                ("fps", self.cfg.window.fps),
            ],
        )
        self._running = True
        self._is_initialized = True

    def _start_host(self, scene: Path, replication: Replication | None, device: int) -> None:
        """Run the Isaac Sim 7 application host, with its panes and transport controls, on the shared stage."""
        from sim_viewer.cli import client_roots_for_backend
        from sim_viewer.scene import CAMERA_SCOPE, RENDER_SCOPE, dynamic_viewport_spec, read_scene_metadata

        from isaacsim.application.host import AppHost
        from isaacsim.foundation.orchestration import RationalRate
        from isaacsim.viewports.presentation import FrameEncoder

        from .isaac_sim_shared_world import SharedWorldOpener

        started = time.perf_counter()
        metadata = read_scene_metadata(scene)
        specs: list = []
        rate = RationalRate(int(self.cfg.window.fps))
        self._shared = SharedWorldOpener(scene, specs, rate, device=device, replication=replication)
        self._resources.callback(self._shared.close)
        self.timings["shared_stage"] = time.perf_counter() - started
        client_roots = client_roots_for_backend("web")
        if self.backend_cfg.app_client_root is not None:
            client_roots = [("/app/", Path(self.backend_cfg.app_client_root) / "renderer"), *client_roots[1:]]
        if self.backend_cfg.camera_controls == "newton":
            from .isaac_sim_camera_controls import newton_controls_client

            directory = Path(self._resources.enter_context(tempfile.TemporaryDirectory(prefix="isaaclab_client_")))
            client_roots = [
                (prefix, newton_controls_client(root, directory) or root) if prefix == "/" else (prefix, root)
                for prefix, root in client_roots
            ]
        self._host = AppHost(
            open_world=self._shared,
            scene=str(scene),
            viewports=specs,
            viewport_factory=lambda index: dynamic_viewport_spec(index, tuple(self.cfg.window.size)),
            encoder=FrameEncoder.create("auto"),
            presentation=self.backend_cfg.presentation,
            renderer_name="ovrtx",
            hidden_roots=(CAMERA_SCOPE, RENDER_SCOPE),
            world_up=metadata.up_vector,
            host="127.0.0.1",
            port=self.backend_cfg.port,
            client_roots=client_roots,
            content_writes=True,
            stage_writes=True,
        )
        self._resources.callback(self._host.stop)
        # The host steps, and so publishes frames, only on request; each rendered frame requests one.
        self._shared.on_frame = self._host.simulation.request_redraw
        self._host.start()
        # Isaac Lab is already running, so the viewer's transport controls start in Playing.
        self._play()
        if not self.cfg.headless and self.backend_cfg.ui == "browser":
            self._log_viewer_url("IsaacSimVisualizer", self.client_url)
        elif not self.cfg.headless:
            self._open_desktop()

    def _open_desktop(self) -> None:
        """Open the Electron window attached to the host that this process runs."""
        from isaacsim.application.client import start_desktop

        self._desktop = start_desktop(
            Path(self.backend_cfg.app_client_root) / "main.cjs",
            environment={"SIM_VIEWER_ATTACH_URL": self._host.url, "SIM_VIEWER_ATTACH_TOKEN": self._host.token},
        )
        self._resources.callback(self._desktop.close)

    def _export_prototypes(self, sim: SimulationContext) -> tuple[Path, Replication | None]:
        """Export each asset once; the environments are cloned and placed in the stage.

        This is Isaac Lab's OVStage population path. With a clone plan, the copies follow it, so environments may hold
        different assets. Without one, every environment receives a copy of the first environment's children, and a
        scene without cloned environments is exported whole.

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
            if len(env_paths) > 1:
                source = env_paths[0]
                children = [child.GetName() for child in stage.GetPrimAtPath(source).GetChildren()]
                copies = [(f"{source}/{name}", [f"{env}/{name}" for env in env_paths[1:]]) for name in children]
                # Environment roots keep their authored transforms, so the copies land in place.
                replication = Replication(copies, env_paths, None)
                sources = tuple(source for source, _ in copies)
            else:
                replication, sources = None, ()
            usd = export_stage_to_string(stage, env_paths, sources, keep_env_roots=True, preserve_references=True)
        layer = Sdf.Layer.CreateAnonymous(".usda")
        layer.ImportFromString(usd)
        exported = Usd.Stage.Open(layer)
        for prim in exported.Traverse():
            # The viewer's stage tree omits typeless containers, which Isaac Lab authors.
            if not prim.GetTypeName():
                prim.SetTypeName("Xform")
        directory = Path(self._resources.enter_context(tempfile.TemporaryDirectory(prefix="isaaclab_isaac_sim_")))
        scene = directory / "scene.usdc"
        exported.Export(str(scene))
        return scene, replication

    def _render_frame(self, dt: float) -> None:
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
        first = not self._armed
        markers = self._markers if self.supports_markers() else None
        registry = self._sim.vis_marker_registry if markers is not None else None

        def write_state(stage):
            if markers is not None:
                markers.sync(stage, registry)
            if first:
                # Bodies carry world matrices: their local transform is their world transform.
                stage.write(paths, "omni:resetXformStack", np.ones(len(paths), dtype=np.bool_))
                # Armed only once written: the step skips writing while a render is in flight.
                self._armed = True
            # The stage's hierarchy derives world matrices from ``omni:xform`` when it seals, so poses go there. The
            # matrices stay on the device; ovstage copies them in on the producing stream.
            stage.write_device(paths, "omni:xform", xform_tensor_from_warp(matrices), stream)

        self._shared.step(int(self._sim_time * 1e9), int(dt * 1e9), write_state)
        self.timings["renders"] = self._shared.renders

    def _play(self) -> None:
        """Put the viewer's transport controls in Playing."""
        from isaacsim.application.host import Command, TimeTarget

        self._host.inbox.submit(Command(verb="play", args={}, time_target=TimeTarget.immediate()))

    def is_training_paused(self) -> bool:
        """Return whether the viewer's Pause holds Isaac Lab's loop; each Step lets one simulation step through."""
        if self._host is None or self._shared is None:
            return False
        if self._host.simulation.run_state.value != "paused":
            self._shared.clear_steps()
            return False
        return not self._shared.take_step()

    def supports_markers(self) -> bool:
        """Return whether visualization markers are drawn."""
        return self.cfg.enable_markers

    def step(self, dt: float) -> None:
        """Update the shared stage and the viewer's frame.

        Args:
            dt: Simulation time-step in seconds.
        """
        if not self._is_initialized or not self._running:
            return
        self._sim_time += dt
        start = time.perf_counter()
        if self._shared is not None and self._shared.take_reset():
            # The viewer's Stop resets the environments, which then keep running.
            self._reset_requested = True
            self._play()
        self._render_frame(dt)
        if dt == 0.0:
            # Isaac Lab calls this in a loop while the viewer is paused; keep serving the viewer without spinning.
            time.sleep(0.002)
        self.timings["publish"] += time.perf_counter() - start
        self.timings["steps"] += 1

    def is_running(self) -> bool:
        """Return whether the viewer is still open; closing the desktop window ends it."""
        if self._desktop is not None and self._desktop.returncode is not None:
            self._running = False
        return self._running

    def close(self) -> None:
        """Stop the host, release the shared stage and remove the exported scene."""
        if self._is_closed:
            return
        try:
            self._resources.close()
        finally:
            self._running = False
            self._host = self._desktop = self._provider = self._shared = None
            super().close()
