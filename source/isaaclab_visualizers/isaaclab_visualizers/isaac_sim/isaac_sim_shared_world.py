# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Sim 7 host worlds on a stage that Isaac Lab owns.

:class:`SharedWorldOpener` is the ``open_world`` callable of the host. Every world it opens, including after the user
stops the scene, uses the same shared stage and runs no scheduler. Isaac Lab calls :meth:`step` once per frame: it
writes simulation state, applies the host's queued commands (camera moves, viewport changes, property reads), seals the
ordinal and renders the viewer's products at it with OVRTX. The host's transport controls are recorded for Isaac Lab:
Pause and Step gate its loop, and Stop requests an environment reset.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .isaac_sim_host_adapter import (
    BorrowedContext,
    CloneAuthoring,
    HostClock,
    HostData,
    StageView,
    live_inspection_type,
)
from .isaac_sim_shared_stage import FrameStore, Replication, SharedStage, ViewerRenderer

if TYPE_CHECKING:
    from isaacsim.application.host import ViewportSpec, World
    from isaacsim.foundation.orchestration import RationalRate

logger = logging.getLogger(__name__)


def _shared_world_type() -> type:
    from isaacsim.application.host import World

    class SharedWorld(World):
        """Host world whose queued work Isaac Lab drains on its own steps.

        The host loop steps a stopped world while it has pending work. Here only Isaac Lab drains it, so the loop must
        not spin on it; it steps when Isaac Lab requests a redraw after rendering. A simulation step (Play or Step) is
        recorded as a grant for Isaac Lab's loop instead of advancing a clock.
        """

        on_simulation_step: Callable[[], None] | None = None

        def has_pending(self) -> bool:
            return False

        def set_running(self, running: bool) -> None:
            self.running = running

        def step_simulation_frame(self) -> int:
            # While playing, the host steps at its own rate; Isaac Lab runs freely then, so only a Step while paused
            # is a grant.
            if not self.running and self.on_simulation_step is not None:
                self.on_simulation_step()
            self.frames_stepped += 1
            return self.sim_time_ns

    return SharedWorld


class SharedWorldOpener:
    """Create the shared stage and open host worlds on it.

    Args:
        scene: Exported scene file the stage is populated from; the host's authoring session edits it.
        specs: Viewport slots, shared with the host.
        rate: Host frame rate.
        device: CUDA device ordinal for rendering.
        replication: Environment copies and placement applied once the stage is populated.
    """

    def __init__(
        self,
        scene: Path,
        specs: Sequence[ViewportSpec],
        rate: RationalRate,
        device: int = 0,
        replication: Replication | None = None,
    ):
        from sim_viewer.scene import read_scene_metadata, root_layer

        from isaacsim.viewports.authoring import AuthoringSession

        self._specs = specs
        self._rate = rate
        self._copies = replication.copies if replication is not None else ()
        self.metadata = read_scene_metadata(scene)
        started = time.perf_counter()
        self.authoring = AuthoringSession(str(scene.resolve()))
        self.startup = {"authoring": time.perf_counter() - started}
        """Wall time [s] of each startup phase, for benchmarking."""
        layer = root_layer(scene, list(specs), self.metadata)
        self.stage = SharedStage(self.authoring.compose(layer), replication)
        self.startup.update(self.stage.startup)
        started = time.perf_counter()
        self.renderer = ViewerRenderer(self.stage, device)
        self.startup["renderer"] = time.perf_counter() - started
        self.frames = FrameStore()
        self._lock = threading.Lock()
        self._world: World | None = None
        self._data: HostData | None = None
        self._activation = 0
        self._opened = 0
        self._step_grants = 0
        self._reset_requested = False
        self._revision: int | None = None
        self.renders = 0
        self.timings = {"write": 0.0, "host": 0.0, "seal": 0.0, "render": 0.0, "steps": 0}
        """Accumulated wall time [s] per phase of :meth:`step`, for benchmarking."""
        self.on_frame: Callable[[], None] | None = None
        """Tells the host a new frame exists; its publishers only read frames after a host step."""

    def step(self, time_ns: int, elapsed_ns: int, write_state: Callable[[SharedStage], None]) -> int | None:
        """Publish a finished frame, then, if the renderer is idle, advance the stage by one sealed ordinal and render.

        Args:
            time_ns: Isaac Lab's simulation time [ns].
            elapsed_ns: Time since the previous step [ns].
            write_state: Writes Isaac Lab's state at the open ordinal.

        Returns:
            The sealed ordinal, or ``None`` while a render is in flight and nothing was written.
        """
        clock = time.perf_counter
        published, busy = self.renderer.poll(self.frames)
        self.renders += published
        if published and self.on_frame is not None:
            self.on_frame()
        if busy:
            return None
        with self._lock:
            world, data = self._world, self._data
        t0 = clock()
        write_state(self.stage)
        t1 = clock()
        sources = [world.commands, *world.command_sources] if world is not None else []
        if data is not None:
            data.time_ns = time_ns
            self._activation += 1
            view = StageView(data)
            for source in sources:
                try:
                    source._capture_activation(view, self._activation)
                except Exception:  # noqa: BLE001 - a viewer command must not stop Isaac Lab's simulation
                    logger.exception("An Isaac Sim viewer command failed; continuing.")
        t2 = clock()
        ordinal = self.stage.seal()
        t3 = clock()
        for source in sources:
            # Resolves the commands' futures now that their writes are sealed.
            source._published()
        self.renderer.render(ordinal, time_ns)
        t4 = clock()
        for key, value in (("write", t1 - t0), ("host", t2 - t1), ("seal", t3 - t2), ("render", t4 - t3)):
            self.timings[key] += value
        self.timings["steps"] += 1
        return ordinal

    def __call__(self, scene: str, wake: Callable[[], None]) -> World:
        """Open a host world on the shared stage."""
        from isaacsim.application.host import AuthoringSync, SceneCommands
        from isaacsim.foundation.orchestration import RationalRate
        from isaacsim.viewports.adapters import RendererOutput

        world_type = _shared_world_type()
        context = BorrowedContext(self.stage.context)
        data = HostData(context, self.stage)
        commands = SceneCommands(rate=self._rate, on_submit=wake)
        world = world_type(
            context,
            data,
            commands,
            self._rate,
            scene,
            clock=HostClock(data),
            simulation_participants=[],
            up=self.metadata.up_vector,
            meters_per_unit=self.metadata.meters_per_unit,
            on_external_submit=wake,
        )
        world.on_simulation_step = self._grant_step
        world.renderer_name = "ovrtx"
        world.authoring = CloneAuthoring(self.authoring, self._copies)
        world.authoring_revision = self.authoring.revision
        world.authoring_sync = AuthoringSync(self.authoring.stage)
        # Property inspection supplies the selected prims' live world poses; Isaac Lab drains it with the commands.
        inspection = SceneCommands(rate=RationalRate(10), on_submit=wake)
        world.inspection = live_inspection_type()(inspection, set(), context)
        world.command_sources.append(inspection)
        frames = self.frames
        world.renderer_output_factory = lambda path, product: RendererOutput(read=lambda: frames.latest(product))
        self._attach_viewports(world)
        with self._lock:
            self._world, self._data = world, data
            self._opened += 1
            # The host reopens its world when the user stops the scene, and also to apply authored edits while playing.
            # Only Stop resets Isaac Lab's environments; an edit changes the authoring revision.
            revision = self.authoring.revision
            self._reset_requested = self._opened > 1 and revision == self._revision
            self._revision = revision
        return world

    def _grant_step(self) -> None:
        with self._lock:
            self._step_grants += 1

    def take_step(self) -> bool:
        """Consume one step the user requested while paused; return whether one was pending."""
        with self._lock:
            if self._step_grants:
                self._step_grants -= 1
                return True
            return False

    def clear_steps(self) -> None:
        """Drop steps granted while the host was playing; they only matter while it is paused."""
        with self._lock:
            self._step_grants = 0

    def take_reset(self) -> bool:
        """Consume a reset that the host's Stop requested; return whether one was pending."""
        with self._lock:
            requested, self._reset_requested = self._reset_requested, False
            return requested

    def _attach_viewports(self, world: World) -> None:
        """Create client viewports by cloning the template subtree at Isaac Lab's open ordinal."""
        from sim_viewer.scene import CAMERA_SCOPE

        renderer = self.renderer
        lock = threading.RLock()
        ready: set[str] = set()
        for spec in self._specs:
            renderer.add_product(spec.product)
            ready.add(spec.product)

        def create_viewport(spec: ViewportSpec) -> Future:
            root = spec.camera.rsplit("/", 1)[0]
            with lock:
                if spec.product in ready:
                    done: Future = Future()
                    done.set_result(None)
                    return done
                world.close_renderer_outputs(spec.product)
                cloned = world.commands.submit(
                    root, "viewport_topology", lambda view: view.clone_prim(f"{CAMERA_SCOPE}/Template", [root])
                )

                def register(view: Any) -> None:
                    cloned.result()
                    view.write([spec.product], "resolution", np.asarray([spec.resolution], dtype=np.int32))
                    with lock:
                        renderer.add_product(spec.product)
                        ready.add(spec.product)

                return world.commands.submit(root + "/ready", "viewport_topology", register)

        def remove_viewport(spec: ViewportSpec) -> None:
            root = spec.camera.rsplit("/", 1)[0]
            with lock:
                ready.discard(spec.product)
                renderer.remove_product(spec.product)
                world.commands.submit(root, "viewport_topology", lambda view: view.remove_prim(root))
                world.close_renderer_outputs(spec.product)

        world.viewport_specs = tuple(self._specs)
        world.create_viewport = create_viewport
        world.remove_viewport = remove_viewport
        world.set_render_product_enabled = renderer.set_product_enabled

    def close(self) -> None:
        """Release the renderer, then the stage."""
        try:
            self.renderer.close()
        finally:
            self.stage.close()
