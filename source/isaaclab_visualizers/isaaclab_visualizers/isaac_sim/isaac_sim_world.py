# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""World opener for the Isaac Sim 7 viewer host, without an Isaac Sim physics participant.

Adapted from ``sim_viewer/scene.py`` in omni_isaac_sim (Apache-2.0, NVIDIA CORPORATION & AFFILIATES). The only
change is that no physics participant is attached: Isaac Lab owns simulation and queues its poses into the world.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sim_viewer.scene import (
    CAMERA_SCOPE,
    CAPTURE_POSITION,
    CLOCK_POSITION,
    MANIPULATION_POSITION,
    VIEWPORT_POSITION,
    read_scene_metadata,
    root_layer,
)

from isaacsim.application.host import NativeParticipant, ViewportSpec, World

if TYPE_CHECKING:
    from isaacsim.foundation.orchestration import RationalRate

__all__ = ["make_physics_free_world_opener"]


def make_physics_free_world_opener(
    specs: Sequence[ViewportSpec],
    rate: RationalRate,
    *,
    renderer: str = "ovrtx",
    device_frames: bool = False,
    extra_participants: Callable[[Callable[[], None]], Sequence[NativeParticipant]] = lambda wake: (),
) -> Callable[[str, Callable[[], None]], World]:
    """Return the function the host uses to open a world for a scene.

    The world has a capture participant at :data:`CAPTURE_POSITION` and a simulation clock at :data:`CLOCK_POSITION`.
    Renderer-independent live pose commands run before physics. Physics and the clock
    suspend together while paused; renderer and capture activations may still run.

    Args:
        specs: Viewport slots.
        rate: Activation rate of capture and rendering.
        renderer: Backend participant, ovgl or ovrtx.
        device_frames: Retain OVRTX CUDA pixels for GPU-capable presentation providers.
        extra_participants: Creates further participants for each new world, such as an external pose producer,
            given the callback that wakes the host when they have new output.

    Returns:
        ``open_world(scene, wake) -> World``.
    """
    from isaacsim.foundation.orchestration import RationalRate
    from isaacsim.viewports.authoring import AuthoringSession

    active_scene: str | None = None
    active_authoring: AuthoringSession | None = None

    def open_world(scene: str, wake: Callable[[], None]) -> World:
        """Open the scene with the selected renderer and viewer authoring participant.

        Args:
            scene: Scene identifier or file path.
            wake: Callback waking the host after new work arrives.

        Returns:
            The initialized simulation world.
        """
        nonlocal active_scene, active_authoring

        import isaacsim.foundation.orchestration as foundation_orchestration
        from isaacsim.application.host import (
            AuthoringSync,
            GizmoCapture,
            PropertyInspection,
            RigidBodyPoseBackend,
            SceneCommands,
        )
        from isaacsim.common.interfaces import Query
        from isaacsim.context.simulation import SimulationContext
        from isaacsim.viewports.authoring import LiveManipulation

        if renderer == "ovgl":
            from isaacsim.rendering import ovgl as backend
        elif renderer == "ovrtx":
            from isaacsim.rendering import ovrtx as backend
        else:
            raise ValueError(f"Unknown renderer: {renderer}")

        initial_specs = tuple(specs)
        path = Path(scene)
        metadata = read_scene_metadata(path)
        key = str(path.resolve())
        # Simulation resets retain edits; switching scenes starts a fresh edit layer.
        authoring = active_authoring if key == active_scene and active_authoring is not None else AuthoringSession(key)
        wrapper = root_layer(path, initial_specs, metadata)
        text = authoring.compose(wrapper)
        scene_products: list[str] = []
        attached = []

        def discover_render_products(simulation_context: SimulationContext) -> None:
            """Discover scene products from the loaded stage before renderer registration.

            Args:
                simulation_context: Finalized context owning the loaded runtime stage.
            """
            viewport_products = {spec.product for spec in initial_specs} | {f"{CAMERA_SCOPE}/Template/Product"}
            products = simulation_context.read(Query().in_("usd-prim-type", ["RenderProduct"]), "usd-path")
            scene_products.extend(sorted(set(products) - viewport_products))

        def participants() -> list[NativeParticipant]:
            """Create and retain the renderer for this world’s lifetime.

            Returns:
                Renderer and capture participants retained by the world.
            """
            config = backend.OvglRendererConfig() if renderer == "ovgl" else backend.OvrtxRendererConfig()
            if renderer == "ovgl":
                # This application serves both Electron textures and browser/recording pixels.
                config.gpu_presentation = True
                config.cpu_readback = True
            config.render_products = [spec.product for spec in initial_specs] + scene_products
            config.rate = foundation_orchestration.RationalRate(rate.numerator, rate.denominator)
            if renderer == "ovrtx" and device_frames:
                config.device = backend.FrameDevice.CUDA
            participant = backend.make_renderer(config)
            attached.append(participant)
            return [*extra_participants(wake), NativeParticipant("viewport", participant, VIEWPORT_POSITION)]

        world = World.open(
            text,
            scene=str(path),
            rate=rate,
            participants=participants,
            prepare_simulation_context=discover_render_products,
            capture_position=CAPTURE_POSITION,
            clock_position=CLOCK_POSITION,
            on_capture_submit=wake,
            up=metadata.up_vector,
            meters_per_unit=metadata.meters_per_unit,
        )
        from isaacsim.application.host import OvglPicker, OvrtxPicker

        world.renderer_selection = attached[0]
        world.picker = OvglPicker(attached[0]) if renderer == "ovgl" else OvrtxPicker()
        world.renderer_name = renderer
        world.scene_render_products = tuple(scene_products)
        world.authoring = authoring
        world.authoring_revision = authoring.revision
        world.authoring_sync = AuthoringSync(authoring.stage)
        from isaacsim.application.host.impl.selection_manipulation import SelectionManipulation

        world.manipulation = SelectionManipulation(
            LiveManipulation(RigidBodyPoseBackend.from_stage(authoring.stage, world)), authoring
        )
        worker = SceneCommands(rate=rate, on_submit=wake)
        world.simulation.attach_participant("manipulation", worker.capture, position=MANIPULATION_POSITION)

        def publish_live(view: object) -> dict:
            try:
                return world.manipulation.tick(view)
            except Exception as error:
                world.simulation.fail(f"Live pose publication failed; reset required: {error}")
                raise

        worker.add_reader("holds", publish_live)

        world.command_sources.append(worker)
        world.manipulation_worker = worker
        inspection_worker = SceneCommands(rate=RationalRate(10), on_submit=wake)
        world.simulation.attach_participant(
            "property-inspection", inspection_worker.capture, position=MANIPULATION_POSITION + 1
        )
        world.inspection = PropertyInspection(
            inspection_worker, world.authoring_sync.body_paths, world.simulation_context
        )

        world.command_sources.append(inspection_worker)
        from isaacsim.viewports.adapters import ContextOutput

        def output_factory(path: str, product: str) -> ContextOutput:
            # The publication carries its optional native-image reference.
            return ContextOutput(world.simulation, path, product=product)

        capture = GizmoCapture(
            [spec.product for spec in initial_specs] + scene_products,
            world.authoring_sync.body_paths,
            metadata.meters_per_unit,
            authoring=authoring,
        )

        world.gizmo_capture = capture

        def set_render_product_enabled(product: str, enabled: bool) -> None:
            """Apply the same viewport demand to rendering and overlay calibration.

            Args:
                product: Render product path.
                enabled: Whether the viewport needs frames.
            """
            with lifecycle_lock:
                if product in ready:
                    attached[0].set_product_enabled(product, enabled)
                    capture.set_product_enabled(product, enabled)

        world.set_render_product_enabled = set_render_product_enabled
        desired = {spec.product for spec in initial_specs}
        ready = set(desired)
        lifecycle_lock = threading.RLock()

        def create_viewport(spec: ViewportSpec) -> Any:
            """Clone one camera/product subtree through the capture participant's publication boundary.

            Args:
                spec: Viewport subtree to create.

            Returns:
                Future completed after the clone is published and the product registered.
            """
            from concurrent.futures import Future

            root = spec.camera.rsplit("/", 1)[0]
            with lifecycle_lock:
                desired.add(spec.product)
                if spec.product in ready:
                    done = Future()
                    done.set_result(None)
                    return done
                # A recreated product must not reuse an output, and its retained frame, from a removed one.
                world.close_renderer_outputs(spec.product)
                cloned = world.commands.submit(
                    root, "viewport_topology", lambda view: view.clone_prim(f"{CAMERA_SCOPE}/Template", [root])
                )

                def register(view: Any) -> None:
                    cloned.result()
                    with lifecycle_lock:
                        if spec.product in desired:
                            view.write([spec.product], "resolution", np.asarray([spec.resolution], dtype=np.int32))
                            attached[0].add_product(spec.product)
                            capture.add_product(spec.product)
                            ready.add(spec.product)

                # Sharing the column with clone forces registration into the next activation.
                return world.commands.submit(root + "/ready", "viewport_topology", register)

        def remove_viewport(spec: ViewportSpec) -> None:
            """Release rendering immediately and publish removal of the runtime subtree.

            Args:
                spec: Viewport subtree to retire.
            """
            root = spec.camera.rsplit("/", 1)[0]
            with lifecycle_lock:
                desired.discard(spec.product)
                ready.discard(spec.product)
                attached[0].remove_product(spec.product)
                capture.remove_product(spec.product)
                world.commands.submit(root, "viewport_topology", lambda view: view.remove_prim(root))
                # Retire the prims even if an output fails to close; report that failure afterwards.
                world.close_renderer_outputs(spec.product)

        world.viewport_specs = initial_specs
        world.create_viewport = create_viewport
        world.remove_viewport = remove_viewport
        capture_worker = SceneCommands(rate=rate, on_submit=wake)
        world.simulation.attach_participant("gizmo-capture", capture_worker.capture, position=VIEWPORT_POSITION - 1)
        capture_worker.add_reader("calibration", capture.capture)

        world.command_sources.append(capture_worker)

        # Gizmo capture composites overlays into each output the world creates and caches.
        world.renderer_output_factory = lambda path, product: capture.wrap(output_factory(path, product), product, path)
        # Keep the previous session available for fallback until the new world opens successfully.
        active_scene, active_authoring = key, authoring
        return world

    return open_world
