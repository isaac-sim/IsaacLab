# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton physics manager for Isaac Lab."""

from __future__ import annotations

import contextlib
import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp
from newton import Axis, ModelBuilder, ModelFlags, State
from newton.selection import ArticulationView
from newton.sensors import SensorContact, SensorIMU

from pxr import UsdGeom

import isaaclab.cloner as cloner
from isaaclab.physics import CallbackHandle, PhysicsEvent, PhysicsManager
from isaaclab.scene_data import SceneDataBackend, SceneDataProvider
from isaaclab.sim import SimulationContext
from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.timer import Timer
from isaaclab.utils.version import has_kit

from isaaclab_newton.physics.newton_backend_cfg import NewtonBackendCfg
from isaaclab_newton.physics.newton_builder_cfg import NewtonBuilderCfg
from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg
from isaaclab_newton.renderers.visual_material import (
    VisualMaterialWriter,
    VisualShapeColorWriter,
)

from .newton_backend import NewtonBackend, NewtonCloneRecord, SiteEntry
from .newton_builder import create_builder
from .newton_scene_data import NewtonSceneDataBackend

if TYPE_CHECKING:
    from isaaclab.assets import BaseArticulation
    from isaaclab.renderers.base_renderer import VisualMaterialBatch


logger = logging.getLogger(__name__)


def create_newton_builder(cfg: NewtonBuilderCfg) -> ModelBuilder:
    """Construct the shared builder: solver-aware under Newton physics, plain otherwise.

    Args:
        cfg: The selected physics configuration.

    Returns:
        An empty builder.
    """
    if isinstance(cfg.physics_cfg, NewtonCfg):
        return create_builder(cfg.physics_cfg)
    return ModelBuilder()


class NewtonManager(PhysicsManager):
    """Newton physics manager for Isaac Lab.

    Runs one :class:`NewtonBackend` through the physics lifecycle and keeps what survives hard resets: the builder,
    site requests, replication outputs and decimation.

    Lifecycle: ``initialize() -> reset() -> step()`` (repeated) ``-> close()``.
    """

    backend_name = "newton"

    def __init__(self, cfg: NewtonCfg):
        super().__init__(cfg)
        self._cfg: NewtonCfg = cfg
        self.builder: ModelBuilder | None = None
        """Shared construction builder, acquired at :meth:`initialize` and kept across hard resets."""
        self.backend: NewtonBackend | None = None
        """Data bound to the finalized model; replaced on every hard reset."""
        self.clear()
        self._scene_data_backend = NewtonSceneDataBackend(self)

    @property
    def shares_materials(self) -> bool:
        """Whether contacts read one material table shared by all environments instead of per-shape values."""
        solver_cfg = self._cfg.solver_cfg
        return solver_cfg.class_type.shares_materials(solver_cfg)

    @classmethod
    def provides_implicit_damping(cls) -> bool:
        # Newton's symplectic integrator has no implicit damping.
        return False

    @classmethod
    def setup_deformable_body(cls, prim: Any, deformable_type: str, sim_mesh_prim: Any, vis_mesh_prim: Any) -> None:
        """Apply Newton's token deformable anchor schemas and sync the visual mesh geometry."""
        sim_mesh_path = sim_mesh_prim.GetPath().pathString
        token = "PhysicsVolumeDeformableSimAPI" if deformable_type == "volume" else "PhysicsSurfaceDeformableSimAPI"
        if not sim_mesh_prim.AddAppliedSchema(token):
            raise RuntimeError(f"Failed to set {deformable_type} deformable sim API on prim '{sim_mesh_path}'.")
        # Newton renders the simulation mesh directly: overwrite the visual mesh geometry until
        # separate visual/simulation meshes are supported.
        vis_mesh = UsdGeom.Mesh(vis_mesh_prim)
        if deformable_type == "volume":
            tet_mesh = UsdGeom.TetMesh(sim_mesh_prim)
            surface_indices = tet_mesh.GetSurfaceFaceVertexIndicesAttr().Get()
            if surface_indices is None or len(surface_indices) == 0:
                raise ValueError(
                    f"Deformable body at '{prim.GetPath().pathString}' has no surface indices on its TetMesh"
                    " prim; cannot sync to visual mesh."
                )
            vis_mesh.GetPointsAttr().Set(tet_mesh.GetPointsAttr().Get())
            vis_mesh.GetFaceVertexIndicesAttr().Set(np.asarray(surface_indices).flatten())
            vis_mesh.GetFaceVertexCountsAttr().Set([3] * len(surface_indices))
        else:
            sim_mesh = UsdGeom.Mesh(sim_mesh_prim)
            vis_mesh.GetFaceVertexIndicesAttr().Set(sim_mesh.GetFaceVertexIndicesAttr().Get())
            vis_mesh.GetFaceVertexCountsAttr().Set(sim_mesh.GetFaceVertexCountsAttr().Get())
        if not prim.AddAppliedSchema("PhysicsDeformableBodyAPI"):
            raise RuntimeError(f"Failed to set deformable body API on prim '{prim.GetPath().pathString}'.")

    def initialize(self, sim_context: SimulationContext) -> None:
        """Initialize the manager with simulation context.

        Args:
            sim_context: Parent simulation context.
        """
        super().initialize(sim_context)
        self.builder = sim_context.get_or_create_backend(NewtonBuilderCfg(physics_cfg=self._cfg))

        # This context imports NewtonManager, so it can only be imported after this module initializes.
        from isaaclab_newton.cloner import NewtonReplicateContext  # noqa: PLC0415

        self.clone_context_type = NewtonReplicateContext

    def reset(self, soft: bool = False) -> None:
        """Reset physics simulation.

        A hard reset (``soft=False``) closes the backend, with its solver, contacts, step callbacks and captured graph,
        and finalizes a new one. Consumers bind to it again on ``PHYSICS_READY``; reacquire ``asset.data`` and its
        arrays afterwards. A soft reset keeps the backend.

        Args:
            soft: If True, skip full reinitialization.
        """
        if not soft:
            self._close_backend()
            self.start_simulation()
            self.initialize_solver()

    def _close_backend(self, *, dispatch_stop: bool = True) -> None:
        """Stop consumers, drop their views, and close the backend."""
        backend = self.backend
        if backend is None:
            return
        if dispatch_stop:
            self.dispatch_event(PhysicsEvent.STOP)
        self.views.clear()
        self.backend = None
        SimulationContext.instance().close_backend(backend)

    def forward(self) -> None:
        """Update articulation kinematics without stepping physics; see :meth:`NewtonBackend.forward`."""
        if self.backend is not None:
            self.backend.forward()

    def video_capture_backend(self) -> str:
        """Newton GL headless perspective video capture."""
        return "newton_gl"

    def _mark_transforms_changed(self) -> None:
        """Publish authored rigid-body changes and invalidate cable geometry."""
        self._scene_data_backend.transforms_timestamp += 1
        self._scene_data_backend.geometry_timestamp += 1
        self._flag_outer_capture()

    def _mark_particles_dirty(self) -> None:
        """Invalidate SDP geometry after native particle writes."""
        self._scene_data_backend.geometry_timestamp += 1
        self._flag_outer_capture()

    def _flag_outer_capture(self) -> None:
        """Flag writes recorded into an outer CUDA graph capture."""
        if self.backend.device.is_capturing:
            self.backend.transforms_may_change_on_graph_replay = True

    def step(self) -> None:
        """Advance physics by one environment step, or one physics step when the environment runs decimation."""
        sim = self._sim
        if sim is None or not sim.is_playing():
            return
        backend = self.backend
        physics_steps = backend.decimation if backend.handles_decimation() else 1
        captured = backend.step()
        self._sim_time += backend.dt * physics_steps
        self._mark_transforms_changed()
        solver_cfg = self._cfg.solver_cfg
        solver_cfg.class_type.check_step(solver_cfg, backend.solver, captured, self._cfg.debug_mode)

    def close(self) -> None:
        """Clean up Newton physics resources."""
        super().close()
        self.clear()
        self.builder = None

    def get_scene_data_backend(self) -> SceneDataBackend | None:
        """Return the SceneDataBackend for the SceneDataProvider."""
        return self._scene_data_backend

    def register_callback(
        self,
        callback: Callable,
        event: PhysicsEvent,
        order: int = 0,
        name: str | None = None,
        wrap_weak_ref: bool = True,
    ) -> CallbackHandle:
        """Register a callback. Passes event to parent class."""
        return super().register_callback(callback, event, order, name, wrap_weak_ref)

    def get_physics_sim_view(self) -> list:
        """Return the registered articulation views."""
        return list(self.views.values())

    def is_fabric_enabled(self) -> bool:
        """Check if fabric interface is enabled (not applicable for Newton)."""
        return False

    def clear(self):
        """Clear all Newton-specific state (callbacks cleared by super().close())."""
        self._close_backend(dispatch_stop=False)
        self._decimation = 1
        # Site requests and replication outputs survive hard resets.
        # (body_pattern, per_world, xform floats) -> (label, transform)
        self._site_requests: dict[tuple[str | None, bool, tuple[float, ...]], tuple[str, wp.transform]] = {}
        self._site_index_map: dict[str, SiteEntry] = {}
        self._clone: NewtonCloneRecord | None = None
        self.views.clear()

    def register_site(self, body_pattern: str | None, xform: wp.transform, *, per_world: bool = False) -> str:
        """Request a site for injection into prototypes before replication.

        Sensors call this during ``__init__``. Identical ``(body_pattern, per_world, transform)`` requests share a site,
        and requests persist until :meth:`close`, so sites survive hard resets without being requested again.
        The pattern matches prototype-local body labels (e.g. ``"Robot/finger.*"``) during replication, and wildcard
        patterns create one site per matched body.

        Args:
            body_pattern: Regex matched against body labels, or ``None`` for a global site.
            xform: Site transform relative to the body.
            per_world: Create one bodyless site in each cloned world's frame. Requires ``body_pattern=None``.

        Returns:
            The assigned site label.
        """
        if per_world and body_pattern is not None:
            raise ValueError("per_world site registration requires body_pattern=None.")
        requests = self._site_requests
        key = (body_pattern, per_world, tuple(xform))
        if key not in requests:
            requests[key] = (f"ft_{len(requests)}", xform)
        return requests[key][0]

    def inject_sites(
        self, main_builder: ModelBuilder, source_builders: dict[str, ModelBuilder]
    ) -> tuple[dict[str, int], dict[int, dict[str, list[int]]], dict[str, wp.transform]]:
        """Add unresolved sites to source builders, or to the main builder for global and shared-asset sites.

        Args:
            main_builder: Top-level builder that receives global sites.
            source_builders: Source builders by source path.

        Returns:
            ``(global_site_indices, source_site_indices, env_root_sites)``: global site shape indices by label,
            builder-local site indices by ``id(builder)`` and label, and world-root site transforms by label.
        """
        global_sites: dict[str, int] = {}
        source_sites: dict[int, dict[str, list[int]]] = {}
        root_sites: dict[str, wp.transform] = {}
        for (body_pattern, per_world, _), (label, xform) in self._site_requests.items():
            if label in self._site_index_map:
                continue
            if per_world:
                root_sites[label] = xform
                continue
            if body_pattern is None:
                global_sites[label] = main_builder.add_site(body=-1, xform=xform, label=label)
                continue
            matched_any = False
            for builder in (*source_builders.values(), main_builder):
                if builder is main_builder and matched_any:
                    break
                indices, names = resolve_matching_names(body_pattern, builder.body_label, raise_when_no_match=False)
                if not indices:
                    continue
                matched_any = True
                sites = [
                    builder.add_site(body=body, xform=xform, label=f"{name}/{label}")
                    for body, name in zip(indices, names)
                ]
                source_sites.setdefault(id(builder), {})[label] = sites
            if not matched_any:
                raise ValueError(f"Site '{label}' with body_pattern '{body_pattern}' matched no builder bodies.")
        return global_sites, source_sites, root_sites

    def record_clone(self, record: NewtonCloneRecord, site_index_map: dict[str, SiteEntry]) -> None:
        """Record native replication outputs for model finalization and consumers.

        Args:
            record: Replication outputs.
            site_index_map: Sites resolved during replication, by label.
        """
        self._clone = record
        self._site_index_map.update(site_index_map)

    def get_site_index_map(self) -> dict[str, SiteEntry]:
        """Return resolved sites by label."""
        return self._site_index_map

    def get_world_xforms(self) -> list[wp.transform] | None:
        """Return the root transform of each cloned world, or ``None`` without replication."""
        clone = self._clone
        return None if clone is None else clone.world_xforms

    def get_clone_source_builders(self) -> dict[str, ModelBuilder]:
        """Return per-source builders retained from replication, keyed by clone-plan source path."""
        clone = self._clone
        return {} if clone is None else clone.source_builders

    def request_extended_state_attribute(self, attr: str) -> None:
        """Request an extended state attribute (e.g. ``"body_qdd"``) from the builder before finalization.

        Args:
            attr: State attribute name (must be in ``State.EXTENDED_ATTRIBUTES``).
        """
        self.builder.request_state_attributes(attr)

    def request_extended_contact_attribute(self, attr: str) -> None:
        """Request an extended contact attribute (e.g. ``"force"``) from the builder before finalization.

        Args:
            attr: Contact attribute name.
        """
        self.builder.request_contact_attributes(attr)

    def add_model_change(self, change: ModelFlags) -> None:
        """Register a model change to notify the solver at the next step."""
        self.backend.model_changes.add(change)

    def invalidate_fk(
        self,
        env_mask: wp.array | None = None,
        env_ids: wp.array | None = None,
        articulation_ids: wp.array | None = None,
    ) -> None:
        """Mark environments as needing FK recomputation and solver reset; see :meth:`NewtonBackend.invalidate_fk`.

        Called by asset write methods that modify joint coordinates or root transforms. The masks are consumed by the
        next forward, raw-state, rendering, or physics-step boundary.
        """
        self._mark_transforms_changed()
        self.backend.invalidate_fk(env_mask, env_ids, articulation_ids)

    def invalidate_body_state(
        self, env_ids: wp.array(dtype=wp.int32) | None = None, env_mask: wp.array(dtype=wp.bool) | None = None
    ) -> None:
        """Mark selected maximal-coordinate body state as changed without requesting FK.

        See :meth:`NewtonBackend.invalidate_body_state`.
        """
        self._mark_transforms_changed()
        self.backend.invalidate_body_state(env_ids, env_mask)

    def _drain_stale_cuda_error(self) -> None:
        """Clear a stale CUDA error latched on the device before (re)initialization.

        Warp 1.15 leaves the per-thread CUDA error uncleared when
        ``wp_free_device_async`` fails to add a graph memory free node while a
        capture is still registered (its "capture ended" sibling branch clears
        the identical error as benign), and the next Warp array copy then
        surfaces that stale error as its own failure, aborting simulation
        initialization. Draining here keeps a prior simulation lifecycle's
        latched error from poisoning this one. Remove once the upstream Warp
        fix lands.
        """
        device = wp.get_device(str(self._device))
        if not device.is_cuda:
            return
        # Private Warp API: the drain primitives are not exposed publicly; getting the
        # device above guarantees the runtime is initialized. Guard the whole private
        # interaction so a future Warp internals reshuffle degrades to a skipped drain
        # rather than hard-failing simulation start.
        try:
            from warp._src.context import runtime as _wp_runtime

            core = _wp_runtime.core
            # wp_cuda_context_check drains via cudaGetLastError() but returns the
            # post-sync error state (0 once a non-sticky error was drained), and its
            # internal check_cuda() prints the drained error verbatim to stderr;
            # suppress the print and diff Warp's error buffer to report the drain.
            before = core.wp_get_error_string()
            was_enabled = bool(core.wp_is_error_output_enabled())
            core.wp_set_error_output_enabled(0)
            try:
                persistent = core.wp_cuda_context_check(device.context)
            finally:
                core.wp_set_error_output_enabled(1 if was_enabled else 0)
            after = core.wp_get_error_string()
        except (ImportError, AttributeError) as exc:
            logger.warning("Skipping stale CUDA error drain; Warp internals unavailable: %s", exc)
            return

        if persistent != 0:
            logger.error(
                "CUDA error %d persists after drain; the device context is likely unrecoverable: %s",
                persistent,
                after.decode(errors="replace"),
            )
        elif after != before:
            logger.warning(
                "Drained stale CUDA error latched by a prior lifecycle: %s (last Warp error recorded before drain: %s)",
                after.decode(errors="replace"),
                before.decode(errors="replace") or "<none>",
            )

    def start_simulation(self) -> None:
        """Start simulation by finalizing model and initializing state.

        This function finalizes the model and initializes the simulation state.
        Note: Collision pipeline is initialized later in initialize_solver() after
        we determine whether the solver needs external collision detection.
        """
        sim = self._sim
        builder = self.builder
        self._drain_stale_cuda_error()

        logger.info("Dispatching MODEL_INIT callbacks")
        self.dispatch_event(PhysicsEvent.MODEL_INIT)

        # The builder survives hard resets, so sites already added to it are skipped.
        global_sites, body_sites, root_sites = self.inject_sites(builder, {})
        site_map = self._site_index_map
        site_map.update((label, (index, None)) for label, index in global_sites.items())
        site_map.update((label, (None, [indices])) for label, indices in body_sites.get(id(builder), {}).items())
        for label, xform in root_sites.items():
            site_map[label] = (None, [[builder.add_site(body=-1, xform=xform, label=label)]])

        device = self._device
        logger.info(f"Finalizing model on device: {device}")
        builder.up_axis = Axis.Z
        if (solver_cfg := self._cfg.solver_cfg) is not None:
            solver_cfg.class_type.prepare_builder(solver_cfg, builder)
        with Timer(
            name="newton_finalize_builder",
            msg="Finalize builder took:",
            activity="Finalizing physics model",
            synchronize="both",
            device=device,
        ):
            cfg = NewtonBackendCfg(physics_cfg=self._cfg, device=sim.device)
            self.backend = sim.get_or_create_backend(cfg, builder=builder, dt=self.get_physics_dt())
            self.backend.model.set_gravity(sim.cfg.gravity)

        self.backend.decimation = self._decimation
        clone = self._clone
        if clone is not None:
            self.backend.particle_ranges = clone.particle_ranges
        self._scene_data_backend.initialize_geometry(
            [] if clone is None else clone.geometry_batches, {} if clone is None else clone.cable_bindings
        )
        logger.info("Dispatching PHYSICS_READY callbacks")
        self.dispatch_event(PhysicsEvent.PHYSICS_READY)

    def instantiate_builder_from_stage(self):
        """Import the explicitly declared clone plan into the Newton builder."""
        cloner.replicate(self._sim.get_clone_plan())

    def initialize_solver(self) -> None:
        """Build the solver, its contacts and the initial body state; see :meth:`NewtonBackend.init_solver`."""
        backend = self.backend
        sim = self._sim
        plan = sim.get_clone_plan() if sim is not None else None
        if sim is not None:
            backend.solver_init_callbacks.append(self._prepare_visualizer_for_capture)
        with Timer(
            name="newton_initialize_solver",
            msg="Initialize solver took:",
            activity="Initializing solver",
            synchronize="both",
            device=self.get_device(),
        ):
            relaxed = sim is not None and has_kit() and (sim.has_gui or sim.has_offscreen_render)
            backend.init_solver(None if plan is None else plan.env_template, relaxed_capture=relaxed)
        self._mark_transforms_changed()

    def _prepare_visualizer_for_capture(self) -> None:
        """Let the Newton visualizer register picking forces before the step is captured."""
        solver_cfg = self._cfg.solver_cfg
        if solver_cfg.class_type.applies_body_forces(solver_cfg, self.backend.solver):
            self._sim._prepare_newton_visualizer_for_capture()

    def create_visual_material_writer(self, batches: tuple[VisualMaterialBatch, ...]) -> VisualMaterialWriter:
        """Compile material-to-shape addresses for the active Newton model."""
        return self.backend.create_visual_material_writer(batches)

    def create_visual_shape_color_writer(
        self, asset: BaseArticulation, body_names: tuple[str, ...]
    ) -> VisualShapeColorWriter:
        """Compile selected articulation-body shape addresses for the active Newton model."""
        model = self.backend.model
        view = asset.root_view
        if not isinstance(view, ArticulationView):
            root_expr = asset.cfg.prim_path
            root_expr += (
                "(?:/.*)?" if asset.cfg.articulation_root_prim_path is None else asset.cfg.articulation_root_prim_path
            )
            prim_paths = [path for path in model.articulation_label if re.fullmatch(root_expr, path)]
            view = ArticulationView(model, prim_paths, verbose=False)
        return VisualShapeColorWriter(model, view, body_names)

    def get_scene_data_provider(self) -> SceneDataProvider:
        """Return the active scene data provider."""
        return SimulationContext.instance().get_scene_data_provider()

    def activate_newton_actuator_path(self) -> None:
        """Run the model's Newton actuators inside the step; see :meth:`NewtonBackend.activate_actuators`."""
        self.backend.activate_actuators()

    def register_solver_init_callback(self, callback: Callable[[], None]) -> None:
        """Append a one-shot hook invoked once the solver exists, before the initial FK and any CUDA graph capture."""
        self.backend.solver_init_callbacks.append(callback)

    def register_pre_actuator_callback(self, callback: Callable[[], None]) -> None:
        """Append a hook invoked before the actuator step.

        Each callback runs inside the captured CUDA graph once per decimation iteration when
        :meth:`handles_decimation` is ``True``, and eagerly once per step otherwise.
        Hard resets discard these hooks; register them again when the new model is ready.
        """
        self.backend.pre_actuator_callbacks.append(callback)

    def register_post_actuator_callback(self, callback: Callable[[], None]) -> None:
        """Append a hook invoked after the actuator step on every iteration.

        Each callback runs inside the captured CUDA graph (when :meth:`handles_decimation` is ``True``) right after the
        Newton actuators and before the solver substeps, so its writes to ``state``/``control`` are visible to the
        integrator on the same iteration. Callbacks fire in registration order.
        Hard resets discard these hooks; register them again when the new model is ready.
        """
        self.backend.post_actuator_callbacks.append(callback)

    def register_state_force_callback(self, callback: Callable[[State], None]) -> None:
        """Register a graph-safe callback that applies forces before every solver substep.

        Callbacks must be registered before the step is captured. Hard resets discard these hooks; register them again
        when the new model is ready.

        Args:
            callback: Function that adds forces [N, N·m] to the provided state.
        """
        if callback not in self.backend.state_force_callbacks:
            self.backend.state_force_callbacks.append(callback)

    def register_post_step_callback(self, callback: Callable[[], None]) -> None:
        """Append a hook invoked once per :meth:`step`, after the last solver substep and before sensors update.

        The hook reflects the state after all decimation iterations of the step. Callbacks must be graph-safe (fixed
        shapes, no host branching on device data) because they are recorded into the captured step.
        Hard resets discard these hooks; register them again when the new model is ready.
        """
        self.backend.post_step_callbacks.append(callback)

    def unregister_post_step_callback(self, callback: Callable[[], None]) -> None:
        """Remove a post-step callback; removing an absent callback is a no-op."""
        if self.backend is not None:
            with contextlib.suppress(ValueError):
                self.backend.post_step_callbacks.remove(callback)

    def set_decimation(self, decimation: int) -> None:
        """Set the decimation count; see :meth:`NewtonBackend.set_decimation`.

        When all actuators are graphable the entire decimation loop is captured as a single CUDA graph, right after
        the next requested step runs eagerly.
        """
        self._decimation = max(1, decimation)
        if self.backend is not None:
            self.backend.set_decimation(self._decimation)

    def handles_decimation(self) -> bool:
        """Whether :meth:`step` executes the full decimation loop; see :meth:`NewtonBackend.handles_decimation`."""
        return self.backend.handles_decimation()

    def add_contact_sensor(
        self,
        body_names_expr: str | list[str] | None = None,
        shape_names_expr: str | list[str] | None = None,
        contact_partners_body_expr: str | list[str] | None = None,
        contact_partners_shape_expr: str | list[str] | None = None,
        verbose: bool = False,
    ) -> SensorContact:
        """Add a contact sensor between bodies or shapes; see :meth:`NewtonBackend.add_contact_sensor`."""
        return self.backend.add_contact_sensor(
            body_names_expr,
            shape_names_expr,
            contact_partners_body_expr,
            contact_partners_shape_expr,
            verbose,
        )

    def add_imu_sensor(self, sites: list[int]) -> SensorIMU:
        """Add an IMU sensor at sites; see :meth:`NewtonBackend.add_imu_sensor`."""
        return self.backend.add_imu_sensor(sites)
