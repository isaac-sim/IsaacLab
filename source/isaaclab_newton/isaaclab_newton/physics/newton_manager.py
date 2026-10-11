# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton physics manager for Isaac Lab."""

from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp
from newton import Axis, Contacts, Control, Model, ModelBuilder, ModelFlags, State
from newton.selection import ArticulationView
from newton.sensors import SensorContact, SensorIMU
from newton.solvers import SolverBase

from pxr import UsdGeom

from isaaclab.physics import PhysicsEvent, PhysicsManager
from isaaclab.scene_data import SceneDataBackend, SceneDataProvider
from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.timer import Timer
from isaaclab.utils.version import has_kit

from isaaclab_newton.cloner.replicate import NewtonReplicateContext
from isaaclab_newton.physics.newton_manager_cfg import (
    NewtonBuilderCfg,
    NewtonCfg,
)
from isaaclab_newton.renderers.visual_material import (
    VisualMaterialWriter,
    VisualShapeColorWriter,
)

from . import newton_backend as nb
from .newton_backend import NewtonBackend, NewtonCloneRecord, SiteEntry, StepCallback
from .newton_scene_data import NewtonSceneDataBackend
from .newton_solver import NewtonSolver, create_builder, create_solver_adapter

if TYPE_CHECKING:
    from isaaclab.actuators.newton import NewtonActuatorAdapter
    from isaaclab.assets import BaseArticulation
    from isaaclab.renderers.base_renderer import VisualMaterialBatch


logger = logging.getLogger(__name__)


def create_newton_builder(cfg: NewtonBuilderCfg) -> ModelBuilder:
    """Construct the builder of a render-only Newton model; the simulation registry's factory.

    Args:
        cfg: The selected, non-Newton physics configuration.

    Returns:
        An empty builder.

    Raises:
        ValueError: If Newton is the physics backend, whose manager owns the builder (see :func:`get_newton_builder`).
    """
    if isinstance(cfg.physics_cfg, NewtonCfg):
        raise ValueError("Newton physics owns its builder; use get_newton_builder(sim).")
    return ModelBuilder()


class NewtonManager(PhysicsManager):
    """Newton physics manager for Isaac Lab.

    Owns construction requests and a :class:`NewtonBackend`, and adapts its functional lifecycle to Isaac Lab.
    Solver-specific behavior lives in stateless :class:`NewtonSolver` adapters selected by
    :attr:`NewtonSolverCfg.class_type`. :attr:`NewtonCfg.class_type` can select a downstream manager independently.

    Initialization (:meth:`reset` with ``soft=False``)::

        MODEL_INIT              consumers author the shared builder
        finalize                prepare_builder (sites, solver normalization) -> Model -> NewtonBackend
        PHYSICS_READY           consumers bind views, sensors, step callbacks, and actuators to the backend
        init_solver             solver, contacts, and body state from the initial joint state

    The step graph is built and captured on the first :meth:`step` after any structural change.
    """

    backend_name = "newton"

    clone_context_type = NewtonReplicateContext

    def __init__(
        self,
        builder: ModelBuilder | None = None,
        cfg: NewtonCfg | None = None,
        *,
        dt: float = 1.0 / 60.0,
        device: str = "cuda:0",
    ):
        """Create an independent manager, optionally from a populated Newton builder.

        Args:
            builder: Construction model retained across hard resets. Created lazily when omitted.
            cfg: Physics and solver settings. A context the manager is attached to adopts them. Defaults to the
                context's physics configuration, or to Newton's standard configuration for standalone use.
            dt: Physics timestep for standalone use; an attached context supplies its timestep.
            device: Simulation device for standalone use.
        """
        super().__init__()
        self._cfg = cfg
        self._device = device
        self._dt = dt
        self._builder = builder
        self._solver_adapter: NewtonSolver | None = None
        self.backend: NewtonBackend | None = None
        self._site_requests = {}
        self._site_index_map = {}
        self._world_builder_hooks = []
        self._clone: NewtonCloneRecord | None = None
        self._scene_data_backend = NewtonSceneDataBackend(lambda: self.backend)
        self._control_binding = None
        self._decimation = 1
        self._apply_every_physics_step = None

    @property
    def cfg(self) -> NewtonCfg:
        """Physics and solver settings: the constructor's, else the attached context's, else Newton's defaults."""
        if self._cfg is None:
            self._cfg = NewtonCfg()
        return self._cfg

    @property
    def solver_adapter(self) -> NewtonSolver:
        """Adapter of the configured solver, used for the builder before a backend exists."""
        if self._solver_adapter is None or self._solver_adapter.cfg is not self.cfg.solver_cfg:
            self._solver_adapter = create_solver_adapter(self.cfg.solver_cfg)
        return self._solver_adapter

    # ----- Physics manager lifecycle ---------------------------------------------

    def reset(self, soft: bool = False) -> None:
        """Reset physics simulation.

        A hard reset (``soft=False``) closes the backend and builds a new one from the retained builder, so nothing
        captured against the old buffers survives. Consumers rebind on ``PHYSICS_READY``. A soft reset keeps the
        backend.

        Args:
            soft: If True, skip full reinitialization.
        """
        if soft:
            return
        self._close_backend()
        sim = self._sim
        self.dispatch_event(PhysicsEvent.MODEL_INIT)
        backend = self.backend = self.finalize_backend()
        clone = self._clone
        self._scene_data_backend.initialize_geometry(
            [] if clone is None else clone.geometry_batches, {} if clone is None else clone.cable_bindings
        )
        self.dispatch_event(PhysicsEvent.PHYSICS_READY)
        with Timer(
            name="newton_initialize_solver",
            msg="Initialize solver took:",
            activity="Initializing solver",
            synchronize="both",
            device=self._device,
        ):
            nb.init_solver(
                backend, relaxed_capture=sim is not None and has_kit() and (sim.has_gui or sim.has_offscreen_render)
            )
        if self._control_binding is not None:
            self.bind_control(*self._control_binding)
        self._resolve_steps_per_call()
        self._mark_transforms_changed()

    def forward(self) -> None:
        """Reset solver internals and recompute body state for worlds whose state was authored."""
        backend = self.backend
        if backend is not None and backend.solver is not None:
            nb.notify_model_changes(backend)
            nb.forward(backend)

    def step(self) -> None:
        """Advance physics by one environment step, or one physics step when the environment runs decimation.

        Called while a caller records a CUDA graph on the simulation device (for example a whole environment step),
        the step records into that graph; call :meth:`prepare` first. Host bookkeeping such as simulation time then
        runs once, when the graph is recorded.
        """
        sim = self._sim
        if sim is not None and not sim.is_playing():
            return
        backend = self.backend
        if backend.device.is_capturing:
            # A caller records a larger graph, such as a whole environment step; record into it.
            self.record_step()
        else:
            graph = nb.step(backend)
            backend.solver_adapter.check_status(backend, graph.captured)
            if self.cfg.debug_mode:
                backend.solver_adapter.log_debug(backend)
        self._sim_time += backend.dt * backend.steps_per_call
        scene_data = self._scene_data_backend
        scene_data.transforms_timestamp += 1
        scene_data.geometry_timestamp += 1

    def record_step(self) -> nb.StepGraph:
        """Record physics into the caller's capture, without host time or rendering bookkeeping."""
        return nb.record_step(self.backend)

    def prepare(self) -> None:
        """Build the step graph ahead of a caller's capture of :meth:`step`, without advancing physics."""
        backend = self.backend
        nb.notify_model_changes(backend)
        nb.forward(backend)
        nb.prepare(backend)

    def close(self) -> None:
        """Clean up Newton physics resources."""
        backend = self.backend
        try:
            super().close()
        finally:
            self.backend = None
            try:
                if backend is not None:
                    backend.close()
            finally:
                self.clear()

    def clear(self) -> None:
        """Release the backend and all session state."""
        self._close_backend(dispatch_stop=False)
        self._builder = None
        self._site_requests = {}
        self._site_index_map = {}
        self._world_builder_hooks = []
        self._clone = None
        self._control_binding = None
        self._decimation = 1
        self._apply_every_physics_step = None

    def _close_backend(self, *, dispatch_stop: bool = True) -> None:
        """Stop consumers, drop their views, and close the backend."""
        backend = self.backend
        if backend is None:
            return
        if dispatch_stop:
            self.dispatch_event(PhysicsEvent.STOP)
        self.views.clear()
        self.backend = None
        backend.close()

    # ----- Decimation --------------------------------------------------------------

    def bind_control(self, actions, scene) -> bool:
        """Run the environment's actions and asset command writers inside the step, in the environment's order.

        Actions that hold their command for the whole environment step run once per step (pre-step callbacks);
        the other actions and the asset command writers run before every physics step (control callbacks).

        Args:
            actions: Environment action manager.
            scene: Scene whose assets submit commands.

        Returns:
            True: the step now applies the actions and submits the scene commands.
        """
        for name in actions.active_terms:
            term = actions.get_term(name)
            register = (
                self.register_control_callback if term.apply_every_physics_step else self.register_pre_step_callback
            )
            register(term.apply_actions, graphable=term.supports_graph_capture, name=f"action.{name}")
        for name, asset in scene.assets.items():
            self.register_control_callback(asset.write_data_to_sim, graphable=asset.supports_graph_capture, name=name)
        self._control_binding = (actions, scene)
        self._resolve_steps_per_call()
        return True

    def set_decimation(self, decimation: int, *, apply_every_physics_step: bool | None = None) -> None:
        """Set the physics steps of one environment step.

        Args:
            decimation: Physics steps per environment step.
            apply_every_physics_step: Whether the environment applies actions before every physics step. ``False``
                lets :meth:`step` advance the whole decimation loop; ``None`` does so only when Newton actuators run
                inside the step.
        """
        self._decimation = max(1, decimation)
        self._apply_every_physics_step = apply_every_physics_step
        if self.backend is not None:
            self._resolve_steps_per_call()

    def handles_decimation(self) -> bool:
        """Whether one :meth:`step` advances the whole decimation loop.

        Functions that cannot be captured run eagerly inside the step, so only host work between physics steps (see
        :meth:`require_env_decimation`) or the environment's actions prevent it.
        """
        backend = self.backend
        if backend is None or backend.env_decimation:
            return False
        if self._control_binding is not None:
            return True
        if self._apply_every_physics_step:
            return False
        return self._apply_every_physics_step is False or backend.actuators is not None

    def require_env_decimation(self) -> None:
        """Keep decimation in the environment for consumers requiring scene publication between physics steps."""
        self.backend.env_decimation = True
        self._resolve_steps_per_call()

    def _resolve_steps_per_call(self) -> None:
        """Set the physics steps of one :meth:`step` once the backend's consumers and the decimation are known."""
        self.backend.steps_per_call = self._decimation if self.handles_decimation() else 1

    # ----- Step callbacks and actuators -----------------------------------------------

    def register_pre_step_callback(
        self, fn: Callable[[], None], *, graphable: bool = True, name: str = ""
    ) -> StepCallback:
        """Run ``fn`` once per :meth:`step`, before the first physics step.

        Consumers register on ``PHYSICS_READY``; a hard reset discards callbacks together with the buffers they bind.

        Args:
            fn: Function to run, e.g. an action that holds its command for the whole environment step.
            graphable: Whether the function can be recorded into a CUDA graph. Other functions run eagerly in place.
            name: Label used in errors and profiles.

        Returns:
            The callback, for :meth:`unregister_step_callback`.
        """
        return nb.register_step_callback(
            self.backend, self.backend.callbacks.pre_step, fn, graphable=graphable, name=name
        )

    def register_control_callback(
        self, fn: Callable[[], None], *, graphable: bool = True, name: str = ""
    ) -> StepCallback:
        """Run ``fn`` before the Newton actuators of every physics step, like MuJoCo's ``mjcb_control``.

        Args:
            fn: Function to run, e.g. a controller or an asset command writer.
            graphable: Whether the function can be recorded into a CUDA graph. Other functions run eagerly in place.
            name: Label used in errors and profiles.

        Returns:
            The callback, for :meth:`unregister_step_callback`.
        """
        return nb.register_step_callback(
            self.backend, self.backend.callbacks.control, fn, graphable=graphable, name=name
        )

    def register_post_actuator_callback(
        self, fn: Callable[[], None], *, graphable: bool = True, name: str = ""
    ) -> StepCallback:
        """Run ``fn`` after the Newton actuators of the last physics step of every :meth:`step`.

        Args:
            fn: Function to run, e.g. actuator telemetry.
            graphable: Whether the function can be recorded into a CUDA graph. Other functions run eagerly in place.
            name: Label used in errors and profiles.

        Returns:
            The callback, for :meth:`unregister_step_callback`.
        """
        callbacks = self.backend.callbacks.post_actuator
        return nb.register_step_callback(self.backend, callbacks, fn, graphable=graphable, name=name)

    def register_state_force_callback(
        self, fn: Callable[[State], None], *, graphable: bool = True, name: str = ""
    ) -> StepCallback:
        """Run ``fn`` before every solver substep with the substep's input state.

        Args:
            fn: Function receiving the input :class:`newton.State`, e.g. one that applies interactive forces.
            graphable: Whether the function can be recorded into a CUDA graph. Other functions run eagerly in place.
            name: Label used in errors and profiles.

        Returns:
            The callback, for :meth:`unregister_step_callback`.
        """
        callbacks = self.backend.callbacks.state_force
        return nb.register_step_callback(self.backend, callbacks, fn, graphable=graphable, name=name)

    def register_post_step_callback(
        self, fn: Callable[[], None], *, graphable: bool = True, name: str = ""
    ) -> StepCallback:
        """Run ``fn`` after every physics step, before sensors update.

        Args:
            fn: Function to run, e.g. one that republishes state within decimation.
            graphable: Whether the function can be recorded into a CUDA graph. Other functions run eagerly in place.
            name: Label used in errors and profiles.

        Returns:
            The callback, for :meth:`unregister_step_callback`.
        """
        return nb.register_step_callback(
            self.backend, self.backend.callbacks.post_step, fn, graphable=graphable, name=name
        )

    def invalidate_step_graph(self) -> None:
        """Rebuild the step graph before the next step; see :func:`newton_backend.invalidate_step_graph`."""
        if self.backend is not None:
            nb.invalidate_step_graph(self.backend)

    def unregister_step_callback(self, callback: StepCallback) -> None:
        """Stop running a callback; a no-op after the model is rebuilt.

        Args:
            callback: Callback returned by one of the ``register_*_callback`` methods.
        """
        if self.backend is not None:
            nb.unregister_step_callback(self.backend, callback)

    def activate_actuators(self) -> NewtonActuatorAdapter | None:
        """Run the model's Newton actuators inside the step. Idempotent.

        Returns:
            The adapter, or ``None`` when no articulation has explicit Newton actuators.
        """
        adapter = nb.activate_actuators(self.backend)
        self._resolve_steps_per_call()
        return adapter

    # ----- Authored state --------------------------------------------------------------

    def invalidate_fk(
        self,
        env_mask: wp.array | None = None,
        env_ids: wp.array | None = None,
        articulation_ids: wp.array | None = None,
    ) -> None:
        """Mark articulations as needing FK and their worlds as needing a solver reset.

        Called by asset writes that modify joint coordinates or root transforms. The flags are consumed at the next
        forward, raw-state read, rendering, or step.

        Args:
            env_mask: Boolean mask of dirtied view rows. Shape ``(num_instances,)``.
            env_ids: Integer indices of dirtied view rows.
            articulation_ids: Model articulation index of each ``(row, articulation)``, from
                ``ArticulationView.articulation_ids``. ``None`` flags every articulation.
        """
        self._mark_transforms_changed()
        if self.backend is not None:
            nb.invalidate_fk(self.backend, env_mask, env_ids, articulation_ids)

    def invalidate_body_state(
        self,
        env_ids: wp.array(dtype=wp.int32) | None = None,
        env_mask: wp.array(dtype=wp.bool) | None = None,
        row_worlds: wp.array(dtype=wp.int32) | None = None,
    ) -> None:
        """Mark worlds whose maximal-coordinate body state was written, without requesting FK.

        Args:
            env_ids: Integer indices of dirtied view rows.
            env_mask: Boolean mask of dirtied view rows.
            row_worlds: World of each view row, from :meth:`view_row_worlds`; ``None`` when rows are worlds.
        """
        self._mark_transforms_changed()
        if self.backend is not None:
            nb.invalidate_body_state(self.backend, env_ids, env_mask, row_worlds)

    def view_row_worlds(self, articulation_ids: wp.array) -> wp.array:
        """Return the world of each row of an articulation view; see :func:`newton_backend.view_row_worlds`.

        Args:
            articulation_ids: ``ArticulationView.articulation_ids``.
        """
        return nb.view_row_worlds(self.backend, articulation_ids)

    def add_model_change(self, change: ModelFlags, env_mask: wp.array | None = None) -> None:
        """Queue a property edit for the next state-read or step boundary.

        Args:
            change: Changed model-property categories.
            env_mask: Selected worlds. An empty mask performs no solver refresh.
        """
        if self.backend is not None:
            nb.mark_model_changed(self.backend, change, env_mask)

    def transforms_may_change_on_graph_replay(self) -> bool:
        """Whether state was written during an outer capture, so graph replays may change it without notice."""
        backend = self.backend
        return backend is not None and backend.transforms_may_change_on_graph_replay

    def mark_particles_dirty(self) -> None:
        """Invalidate scene-data geometry after native particle writes."""
        self._scene_data_backend.geometry_timestamp += 1
        self._flag_outer_capture()

    def _mark_transforms_changed(self) -> None:
        """Publish authored rigid-body changes and invalidate cable geometry."""
        self._scene_data_backend.transforms_timestamp += 1
        self._scene_data_backend.geometry_timestamp += 1
        self._flag_outer_capture()

    def _flag_outer_capture(self) -> None:
        """Remember writes recorded into an outer capture, whose replays bypass Python invalidation."""
        backend = self.backend
        if backend is not None and backend.device.is_capturing:
            backend.transforms_may_change_on_graph_replay = True

    # ----- Sensors ------------------------------------------------------------------

    def add_contact_sensor(
        self,
        body_names_expr: str | list[str] | None = None,
        shape_names_expr: str | list[str] | None = None,
        contact_partners_body_expr: str | list[str] | None = None,
        contact_partners_shape_expr: str | list[str] | None = None,
        verbose: bool = False,
    ) -> SensorContact:
        """Add a contact sensor for reporting contacts between bodies or shapes; see :func:`add_contact_sensor`."""
        with Timer(name="newton_contact_sensor", msg="Contact sensor construction took:", synchronize="both"):
            return nb.add_contact_sensor(
                self.backend,
                body_names_expr,
                shape_names_expr,
                contact_partners_body_expr,
                contact_partners_shape_expr,
                verbose,
            )

    def add_imu_sensor(self, sites: list[int]) -> SensorIMU:
        """Add an IMU sensor measuring acceleration and angular velocity at sites; see :func:`add_imu_sensor`."""
        return nb.add_imu_sensor(self.backend, sites)

    # ----- Model construction -------------------------------------------------------

    def prepare_builder(self, builder: ModelBuilder) -> None:
        """Apply unresolved site requests and solver-specific normalization to the builder about to be finalized.

        The builder survives hard resets, and so do the sites added to it, so a site is added only once.

        Args:
            builder: Builder of the simulated model.
        """
        global_sites, body_sites, root_sites = self.inject_sites(builder, {})
        site_map = self._site_index_map
        site_map.update((label, (index, None)) for label, index in global_sites.items())
        site_map.update((label, (None, [indices])) for label, indices in body_sites.get(id(builder), {}).items())
        for label, xform in root_sites.items():
            site_map[label] = (None, [[builder.add_site(body=-1, xform=xform, label=label)]])
        builder.up_axis = Axis.Z
        self.solver_adapter.prepare_solver_builder(builder)

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

    def get_world_builder_hooks(self) -> list[Callable[[ModelBuilder, int, np.ndarray, np.ndarray], None]]:
        """Return the hooks extending every world built by Newton replication."""
        return self._world_builder_hooks

    def record_clone(self, record: NewtonCloneRecord, site_index_map: dict[str, SiteEntry]) -> None:
        """Record native replication outputs for model finalization and consumers.

        Args:
            record: Replication outputs.
            site_index_map: Sites resolved during replication, by label.
        """
        self._clone = record
        self._site_index_map.update(site_index_map)

    def get_clone_record(self) -> NewtonCloneRecord | None:
        """Return the outputs of the last native replication, or ``None`` without replication."""
        return self._clone

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

    @property
    def builder(self) -> ModelBuilder:
        """This manager's construction model, retained across hard resets."""
        if self._builder is None:
            self._builder = create_builder(self.cfg)
        return self._builder

    def finalize_backend(self) -> NewtonBackend:
        """Finalize this manager's builder into independently owned simulation buffers."""
        builder = self.builder
        self.prepare_builder(builder)
        model = builder.finalize(device=self._device)
        if self._sim is not None:
            model.set_gravity(self._sim.cfg.gravity)
        ranges = {
            label: (start, end - start, kind)
            for kind, labels, starts, ends in (
                ("surface", builder.surface_label, builder._surface_particle_start, builder._surface_particle_end),
                ("volume", builder.volume_label, builder._volume_particle_start, builder._volume_particle_end),
            )
            for label, start, end in zip(labels, starts, ends, strict=True)
        }
        return NewtonBackend(model, self.cfg, dt=self.get_physics_dt(), deformable_ranges=ranges)

    def get_physics_dt(self) -> float:
        """Return the physics timestep: the attached context's, or the constructor's for standalone use."""
        return self._dt if self._sim is None else self._sim.cfg.dt

    # ----- Deprecated accessors ------------------------------------------------------------

    def _deprecated_backend_field(self, name: str) -> Any:
        warnings.warn(
            f"NewtonManager.get_{name}() is deprecated; read manager.backend.{name} instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        backend = self.backend
        return None if backend is None else getattr(backend, name)

    def get_solver(self) -> SolverBase | None:
        """Return :attr:`NewtonBackend.solver` of the active backend.

        .. deprecated::
            Read ``manager.backend.solver`` instead.
        """
        return self._deprecated_backend_field("solver")

    def get_model(self) -> Model | None:
        """Return :attr:`NewtonBackend.model` of the active backend.

        .. deprecated::
            Read ``manager.backend.model`` instead.
        """
        return self._deprecated_backend_field("model")

    def get_state_0(self) -> State | None:
        """Return :attr:`NewtonBackend.state_0` of the active backend.

        .. deprecated::
            Read ``manager.backend.state_0`` instead.
        """
        return self._deprecated_backend_field("state_0")

    def get_state_1(self) -> State | None:
        """Return :attr:`NewtonBackend.state_1` of the active backend.

        .. deprecated::
            Read ``manager.backend.state_1`` instead.
        """
        return self._deprecated_backend_field("state_1")

    def get_control(self) -> Control | None:
        """Return :attr:`NewtonBackend.control` of the active backend.

        .. deprecated::
            Read ``manager.backend.control`` instead.
        """
        return self._deprecated_backend_field("control")

    def get_contacts(self) -> Contacts | None:
        """Return :attr:`NewtonBackend.contacts` of the active backend.

        .. deprecated::
            Read ``manager.backend.contacts`` instead.
        """
        return self._deprecated_backend_field("contacts")

    def get_scene_data_backend(self) -> SceneDataBackend | None:
        """Return the SceneDataBackend for the SceneDataProvider."""
        return self._scene_data_backend

    def get_scene_data_provider(self) -> SceneDataProvider:
        """Return the active scene data provider."""
        return self._sim.get_scene_data_provider()

    def get_physics_sim_view(self) -> list:
        """Return the registered articulation views."""
        return list(self.views.values())

    def create_visual_material_writer(self, batches: tuple[VisualMaterialBatch, ...]) -> VisualMaterialWriter:
        """Compile material-to-shape addresses for the active Newton model."""
        return self.backend.create_visual_material_writer(batches)

    def create_visual_shape_color_writer(
        self, asset: BaseArticulation, body_names: tuple[str, ...]
    ) -> VisualShapeColorWriter:
        """Compile selected articulation-body shape addresses for the active Newton model."""
        model = None if self.backend is None else self.backend.model
        view = asset.root_view
        if not isinstance(view, ArticulationView):
            root_expr = asset.cfg.prim_path
            root_expr += (
                "(?:/.*)?" if asset.cfg.articulation_root_prim_path is None else asset.cfg.articulation_root_prim_path
            )
            prim_paths = [path for path in model.articulation_label if re.fullmatch(root_expr, path)]
            view = ArticulationView(model, prim_paths, verbose=False)
        return VisualShapeColorWriter(model, view, body_names)

    # ----- Backend descriptors -----------------------------------------------

    def video_capture_backend(self) -> str:
        """Newton GL headless perspective video capture."""
        return "newton_gl"

    @staticmethod
    def setup_deformable_body(prim: Any, deformable_type: str, sim_mesh_prim: Any, vis_mesh_prim: Any) -> None:
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
