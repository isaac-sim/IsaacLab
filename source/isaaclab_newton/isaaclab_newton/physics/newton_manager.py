# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton physics manager for Isaac Lab."""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import warp as wp
from newton import Axis, Contacts, Control, Heightfield, Model, ModelBuilder, ModelFlags, ShapeFlags, State, eval_fk
from newton.selection import ArticulationView
from newton.sensors import SensorContact, SensorIMU
from newton.solvers import SolverBase, SolverMuJoCo
from newton.usd import SchemaResolver, SchemaResolverMjc, SchemaResolverNewton, SchemaResolverPhysx

from pxr import Usd, UsdGeom

from isaaclab.physics import PhysicsEvent, PhysicsManager
from isaaclab.scene_data import SceneDataBackend, SceneDataFormat, SceneDataProvider
from isaaclab.sim import SimulationContext
from isaaclab.utils import checked_apply, to_dict
from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.timer import Timer
from isaaclab.utils.version import has_kit

from isaaclab_newton.physics.newton_manager_cfg import (
    NewtonBackendCfg,
    NewtonBuilderCfg,
    NewtonCfg,
    NewtonShapeCfg,
)
from isaaclab_newton.renderers.visual_material import (
    VisualMaterialWriter,
    VisualShapeColorWriter,
)

from . import newton_backend as nb
from .newton_backend import NewtonBackend, NewtonCloneRecord, SiteEntry, StepCallback, StepPhase

if TYPE_CHECKING:
    from isaaclab.actuators.newton import NewtonActuatorAdapter
    from isaaclab.assets import BaseArticulation
    from isaaclab.renderers.base_renderer import VisualMaterialBatch

    from .newton_manager_cfg import NewtonSolverCfg


logger = logging.getLogger(__name__)

_SENSORS_BY_STATE_ATTRIBUTE = {
    "body_qdd": "the IMU or PVA sensor",
    "body_parent_f": "the joint-wrench sensor",
}
"""Isaac Lab sensors that read each state attribute MuJoCo Warp's sensor stage fills."""


def create_newton_builder(cfg: NewtonBuilderCfg) -> ModelBuilder:
    """Construct a native builder through the selected physics manager's factory.

    Args:
        cfg: The selected physics configuration.

    Returns:
        An empty builder with the selected solver schemas and shape/BVH defaults.
    """
    if isinstance(cfg.physics_cfg, NewtonCfg):
        return cfg.physics_cfg.class_type.create_builder(physics_cfg=cfg.physics_cfg)
    return ModelBuilder()


class NewtonSceneDataBackend(SceneDataBackend):
    """Scene data backend that reads rigid body transforms from Newton's simulation state.

    The backend reads ``body_q`` (an array of :class:`wp.transformf`) from Newton's current state and exposes it as
    :class:`SceneDataFormat.Transform`. Body paths come from the model's ``body_label`` attribute.
    """

    def __init__(self, backend: Callable[[], NewtonBackend | None]):
        """Initialize the scene data backend.

        Args:
            backend: Returns the Newton backend whose state to publish; it changes on every hard reset.
        """
        self._backend = backend
        self._transforms = SceneDataFormat.Transform()
        self.transforms_timestamp = 0
        self.geometry_timestamp = 0
        self._geometry_batches = []

    def initialize_geometry(self, geometry_batches: list, cable_bindings: dict[str, list[int]]) -> None:
        """Bind imported geometry paths to native particle ranges and capsule endpoints.

        Args:
            geometry_batches: Deformable and particle geometry recorded by replication.
            cable_bindings: Native capsule shape indices of each open cable.
        """
        self._geometry_batches = list(geometry_batches)
        endpoints, ranges = [], {}
        offset = 0
        for path, shapes in cable_bindings.items():
            ids = np.asarray(shapes, dtype=np.int32)
            left = np.concatenate((ids[:1], ids))
            right = np.concatenate((ids, ids[-1:]))
            left_sign = np.ones(len(left), dtype=np.int32)
            right_sign = -np.ones(len(right), dtype=np.int32)
            left_sign[0], right_sign[-1] = -1, 1
            endpoints.append(np.column_stack((left, left_sign, right, right_sign)))
            ranges[path] = (offset, len(left))
            offset += len(left)
        if endpoints:
            model = self.model
            source = SceneDataFormat.CapsuleEndpoints()
            source.transforms = self._backend().state_0.body_q
            source.shape_body = model.shape_body
            source.shape_transform = model.shape_transform
            source.shape_scale = model.shape_scale
            source.endpoints = wp.array(np.concatenate(endpoints), dtype=wp.vec4i, device=model.device)
            self._geometry_batches.append((source, ranges))

    @property
    def native_geometry_formats(self) -> tuple[type, ...]:
        return (SceneDataFormat.Points, SceneDataFormat.WeightedPoints, SceneDataFormat.CapsuleEndpoints)

    def get_geometry_batches(self, output_format=SceneDataFormat.Points):
        """Publish native arrays; SDP derives cable endpoints and applies destination layouts."""
        state = self.state
        for source, _ in self._geometry_batches:
            attribute = "transforms" if source._cls is SceneDataFormat.CapsuleEndpoints else "points"
            data = state.particle_q if attribute == "points" else state.body_q
            if getattr(source, attribute) is not data:
                setattr(source, attribute, data)
                self.geometry_timestamp += 1
        return self._geometry_batches

    @property
    def transforms(self) -> SceneDataFormat.Transform:
        """Publish the authoritative native pointer, including solver state-buffer swaps."""
        transforms = self.state.body_q
        if self._transforms.transforms is not transforms:
            self._transforms.transforms = transforms
            self.transforms_timestamp += 1
        return self._transforms

    @property
    def transform_count(self) -> int:
        """Return the number of rigid body transforms in the Newton sim."""
        return self.model.body_count

    @property
    def transform_paths(self) -> list[str]:
        """Return the prim paths for each rigid body transform."""
        if self.model.body_label is not None:
            return list(self.model.body_label)
        return []

    @property
    def model(self) -> Model | None:
        backend = self._backend()
        return None if backend is None else backend.model

    @property
    def state(self) -> State | None:
        """Return native physics state, consistent with authored state, without entering the rendering path."""
        backend = self._backend()
        if backend is None:
            return None
        if backend.transforms_may_change_on_graph_replay:
            # Raw external graph replays bypass Python invalidation, so these reads must stay conservative.
            self.transforms_timestamp += 1
            self.geometry_timestamp += 1
        if backend.solver is not None:
            nb.forward(backend)
        return backend.state_0


class NewtonManager(PhysicsManager):
    """Newton physics manager for Isaac Lab.

    The manager has two roles:

    * **Physics manager.** It drives the active :class:`~isaaclab_newton.physics.NewtonBackend` (:attr:`backend`)
      through the :class:`~isaaclab.physics.PhysicsManager` lifecycle. The backend holds every model-bound object
      (state, control, solver, contacts, sensors, actuators, step callbacks, and the compiled step graph); the
      functions in :mod:`isaaclab_newton.physics.newton_backend` operate on it explicitly. The manager itself only
      keeps what must survive a hard reset: site requests, replication outputs, and the decimation setting.
    * **Solver manager.** Each solver has a subclass (e.g. :class:`NewtonMJWarpManager`) that overrides the solver
      hooks below: construction, stepping, reset, forward kinematics, and capabilities. Hooks are stateless
      classmethods over an explicit backend, so the same subclass can drive several backends.
      :meth:`NewtonCfg.__post_init__` selects the subclass from :attr:`NewtonSolverCfg.class_type`.

    Initialization (:meth:`reset` with ``soft=False``)::

        MODEL_INIT              consumers author the shared builder
        finalize                prepare_builder (sites, solver normalization) -> Model -> NewtonBackend
        PHYSICS_READY           consumers bind views, sensors, step callbacks, and actuators to the backend
        init_solver             solver, contacts, and body state from the initial joint state

    The step graph is built and captured on the first :meth:`step` after any structural change.
    """

    backend: ClassVar[NewtonBackend | None] = None
    """Active Newton backend; borrowed from the simulation registry, replaced on every hard reset."""

    # Session state; survives hard resets, which rebuild the model from the same builder.
    _site_requests: ClassVar[dict[tuple[str | None, bool, tuple[float, ...]], tuple[str, wp.transform]]] = {}
    _site_index_map: ClassVar[dict[str, SiteEntry]] = {}
    _world_builder_hooks: ClassVar[list[Callable[[ModelBuilder, int, np.ndarray, np.ndarray], None]]] = []
    _clone: ClassVar[NewtonCloneRecord | None] = None
    _scene_data_backend: ClassVar[NewtonSceneDataBackend | None] = None
    _decimation: ClassVar[int] = 1
    _apply_every_physics_step: ClassVar[bool | None] = None

    # ----- Physics manager lifecycle ---------------------------------------------

    @classmethod
    def initialize(cls, sim_context: SimulationContext) -> None:
        """Initialize the manager with simulation context.

        Args:
            sim_context: Parent simulation context.
        """
        super().initialize(sim_context)
        # This context imports NewtonManager, so it can only be imported after this module initializes.
        from isaaclab_newton.cloner import NewtonReplicateContext  # noqa: PLC0415

        cls.clone_context_type = NewtonReplicateContext
        NewtonManager._scene_data_backend = NewtonSceneDataBackend(lambda: NewtonManager.backend)

    @classmethod
    def reset(cls, soft: bool = False) -> None:
        """Reset physics simulation.

        A hard reset (``soft=False``) closes the backend and builds a new one from the retained builder, so nothing
        captured against the old buffers survives. Consumers rebind on ``PHYSICS_READY``. A soft reset keeps the
        backend.

        Args:
            soft: If True, skip full reinitialization.
        """
        if soft:
            return
        cls._close_backend()
        sim = SimulationContext.instance()
        cls._drain_stale_cuda_error()
        cls.dispatch_event(PhysicsEvent.MODEL_INIT)
        cfg = NewtonBackendCfg(physics_cfg=sim.cfg.physics, device=sim.device)
        with Timer(
            name="newton_finalize_builder",
            msg="Finalize builder took:",
            activity="Finalizing physics model",
            synchronize="both",
            device=sim.device,
        ):
            backend = NewtonManager.backend = sim.get_or_create_backend(cfg)
        clone = NewtonManager._clone
        cls._scene_data_backend.initialize_geometry(
            [] if clone is None else clone.geometry_batches, {} if clone is None else clone.cable_bindings
        )
        cls.dispatch_event(PhysicsEvent.PHYSICS_READY)
        with Timer(
            name="newton_initialize_solver",
            msg="Initialize solver took:",
            activity="Initializing solver",
            synchronize="both",
            device=sim.device,
        ):
            nb.init_solver(backend)
        cls._mark_transforms_changed()

    @classmethod
    def forward(cls) -> None:
        """Reset solver internals and recompute body state for worlds whose state was authored."""
        backend = NewtonManager.backend
        if backend is not None and backend.solver is not None:
            nb.forward(backend)

    @classmethod
    def step(cls) -> None:
        """Advance physics by one environment step, or one physics step when the environment runs decimation.

        Called while a caller records a CUDA graph on the simulation device (for example a whole environment step),
        the step records into that graph; call :meth:`prepare` first. Host bookkeeping such as simulation time then
        runs once, when the graph is recorded.
        """
        sim = PhysicsManager._sim
        if sim is None or not sim.is_playing():
            return
        backend = NewtonManager.backend
        steps = NewtonManager._decimation if cls.handles_decimation() else 1
        if wp.get_device(backend.device).is_capturing:
            # A caller records a larger graph, such as a whole environment step; record into it.
            nb.record_step(backend, steps)
        else:
            graph = nb.step(backend, steps, cls._capture_graph if cls._uses_cuda_graph() else None)
            backend.manager.check_status(backend, graph.captured)
            if PhysicsManager._cfg.debug_mode:
                backend.manager.log_debug(backend)
        PhysicsManager._sim_time += backend.dt * steps
        scene_data = NewtonManager._scene_data_backend
        scene_data.transforms_timestamp += 1
        scene_data.geometry_timestamp += 1

    @classmethod
    def prepare(cls) -> None:
        """Build the step graph ahead of a caller's capture of :meth:`step`, without advancing physics."""
        backend = NewtonManager.backend
        nb.notify_model_changes(backend)
        nb.forward(backend)
        nb.prepare(backend, NewtonManager._decimation if cls.handles_decimation() else 1)

    @classmethod
    def close(cls) -> None:
        """Clean up Newton physics resources."""
        super().close()
        cls.clear()

    @classmethod
    def clear(cls) -> None:
        """Release the backend and all session state."""
        cls._close_backend()
        NewtonManager._site_requests = {}
        NewtonManager._site_index_map = {}
        NewtonManager._world_builder_hooks = []
        NewtonManager._clone = None
        NewtonManager._scene_data_backend = None
        NewtonManager._decimation = 1
        NewtonManager._apply_every_physics_step = None

    @classmethod
    def _close_backend(cls) -> None:
        """Stop consumers, drop their views, and return the backend to the simulation registry."""
        backend = NewtonManager.backend
        if backend is None:
            return
        cls.dispatch_event(PhysicsEvent.STOP)
        for key in [key for key in NewtonManager.views if key[0] is NewtonManager]:
            del NewtonManager.views[key]
        NewtonManager.backend = None
        SimulationContext.instance().close_backend(backend)

    @classmethod
    def _drain_stale_cuda_error(cls) -> None:
        """Clear a stale CUDA error latched on the device before (re)initialization.

        Warp 1.15 leaves the per-thread CUDA error uncleared when ``wp_free_device_async`` fails to add a graph memory
        free node while a capture is still registered, and the next Warp array copy then surfaces that stale error as
        its own failure. Draining here keeps a prior lifecycle's latched error from poisoning this one. Remove once the
        upstream Warp fix lands.
        """
        device = wp.get_device(str(PhysicsManager._device))
        if not device.is_cuda:
            return
        # Private Warp API: guard the whole interaction so a Warp internals reshuffle skips the drain
        # instead of failing simulation start.
        try:
            from warp._src.context import runtime as _wp_runtime

            core = _wp_runtime.core
            # wp_cuda_context_check drains via cudaGetLastError() and prints the drained error; suppress the print
            # and diff Warp's error buffer to report the drain.
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

    # ----- Decimation --------------------------------------------------------------

    @classmethod
    def set_decimation(cls, decimation: int, *, apply_every_physics_step: bool | None = None) -> None:
        """Set the physics steps of one environment step.

        Args:
            decimation: Physics steps per environment step.
            apply_every_physics_step: Whether the environment applies actions before every physics step. ``False``
                lets :meth:`step` advance the whole decimation loop; ``None`` does so only when Newton actuators run
                inside the step.
        """
        NewtonManager._decimation = max(1, decimation)
        NewtonManager._apply_every_physics_step = apply_every_physics_step

    @classmethod
    def handles_decimation(cls) -> bool:
        """Whether one :meth:`step` advances the whole decimation loop.

        Functions that cannot be captured run eagerly inside the step, so only host work between physics steps (see
        :meth:`require_env_decimation`) or the environment's actions prevent it.
        """
        backend = NewtonManager.backend
        if backend is None or backend.env_decimation or NewtonManager._apply_every_physics_step:
            return False
        return NewtonManager._apply_every_physics_step is False or backend.actuators is not None

    @classmethod
    def require_env_decimation(cls) -> None:
        """Make the environment run the decimation loop, because a consumer does host work between physics steps.

        Articulations whose explicit actuators run as Isaac Lab actuator models call this on ``PHYSICS_READY``.
        """
        NewtonManager.backend.env_decimation = True

    @classmethod
    def _uses_cuda_graph(cls) -> bool:
        """Whether the step graph is captured into CUDA graphs."""
        cfg, backend = PhysicsManager._cfg, NewtonManager.backend
        return cfg.use_cuda_graph and "cuda" in backend.device

    @classmethod
    def _capture_graph(cls, fn: Callable[[], None]) -> wp.Graph:
        """Record a step graph segment without executing it."""
        sim = PhysicsManager._sim
        relaxed = has_kit() and (sim.has_gui or sim.has_offscreen_render)
        return nb.capture_graph(PhysicsManager._device, fn, relaxed=relaxed)

    # ----- Step callbacks and actuators -----------------------------------------------

    @classmethod
    def register_step_callback(
        cls, fn: Callable[..., None], phase: StepPhase, *, graphable: bool = True, name: str = ""
    ) -> StepCallback:
        """Run ``fn`` at ``phase`` of every step until it is unregistered or the model is rebuilt.

        Consumers register on ``PHYSICS_READY``; a hard reset discards callbacks together with the buffers they bind.

        Args:
            fn: Function to run. :attr:`StepPhase.STATE_FORCE` functions receive the substep's input state.
            phase: Where the function runs.
            graphable: Whether the function can be recorded into a CUDA graph. Other functions run eagerly in place.
            name: Label used in errors and profiles.

        Returns:
            The callback, for :meth:`unregister_step_callback`.
        """
        return nb.register_step_callback(NewtonManager.backend, fn, phase, graphable=graphable, name=name)

    @classmethod
    def unregister_step_callback(cls, callback: StepCallback) -> None:
        """Stop running a callback; a no-op after the model is rebuilt.

        Args:
            callback: Callback returned by :meth:`register_step_callback`.
        """
        if NewtonManager.backend is not None:
            nb.unregister_step_callback(NewtonManager.backend, callback)

    @classmethod
    def activate_actuators(cls) -> NewtonActuatorAdapter | None:
        """Run the model's Newton actuators inside the step. Idempotent.

        Returns:
            The adapter, or ``None`` when no articulation has explicit Newton actuators.
        """
        return nb.activate_actuators(NewtonManager.backend)

    # ----- Authored state --------------------------------------------------------------

    @classmethod
    def invalidate_fk(
        cls, env_mask: wp.array | None = None, env_ids: wp.array | None = None, articulation_ids: wp.array | None = None
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
        cls._mark_transforms_changed()
        if NewtonManager.backend is not None:
            nb.invalidate_fk(NewtonManager.backend, env_mask, env_ids, articulation_ids)

    @classmethod
    def invalidate_body_state(
        cls,
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
        cls._mark_transforms_changed()
        if NewtonManager.backend is not None:
            nb.invalidate_body_state(NewtonManager.backend, env_ids, env_mask, row_worlds)

    @classmethod
    def view_row_worlds(cls, articulation_ids: wp.array) -> wp.array:
        """Return the world of each row of an articulation view; see :func:`newton_backend.view_row_worlds`.

        Args:
            articulation_ids: ``ArticulationView.articulation_ids``.
        """
        return nb.view_row_worlds(NewtonManager.backend, articulation_ids)

    @classmethod
    def add_model_change(cls, change: ModelFlags) -> None:
        """Notify the solver of a model change before the next step.

        Changes authored before the solver exists are part of the model it is constructed from.
        """
        if NewtonManager.backend is not None and NewtonManager.backend.solver is not None:
            NewtonManager.backend.model_changes.add(change)

    @classmethod
    def transforms_may_change_on_graph_replay(cls) -> bool:
        """Whether state was written during an outer capture, so graph replays may change it without notice."""
        backend = NewtonManager.backend
        return backend is not None and backend.transforms_may_change_on_graph_replay

    @classmethod
    def mark_particles_dirty(cls) -> None:
        """Invalidate scene-data geometry after native particle writes."""
        NewtonManager._scene_data_backend.geometry_timestamp += 1
        cls._flag_outer_capture()

    @classmethod
    def _mark_transforms_changed(cls) -> None:
        """Publish authored rigid-body changes and invalidate cable geometry."""
        if NewtonManager._scene_data_backend is not None:
            NewtonManager._scene_data_backend.transforms_timestamp += 1
            NewtonManager._scene_data_backend.geometry_timestamp += 1
        cls._flag_outer_capture()

    @classmethod
    def _flag_outer_capture(cls) -> None:
        """Remember writes recorded into an outer capture, whose replays bypass Python invalidation."""
        backend = NewtonManager.backend
        if backend is not None and backend.cfg is not None and wp.get_device(backend.device).is_capturing:
            backend.transforms_may_change_on_graph_replay = True

    # ----- Sensors ------------------------------------------------------------------

    @classmethod
    def add_contact_sensor(
        cls,
        body_names_expr: str | list[str] | None = None,
        shape_names_expr: str | list[str] | None = None,
        contact_partners_body_expr: str | list[str] | None = None,
        contact_partners_shape_expr: str | list[str] | None = None,
        verbose: bool = False,
    ) -> SensorContact:
        """Add a contact sensor for reporting contacts between bodies or shapes; see :func:`add_contact_sensor`."""
        with Timer(name="newton_contact_sensor", msg="Contact sensor construction took:", synchronize="both"):
            return nb.add_contact_sensor(
                NewtonManager.backend,
                body_names_expr,
                shape_names_expr,
                contact_partners_body_expr,
                contact_partners_shape_expr,
                verbose,
            )

    @classmethod
    def add_imu_sensor(cls, sites: list[int]) -> SensorIMU:
        """Add an IMU sensor measuring acceleration and angular velocity at sites; see :func:`add_imu_sensor`."""
        return nb.add_imu_sensor(NewtonManager.backend, sites)

    # ----- Model construction -------------------------------------------------------

    @classmethod
    def create_builder(
        cls, up_axis: str | None = None, *, physics_cfg: NewtonCfg | None = None, **kwargs
    ) -> ModelBuilder:
        """Create a :class:`ModelBuilder` configured with default settings.

        Forwards :class:`NewtonShapeCfg` defaults onto Newton's upstream
        ``ModelBuilder.default_shape_cfg`` via :func:`~isaaclab.utils.checked_apply`.
        Falls back to wrapper defaults when no Newton config is active so
        rough-terrain margin/gap still apply during early construction.

        Args:
            up_axis: Override for the up-axis. Defaults to ``"Z"``.
            physics_cfg: Explicit builder settings; None uses the active physics configuration.
            **kwargs: Forwarded to :class:`ModelBuilder`.

        Returns:
            New builder with up-axis and per-shape defaults (gap, margin) applied.
        """
        cfg = PhysicsManager._cfg if physics_cfg is None else physics_cfg
        newton_cfg = cfg if isinstance(cfg, NewtonCfg) else None
        builder = ModelBuilder(up_axis=up_axis or "Z", **kwargs)
        builder.default_bvh_cfg = ModelBuilder.BvhConfig(
            mesh_constructor=newton_cfg.bvh_constructor_geometry if newton_cfg else None,
            gaussian_constructor=newton_cfg.bvh_constructor_gaussian if newton_cfg else None,
            shape_constructor=newton_cfg.bvh_constructor_scene if newton_cfg else None,
            shape_flags=ShapeFlags.VISIBLE,
        )
        cls.register_builder_attributes(builder, newton_cfg.solver_cfg if newton_cfg else None)
        checked_apply(newton_cfg.default_shape_cfg if newton_cfg else NewtonShapeCfg(), builder.default_shape_cfg)
        return builder

    @classmethod
    def get_usd_import_schema_resolvers(cls, solver_cfg: NewtonSolverCfg | None) -> list[SchemaResolver]:
        """Return ordered schema resolvers for physics-model USD imports.

        MJC is enabled for managers that register ``SolverMuJoCo`` attributes. Visualization and articulation-ordering
        builders keep their fixed pair because solver attributes do not affect their outputs.

        Args:
            solver_cfg: Solver configuration of the imported model; ``None`` without Newton physics.
        """
        resolvers: list[SchemaResolver] = [SchemaResolverNewton(), SchemaResolverPhysx()]
        if cls.registers_builder_attributes_from(SolverMuJoCo, solver_cfg):
            resolvers.append(SchemaResolverMjc())
        return resolvers

    @staticmethod
    def inject_terrain_heightfields(stage: Usd.Stage, builder: ModelBuilder, *, root_paths: Sequence[str]) -> list[str]:
        """Replace height-field-tagged terrain colliders with Newton heightfields.

        Scans the stage for prims carrying the ``newton:heightfield:resolution`` attribute authored by
        :class:`~isaaclab.terrains.TerrainImporter`. Each collision mesh is rasterized into a
        :class:`newton.Heightfield` and added to ``builder`` as a static shape. Heightfields compile on MuJoCo roughly
        two orders of magnitude faster than the equivalent terrain mesh while colliding identically at the same
        horizontal resolution.

        Args:
            stage: The USD stage being imported.
            builder: The Newton model builder receiving the heightfield shapes.
            root_paths: Concrete subtree roots to scan.

        Returns:
            Prim paths of converted colliders, which the caller excludes from the USD import.
        """
        ignore_paths: list[str] = []
        xform_cache = UsdGeom.XformCache()
        for prim in (prim for root_path in root_paths for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path))):
            attr = prim.GetAttribute("newton:heightfield:resolution")
            if not attr or not attr.HasAuthoredValue():
                continue
            resolution = float(attr.Get())
            mesh_prim = (
                prim if prim.IsA(UsdGeom.Mesh) else next((p for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Mesh)), None)
            )
            if mesh_prim is None:
                continue
            mesh = UsdGeom.Mesh(mesh_prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
            # Transform vertices into world frame (USD uses row-vector convention).
            mat = np.array(xform_cache.GetLocalToWorldTransform(mesh_prim), dtype=np.float64).reshape(4, 4)
            world = (points @ mat[:3, :3] + mat[3, :3]).astype(np.float32)
            device = str(PhysicsManager._device)
            wp_mesh = wp.Mesh(
                points=wp.array(world, dtype=wp.vec3, device=device),
                indices=wp.array(faces, dtype=wp.int32, device=device),
            )
            heightfield, xform = Heightfield.create_from_mesh(wp_mesh, resolution)
            builder.add_shape_heightfield(heightfield=heightfield, xform=xform)
            logger.info(
                "Converted terrain collider %s (%d faces) to a %dx%d heightfield.",
                prim.GetPath().pathString,
                faces.shape[0] // 3,
                heightfield.nrow,
                heightfield.ncol,
            )
            ignore_paths.append(prim.GetPath().pathString)
        return ignore_paths

    @classmethod
    def prepare_builder(cls, builder: ModelBuilder, solver_cfg: NewtonSolverCfg) -> None:
        """Apply unresolved site requests and solver-specific normalization to the builder about to be finalized.

        The builder survives hard resets, and so do the sites added to it, so a site is added only once.

        Args:
            builder: Builder of the simulated model.
            solver_cfg: Solver configuration of the model.
        """
        global_sites, body_sites, root_sites = cls.inject_sites(builder, {})
        site_map = NewtonManager._site_index_map
        site_map.update((label, (index, None)) for label, index in global_sites.items())
        site_map.update((label, (None, [indices])) for label, indices in body_sites.get(id(builder), {}).items())
        for label, xform in root_sites.items():
            site_map[label] = (None, [[builder.add_site(body=-1, xform=xform, label=label)]])
        builder.up_axis = Axis.Z
        cls.prepare_solver_builder(builder, solver_cfg)

    @classmethod
    def register_site(cls, body_pattern: str | None, xform: wp.transform, *, per_world: bool = False) -> str:
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
        requests = NewtonManager._site_requests
        key = (body_pattern, per_world, tuple(xform))
        if key not in requests:
            requests[key] = (f"ft_{len(requests)}", xform)
        return requests[key][0]

    @classmethod
    def inject_sites(
        cls, main_builder: ModelBuilder, source_builders: dict[str, ModelBuilder]
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
        for (body_pattern, per_world, _), (label, xform) in NewtonManager._site_requests.items():
            if label in NewtonManager._site_index_map:
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

    @classmethod
    def get_world_builder_hooks(cls) -> list[Callable[[ModelBuilder, int, np.ndarray, np.ndarray], None]]:
        """Return the hooks extending every world built by Newton replication."""
        return NewtonManager._world_builder_hooks

    @classmethod
    def record_clone(cls, record: NewtonCloneRecord, site_index_map: dict[str, SiteEntry]) -> None:
        """Record native replication outputs for model finalization and consumers.

        Args:
            record: Replication outputs.
            site_index_map: Sites resolved during replication, by label.
        """
        NewtonManager._clone = record
        NewtonManager._site_index_map.update(site_index_map)

    @classmethod
    def get_clone_record(cls) -> NewtonCloneRecord | None:
        """Return the outputs of the last native replication, or ``None`` without replication."""
        return NewtonManager._clone

    @classmethod
    def get_site_index_map(cls) -> dict[str, SiteEntry]:
        """Return resolved sites by label."""
        return NewtonManager._site_index_map

    @classmethod
    def get_world_xforms(cls) -> list[wp.transform] | None:
        """Return the root transform of each cloned world, or ``None`` without replication."""
        clone = NewtonManager._clone
        return None if clone is None else clone.world_xforms

    @classmethod
    def get_clone_source_builders(cls) -> dict[str, ModelBuilder]:
        """Return per-source builders retained from replication, keyed by clone-plan source path."""
        clone = NewtonManager._clone
        return {} if clone is None else clone.source_builders

    @classmethod
    def request_extended_state_attribute(cls, attr: str) -> None:
        """Request an extended state attribute (e.g. ``"body_qdd"``) from the builder before finalization.

        Args:
            attr: State attribute name (must be in ``State.EXTENDED_ATTRIBUTES``).
        """
        cls._shared_builder().request_state_attributes(attr)

    @classmethod
    def request_extended_contact_attribute(cls, attr: str) -> None:
        """Request an extended contact attribute (e.g. ``"force"``) from the builder before finalization.

        Args:
            attr: Contact attribute name.
        """
        cls._shared_builder().request_contact_attributes(attr)

    @staticmethod
    def _shared_builder() -> ModelBuilder:
        """Return the simulation's shared builder, which survives hard resets."""
        sim = SimulationContext.instance()
        return sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))

    # ----- Accessors -------------------------------------------------------------------

    @classmethod
    def get_solver(cls) -> SolverBase | None:
        """Return the active Newton solver, or ``None`` before it is constructed."""
        backend = NewtonManager.backend
        return None if backend is None else backend.solver

    @classmethod
    def get_model(cls) -> Model | None:
        """Return the active physics model."""
        backend = NewtonManager.backend
        return None if backend is None else backend.model

    @classmethod
    def get_state_0(cls) -> State | None:
        """Return the current state."""
        backend = NewtonManager.backend
        return None if backend is None else backend.state_0

    @classmethod
    def get_state_1(cls) -> State | None:
        """Return the spare state of double-buffered solvers."""
        backend = NewtonManager.backend
        return None if backend is None else backend.state_1

    @classmethod
    def get_control(cls) -> Control | None:
        """Return the control inputs."""
        backend = NewtonManager.backend
        return None if backend is None else backend.control

    @classmethod
    def get_contacts(cls) -> Contacts | None:
        """Return the current Newton contacts, if the active solver exposes them."""
        backend = NewtonManager.backend
        return None if backend is None else backend.contacts

    @classmethod
    def get_scene_data_backend(cls) -> SceneDataBackend | None:
        """Return the SceneDataBackend for the SceneDataProvider."""
        return NewtonManager._scene_data_backend

    @classmethod
    def get_scene_data_provider(cls) -> SceneDataProvider:
        """Return the active scene data provider."""
        return SimulationContext.instance().get_scene_data_provider()

    @classmethod
    def get_physics_sim_view(cls) -> list:
        """Return the registered articulation views."""
        return [view for (manager, _), view in cls.views.items() if manager is NewtonManager]

    @classmethod
    def create_visual_material_writer(cls, batches: tuple[VisualMaterialBatch, ...]) -> VisualMaterialWriter:
        """Compile material-to-shape addresses for the active Newton model."""
        return NewtonManager.backend.create_visual_material_writer(batches)

    @classmethod
    def create_visual_shape_color_writer(
        cls, asset: BaseArticulation, body_names: tuple[str, ...]
    ) -> VisualShapeColorWriter:
        """Compile selected articulation-body shape addresses for the active Newton model."""
        model = cls.get_model()
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

    @classmethod
    def video_capture_backend(cls) -> str:
        """Newton GL headless perspective video capture."""
        return "newton_gl"

    # ----- Solver hooks ------------------------------------------------------------------
    # Overridden by solver managers. They are stateless and take every input explicitly, so a solver manager can
    # drive several backends at once.

    solver_class: ClassVar[type[SolverBase] | None] = None
    """Newton solver the default :meth:`create_solver` constructs from the matching configuration fields."""

    builder_attribute_solvers: ClassVar[tuple[type[SolverBase], ...]] = ()
    """Solvers whose custom builder attributes are registered before import."""

    single_state: ClassVar[bool] = False
    """Whether the solver steps in place on one :class:`newton.State`."""

    supports_deterministic: ClassVar[bool] = False
    """Whether the solver can honor a Warp determinism guarantee."""

    supports_contact_sensors: ClassVar[bool] = True
    """Whether Newton contact sensors can read the solver's contacts."""

    prepares_step: ClassVar[bool] = False
    """Whether :meth:`prepare_step` does work, so the step graph runs it."""

    ignored_model_changes: ClassVar[dict[int, str]] = {}
    """Model changes the solver does not apply after construction, mapped to the warning logged once."""

    @classmethod
    def create_solver(
        cls,
        model: Model,
        solver_cfg: NewtonSolverCfg,
        deterministic_mode: wp.DeterministicMode = wp.DeterministicMode.NOT_GUARANTEED,
    ) -> SolverBase:
        """Construct the configured solver. Coupled solvers call this to build their entries.

        The default constructs :attr:`solver_class` from the configuration fields its constructor accepts.

        Args:
            model: Model or model view the solver runs on.
            solver_cfg: Solver configuration.
            deterministic_mode: Determinism guarantee requested for the solver.

        Returns:
            The solver.
        """
        if cls.solver_class is None:
            raise NotImplementedError(f"{cls.__name__} does not implement solver construction.")
        return cls.solver_class(model, **cls.solver_kwargs(cls.solver_class, solver_cfg, deterministic_mode))

    @staticmethod
    def solver_kwargs(solver_cls: type, solver_cfg: Any, deterministic_mode: wp.DeterministicMode) -> dict:
        """Return the configuration fields that match the solver constructor.

        Args:
            solver_cls: Solver class to construct.
            solver_cfg: Solver configuration.
            deterministic_mode: Forwarded when the constructor accepts ``deterministic``.

        Returns:
            Constructor keyword arguments, excluding ``self`` and ``model``.
        """
        valid = set(inspect.signature(solver_cls.__init__).parameters) - {"self", "model"}
        kwargs = {key: value for key, value in to_dict(solver_cfg).items() if key in valid}
        if "deterministic" in valid:
            kwargs["deterministic"] = deterministic_mode
        return kwargs

    @classmethod
    def validate_cfg(cls, backend: NewtonBackend) -> None:
        """Reject configurations the solver cannot run, before it is constructed.

        Args:
            backend: Backend whose solver is about to be constructed.
        """
        mode = backend.deterministic_mode
        if mode != wp.DeterministicMode.NOT_GUARANTEED and not cls.supports_deterministic:
            raise ValueError(
                f"Newton deterministic mode {mode.name} is not supported by {type(backend.cfg.solver_cfg).__name__}."
                " Use MJWarp on the GPU, XPBD, or Featherstone, or disable deterministic mode."
            )

    @classmethod
    def register_builder_attributes(cls, builder: ModelBuilder, solver_cfg: NewtonSolverCfg | None) -> None:
        """Register custom attributes the solver reads from the model.

        Args:
            builder: Builder that receives imported assets.
            solver_cfg: Solver configuration of the model.
        """
        for solver_cls in cls.builder_attribute_solvers:
            solver_cls.register_custom_attributes(builder)

    @classmethod
    def registers_builder_attributes_from(
        cls, solver_cls: type[SolverBase], solver_cfg: NewtonSolverCfg | None
    ) -> bool:
        """Return whether this manager registers ``solver_cls``'s custom builder attributes.

        Args:
            solver_cls: Solver whose attributes, and matching USD schemas, may be imported.
            solver_cfg: Solver configuration of the model.
        """
        return solver_cls in cls.builder_attribute_solvers

    @classmethod
    def prepare_solver_builder(cls, builder: ModelBuilder, solver_cfg: NewtonSolverCfg) -> None:
        """Normalize a complete builder for the solver before finalization. The default is a no-op.

        Args:
            builder: Builder about to be finalized.
            solver_cfg: Solver configuration of the model.
        """

    @classmethod
    def uses_collision_pipeline(cls, backend: NewtonBackend) -> bool:
        """Whether contacts come from Newton's :class:`~newton.CollisionPipeline` instead of the solver."""
        return True

    @classmethod
    def supports_body_forces(cls, backend: NewtonBackend) -> bool:
        """Whether the solver consumes applied rigid-body forces from :attr:`newton.State.body_f`."""
        return True

    @classmethod
    def supports_graph_capture(cls, backend: NewtonBackend) -> bool:
        """Whether the configured solver can be recorded into a CUDA graph."""
        return True

    @classmethod
    def create_contacts(cls, backend: NewtonBackend) -> Contacts | None:
        """Allocate contacts the solver reports from internal collision detection.

        Called only when :meth:`uses_collision_pipeline` is ``False``.
        """
        return None

    @classmethod
    def prepare_contacts(cls, backend: NewtonBackend) -> None:
        """Bind solver-owned buffers to newly allocated contacts. The default is a no-op."""

    @classmethod
    def initialize_output_state(cls, backend: NewtonBackend, state: State) -> None:
        """Initialize solver-owned buffers of the output state of double-buffered solvers. The default is a no-op."""

    @classmethod
    def prepare_step(cls, backend: NewtonBackend, state: State) -> None:
        """Refresh solver acceleration structures once per physics step, before collision; set :attr:`prepares_step`.

        Overrides must be graphable.
        """

    @classmethod
    def step_solver(
        cls, backend: NewtonBackend, state_in: State, state_out: State, contacts: Contacts | None, dt: float
    ) -> None:
        """Advance one solver substep.

        Args:
            backend: Backend whose solver to step.
            state_in: Input state.
            state_out: Output state; the same object as ``state_in`` for single-state solvers.
            contacts: Contacts from the collision pipeline, or ``None`` when the solver detects contacts internally.
            dt: Substep duration [s].
        """
        backend.solver.step(state_in, state_out, backend.control, contacts, dt)

    @classmethod
    def reset_solver(cls, backend: NewtonBackend, state: State, world_mask: wp.array) -> None:
        """Clear solver-owned history for masked worlds while keeping authored joint state.

        Args:
            backend: Backend whose solver to reset.
            state: State whose solver-owned buffers are reset.
            world_mask: Per-world mask of shape ``(world_count + 1,)``; the final entry selects global entities.
        """
        backend.solver.reset(state, world_mask=world_mask, flags=0)

    @classmethod
    def eval_fk(
        cls, backend: NewtonBackend, state: State, world_mask: wp.array | None, fk_mask: wp.array | None
    ) -> None:
        """Update body state from joint coordinates for masked articulations.

        Args:
            backend: Backend whose model to evaluate.
            state: State to update in place.
            world_mask: Per-world mask of reset worlds, or ``None`` for all worlds.
            fk_mask: Per-articulation mask, or ``None`` for all articulations.
        """
        eval_fk(backend.model, state.joint_q, state.joint_qd, state, fk_mask)

    @classmethod
    def check_status(cls, backend: NewtonBackend, captured: bool) -> None:
        """Raise asynchronous solver failures after a step. The default is a no-op.

        Args:
            backend: Backend that stepped.
            captured: Whether the step replayed a captured graph.
        """

    @classmethod
    def log_debug(cls, backend: NewtonBackend) -> None:
        """Log solver diagnostics after a step when debug mode is enabled. The default is a no-op."""

    @classmethod
    def create_fixed_tendon_control(cls, articulation: Any, model: Model) -> Any:
        """Build the solver's fixed-tendon command adapter for ``articulation``.

        Tendon state is backend-neutral and lives on the articulation; how a target reaches the solver is not. Only
        solvers that transmit to tendons implement this.

        Args:
            articulation: Newton articulation to drive.
            model: Finalized model of the articulation.

        Raises:
            NotImplementedError: For solvers without fixed-tendon transmission.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not drive fixed tendons. Fixed-tendon targets require a solver"
            " that transmits to tendons, such as MJWarp."
        )

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
