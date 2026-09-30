# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton physics manager for Isaac Lab."""

from __future__ import annotations

import contextlib
import gc
import logging
import re
from collections.abc import Callable, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
import warp as wp
from newton import Axis, Contacts, Control, Heightfield, Model, ModelBuilder, ModelFlags, ShapeFlags, State
from newton.selection import ArticulationView
from newton.sensors import SensorContact, SensorFrameTransform, SensorIMU
from newton.solvers import SolverBase, SolverMuJoCo
from newton.usd import SchemaResolver, SchemaResolverMjc, SchemaResolverNewton, SchemaResolverPhysx

from pxr import Usd, UsdGeom

from isaaclab.physics import CallbackHandle, PhysicsEvent, PhysicsManager
from isaaclab.scene_data import SceneDataBackend, SceneDataFormat, SceneDataProvider
from isaaclab.sim import SimulationContext
from isaaclab.utils import checked_apply
from isaaclab.utils.buffers import TimestampedBuffer
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

from . import runtime as newton_runtime
from .runtime import (
    NewtonBuildRequests,
    NewtonCloneRecord,
    NewtonRuntime,
    NewtonSchema,
    SiteEntry,
    resolve_deterministic_mode,
    validate_deterministic_mode,
)
from .solver_binding import NewtonSolverBinding
from .step_program import StepPhase, StepStage

if TYPE_CHECKING:
    from isaaclab.actuators.newton import NewtonActuatorAdapter
    from isaaclab.assets import BaseArticulation
    from isaaclab.renderers.base_renderer import VisualMaterialBatch


logger = logging.getLogger(__name__)


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


class NewtonBackend:
    """Own one finalized Newton model and its native state and control buffers."""

    def __init__(self, cfg: NewtonBackendCfg):
        builder = SimulationContext.instance().get_or_create_backend(NewtonBuilderCfg(physics_cfg=cfg.physics_cfg))
        self.model = builder.finalize(device=cfg.device)
        self.particle_ranges: dict[str, tuple[int, int]] = {}
        # Newton 1.6 preserves groups through builder replication but not finalization.
        # Remove this snapshot when the pinned Newton includes newton-physics/newton#3326.
        self.deformable_ranges = {
            label: (start, end - start, kind)
            for family, kind in (("cloth", "surface"), ("soft", "volume"))
            for label, start, end in zip(
                getattr(builder, f"_{family}_label"),
                getattr(builder, f"_{family}_particle_start"),
                getattr(builder, f"_{family}_particle_end"),
                strict=True,
            )
        }
        self.model.num_envs = self.model.world_count
        simulation = isinstance(cfg.physics_cfg, NewtonCfg)
        soft_contact = cfg.physics_cfg.soft_contact_cfg if simulation else None
        if soft_contact is not None:
            self.model.soft_contact_ke = float(soft_contact.soft_contact_ke)
            self.model.soft_contact_kd = float(soft_contact.soft_contact_kd)
            self.model.soft_contact_mu = float(soft_contact.soft_contact_mu)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state() if simulation else None
        self.control = self.model.control() if simulation else None
        self.geometry_offsets = (
            {} if simulation else {path: bounds[0] for path, bounds in self.deformable_ranges.items()}
        )
        self.bvh_refit = TimestampedBuffer()

    def create_visual_material_writer(self, batches: tuple[VisualMaterialBatch, ...]) -> VisualMaterialWriter:
        """Bind material writes to this resource's native shape-color buffer."""
        return VisualMaterialWriter(self.model, batches)

    def close(self) -> None:
        """Drop native handles after consumers release their bindings."""
        self.control = self.state_1 = self.state_0 = self.model = None
        self.deformable_ranges.clear()
        self.particle_ranges = {}
        self.geometry_offsets = {}
        self.bvh_refit = TimestampedBuffer()


class NewtonQueries:
    """Stateless query and capture functions operating on an explicit Newton resource."""

    @staticmethod
    def run_query(
        backend: NewtonBackend,
        timestamp: int,
        query: Callable[[], None],
        graph: tuple[tuple[int, ...], wp.Graph] | None,
        *,
        use_cuda_graph: bool = True,
    ) -> tuple[tuple[int, ...], wp.Graph] | None:
        """Refresh shared BVHs once per publication and run a consumer-owned query graph.

        Args:
            backend: Shared model, state, and acceleration structures.
            timestamp: Sum of the monotonic SDP transform and geometry timestamps for this state.
            query: Camera or ray-cast work using the supplied backend.
            graph: This consumer's previous captured query and pointer layout, or None.
            use_cuda_graph: Whether to capture GPU queries. CPU queries always run eagerly.

        Returns:
            The consumer's captured query and pointer layout, or None for eager execution.
        """
        model, state = backend.model, backend.state_0
        pointers = tuple(array.ptr if array is not None else 0 for array in (state.body_q, state.particle_q))
        cached = backend.bvh_refit
        if cached.timestamp != timestamp:
            if model.device.is_cuda and use_cuda_graph:
                if cached.data is None or cached.data[0] != pointers:
                    refit = partial(NewtonQueries._refit_bvh, backend)
                    cached.data = pointers, NewtonQueries.capture_graph(str(model.device), refit, relaxed=has_kit())
                wp.capture_launch(cached.data[1])
            else:
                NewtonQueries._refit_bvh(backend)
            cached.timestamp = timestamp
        if not model.device.is_cuda or not use_cuda_graph:
            query()
            return None
        if graph is None or graph[0] != pointers:
            graph = pointers, NewtonQueries.capture_graph(str(model.device), query, relaxed=has_kit())
        wp.capture_launch(graph[1])
        return graph

    @staticmethod
    def capture_graph(device: str, capture_target: Callable[[], None], *, relaxed: bool = False) -> wp.Graph:
        """Record work without an eager warmup or graph replay, using Warp's public capture API."""
        stream = wp.get_stream(device)
        mode = wp.CaptureMode.THREAD_LOCAL
        if relaxed:
            # RTX uses the legacy CUDA stream. A nonblocking stream avoids implicit synchronization.
            stream = wp.stream_from_torch(torch.cuda.Stream(device=device))
            mode = wp.CaptureMode.RELAXED
        with NewtonQueries._paused_gc(), wp.ScopedStream(stream):
            with wp.ScopedCapture(stream=stream, capture_mode=mode) as capture:
                capture_target()
        return capture.graph

    @staticmethod
    def _refit_bvh(backend: NewtonBackend) -> None:
        model, state = backend.model, backend.state_0
        if model.shape_count:
            model.bvh_refit_shapes(state)
        if model.particle_count:
            model.bvh_refit_particles(state)

    @staticmethod
    @contextlib.contextmanager
    def _paused_gc():
        """Keep collection-driven frees out of CUDA graph capture."""
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            yield
        finally:
            if was_enabled:
                gc.enable()
                gc.collect()


class NewtonSceneDataBackend(SceneDataBackend):
    """Scene data backend that reads rigid body transforms from Newton's simulation state.

    The backend reads ``body_q`` (an array of :class:`wp.transformf`) from
    Newton's current state and exposes it as :class:`SceneDataFormat.Transform`.
    Body paths come from the model's ``body_label`` attribute.
    """

    def __init__(self, runtime: Callable[[], NewtonRuntime | None]):
        """Initialize the backend.

        Args:
            runtime: Returns the runtime whose state to publish; the runtime changes on every hard reset.
        """
        self._runtime = runtime
        self._transforms = SceneDataFormat.Transform()
        self.transforms_timestamp = 0
        self.geometry_timestamp = 0
        self._geometry_batches = []

    def initialize_geometry(self, cable_bindings: dict[str, list[int]]) -> None:
        """Bind imported geometry paths to native particle ranges and capsule endpoints.

        Args:
            cable_bindings: Native capsule shape indices of each open cable.
        """
        state = self._runtime().backend.state_0
        self._geometry_batches = [
            (source, ranges)
            for source, ranges in self._geometry_batches
            if source._cls is not SceneDataFormat.CapsuleEndpoints
        ]
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
            source.transforms = state.body_q
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
        runtime = self._runtime()
        return None if runtime is None else runtime.backend.model

    @property
    def state(self) -> State | None:
        """Return native physics state, reconciled with authored state, without entering the rendering path."""
        runtime = self._runtime()
        if runtime is None:
            return None
        if runtime.transforms_may_change_on_graph_replay:
            # Raw external graph replays bypass Python invalidation, so these reads must stay conservative.
            self.transforms_timestamp += 1
            self.geometry_timestamp += 1
        newton_runtime.reconcile(runtime)
        return runtime.backend.state_0


class NewtonManager(PhysicsManager):
    """Newton physics manager for Isaac Lab.

    The manager is a thin facade over three pieces of data with separate lifetimes:

    * :class:`~isaaclab_newton.physics.runtime.NewtonBuildRequests`: construction requests (sites, extended
      attributes, world hooks, clone outputs) collected before the model is finalized and kept across hard resets.
    * :class:`~isaaclab_newton.physics.runtime.NewtonRuntime`: everything bound to one finalized model (solver,
      contacts, sensors, actuators, consumer stages, and the compiled step program). A hard reset discards it.
    * :class:`~isaaclab_newton.physics.step_program.StepProgram`: the whole step, including the decimation loop,
      Newton actuators, and consumer stages, compiled to straight-line operations over explicit buffers. Graph-safe
      segments replay from CUDA graphs; other operations run eagerly at the same position.

    The runtime is plain data driven by the functions in :mod:`isaaclab_newton.physics.runtime`, which take it
    explicitly (``runtime.step(rt, steps, capture)``). The manager only supplies the active simulation's runtime, so
    the functional core already supports several runtimes bound to different Newton backends.

    Solver-specific behavior lives in a :class:`~isaaclab_newton.physics.solver_binding.NewtonSolverBinding`.
    Concrete managers only select the binding through :attr:`solver_binding`, and
    :meth:`NewtonCfg.__post_init__` selects the manager from :attr:`NewtonSolverCfg.class_type`.

    Lifecycle: ``initialize() -> reset() -> step()`` (repeated) ``-> close()``.
    """

    solver_binding: ClassVar[type[NewtonSolverBinding] | None] = None
    """Solver binding constructed by :meth:`initialize_solver`; set by concrete managers."""

    _requests: ClassVar[NewtonBuildRequests] = NewtonBuildRequests()
    _runtime: ClassVar[NewtonRuntime | None] = None
    _scene_data_backend: ClassVar[NewtonSceneDataBackend | None] = None
    _decimation: ClassVar[int] = 1
    _fold_decimation: ClassVar[bool | None] = None

    # ----- PhysicsManager lifecycle -------------------------------------------

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
        NewtonManager._scene_data_backend = NewtonSceneDataBackend(lambda: NewtonManager._runtime)

    @classmethod
    def reset(cls, soft: bool = False) -> None:
        """Reset physics simulation.

        A hard reset (``soft=False``) discards the runtime, releases the finalized model, and rebuilds both from the
        retained builder, so nothing captured against the old buffers survives. A soft reset keeps the runtime.

        Args:
            soft: If True, skip full reinitialization.
        """
        if soft:
            return
        if NewtonManager._runtime is not None:
            cls.dispatch_event(PhysicsEvent.STOP)
            cls._release_runtime()
        cls.start_simulation()
        cls.initialize_solver()

    @classmethod
    def forward(cls) -> None:
        """Reset solver internals and run FK for worlds whose state was authored since the last boundary."""
        runtime = NewtonManager._runtime
        if runtime is not None:
            newton_runtime.reconcile(runtime)

    @classmethod
    def step(cls) -> None:
        """Advance physics through the compiled step program.

        The program covers the whole decimation loop when Newton actuators are active (see :meth:`handles_decimation`)
        and one physics step otherwise. It is compiled, and captured on CUDA, on the first step after any change to
        its structure, once authored state is reconciled. Capture does not advance physics.
        """
        sim = PhysicsManager._sim
        if sim is None or not sim.is_playing():
            return
        runtime = NewtonManager._runtime
        steps = NewtonManager._decimation if cls.handles_decimation() else 1
        capture = cls._capture_graph if cls._uses_cuda_graph() else None
        program = newton_runtime.step(runtime, steps, capture)
        PhysicsManager._sim_time += runtime.schema.physics_dt * program.steps
        cls._mark_transforms_changed()
        runtime.solver.check_status(program.is_captured)
        if PhysicsManager._cfg.debug_mode:
            runtime.solver.log_debug()

    @classmethod
    def prepare(cls) -> None:
        """Compile and capture the step program ahead of the next step, without advancing physics.

        Authored state is reconciled first, so capture sees the state the next step starts from.
        """
        runtime = NewtonManager._runtime
        newton_runtime.apply_model_changes(runtime)
        newton_runtime.reconcile(runtime)
        steps = NewtonManager._decimation if cls.handles_decimation() else 1
        with Timer(name="newton_cuda_graph", msg="CUDA graph took:"):
            newton_runtime.prepare(runtime, steps, cls._capture_graph if cls._uses_cuda_graph() else None)

    @classmethod
    def close(cls) -> None:
        """Clean up Newton physics resources."""
        super().close()
        cls.clear()

    @classmethod
    def clear(cls) -> None:
        """Release the runtime and construction requests (callbacks are cleared by :meth:`close`)."""
        cls._release_runtime()
        NewtonManager._requests = NewtonBuildRequests()
        NewtonManager._scene_data_backend = None
        NewtonManager._decimation = 1
        NewtonManager._fold_decimation = None
        for key in [key for key in NewtonManager.views if key[0] is NewtonManager]:
            del NewtonManager.views[key]

    @classmethod
    def _release_runtime(cls) -> None:
        """Drop the runtime and return its model to the simulation registry."""
        runtime = NewtonManager._runtime
        NewtonManager._runtime = None
        if runtime is not None:
            SimulationContext.instance().close_backend(runtime.backend)

    # ----- Model construction ------------------------------------------------

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
        if cls.solver_binding is not None:
            cls.solver_binding.register_builder_attributes(builder)
        checked_apply(newton_cfg.default_shape_cfg if newton_cfg else NewtonShapeCfg(), builder.default_shape_cfg)
        return builder

    @classmethod
    def get_usd_import_schema_resolvers(cls) -> list[SchemaResolver]:
        """Return ordered schema resolvers for physics-model USD imports.

        MJC is enabled for managers that register ``SolverMuJoCo`` attributes. Visualization and articulation-ordering
        builders keep their fixed pair because solver attributes do not affect their outputs.
        """
        resolvers: list[SchemaResolver] = [SchemaResolverNewton(), SchemaResolverPhysx()]
        if cls.solver_binding is not None and cls.solver_binding.registers_builder_attributes_from(SolverMuJoCo):
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
    def build_requests(cls) -> NewtonBuildRequests:
        """Return the construction requests collected before model finalization."""
        return NewtonManager._requests

    @classmethod
    def record_clone(cls, record: NewtonCloneRecord, geometry_batches: list) -> None:
        """Record native replication outputs for model finalization and consumers.

        Args:
            record: Replication outputs.
            geometry_batches: Deformable and particle geometry published through scene data.
        """
        requests = NewtonManager._requests
        requests.clone = record
        requests.site_index_map = dict(record.site_index_map)
        NewtonManager._scene_data_backend._geometry_batches = geometry_batches

    @classmethod
    def cl_register_site(cls, body_pattern: str | None, xform: wp.transform, *, per_world: bool = False) -> str:
        """Request a site for injection into prototypes before replication.

        Sensors call this during ``__init__``. Identical ``(body_pattern, per_world, transform)`` requests share a site.
        The pattern matches prototype-local body labels (e.g. ``"Robot/finger.*"``) during replication, and wildcard
        patterns create one site per matched body.

        Args:
            body_pattern: Regex matched against body labels, or ``None`` for a global site.
            xform: Site transform relative to the body.
            per_world: Create one bodyless site in each cloned world's frame. Requires ``body_pattern=None``.

        Returns:
            The assigned site label.
        """
        return NewtonManager._requests.register_site(body_pattern, xform, per_world=per_world)

    @classmethod
    def get_site_index_map(cls) -> dict[str, SiteEntry]:
        """Return resolved sites by label."""
        return NewtonManager._requests.site_index_map

    @classmethod
    def get_world_xforms(cls) -> list[wp.transform] | None:
        """Return the root transform of each cloned world, or ``None`` without replication."""
        clone = NewtonManager._requests.clone
        return None if clone is None else clone.world_xforms

    @classmethod
    def get_clone_source_builders(cls) -> dict[str, ModelBuilder]:
        """Return per-source builders retained from replication, keyed by clone-plan source path."""
        clone = NewtonManager._requests.clone
        return {} if clone is None else clone.source_builders

    @classmethod
    def request_extended_state_attribute(cls, attr: str) -> None:
        """Request an extended state attribute (e.g. ``"body_qdd"``) before model finalization.

        Args:
            attr: State attribute name (must be in ``State.EXTENDED_ATTRIBUTES``).
        """
        NewtonManager._requests.state_attributes.add(attr)

    @classmethod
    def request_extended_contact_attribute(cls, attr: str) -> None:
        """Request an extended contact attribute (e.g. ``"force"``) before model finalization.

        Args:
            attr: Contact attribute name.
        """
        NewtonManager._requests.contact_attributes.add(attr)

    @classmethod
    def start_simulation(cls) -> None:
        """Finalize the model and create the runtime bound to it.

        Dispatches ``MODEL_INIT`` before finalization, so consumers can still author the builder, and
        ``PHYSICS_READY`` after it, when consumers bind views, sensors, actuators, and stages to the runtime.
        """
        sim = SimulationContext.instance()
        cfg = NewtonBackendCfg(physics_cfg=sim.cfg.physics, device=sim.device)
        builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=cfg.physics_cfg))
        cls._drain_stale_cuda_error()

        logger.info("Dispatching MODEL_INIT callbacks")
        cls.dispatch_event(PhysicsEvent.MODEL_INIT)

        requests = NewtonManager._requests
        requests.prepare_builder(builder)
        builder.up_axis = Axis.Z
        cls.solver_binding.prepare_builder(builder)
        device = PhysicsManager._device
        logger.info(f"Finalizing model on device: {device}")
        with Timer(
            name="newton_finalize_builder",
            msg="Finalize builder took:",
            activity="Finalizing physics model",
            synchronize="both",
            device=device,
        ):
            backend = sim.get_or_create_backend(cfg)
        model = backend.model
        clone = requests.clone
        backend.particle_ranges = {} if clone is None else clone.particle_ranges
        model.set_gravity(sim.cfg.gravity)
        if clone is not None:
            model.num_envs = clone.num_envs
        physics_cfg = PhysicsManager._cfg
        schema = NewtonSchema.from_model(
            model,
            physics_dt=cls.get_physics_dt(),
            num_substeps=physics_cfg.num_substeps,
            collision_decimation=physics_cfg.collision_decimation,
            world_prototypes=None if clone is None else clone.world_prototypes,
        )
        NewtonManager._runtime = newton_runtime.create_runtime(backend, schema)

        NewtonManager._scene_data_backend.initialize_geometry({} if clone is None else clone.cable_bindings)
        logger.info("Dispatching PHYSICS_READY callbacks")
        cls.dispatch_event(PhysicsEvent.PHYSICS_READY)

    @classmethod
    def initialize_solver(cls) -> None:
        """Construct the solver and contacts, and establish the initial body state.

        The step program is compiled and captured on the first step, after the environment authors its initial state.
        Initialization does not advance physics.
        """
        cfg = PhysicsManager._cfg
        if cfg is None:
            return
        runtime = NewtonManager._runtime
        with Timer(
            name="newton_initialize_solver",
            msg="Initialize solver took:",
            activity="Initializing solver",
            synchronize="both",
            device=cls.get_device(),
        ):
            mode = resolve_deterministic_mode(cfg)
            validate_deterministic_mode(cfg, mode, NewtonManager._requests.state_attributes)
            newton_runtime.bind_solver(runtime, cls.solver_binding, cfg, mode)

        # Picking applies forces inside solver substeps, so its stage must exist before the first capture.
        sim = PhysicsManager._sim
        if runtime.solver.supports_body_forces and sim is not None:
            sim._prepare_newton_visualizer_for_capture()

        runtime.solver.eval_fk(runtime.backend.state_0, None, None)
        cls._mark_transforms_changed()
        if cfg.use_cuda_graph and "cuda" in PhysicsManager._device and not runtime.solver.supports_graph_capture:
            logger.warning(
                "%s does not support CUDA graph capture for the current solver configuration; using eager execution.",
                cls.__name__,
            )

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

    # ----- Step program ------------------------------------------------------

    @classmethod
    def add_stage(
        cls, fn: Callable[..., None], phase: StepPhase, *, graph_safe: bool = True, name: str = ""
    ) -> StepStage:
        """Schedule an operation into every Newton step program until the model is rebuilt.

        Consumers add stages while ``PHYSICS_READY`` dispatches; a hard reset discards them together with the buffers
        they bind, and consumers add them again on the next ``PHYSICS_READY``.

        Args:
            fn: Operation to run. :attr:`StepPhase.SUBSTEP` operations receive the substep's input state.
            phase: Program position.
            graph_safe: Whether the operation can be recorded into a CUDA graph. Other operations run eagerly.
            name: Label used in profiles.

        Returns:
            The stage, for :meth:`remove_stage`.
        """
        stage = StepStage(fn, phase, graph_safe, name or getattr(fn, "__name__", ""))
        return newton_runtime.add_stage(NewtonManager._runtime, stage)

    @classmethod
    def remove_stage(cls, stage: StepStage) -> None:
        """Remove a stage added by :meth:`add_stage`; a no-op after the model is rebuilt.

        Args:
            stage: Stage returned by :meth:`add_stage`.
        """
        if NewtonManager._runtime is not None:
            newton_runtime.remove_stage(NewtonManager._runtime, stage)

    @classmethod
    def activate_newton_actuator_path(cls) -> None:
        """Run the articulations' Newton actuators inside the step program.

        Idempotent; called by every articulation whose explicit actuators run as Newton actuators. The first call
        builds the single :class:`NewtonActuatorAdapter` over the model's actuators. Actuators that are not graph-safe
        run eagerly between captured segments of the same program.
        """
        newton_runtime.activate_actuators(NewtonManager._runtime)

    @classmethod
    def get_actuator_adapter(cls) -> NewtonActuatorAdapter | None:
        """Return the adapter running Newton actuators inside the step program, if any articulation activated it."""
        runtime = NewtonManager._runtime
        return None if runtime is None else runtime.actuators

    @classmethod
    def set_decimation(cls, decimation: int, *, fold: bool | None = None) -> None:
        """Set the physics steps one :meth:`step` advances when the manager owns the decimation loop.

        Args:
            decimation: Physics steps per environment step.
            fold: Whether the environment allows folding the loop into one :meth:`step`. ``None`` folds only when
                Newton actuators run inside the step program.
        """
        NewtonManager._decimation = max(1, decimation)
        NewtonManager._fold_decimation = fold

    @classmethod
    def handles_decimation(cls) -> bool:
        """Whether one :meth:`step` advances the whole decimation loop.

        The loop folds unless a consumer does host work between physics steps; graph safety of individual
        operations does not matter, because operations that cannot be captured run eagerly inside the program.
        """
        runtime = NewtonManager._runtime
        return runtime is not None and newton_runtime.owns_decimation(runtime, NewtonManager._fold_decimation)

    @classmethod
    def require_host_physics_steps(cls) -> None:
        """Declare host work between physics steps, so the environment drives the decimation loop.

        Consumers call this while ``PHYSICS_READY`` dispatches, for example articulations whose explicit actuators run
        as Isaac Lab actuator models.
        """
        NewtonManager._runtime.host_physics_steps = True

    @classmethod
    def _uses_cuda_graph(cls) -> bool:
        """Whether the step program is captured into CUDA graphs."""
        cfg, device = PhysicsManager._cfg, PhysicsManager._device
        return (
            cfg.use_cuda_graph
            and device is not None
            and "cuda" in device
            and NewtonManager._runtime.solver.supports_graph_capture
        )

    @classmethod
    def _capture_graph(cls, capture_target: Callable[[], None]) -> wp.Graph:
        """Record a program segment without an eager warmup or graph replay."""
        sim = PhysicsManager._sim
        relaxed = has_kit() and (sim.has_gui or sim.has_offscreen_render)
        return NewtonQueries.capture_graph(PhysicsManager._device, capture_target, relaxed=relaxed)

    # ----- Authored-state invalidation ----------------------------------------

    @classmethod
    def invalidate_fk(
        cls, env_mask: wp.array | None = None, env_ids: wp.array | None = None, articulation_ids: wp.array | None = None
    ) -> None:
        """Mark articulations as needing FK and their worlds as needing a solver reset.

        Called by asset writes that modify joint coordinates or root transforms. The flags are consumed at the next
        forward, raw-state read, rendering, or step boundary.

        Args:
            env_mask: Boolean mask of dirtied view rows. Shape ``(num_instances,)``.
            env_ids: Integer indices of dirtied view rows.
            articulation_ids: Model articulation index of each ``(row, articulation)``, from
                ``ArticulationView.articulation_ids``. ``None`` flags every articulation.
        """
        cls._mark_transforms_changed()
        if NewtonManager._runtime is not None:
            newton_runtime.invalidate_fk(NewtonManager._runtime, env_mask, env_ids, articulation_ids)

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
        if NewtonManager._runtime is not None:
            newton_runtime.invalidate_body_state(NewtonManager._runtime, env_ids, env_mask, row_worlds)

    @classmethod
    def view_row_worlds(cls, articulation_ids: wp.array) -> wp.array:
        """Return the world of each row of an articulation view; see :func:`runtime.view_row_worlds`.

        Args:
            articulation_ids: ``ArticulationView.articulation_ids``.
        """
        return newton_runtime.view_row_worlds(NewtonManager._runtime, articulation_ids)

    @classmethod
    def add_model_change(cls, change: ModelFlags) -> None:
        """Register a model change to notify the solver before the next step.

        Changes authored before model finalization are part of the finalized model, so they need no notification.
        """
        if NewtonManager._runtime is not None:
            NewtonManager._runtime.model_changes.add(change)

    @classmethod
    def transforms_may_change_on_graph_replay(cls) -> bool:
        """Whether state was written during an outer capture, so graph replays may change it without notice."""
        runtime = NewtonManager._runtime
        return runtime is not None and runtime.transforms_may_change_on_graph_replay

    @classmethod
    def _mark_transforms_changed(cls) -> None:
        """Publish authored rigid-body changes and invalidate cable geometry."""
        if NewtonManager._scene_data_backend is not None:
            NewtonManager._scene_data_backend.transforms_timestamp += 1
            NewtonManager._scene_data_backend.geometry_timestamp += 1
        cls._flag_outer_capture()

    @classmethod
    def mark_particles_dirty(cls) -> None:
        """Invalidate scene-data geometry after native particle writes."""
        NewtonManager._scene_data_backend.geometry_timestamp += 1
        cls._flag_outer_capture()

    @classmethod
    def _flag_outer_capture(cls) -> None:
        """Remember writes recorded into an outer capture, whose replays bypass Python invalidation."""
        device, runtime = PhysicsManager._device, NewtonManager._runtime
        if device is None or runtime is None:
            return
        device = wp.get_device(device)
        if device.is_cuda and device.stream.is_capturing:
            runtime.transforms_may_change_on_graph_replay = True

    # ----- Sensors -----------------------------------------------------------

    @classmethod
    def add_contact_sensor(
        cls,
        body_names_expr: str | list[str] | None = None,
        shape_names_expr: str | list[str] | None = None,
        contact_partners_body_expr: str | list[str] | None = None,
        contact_partners_shape_expr: str | list[str] | None = None,
        verbose: bool = False,
    ) -> SensorContact:
        """Add a contact sensor for reporting contacts between bodies or shapes.

        Compiles Isaac Lab regular expressions and delegates to :class:`newton.sensors.SensorContact`, which
        full-matches compiled patterns against model labels. Identical requests share one sensor.

        Args:
            body_names_expr: Expression for body names to sense.
            shape_names_expr: Expression for shape names to sense.
            contact_partners_body_expr: Expression for contact partner body names.
            contact_partners_shape_expr: Expression for contact partner shape names.
            verbose: Print verbose information.

        Returns:
            The Newton contact sensor updated at the end of every step.
        """
        with Timer(
            name="newton_contact_sensor",
            msg="Contact sensor construction took:",
            synchronize="both",
            device=cls.get_device(),
        ):
            return newton_runtime.add_contact_sensor(
                NewtonManager._runtime,
                body_names_expr,
                shape_names_expr,
                contact_partners_body_expr,
                contact_partners_shape_expr,
                verbose,
            )

    @classmethod
    def add_frame_transform_sensor(cls, shapes: list[int], reference_sites: list[int]) -> SensorFrameTransform:
        """Add a frame transform sensor for measuring relative transforms.

        Args:
            shapes: Ordered shape indices to measure.
            reference_sites: Reference site index of each shape.

        Returns:
            The Newton sensor updated at the end of every step.
        """
        return newton_runtime.add_frame_transform_sensor(NewtonManager._runtime, shapes, reference_sites)

    @classmethod
    def add_imu_sensor(cls, sites: list[int]) -> SensorIMU:
        """Add an IMU sensor measuring acceleration and angular velocity at sites.

        Args:
            sites: Site index of each environment.

        Returns:
            The Newton sensor updated at the end of every step.
        """
        if NewtonManager._runtime is None:
            raise RuntimeError("add_imu_sensor called before model finalization (start_simulation).")
        return newton_runtime.add_imu_sensor(NewtonManager._runtime, sites)

    # ----- Accessors ---------------------------------------------------------

    @classmethod
    def get_newton_backend(cls) -> NewtonBackend | None:
        """Return the borrowed native model, state, and control, or ``None`` before finalization."""
        runtime = NewtonManager._runtime
        return None if runtime is None else runtime.backend

    @classmethod
    def get_schema(cls) -> NewtonSchema | None:
        """Return the immutable description of the finalized worlds, or ``None`` before finalization."""
        runtime = NewtonManager._runtime
        return None if runtime is None else runtime.schema

    @classmethod
    def get_solver(cls) -> SolverBase | None:
        """Return the active Newton solver, or ``None`` before :meth:`initialize_solver`."""
        runtime = NewtonManager._runtime
        return None if runtime is None or runtime.solver is None else runtime.solver.solver

    @classmethod
    def get_model(cls) -> Model | None:
        """Return the active physics model. Render consumers acquire their backend from the registry."""
        backend = cls.get_newton_backend()
        return None if backend is None else backend.model

    @classmethod
    def get_state_0(cls) -> State | None:
        """Return the current state."""
        backend = cls.get_newton_backend()
        return None if backend is None else backend.state_0

    @classmethod
    def get_state_1(cls) -> State | None:
        """Return the spare state of double-buffered solvers."""
        backend = cls.get_newton_backend()
        return None if backend is None else backend.state_1

    @classmethod
    def get_control(cls) -> Control | None:
        """Return the control inputs."""
        backend = cls.get_newton_backend()
        return None if backend is None else backend.control

    @classmethod
    def get_contacts(cls) -> Contacts | None:
        """Return the current Newton contact buffer, if the active solver exposes one."""
        runtime = NewtonManager._runtime
        return None if runtime is None else runtime.contacts

    @classmethod
    def get_num_envs(cls) -> int | None:
        """Return the number of simulated environments, or ``None`` before replication."""
        model = cls.get_model()
        if model is not None:
            return model.num_envs
        clone = NewtonManager._requests.clone
        return None if clone is None else clone.num_envs

    @classmethod
    def get_dt(cls) -> float:
        """Get the physics timestep. Alias for get_physics_dt()."""
        return cls.get_physics_dt()

    @classmethod
    def get_solver_dt(cls) -> float:
        """Get the solver substep timestep."""
        return NewtonManager._runtime.schema.solver_dt

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
    def create_fixed_tendon_control(cls, articulation: Any) -> Any:
        """Build the solver's fixed-tendon command adapter for ``articulation``.

        Tendon state is backend-neutral and lives on the articulation; how a target reaches the solver is not, so the
        solver binding builds the adapter.

        Args:
            articulation: Newton articulation to drive.

        Returns:
            The adapter, or ``None`` when all of the articulation's tendons are passive.
        """
        return cls.solver_binding.create_fixed_tendon_control(articulation, cls.get_model())

    @classmethod
    def create_visual_material_writer(cls, batches: tuple[VisualMaterialBatch, ...]) -> VisualMaterialWriter:
        """Compile material-to-shape addresses for the active Newton model."""
        return cls.get_newton_backend().create_visual_material_writer(batches)

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
    def provides_implicit_damping(cls) -> bool:
        # Newton's symplectic integrator has no implicit damping.
        return False

    @classmethod
    def video_capture_backend(cls) -> str:
        """Newton GL headless perspective video capture."""
        return "newton_gl"

    @classmethod
    def is_fabric_enabled(cls) -> bool:
        """Check if fabric interface is enabled (not applicable for Newton)."""
        return False

    @classmethod
    def register_callback(
        cls,
        callback: Callable,
        event: PhysicsEvent,
        order: int = 0,
        name: str | None = None,
        wrap_weak_ref: bool = True,
    ) -> CallbackHandle:
        """Register a callback. Passes event to parent class."""
        return PhysicsManager.register_callback(callback, event, order, name, wrap_weak_ref)

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
