# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton simulation data and the functions that step it.

:class:`NewtonBackend` holds everything bound to one finalized :class:`newton.Model`: state, control, solver,
contacts, sensors, actuators, step callbacks, and the captured step. The functions in this module take
the backend explicitly, in the style of ``mj_step(m, d)``, and read no class or module state, so several backends can
be built and stepped side by side.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
import warp as wp
from newton import CollisionPipeline, Contacts, ModelBuilder, ModelFlags, State
from newton.sensors import SensorContact, SensorIMU
from newton.solvers import SolverBase

from isaaclab.sim import SimulationContext
from isaaclab.utils import instantiate
from isaaclab.utils.buffers import TimestampedBuffer
from isaaclab.utils.version import has_kit
from isaaclab.utils.warp.index_kernel import IndexKernelDispatcher

from isaaclab_newton.renderers.visual_material import VisualMaterialWriter

from .newton_builder_cfg import NewtonBuilderCfg
from .newton_manager_cfg import NewtonCfg
from .solver_specs import solver_spec

if TYPE_CHECKING:
    from isaaclab.actuators.newton import NewtonActuatorAdapter
    from isaaclab.renderers.base_renderer import VisualMaterialBatch

    from .newton_backend_cfg import NewtonBackendCfg

logger = logging.getLogger(__name__)

SiteEntry = tuple[int, None] | tuple[None, list[list[int]]]
"""Resolved site: ``(global_shape_index, None)`` for a global site, or ``(None, per_world_indices)``."""


# ----- Reset-mask kernels ------------------------------------------------------


@wp.kernel(enable_backward=False)
def _or_reset_masks_from_mask(
    env_mask: wp.array(dtype=wp.bool),
    articulation_ids: wp.array2d(dtype=int),
    world_mask: wp.array(dtype=wp.bool),
    fk_mask: wp.array(dtype=wp.bool),
):
    """OR env_mask into world_mask and set corresponding articulation bits in fk_mask."""
    world, arti = wp.tid()
    if env_mask[world]:
        world_mask[world] = True
        fk_mask[articulation_ids[world, arti]] = True


@wp.kernel(enable_backward=False)
def _scatter_reset_masks_from_ids(
    env_ids: wp.array(dtype=Any),
    articulation_ids: wp.array2d(dtype=int),
    world_mask: wp.array(dtype=wp.bool),
    fk_mask: wp.array(dtype=wp.bool),
):
    """Scatter-set world_mask and fk_mask from sparse env_ids."""
    i, arti = wp.tid()
    world = wp.int32(env_ids[i])
    world_mask[world] = True
    fk_mask[articulation_ids[world, arti]] = True


_SCATTER_RESET_MASKS_FROM_IDS = IndexKernelDispatcher(_scatter_reset_masks_from_ids, ("env_ids",))


@wp.kernel(enable_backward=False)
def _or_world_reset_mask_from_mask(env_mask: wp.array(dtype=wp.bool), world_mask: wp.array(dtype=wp.bool)):
    """Mark masked worlds for solver reset without requesting FK."""
    world = wp.tid()
    if env_mask[world]:
        world_mask[world] = True


@wp.kernel(enable_backward=False)
def _scatter_world_reset_mask_from_ids(env_ids: wp.array(dtype=wp.int32), world_mask: wp.array(dtype=wp.bool)):
    """Mark selected worlds for solver reset without requesting FK."""
    world_mask[env_ids[wp.tid()]] = True


# ----- Replication outputs ------------------------------------------------------


@dataclass
class NewtonCloneRecord:
    """Native replication outputs, kept across hard resets."""

    world_xforms: list[wp.transform] | None
    source_builders: dict[str, ModelBuilder]
    particle_ranges: dict[str, tuple[int, int]]
    cable_bindings: dict[str, list[int]]
    geometry_batches: list


# ----- The backend ---------------------------------------------------------------


class NewtonBackend:
    """Everything bound to one finalized Newton model.

    The simulation registry owns the backend; physics, renderers, sensors and visualizers borrow it. A render-only
    backend (non-Newton physics) allocates only :attr:`model` and :attr:`state_0`. A simulation backend also allocates
    the second state, control and reset masks; :func:`init_solver` constructs the solver and contacts, and
    :func:`step` advances it. A hard reset closes it and finalizes a new one.
    """

    def __init__(self, cfg: NewtonBackendCfg, builder: ModelBuilder | None = None, *, dt: float | None = None):
        """Finalize a builder and allocate the buffers bound to its model.

        Args:
            cfg: Physics settings and device. :class:`NewtonCfg` physics makes this a simulation backend; any other
                physics makes it render-only.
            builder: Builder to finalize. Defaults to the simulation's shared builder.
            dt: Duration of one physics step [s]. Required for a simulation backend.
        """
        if builder is None:
            builder = SimulationContext.instance().get_or_create_backend(NewtonBuilderCfg(physics_cfg=cfg.physics_cfg))
        model = self.model = builder.finalize(device=cfg.device)
        model.num_envs = model.world_count
        self.device: wp.Device = model.device
        self.particle_ranges: dict[str, tuple[int, int]] = {}
        self.deformable_ranges = {
            label: (start, end - start, kind)
            for kind, labels, starts, ends in (
                ("surface", builder.surface_label, builder._surface_particle_start, builder._surface_particle_end),
                ("volume", builder.volume_label, builder._volume_particle_start, builder._volume_particle_end),
            )
            for label, start, end in zip(labels, starts, ends, strict=True)
        }
        physics_cfg = cfg.physics_cfg
        simulation = isinstance(physics_cfg, NewtonCfg)
        soft_contact = physics_cfg.soft_contact_cfg if simulation else None
        if soft_contact is not None:
            model.soft_contact_ke = float(soft_contact.soft_contact_ke)
            model.soft_contact_kd = float(soft_contact.soft_contact_kd)
            model.soft_contact_mu = float(soft_contact.soft_contact_mu)
        self.state_0: State = model.state()
        self.geometry_offsets = (
            {} if simulation else {path: bounds[0] for path, bounds in self.deformable_ranges.items()}
        )
        self.bvh_refit = TimestampedBuffer()
        self.cfg: NewtonCfg | None = physics_cfg if simulation else None
        if not simulation:
            self.state_1 = self.control = None
            return
        if dt is None:
            raise ValueError("A simulation NewtonBackend requires the physics step duration dt.")

        self.solver_cfg = physics_cfg.solver_cfg  # init_solver resolves its model-dependent defaults
        self.dt = dt  # [s]
        self.num_substeps = physics_cfg.num_substeps
        self.solver_dt = dt / self.num_substeps  # [s]
        self.deterministic_mode = resolve_deterministic_mode(physics_cfg)
        # Every physics step ends in state_0; double-buffered solvers alternate with state_1.
        self.state_1: State = self.state_0 if solver_spec(self.solver_cfg).single_state else model.state()
        self.control = model.control()
        self.solver: SolverBase | None = None  # Newton's solver, constructed by init_solver
        self.collision_pipeline: CollisionPipeline | None = None
        self.contacts: Contacts | None = None

        # Reset bookkeeping for forward(). Isaac Lab resets local worlds only, so the final (global) slot stays false.
        self.world_mask = wp.zeros(model.world_count + 1, dtype=wp.bool, device=self.device)
        self.fk_mask = wp.zeros(model.articulation_count, dtype=wp.bool, device=self.device)
        self.kinematics_dirty = False
        # Sticky: outer graph replays author state without passing through Python invalidation.
        self.transforms_may_change_on_graph_replay = False
        self.model_changes: set[int] = set()
        self.warned_model_changes: set[int] = set()

        # Consumers and the step. Callbacks run at fixed points of :func:`step`; see :func:`_simulate_full`.
        self.pre_actuator_callbacks: list[Callable[[], None]] = []
        self.post_actuator_callbacks: list[Callable[[], None]] = []
        self.state_force_callbacks: list[Callable[[State], None]] = []
        self.post_step_callbacks: list[Callable[[], None]] = []
        self.contact_sensors: dict[tuple, SensorContact] = {}
        self.imu_sensors: list[SensorIMU] = []
        self.actuators: NewtonActuatorAdapter | None = None
        self.actuators_active = False  # set once an articulation runs its Newton actuators in the step
        self.decimation = 1
        self.staged_body_f: wp.array | None = None  # allocated by init_solver
        self.graph: wp.Graph | None = None
        self.graph_capture_pending = False
        self.capture: Callable[[Callable[[], None]], wp.Graph] | None = None  # set by init_solver

    def create_visual_material_writer(self, batches: tuple[VisualMaterialBatch, ...]) -> VisualMaterialWriter:
        """Bind material writes to this resource's native shape-color buffer."""
        return VisualMaterialWriter(self.model, batches)

    def close(self) -> None:
        """Drop native handles after consumers release their bindings."""
        self.model = self.state_0 = self.state_1 = self.control = None
        self.particle_ranges, self.deformable_ranges, self.geometry_offsets = {}, {}, {}
        self.bvh_refit = TimestampedBuffer()
        if self.cfg is not None:
            self.solver = self.collision_pipeline = self.contacts = self.actuators = self.graph = None
            self.staged_body_f = None
            self.pre_actuator_callbacks, self.post_actuator_callbacks = [], []
            self.state_force_callbacks, self.post_step_callbacks = [], []
            self.contact_sensors, self.imu_sensors = {}, []
            self.world_mask = self.fk_mask = None


# ----- Solver ------------------------------------------------------------------------


def init_solver(backend: NewtonBackend, *, relaxed_capture: bool = False) -> None:
    """Construct the solver and contacts, compute body state from the initial joint state, and resolve capture.

    Consumers bind views, sensors, callbacks, and model properties before this, so solvers that copy model data at
    construction see the authored model.

    Args:
        backend: Simulation backend without a solver.
        relaxed_capture: Capture step graphs in relaxed mode; see :func:`capture_graph`.
    """
    spec, model, mode = solver_spec(backend.solver_cfg), backend.model, backend.deterministic_mode
    solver_cfg = backend.solver_cfg = spec.resolve_cfg(model, backend.solver_cfg)
    if mode != wp.DeterministicMode.NOT_GUARANTEED and not spec.deterministic:
        raise ValueError(
            f"Newton deterministic mode {mode.name} is not supported by {type(solver_cfg).__name__}."
            " Use MJWarp on the GPU, XPBD, or Featherstone, or disable deterministic mode."
        )
    spec.validate(model, solver_cfg, backend.cfg, mode)
    solver = backend.solver = instantiate(solver_cfg, model, mode)
    if not spec.single_state:
        spec.init_state(solver, solver_cfg, backend.state_1)
    _allocate_contacts(backend)
    state_0 = backend.state_0
    backend.staged_body_f = wp.zeros_like(state_0.body_f) if state_0.body_count else None
    spec.eval_fk(solver, solver_cfg, state_0, None, None)
    backend.capture = None
    if backend.cfg.use_cuda_graph and backend.device.is_cuda:
        if spec.supports_capture(solver, solver_cfg):
            backend.capture = partial(capture_graph, backend.device, relaxed=relaxed_capture)
        else:
            logger.warning(
                "%s cannot be captured in its current configuration; stepping eagerly.", type(solver_cfg).__name__
            )
    invalidate_graph(backend)


def _allocate_contacts(backend: NewtonBackend) -> None:
    """Allocate contacts, and the collision pipeline when the solver does not detect contacts internally."""
    spec, model, solver, solver_cfg = solver_spec(backend.solver_cfg), backend.model, backend.solver, backend.solver_cfg
    invalidate_graph(backend)
    if not spec.external_contacts(solver, solver_cfg):
        backend.contacts = spec.create_contacts(solver, solver_cfg)
    else:
        collision_cfg = backend.cfg.collision_cfg
        args = collision_cfg.to_pipeline_args() if collision_cfg is not None else {"broad_phase": "explicit"}
        args["deterministic"] = backend.deterministic_mode != wp.DeterministicMode.NOT_GUARANTEED
        required = getattr(solver, "get_max_contact_count", lambda: 0)()
        if backend.collision_pipeline is None:
            backend.collision_pipeline = spec.create_collision_pipeline(model, solver, solver_cfg, **args)
        backend.contacts = backend.collision_pipeline.contacts()
        # MuJoCo Warp can require more contacts than the pipeline estimates.
        if required > backend.contacts.rigid_contact_max:
            if args["deterministic"]:
                # The deterministic sort buffer is sized at construction, so rebuild the pipeline to match.
                backend.collision_pipeline = spec.create_collision_pipeline(
                    model, solver, solver_cfg, **args, rigid_contact_max=required
                )
                backend.contacts = backend.collision_pipeline.contacts()
            else:
                backend.contacts = Contacts(
                    rigid_contact_max=required,
                    soft_contact_max=0,
                    device=backend.device,
                    requested_attributes=model.get_requested_contact_attributes(),
                )
    if backend.contacts is not None:
        spec.prepare_contacts(solver, solver_cfg, backend.contacts, backend.collision_pipeline)


def resolve_deterministic_mode(cfg: NewtonCfg) -> wp.DeterministicMode:
    """Translate the determinism request of a Newton configuration into a Warp mode.

    An explicit :attr:`NewtonCfg.deterministic_mode` wins over the generic
    :attr:`~isaaclab.physics.PhysicsCfg.deterministic` request. MuJoCo on the CPU is reproducible on its own and Warp's
    mode does not reach it, so no mode applies there.

    Args:
        cfg: Newton physics configuration.

    Returns:
        The mode for the solver and collision pipeline.
    """
    if getattr(cfg.solver_cfg, "use_mujoco_cpu", False):
        return wp.DeterministicMode.NOT_GUARANTEED
    if cfg.deterministic_mode != "not_guaranteed":
        return {"run_to_run": wp.DeterministicMode.RUN_TO_RUN, "gpu_to_gpu": wp.DeterministicMode.GPU_TO_GPU}[
            cfg.deterministic_mode
        ]
    return wp.DeterministicMode.RUN_TO_RUN if cfg.deterministic else wp.DeterministicMode.NOT_GUARANTEED


# ----- Authored state ----------------------------------------------------------------


def invalidate_fk(
    backend: NewtonBackend,
    env_mask: wp.array | None = None,
    env_ids: wp.array | None = None,
    articulation_ids: wp.array | None = None,
) -> None:
    """Flag articulations for forward kinematics and their worlds for a solver reset, without host synchronization.

    Args:
        backend: Simulation backend.
        env_mask: Mask over environments. Shape ``(num_envs,)``.
        env_ids: Selected environments.
        articulation_ids: Model articulation index of each ``(world, articulation)``. Shape
            ``(world_count, count_per_world)``; ``None`` flags everything.
    """
    backend.kinematics_dirty = True
    outputs = [backend.world_mask, backend.fk_mask]
    if articulation_ids is not None and env_mask is not None:
        wp.launch(
            _or_reset_masks_from_mask,
            articulation_ids.shape,
            [env_mask, articulation_ids],
            outputs,
            device=backend.device,
        )
    elif articulation_ids is not None and env_ids is not None:
        dim = (env_ids.shape[0], articulation_ids.shape[1])
        kernel = _SCATTER_RESET_MASKS_FROM_IDS.select(env_ids)
        wp.launch(kernel, dim, [env_ids, articulation_ids], outputs, device=backend.device)
    else:
        backend.world_mask[: backend.model.world_count].fill_(True)
        backend.fk_mask.fill_(True)


def invalidate_body_state(
    backend: NewtonBackend,
    env_ids: wp.array | None = None,
    env_mask: wp.array | None = None,
) -> None:
    """Flag worlds whose maximal-coordinate body state was written, without requesting FK.

    Args:
        backend: Simulation backend.
        env_ids: Selected environments.
        env_mask: Mask over environments.
    """
    backend.kinematics_dirty = True
    if env_mask is not None:
        wp.launch(
            _or_world_reset_mask_from_mask, env_mask.shape[0], [env_mask], [backend.world_mask], device=backend.device
        )
    elif env_ids is not None:
        kernel = _scatter_world_reset_mask_from_ids
        wp.launch(kernel, env_ids.shape[0], [env_ids], [backend.world_mask], device=backend.device)
    else:
        backend.world_mask[: backend.model.world_count].fill_(True)


def forward(backend: NewtonBackend, *, force: bool = False) -> None:
    """Reset solver internals and recompute body state for the flagged worlds, then clear the flags.

    Like ``mj_forward``, this makes derived state consistent with authored state without advancing time. It runs only
    when state was authored since the last call, when an outer graph replay may have authored it, or when forced; the
    work is masked, so forcing it changes nothing unflagged.

    Args:
        backend: Simulation backend with a solver.
        force: Launch the masked work even without known authored state, e.g. while recording a graph whose replays
            follow writes made outside it.
    """
    if not (force or backend.kinematics_dirty or backend.transforms_may_change_on_graph_replay):
        return
    if backend.solver is None:
        raise RuntimeError("Newton state was authored before the solver was initialized (init_solver).")
    spec, solver, solver_cfg, state = (
        solver_spec(backend.solver_cfg),
        backend.solver,
        backend.solver_cfg,
        backend.state_0,
    )
    spec.reset(solver, solver_cfg, state, backend.world_mask)
    spec.eval_fk(solver, solver_cfg, state, backend.world_mask, backend.fk_mask)
    backend.fk_mask.zero_()
    backend.world_mask.zero_()
    backend.kinematics_dirty = False


def mark_model_changed(backend: NewtonBackend, change: ModelFlags) -> None:
    """Queue a model-property change for the solver, notified once at the next step.

    Args:
        backend: Simulation backend.
        change: Property categories the solver must refresh.
    """
    backend.model_changes.add(change)


def notify_model_changes(backend: NewtonBackend) -> None:
    """Notify the solver of the queued property changes once, combining their categories."""
    if not backend.model_changes:
        return
    spec = solver_spec(backend.solver_cfg)
    flags = 0
    for change in backend.model_changes:
        if change in spec.ignored_model_changes and change not in backend.warned_model_changes:
            logger.warning(spec.ignored_model_changes[change])
            backend.warned_model_changes.add(change)
        flags |= change
    with wp.ScopedDevice(backend.device):
        # Refresh shared solver constants once after applying all pending property changes.
        backend.solver.notify_model_changed(flags)
    backend.model_changes.clear()


# ----- Consumers -----------------------------------------------------------------


def activate_actuators(backend: NewtonBackend) -> NewtonActuatorAdapter | None:
    """Run the model's Newton actuators inside the step. Idempotent.

    Together with the actuators' graph safety, activation lets one :func:`step` run the whole decimation loop (see
    :func:`handles_decimation`).

    Returns:
        The adapter, or ``None`` when the model has no actuators.
    """
    backend.actuators_active = True
    model = backend.model
    if backend.actuators is None and model.actuators:
        from isaaclab.actuators.newton import NewtonActuatorAdapter  # noqa: PLC0415

        num_envs = model.world_count
        backend.actuators = NewtonActuatorAdapter(
            actuators=list(model.actuators),
            num_envs=num_envs,
            num_joints=model.joint_dof_count // num_envs,
            dof_offset=0,
            device=str(backend.device),
        )
        backend.actuators.finalize(backend.control)
    return backend.actuators


def add_contact_sensor(
    backend: NewtonBackend,
    body_names_expr: str | list[str] | None = None,
    shape_names_expr: str | list[str] | None = None,
    contact_partners_body_expr: str | list[str] | None = None,
    contact_partners_shape_expr: str | list[str] | None = None,
    verbose: bool = False,
) -> SensorContact:
    """Add a contact sensor between bodies or shapes; identical requests share one sensor.

    Args:
        backend: Simulation backend.
        body_names_expr: Expression for body names to sense.
        shape_names_expr: Expression for shape names to sense.
        contact_partners_body_expr: Expression for contact partner body names.
        contact_partners_shape_expr: Expression for contact partner shape names.
        verbose: Print verbose information.

    Returns:
        The Newton contact sensor, updated at the end of every step.
    """
    if not solver_spec(backend.solver_cfg).contact_sensors:
        raise NotImplementedError(
            f"Newton contact sensors are not yet supported by {type(backend.solver_cfg).__name__}:"
            " contact forces live in per-entry buffers."
        )
    if (body_names_expr is None) == (shape_names_expr is None):
        raise ValueError("Exactly one of body_names_expr or shape_names_expr must be provided.")
    if contact_partners_body_expr is not None and contact_partners_shape_expr is not None:
        raise ValueError("Only one of contact_partners_body_expr or contact_partners_shape_expr must be provided.")
    exprs = (body_names_expr, shape_names_expr, contact_partners_body_expr, contact_partners_shape_expr)
    key = tuple(tuple(expr) if isinstance(expr, list) else expr for expr in exprs)
    sensor = backend.contact_sensors.get(key)
    if sensor is not None:
        return sensor
    sensor = SensorContact(
        backend.model,
        sensing_bodies=compile_label_pattern(body_names_expr),
        sensing_shapes=compile_label_pattern(shape_names_expr),
        counterpart_bodies=compile_label_pattern(contact_partners_body_expr),
        counterpart_shapes=compile_label_pattern(contact_partners_shape_expr),
        measure_total=True,
        verbose=verbose,
    )
    backend.contact_sensors[key] = sensor
    invalidate_graph(backend)
    # Contacts allocated before the sensor requested the force attribute must be reallocated.
    if backend.contacts is not None and backend.contacts.force is None:
        _allocate_contacts(backend)
    return sensor


def add_imu_sensor(backend: NewtonBackend, sites: list[int]) -> SensorIMU:
    """Add an IMU sensor at sites; the ``body_qdd`` state attribute must be requested before finalization.

    Args:
        backend: Simulation backend.
        sites: Site index of each environment.

    Returns:
        The Newton sensor, updated at the end of every step.
    """
    sensor = SensorIMU(backend.model, sites=sites, request_state_attributes=False)
    backend.imu_sensors.append(sensor)
    invalidate_graph(backend)
    return sensor


def compile_label_pattern(expr: str | list[str] | None) -> re.Pattern[str] | None:
    """Compile Isaac Lab selector expressions for Newton's full label matching."""
    if not expr:
        return None
    return re.compile("|".join((expr,) if isinstance(expr, str) else expr))


# ----- Stepping ----------------------------------------------------------------------


def handles_decimation(backend: NewtonBackend) -> bool:
    """Whether one :func:`step` runs the whole decimation loop.

    This is the case once an articulation runs its Newton actuators in the step (:func:`activate_actuators`) and every
    actuator can be recorded into a CUDA graph.
    """
    return backend.actuators_active and (backend.actuators is None or backend.actuators.is_all_graphable)


def set_decimation(backend: NewtonBackend, decimation: int) -> None:
    """Set the physics steps of one environment step, and recapture a step graph that runs them all.

    Args:
        backend: Simulation backend.
        decimation: Physics steps per environment step.
    """
    backend.decimation = max(1, decimation)
    if handles_decimation(backend):
        invalidate_graph(backend)


def invalidate_graph(backend: NewtonBackend) -> None:
    """Drop the captured step and capture a new one right after the next step runs eagerly, when capture is enabled."""
    backend.graph = None
    backend.graph_capture_pending = backend.capture is not None


def step(backend: NewtonBackend) -> bool:
    """Advance one physics step, or the whole decimation loop when :func:`handles_decimation`.

    Notifies model changes and runs :func:`forward` for authored state. When the decimation loop runs here, Newton
    actuators and the actuator callbacks run inside the step for every physics step (:func:`_simulate_full`).
    Otherwise they run eagerly once, before a step of collision, substeps, and sensors
    (:func:`_simulate_physics_only`). The first step after :func:`invalidate_graph` runs eagerly, which also performs
    lazy solver allocations, and then the step is captured with :attr:`NewtonBackend.capture` for later steps.

    Returns:
        Whether a captured graph ran.
    """
    notify_model_changes(backend)
    forward(backend)
    if handles_decimation(backend):
        simulate = partial(_simulate_full, backend)
    else:
        for callback in backend.pre_actuator_callbacks:
            callback()
        if backend.actuators is not None:
            backend.actuators.step(backend.state_0, backend.control, backend.dt)
        for callback in backend.post_actuator_callbacks:
            callback()
        simulate = partial(_simulate_physics_only, backend)
    graph = backend.graph
    if graph is not None:
        wp.capture_launch(graph)
        return True
    with wp.ScopedDevice(backend.device):
        simulate()
    if backend.graph_capture_pending:
        backend.graph = backend.capture(simulate)
        backend.graph_capture_pending = False
    return False


def _simulate_full(backend: NewtonBackend) -> None:
    """Run ``decimation x (collide, actuators, substeps)``, then post-step callbacks and sensors.

    Every Newton actuator must be graph-safe, so the whole loop can be captured as one CUDA graph.
    """
    pipeline = backend.collision_pipeline
    contacts = backend.contacts if pipeline is not None else None
    actuators, decimation = backend.actuators, backend.decimation
    _stage_body_forces(backend)
    for i in range(decimation):
        _prepare_step(backend)
        if pipeline is not None:
            pipeline.collide(backend.state_0, contacts)
        for callback in backend.pre_actuator_callbacks:
            callback()
        if actuators is not None:
            # An odd decimation copies the last actuator output back, keeping buffer addresses stable across replays.
            swap_state = decimation % 2 == 0 or i < decimation - 1
            actuators.step(backend.state_0, backend.control, backend.dt, swap_state=swap_state)
        for callback in backend.post_actuator_callbacks:
            callback()
        _run_solver_substeps(backend, contacts)
    for callback in backend.post_step_callbacks:
        callback()
    _update_sensors(backend)


def _simulate_physics_only(backend: NewtonBackend) -> None:
    """Run collision, solver substeps, post-step callbacks, and sensors, without actuators."""
    pipeline = backend.collision_pipeline
    contacts = backend.contacts if pipeline is not None else None
    _stage_body_forces(backend)
    _prepare_step(backend)
    if pipeline is not None:
        pipeline.collide(backend.state_0, contacts)
    _run_solver_substeps(backend, contacts)
    for callback in backend.post_step_callbacks:
        callback()
    _update_sensors(backend)


def _prepare_step(backend: NewtonBackend) -> None:
    """Refresh solver structures before the physics step, when the solver's spec does."""
    prepare_step = solver_spec(backend.solver_cfg).prepare_step
    if prepare_step is not None:
        prepare_step(backend.solver, backend.solver_cfg, backend.state_0)


def _stage_body_forces(backend: NewtonBackend) -> None:
    """Copy the body forces written to ``state_0`` before the step, so that every solver substep applies them."""
    if backend.staged_body_f is not None:
        wp.copy(backend.staged_body_f, backend.state_0.body_f)


def _run_solver_substeps(backend: NewtonBackend, contacts: Contacts | None) -> None:
    """Run the solver substeps of one physics step, alternating double-buffered states.

    The staged body forces overwrite the input state's ``body_f`` before every substep, so state-force callbacks add
    on top of them. Every physics step ends in :attr:`NewtonBackend.state_0`.
    """
    spec, solver, solver_cfg = solver_spec(backend.solver_cfg), backend.solver, backend.solver_cfg
    num_substeps, control, dt = backend.num_substeps, backend.control, backend.solver_dt
    collide_every = backend.cfg.collision_decimation
    # The last substep is skipped: its contacts would only feed the next physics step's collision.
    collide_mid_loop = collide_every > 0 and backend.collision_pipeline is not None and contacts is not None
    for i in range(num_substeps):
        state = backend.state_0
        if backend.staged_body_f is not None:
            wp.copy(state.body_f, backend.staged_body_f)
        for callback in backend.state_force_callbacks:
            callback(state)
        if spec.single_state:
            spec.step(solver, solver_cfg, state, state, control, contacts, dt)
        else:
            spec.step(solver, solver_cfg, state, backend.state_1, control, contacts, dt)
            if num_substeps % 2 == 1 and i == num_substeps - 1:
                state.assign(backend.state_1)
            else:
                backend.state_0, backend.state_1 = backend.state_1, backend.state_0
        backend.state_0.clear_forces()
        if collide_mid_loop and (i + 1) % collide_every == 0 and i + 1 < num_substeps:
            backend.collision_pipeline.collide(backend.state_0, contacts)
    if spec.finish_step is not None:
        spec.finish_step(solver, solver_cfg, backend.state_0)


def _update_sensors(backend: NewtonBackend) -> None:
    """Push the current state to IMU and contact sensors."""
    for sensor in backend.imu_sensors:
        sensor.update(backend.state_0)
    if backend.contact_sensors:
        backend.solver.update_contacts(backend.contacts, backend.state_0)
        for sensor in backend.contact_sensors.values():
            sensor.update(backend.state_0, backend.contacts)


# ----- CUDA graph capture and scene queries ------------------------------------------


def capture_graph(device: wp.DeviceLike, fn: Callable[[], None], *, relaxed: bool = False) -> wp.Graph:
    """Record work without an eager warmup or graph replay, using Warp's public capture API.

    Args:
        device: CUDA device.
        fn: Work to record. Perform lazy initialization before capture.
        relaxed: Allow unrelated work on other threads, as required by Kit rendering.

    Returns:
        The captured graph, replayed with :func:`warp.capture_launch`.
    """
    stream = wp.get_stream(device)
    mode = wp.CaptureMode.THREAD_LOCAL
    if relaxed:
        # RTX uses the legacy CUDA stream. A nonblocking stream avoids implicit synchronization.
        stream = wp.stream_from_torch(torch.cuda.Stream(device=wp.device_to_torch(device)))
        mode = wp.CaptureMode.RELAXED
    with _paused_gc(), wp.ScopedStream(stream):
        with wp.ScopedCapture(stream=stream, capture_mode=mode) as capture:
            fn()
    return capture.graph


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
                refit = partial(refit_bvh, backend)
                cached.data = pointers, capture_graph(str(model.device), refit, relaxed=has_kit())
            wp.capture_launch(cached.data[1])
        else:
            refit_bvh(backend)
        cached.timestamp = timestamp
    if not model.device.is_cuda or not use_cuda_graph:
        query()
        return None
    if graph is None or graph[0] != pointers:
        graph = pointers, capture_graph(str(model.device), query, relaxed=has_kit())
    wp.capture_launch(graph[1])
    return graph


def refit_bvh(backend: NewtonBackend) -> None:
    """Refit the shape and particle BVHs of the backend's model to its current state."""
    model, state = backend.model, backend.state_0
    if model.shape_count:
        model.bvh_refit_shapes(state)
    if model.particle_count:
        model.bvh_refit_particles(state)
