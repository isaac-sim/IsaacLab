# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton simulation data and the functions that step it.

:class:`NewtonBackend` holds everything bound to one finalized :class:`newton.Model`: state, control, solver,
contacts, sensors, actuators, step callbacks, and the compiled :class:`StepGraph`. The functions in this module take
the backend explicitly, in the style of ``mj_step(m, d)``, and read no class or module state, so several backends can
be built and stepped side by side.
"""

from __future__ import annotations

import contextlib
import enum
import gc
import itertools
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import warp as wp
from newton import CollisionPipeline, Contacts, Model, ModelBuilder, ModelFlags, State
from newton.sensors import SensorContact, SensorIMU

from isaaclab.utils.buffers import TimestampedBuffer
from isaaclab.utils.string import string_to_callable
from isaaclab.utils.version import has_kit
from isaaclab.utils.warp.index_kernel import IndexKernelDispatcher

from isaaclab_newton.renderers.visual_material import VisualMaterialWriter

from .newton_manager_cfg import NewtonBuilderCfg, NewtonCfg

if TYPE_CHECKING:
    from isaaclab.actuators.newton import NewtonActuatorAdapter
    from isaaclab.physics import PhysicsCfg
    from isaaclab.renderers.base_renderer import VisualMaterialBatch

    from .newton_manager_cfg import NewtonBackendCfg
    from .newton_solver import NewtonSolver

logger = logging.getLogger(__name__)

SiteEntry = tuple[int, None] | tuple[None, list[list[int]]]
"""Resolved site: ``(global_shape_index, None)`` for a global site, or ``(None, per_world_indices)``."""


# ----- Reset-mask kernels ------------------------------------------------------


@wp.kernel(enable_backward=False)
def _mark_model_changes(mask: wp.array(dtype=wp.bool), dirty: wp.array(dtype=wp.int32)):
    if mask[wp.tid()]:
        wp.atomic_max(dirty, 0, 1)


@wp.kernel(enable_backward=False)
def _mark_fk_from_mask(
    env_mask: wp.array(dtype=wp.bool),
    articulation_ids: wp.array2d(dtype=int),
    articulation_world: wp.array(dtype=wp.int32),
    world_mask: wp.array(dtype=wp.bool),
    fk_mask: wp.array(dtype=wp.bool),
):
    """Flag the articulations of masked view rows and the local worlds that contain them."""
    row, arti = wp.tid()
    if env_mask[row]:
        articulation = articulation_ids[row, arti]
        fk_mask[articulation] = True
        world = articulation_world[articulation]
        if world >= 0:
            world_mask[world] = True


@wp.kernel(enable_backward=False)
def _mark_fk_from_ids(
    env_ids: wp.array(dtype=Any),
    articulation_ids: wp.array2d(dtype=int),
    articulation_world: wp.array(dtype=wp.int32),
    world_mask: wp.array(dtype=wp.bool),
    fk_mask: wp.array(dtype=wp.bool),
):
    """Flag the articulations of selected view rows and the local worlds that contain them."""
    i, arti = wp.tid()
    articulation = articulation_ids[wp.int32(env_ids[i]), arti]
    fk_mask[articulation] = True
    world = articulation_world[articulation]
    if world >= 0:
        world_mask[world] = True


_MARK_FK_FROM_IDS = IndexKernelDispatcher(_mark_fk_from_ids, ("env_ids",))


@wp.kernel(enable_backward=False)
def _mark_worlds_from_mask(
    env_mask: wp.array(dtype=wp.bool), row_worlds: wp.array(dtype=wp.int32), world_mask: wp.array(dtype=wp.bool)
):
    """Flag the worlds of masked view rows for a solver reset without requesting FK."""
    row = wp.tid()
    if env_mask[row]:
        world_mask[row_worlds[row]] = True


@wp.kernel(enable_backward=False)
def _mark_articulations_of_worlds(
    worlds: wp.array(dtype=wp.bool), articulation_world: wp.array(dtype=wp.int32), fk_mask: wp.array(dtype=wp.bool)
):
    """Flag the articulations of the masked worlds for forward kinematics."""
    articulation = wp.tid()
    world = articulation_world[articulation]
    if world >= 0 and worlds[world]:
        fk_mask[articulation] = True


@wp.kernel(enable_backward=False)
def _mark_worlds(worlds: wp.array(dtype=wp.bool), world_mask: wp.array(dtype=wp.bool)):
    """Flag the masked worlds for a solver reset."""
    world = wp.tid()
    if worlds[world]:
        world_mask[world] = True


@wp.kernel(enable_backward=False)
def _mark_worlds_from_ids(
    env_ids: wp.array(dtype=wp.int32), row_worlds: wp.array(dtype=wp.int32), world_mask: wp.array(dtype=wp.bool)
):
    """Flag the worlds of selected view rows for a solver reset without requesting FK."""
    world_mask[row_worlds[env_ids[wp.tid()]]] = True


# ----- Step callbacks and the compiled step -------------------------------------


class StepPhase(enum.Enum):
    """Where a step callback runs inside one physics step, following MuJoCo's ``mjcb_*`` callback hooks."""

    CONTROL = "control"
    """After collision and before Newton actuators, like ``mjcb_control``. Controllers write targets and efforts."""

    POST_ACTUATOR = "post_actuator"
    """After Newton actuators of the last physics step of :func:`step`, before its solver substeps, e.g. actuator
    telemetry, which consumers observe after the step."""

    STATE_FORCE = "state_force"
    """Before every solver substep. The callback receives the substep's input :class:`newton.State`."""

    POST_STEP = "post_step"
    """After every physics step, before sensors update; state and feedback stay current within decimation."""


@dataclass(frozen=True, eq=False)
class StepCallback:
    """A function run at a :class:`StepPhase` of every step, until it is unregistered or the backend closes."""

    fn: Callable[..., None]
    """Function to run. :attr:`StepPhase.STATE_FORCE` callbacks receive the input state; others take no arguments."""

    phase: StepPhase
    """Where the callback runs."""

    graphable: bool = True
    """Whether the function can be recorded into a CUDA graph: fixed buffers and no host branching on device data.
    Other callbacks run eagerly at the same position."""

    name: str = ""
    """Label used in errors and profiles."""


@dataclass(frozen=True)
class StepOp:
    """One operation of a :class:`StepGraph`, with every buffer bound."""

    fn: Callable[[], None]
    graphable: bool
    name: str


class StepGraph:
    """One call to :func:`step` as straight-line operations, grouped into CUDA graphs.

    Consecutive graphable operations form a segment that replays from one captured graph. Other operations, such as
    TorchScript actuators, run eagerly between segments, so the eager and captured paths run the same operations in
    the same order.
    """

    def __init__(self, ops: Sequence[StepOp], steps: int):
        """Initialize the graph.

        Args:
            ops: Operations in execution order.
            steps: Physics steps one launch advances.
        """
        self.ops = tuple(ops)
        self.steps = steps
        self.segments = tuple(
            (graphable, tuple(group)) for graphable, group in itertools.groupby(self.ops, key=lambda op: op.graphable)
        )
        self.graphable = all(op.graphable for op in self.ops)
        self.graphs: tuple[CapturedGraph | None, ...] | None = None

    @property
    def captured(self) -> bool:
        """Whether graphable segments replay from captured graphs."""
        return self.graphs is not None

    def capture(self, capture: Callable[[Callable[[], None]], CapturedGraph]) -> None:
        """Record every graphable segment without executing it.

        Args:
            capture: Records a function into a graph on the simulation device.
        """
        self.graphs = tuple(capture(partial(_run, ops)) if graphable else None for graphable, ops in self.segments)

    def launch(self) -> None:
        """Advance :attr:`steps` physics steps, replaying captured segments."""
        if self.graphs is None:
            _run(self.ops)
            return
        for (_, ops), graph in zip(self.segments, self.graphs):
            if graph is None:
                _run(ops)
            else:
                graph.launch()

    def record(self) -> None:
        """Launch every operation so that the caller's active capture records them.

        Raises:
            RuntimeError: If an operation cannot be recorded into a CUDA graph.
        """
        if not self.graphable:
            eager = sorted({op.name for op in self.ops if not op.graphable})
            raise RuntimeError(
                f"The Newton step cannot be recorded into an outer CUDA graph: {eager} are not graphable."
            )
        _run(self.ops)


def _run(ops: Sequence[StepOp]) -> None:
    for op in ops:
        op.fn()


# ----- Replication outputs ------------------------------------------------------


@dataclass
class NewtonCloneRecord:
    """Native replication outputs, kept across hard resets and consumed when the model is finalized."""

    world_xforms: list[wp.transform] | None
    """Root transform of each cloned world."""

    source_builders: dict[str, ModelBuilder]
    """Per-source builders retained so single-model consumers can finalize one environment."""

    particle_ranges: dict[str, tuple[int, int]]
    """Native ``(start, count)`` particle range of each imported particle prim."""

    cable_bindings: dict[str, list[int]]
    """Native capsule shape indices of each open cable."""

    geometry_batches: list
    """Deformable and particle geometry published through scene data."""


# ----- The backend ---------------------------------------------------------------


class NewtonBackend:
    """Everything bound to one finalized Newton model.

    The simulation registry owns the backend (see :class:`~isaaclab_newton.physics.NewtonBackendCfg`), and physics,
    renderers, sensors, and visualizers borrow it. A hard reset closes it and finalizes a new one, so no solver, graph,
    or callback outlives the buffers it binds.

    A render-only backend (for a non-Newton physics configuration) allocates only :attr:`model` and :attr:`state_0`.
    A simulation backend also allocates the second state, control, and reset masks; :func:`init_solver` then
    constructs the solver and contacts, and :func:`step` advances it.
    """

    def __init__(
        self,
        model: Model,
        physics_cfg: PhysicsCfg | None = None,
        *,
        dt: float | None = None,
        deformable_ranges: dict[str, tuple[int, int, str]] | None = None,
    ):
        """Bind native buffers to a finalized model.

        Args:
            model: Finalized model.
            physics_cfg: Physics configuration. A :class:`NewtonCfg` makes this a simulation backend driven by
                ``physics_cfg.solver_cfg.class_type``; anything else makes it render-only.
            dt: Duration of one physics step [s]. Required for a simulation backend.
            deformable_ranges: Native ``(start, count, kind)`` particle range of each deformable mesh, by label.
        """
        self.model = model
        self.is_stepping = False
        self.deformable_ranges = deformable_ranges or {}
        self.device: wp.Device = model.device
        # Physics settings apply to the model before any state is allocated from it.
        soft_contact = physics_cfg.soft_contact_cfg if isinstance(physics_cfg, NewtonCfg) else None
        if soft_contact is not None:
            model.soft_contact_ke = float(soft_contact.soft_contact_ke)
            model.soft_contact_kd = float(soft_contact.soft_contact_kd)
            model.soft_contact_mu = float(soft_contact.soft_contact_mu)
        self.state_0: State = model.state()
        self.bvh_refit = TimestampedBuffer()

        simulation = isinstance(physics_cfg, NewtonCfg)
        self.cfg: NewtonCfg | None = physics_cfg if simulation else None
        self.geometry_offsets = (
            {} if simulation else {path: bounds[0] for path, bounds in self.deformable_ranges.items()}
        )
        if not simulation:
            self.state_1 = self.control = None
            return
        if dt is None:
            raise ValueError("A simulation NewtonBackend requires the physics step duration dt.")

        adapter = physics_cfg.solver_cfg.class_type
        self.solver_adapter: type[NewtonSolver] = string_to_callable(adapter) if isinstance(adapter, str) else adapter
        """Stateless solver adapter whose classmethods construct, step, and reset the solver of this backend."""
        self.dt = dt
        """Duration of one physics step [s]."""
        self.num_substeps = physics_cfg.num_substeps
        """Solver substeps per physics step."""
        self.solver_dt = dt / self.num_substeps
        """Duration of one solver substep [s]."""
        self.deterministic_mode = resolve_deterministic_mode(physics_cfg)
        """Determinism guarantee for the solver and collision pipeline."""

        self.state_1: State = self.state_0 if self.solver_adapter.single_state else model.state()
        """Spare state of double-buffered solvers; every physics step ends in :attr:`state_0`."""
        self.control = model.control()
        self.solver = None
        """Newton solver, constructed by :func:`init_solver`."""
        self.collision_pipeline: CollisionPipeline | None = None
        """Newton collision pipeline, when the solver does not detect contacts internally."""
        self.contacts: Contacts | None = None
        """Contacts filled by the collision pipeline or reported by the solver."""

        # Reset bookkeeping. Isaac Lab resets local worlds only, so the final (global) world-mask slot stays false.
        self.world_mask = wp.zeros(model.world_count + 1, dtype=wp.bool, device=self.device)
        """Worlds whose solver internals reset at the next :func:`forward`; the final slot selects global entities."""
        self.fk_mask = wp.zeros(model.articulation_count, dtype=wp.bool, device=self.device)
        """Articulations whose body state :func:`forward` recomputes from joint coordinates."""
        self.world_ids = wp.array(np.arange(model.world_count, dtype=np.int32), device=self.device)
        """Identity row-to-world map of views that span every world."""
        self.kinematics_dirty = False
        """Whether state was authored since the last :func:`forward`."""
        self.transforms_may_change_on_graph_replay = False
        """Whether state was written inside an outer capture, whose replays bypass Python invalidation."""
        self.model_changes: set[int] = set()
        """:class:`newton.ModelFlags` to notify the solver of before the next step."""
        self.warned_model_changes: set[int] = set()
        self.model_dirty = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.model_change_graphs: dict[int, CapturedGraph] = {}

        # Consumers of the step.
        self.callbacks: list[StepCallback] = []
        self.contact_sensors: dict[tuple, SensorContact] = {}
        self.imu_sensors: list[SensorIMU] = []
        self.actuators: NewtonActuatorAdapter | None = None
        """Newton actuators run inside the step, once an articulation activates them."""
        self.env_decimation = False
        """Whether a consumer does host work between physics steps, so the environment runs the decimation loop."""

        # Compiled step.
        self.staged_forces: dict[str, wp.array] = {}
        """Copies of the forces authored before a step, re-applied before every solver substep."""
        self.step_graph: StepGraph | None = None
        """Compiled step; ``None`` until the next :func:`step` builds it."""
        self.steps_per_call = 1
        """Physics steps one :func:`step` advances, e.g. the environment's decimation when physics runs the loop."""
        self.capture: Callable[[Callable[[], None]], CapturedGraph] | None = None
        """Records step-graph segments into CUDA graphs; ``None`` steps eagerly. Set by :func:`init_solver`."""

    def create_visual_material_writer(self, batches: tuple[VisualMaterialBatch, ...]) -> VisualMaterialWriter:
        """Bind material writes to this resource's native shape-color buffer."""
        return VisualMaterialWriter(self.model, batches)

    def close(self) -> None:
        """Drop native handles after consumers release their bindings."""
        self.model = self.state_0 = self.state_1 = self.control = None
        self.deformable_ranges, self.geometry_offsets = {}, {}
        self.bvh_refit = TimestampedBuffer()
        if self.cfg is not None:
            self.solver = self.collision_pipeline = self.contacts = self.actuators = self.step_graph = None
            self.callbacks, self.contact_sensors, self.imu_sensors = [], {}, []
            self.staged_forces.clear()
            self.model_change_graphs.clear()
            self.model_dirty = None
            self.world_mask = self.fk_mask = self.world_ids = None


def create_newton_backend(cfg: NewtonBackendCfg) -> NewtonBackend:
    """Finalize the shared builder of the simulation and bind a backend to the model.

    This is the simulation registry's constructor for :class:`~isaaclab_newton.physics.NewtonBackendCfg`. For Newton
    physics, the selected manager first applies pending sites, attribute requests, and solver-specific builder
    normalization.

    Args:
        cfg: Physics configuration and device.

    Returns:
        The backend.
    """
    simulation = isinstance(cfg.physics_cfg, NewtonCfg)
    if simulation:
        return cfg.manager.finalize_backend()
    sim = cfg.manager._sim
    builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=cfg.physics_cfg, manager=cfg.manager))
    model = builder.finalize(device=cfg.device)
    deformable_ranges = {
        label: (start, end - start, kind)
        for kind, labels, starts, ends in (
            ("surface", builder.surface_label, builder._surface_particle_start, builder._surface_particle_end),
            ("volume", builder.volume_label, builder._volume_particle_start, builder._volume_particle_end),
        )
        for label, start, end in zip(labels, starts, ends, strict=True)
    }
    return NewtonBackend(model, cfg.physics_cfg, deformable_ranges=deformable_ranges)


# ----- Solver ------------------------------------------------------------------------


def init_solver(backend: NewtonBackend, *, relaxed_capture: bool = False) -> None:
    """Construct the solver and contacts, compute body state from the initial joint state, and resolve capture.

    Consumers bind views, sensors, callbacks, and model properties before this, so solvers that copy model data at
    construction see the authored model.

    Args:
        backend: Simulation backend without a solver.
        relaxed_capture: Capture step graphs in relaxed mode; see :func:`capture_graph`.
    """
    adapter = backend.solver_adapter
    adapter.validate_cfg(backend)
    cfg = backend.cfg
    backend.solver = adapter.create_solver(backend.model, cfg.solver_cfg, backend.deterministic_mode)
    if not adapter.single_state:
        adapter.initialize_output_state(backend, backend.state_1)
    _allocate_contacts(backend)
    adapter.eval_fk(backend, backend.state_0, None, None)
    backend.capture = None
    if cfg.use_cuda_graph and backend.device.is_cuda:
        if adapter.supports_graph_capture(backend):
            backend.capture = partial(capture_graph, backend.device, relaxed=relaxed_capture)
        else:
            logger.warning("%s cannot capture the current solver configuration; stepping eagerly.", adapter.__name__)


def _allocate_contacts(backend: NewtonBackend) -> None:
    """Allocate contacts, and the collision pipeline when the solver does not detect contacts internally."""
    adapter, model = backend.solver_adapter, backend.model
    backend.step_graph = None
    if not adapter.uses_collision_pipeline(backend):
        backend.contacts = adapter.create_contacts(backend)
    else:
        collision_cfg = backend.cfg.collision_cfg
        args = collision_cfg.to_pipeline_args() if collision_cfg is not None else {"broad_phase": "explicit"}
        args["deterministic"] = backend.deterministic_mode != wp.DeterministicMode.NOT_GUARANTEED
        required = getattr(backend.solver, "get_max_contact_count", lambda: 0)()
        if backend.collision_pipeline is None:
            backend.collision_pipeline = CollisionPipeline(model, **args)
        backend.contacts = backend.collision_pipeline.contacts()
        # MuJoCo Warp can require more contacts than the pipeline estimates.
        if required > backend.contacts.rigid_contact_max:
            if args["deterministic"]:
                # The deterministic sort buffer is sized at construction, so rebuild the pipeline to match.
                backend.collision_pipeline = CollisionPipeline(model, **args, rigid_contact_max=required)
                backend.contacts = backend.collision_pipeline.contacts()
            else:
                backend.contacts = Contacts(
                    rigid_contact_max=required,
                    soft_contact_max=0,
                    device=backend.device,
                    requested_attributes=model.get_requested_contact_attributes(),
                )
    if backend.contacts is not None:
        adapter.prepare_contacts(backend)


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

    View rows map to worlds through the model's articulation-to-world table, so views over a subset of worlds flag the
    right worlds, and global articulations flag FK only.

    Args:
        backend: Simulation backend.
        env_mask: Mask over view rows.
        env_ids: Selected view rows.
        articulation_ids: Model articulation index of each ``(row, articulation)``; ``None`` flags everything.
    """
    backend.kinematics_dirty = True
    outputs = [backend.world_mask, backend.fk_mask]
    articulation_world = backend.model.articulation_world
    if articulation_ids is not None and env_mask is not None:
        inputs = [env_mask, articulation_ids, articulation_world]
        wp.launch(_mark_fk_from_mask, articulation_ids.shape, inputs, outputs, device=backend.device)
    elif articulation_ids is not None and env_ids is not None:
        dim = (env_ids.shape[0], articulation_ids.shape[1])
        inputs = [env_ids, articulation_ids, articulation_world]
        wp.launch(_MARK_FK_FROM_IDS.select(env_ids), dim, inputs, outputs, device=backend.device)
    else:
        backend.world_mask[: backend.model.world_count].fill_(True)
        backend.fk_mask.fill_(True)


def invalidate_body_state(
    backend: NewtonBackend,
    env_ids: wp.array | None = None,
    env_mask: wp.array | None = None,
    row_worlds: wp.array | None = None,
) -> None:
    """Flag worlds whose maximal-coordinate body state was written, without requesting FK.

    Args:
        backend: Simulation backend.
        env_ids: Selected view rows.
        env_mask: Mask over view rows.
        row_worlds: World of each view row (see :func:`view_row_worlds`); ``None`` when rows are worlds.
    """
    backend.kinematics_dirty = True
    selection = env_mask if env_mask is not None else env_ids
    if selection is None:
        backend.world_mask[: backend.model.world_count].fill_(True)
        return
    row_worlds = backend.world_ids if row_worlds is None else row_worlds
    kernel = _mark_worlds_from_mask if env_mask is not None else _mark_worlds_from_ids
    wp.launch(kernel, selection.shape[0], [selection, row_worlds], [backend.world_mask], device=backend.device)


def invalidate_worlds(backend: NewtonBackend, worlds: wp.array) -> None:
    """Flag whole worlds for a solver reset and forward kinematics, without host synchronization.

    This is the commit of a world-masked reset: write the state of the masked worlds, call this, then
    :func:`forward`. Both launch masked kernels whose effect follows the mask contents at launch time, so they can be
    recorded into a caller's CUDA graph and replayed with a different mask each time.

    Args:
        backend: Simulation backend.
        worlds: Boolean mask over local worlds. Shape is (world_count,).
    """
    backend.kinematics_dirty = True
    model = backend.model
    wp.launch(_mark_worlds, model.world_count, [worlds], [backend.world_mask], device=backend.device)
    if model.articulation_count:
        inputs = [worlds, model.articulation_world]
        wp.launch(
            _mark_articulations_of_worlds, model.articulation_count, inputs, [backend.fk_mask], device=backend.device
        )


def view_row_worlds(backend: NewtonBackend, articulation_ids: wp.array) -> wp.array:
    """Return the world of each row of an articulation view. Call once at bind time; it reads device arrays.

    Args:
        backend: Backend of the view's model.
        articulation_ids: Model articulation index of each ``(row, articulation)``.

    Returns:
        World index of each row, shape ``(num_rows,)``; ``-1`` for global articulations.
    """
    worlds = backend.model.articulation_world.numpy()[articulation_ids.numpy()[:, 0]]
    return wp.array(worlds.astype(np.int32), dtype=wp.int32, device=backend.device)


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
    adapter, state = backend.solver_adapter, backend.state_0
    adapter.reset_solver(backend, state, backend.world_mask)
    adapter.eval_fk(backend, state, backend.world_mask, backend.fk_mask)
    backend.fk_mask.zero_()
    backend.world_mask.zero_()
    backend.kinematics_dirty = False


def mark_model_changed(backend: NewtonBackend, change: ModelFlags, env_mask: wp.array | None = None) -> None:
    """Queue a model-property transaction, selecting affected worlds entirely on the device.

    Args:
        backend: Simulation backend.
        change: Property categories the solver must refresh.
        env_mask: Changed worlds; ``None`` means an unconditional edit.
    """
    if backend.solver is None:
        return
    backend.model_changes.add(change)
    if env_mask is None:
        backend.model_dirty.fill_(1)
    else:
        wp.launch(_mark_model_changes, env_mask.size, [env_mask, backend.model_dirty], device=backend.device)


def notify_model_changes(backend: NewtonBackend) -> None:
    """Commit queued property edits once, skipping solver work when every edit mask was empty.

    CUDA uses a conditional graph node so deciding whether to rebuild constants never reads a mask back to the host.
    Multiple property categories are combined before notification, avoiding repeated constant rebuilds.

    Args:
        backend: Simulation backend with a solver.
    """
    if not backend.model_changes:
        return
    flags = 0
    for change in backend.model_changes:
        flags |= change
        if change in backend.solver_adapter.ignored_model_changes and change not in backend.warned_model_changes:
            logger.warning(backend.solver_adapter.ignored_model_changes[change])
            backend.warned_model_changes.add(change)

    def refresh():
        backend.solver.notify_model_changed(flags)
        backend.model_dirty.zero_()

    def conditional_refresh():
        wp.capture_if(backend.model_dirty, refresh)

    with wp.ScopedDevice(backend.device):
        if not backend.device.is_cuda:
            if backend.model_dirty.numpy()[0]:
                refresh()
        elif backend.device.is_capturing:
            if not isinstance(wp.get_device_allocator(backend.device), _GraphAllocator):
                raise RuntimeError(
                    "Record model-property edits with capture_graph() so conditional updates own their scratch storage."
                )
            conditional_refresh()
        elif backend.solver_adapter.supports_graph_capture(backend):
            graph = backend.model_change_graphs.get(flags)
            if graph is None:
                graph = backend.model_change_graphs[flags] = capture_graph(backend.device, conditional_refresh)
            graph.launch()
        elif backend.model_dirty.numpy()[0]:
            refresh()
    backend.model_changes.clear()


# ----- Consumers -----------------------------------------------------------------


def register_step_callback(
    backend: NewtonBackend, fn: Callable[..., None], phase: StepPhase, *, graphable: bool = True, name: str = ""
) -> StepCallback:
    """Run ``fn`` at ``phase`` of every subsequent step.

    Args:
        backend: Simulation backend.
        fn: Function to run; :attr:`StepPhase.STATE_FORCE` functions receive the substep's input state.
        phase: Where the function runs.
        graphable: Whether the function can be recorded into a CUDA graph. Other functions run eagerly in place.
        name: Label used in errors and profiles.

    Returns:
        The callback, for :func:`unregister_step_callback`. Registering the same function and phase again returns
        the existing callback.
    """
    for callback in backend.callbacks:
        if callback.fn == fn and callback.phase == phase:
            return callback
    callback = StepCallback(fn, phase, graphable, name or getattr(fn, "__name__", ""))
    backend.callbacks.append(callback)
    backend.step_graph = None
    return callback


def unregister_step_callback(backend: NewtonBackend, callback: StepCallback) -> None:
    """Stop running a callback; unregistering an absent callback is a no-op.

    Args:
        backend: Simulation backend.
        callback: Callback returned by :func:`register_step_callback`.
    """
    if callback in backend.callbacks:
        backend.callbacks.remove(callback)
        backend.step_graph = None


def activate_actuators(backend: NewtonBackend) -> NewtonActuatorAdapter | None:
    """Run the model's Newton actuators inside the step. Idempotent.

    The adapter addresses the model's flat DOF space and articulations bind their own views of it, so worlds may hold
    different DOF layouts.

    Args:
        backend: Simulation backend.

    Returns:
        The adapter, or ``None`` when the model has no actuators.
    """
    model = backend.model
    if backend.actuators is None and model.actuators:
        from isaaclab.actuators.newton import NewtonActuatorAdapter  # noqa: PLC0415

        backend.actuators = NewtonActuatorAdapter(
            actuators=list(model.actuators),
            num_envs=1,
            num_joints=model.joint_dof_count,
            dof_offset=0,
            device=str(backend.device),
        )
        backend.actuators.finalize(backend.control)
        backend.step_graph = None
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
    if not backend.solver_adapter.supports_contact_sensors:
        raise NotImplementedError(
            f"Newton contact sensors are not yet supported by {backend.solver_adapter.__name__}:"
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
    backend.step_graph = None
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
    backend.step_graph = None
    return sensor


def compile_label_pattern(expr: str | list[str] | None) -> re.Pattern[str] | None:
    """Compile Isaac Lab selector expressions for Newton's full label matching."""
    if not expr:
        return None
    return re.compile("|".join((expr,) if isinstance(expr, str) else expr))


# ----- Stepping ----------------------------------------------------------------------


def build_step_graph(backend: NewtonBackend, steps: int) -> StepGraph:
    """Unroll ``steps`` physics steps into straight-line operations with every buffer bound.

    Each physics step runs ``collide -> CONTROL -> Newton actuators -> substeps``, where each substep runs
    ``staged forces -> STATE_FORCE -> solver`` and an optional mid-step collision. ``POST_ACTUATOR`` callbacks run
    after the actuators of the last physics step, and ``POST_STEP`` callbacks and sensors run after every physics step.
    Double-buffered states alternate when the graph is built, and every physics step ends in
    :attr:`NewtonBackend.state_0`, so callbacks and bound views always observe the current state.

    Forces written to ``state_0`` before the step apply to every substep of every physics step and are cleared at the
    end, so each write applies to exactly the next :func:`step`.

    Args:
        backend: Simulation backend with a solver.
        steps: Physics steps one launch advances.

    Returns:
        The step graph, not yet captured.
    """
    adapter, state_0, control = backend.solver_adapter, backend.state_0, backend.control
    spare = state_0 if adapter.single_state else backend.state_1
    pipeline = backend.collision_pipeline
    contacts = backend.contacts if pipeline is not None else None
    collide_every = backend.cfg.collision_decimation if pipeline is not None else 0
    phases = {phase: [cb for cb in backend.callbacks if cb.phase == phase] for phase in StepPhase}
    ops: list[StepOp] = []

    def emit(fn: Callable[[], None], name: str, graphable: bool = True) -> None:
        ops.append(StepOp(fn, graphable, name))

    def emit_callbacks(phase: StepPhase, *args) -> None:
        for cb in phases[phase]:
            emit(partial(cb.fn, *args) if args else cb.fn, cb.name, cb.graphable)

    # Forces persist in a single state across substeps. Double-buffered solvers alternate input states, and STATE_FORCE
    # callbacks add to the input forces, so both re-apply staged copies before every substep.
    if not adapter.single_state or phases[StepPhase.STATE_FORCE]:
        for name in ("body_f", "particle_f"):
            forces = getattr(state_0, name, None)
            if forces is not None and name not in backend.staged_forces:
                backend.staged_forces[name] = wp.empty_like(forces)
        staged = list(backend.staged_forces.items())
    else:
        staged = []
    actuator_steps = 0
    for physics_step in range(steps):
        if adapter.prepares_step:
            emit(partial(adapter.prepare_step, backend, state_0), "solver.prepare_step")
        if pipeline is not None:
            emit(partial(pipeline.collide, state_0, contacts), "collide")
        emit_callbacks(StepPhase.CONTROL)
        for name, buffer in staged:
            emit(partial(wp.copy, buffer, getattr(state_0, name)), f"stage_forces.{name}")
        if backend.actuators is not None:
            _emit_actuators(emit, backend.actuators, state_0, control, backend.dt, actuator_steps % 2)
            actuator_steps += 1
        if physics_step == steps - 1:
            emit_callbacks(StepPhase.POST_ACTUATOR)
        state_in, state_out = state_0, spare
        for substep in range(backend.num_substeps):
            for name, buffer in staged:
                emit(partial(wp.copy, getattr(state_in, name), buffer), f"apply_forces.{name}")
            emit_callbacks(StepPhase.STATE_FORCE, state_in)
            emit(
                partial(adapter.step_solver, backend, state_in, state_out, contacts, backend.solver_dt),
                "solver",
                adapter.supports_graph_capture(backend),
            )
            if not adapter.single_state:
                state_in, state_out = state_out, state_in
            if collide_every > 0 and (substep + 1) % collide_every == 0 and substep + 1 < backend.num_substeps:
                emit(partial(pipeline.collide, state_in, contacts), "collide")
        if state_in is not state_0:
            emit(partial(state_0.assign, state_in), "state.assign")
        for name, buffer in staged:
            emit(partial(wp.copy, getattr(state_0, name), buffer), f"restore_forces.{name}")
        emit_callbacks(StepPhase.POST_STEP)
        if backend.contact_sensors or backend.imu_sensors:
            emit(partial(_update_sensors, backend), "sensors")
    if actuator_steps % 2:
        # Keep actuator history in the first buffers so every replay starts from the same addresses.
        adapter = backend.actuators
        for actuator, current, previous in zip(adapter.actuators, *adapter.state_buffers):
            if current is not None:
                emit(partial(current.assign, previous), "actuators.assign", actuator.is_graphable())

    # Forces apply to one step; consumers author them again before the next.
    emit(state_0.clear_forces, "clear_forces")
    return StepGraph(ops, steps)


def _emit_actuators(
    emit: Callable[..., None], adapter: NewtonActuatorAdapter, state: State, control: Any, dt: float, parity: int
) -> None:
    """Emit Newton actuator operations reading one history buffer and writing the other."""
    states_a, states_b = adapter.state_buffers
    states_in, states_out = (states_a, states_b) if parity == 0 else (states_b, states_a)
    emit(partial(adapter.zero_outputs, control), "actuators.zero")
    for actuator, state_in, state_out in zip(adapter.actuators, states_in, states_out):
        step_actuator = partial(actuator.step, state, control, state_in, state_out, dt=dt)
        emit(step_actuator, f"actuator.{type(actuator.controller).__name__}", actuator.is_graphable())


def _update_sensors(backend: NewtonBackend) -> None:
    """Push the current state to IMU and contact sensors."""
    for sensor in backend.imu_sensors:
        sensor.update(backend.state_0)
    if backend.contact_sensors:
        backend.solver.update_contacts(backend.contacts, backend.state_0)
        for sensor in backend.contact_sensors.values():
            sensor.update(backend.state_0, backend.contacts)


def prepare(backend: NewtonBackend) -> StepGraph:
    """Build the step graph for :attr:`NewtonBackend.steps_per_call` unless the current one matches.

    This allocates any required force staging buffers without advancing physics. Call it before
    recording :func:`step` into an outer capture.

    Args:
        backend: Simulation backend with a solver.

    Returns:
        The step graph.
    """
    graph = backend.step_graph
    if graph is None or graph.steps != backend.steps_per_call:
        graph = backend.step_graph = build_step_graph(backend, backend.steps_per_call)
    return graph


def step(backend: NewtonBackend) -> StepGraph:
    """Advance the prepared schedule with live reads throughout its controller phases."""
    backend.is_stepping = True
    try:
        return _step(backend)
    finally:
        backend.is_stepping = False


def _step(backend: NewtonBackend) -> StepGraph:
    """Advance :attr:`NewtonBackend.steps_per_call` physics steps.

    Notifies model changes, runs :func:`forward` for authored state, and launches the step graph. The first launch
    after the graph is built runs eagerly, which also performs lazy solver allocations, and then the graph is captured
    with :attr:`NewtonBackend.capture` for later launches.

    Args:
        backend: Simulation backend with a solver.

    Returns:
        The step graph that ran.
    """
    notify_model_changes(backend)
    forward(backend)
    graph = prepare(backend)
    if graph.captured and graph.graphable:
        # Graph replays carry their device; skip the host-side device scope.
        graph.launch()
        return graph
    with execution_stream(backend.device):
        graph.launch()
    if backend.capture is not None and not graph.captured:
        graph.capture(backend.capture)
    return graph


def record_step(backend: NewtonBackend) -> StepGraph:
    """Record :attr:`NewtonBackend.steps_per_call` physics steps into the caller's active CUDA graph capture.

    The caller owns the capture of a larger graph, such as a whole environment step, on whichever stream it records:
    a Warp capture, or a Torch capture that Warp joins on the same stream with ``wp.capture_begin(stream,
    external=True)`` so that Warp allocations stay graph safe. :func:`prepare` the step before the
    capture; the solvers need no warm-up step, but step callbacks must not allocate when first called. The masked
    :func:`forward` is always recorded, so each replay applies state authored since the previous one, inside or outside
    the graph. The recording stays valid while :attr:`NewtonBackend.step_graph` is the returned graph; registering
    callbacks or changing the step count builds a new one, which must be recorded again.

    Args:
        backend: Simulation backend with a solver.

    Returns:
        The recorded step graph.

    Raises:
        RuntimeError: If no step graph for the current step count was prepared, or it contains operations that cannot
            be recorded.
    """
    graph = backend.step_graph
    if graph is None or graph.steps != backend.steps_per_call:
        raise RuntimeError("Prepare the Newton step before recording it into an outer CUDA graph.")
    if not graph.graphable:
        eager = sorted({op.name for op in graph.ops if not op.graphable})
        raise RuntimeError(f"The Newton step cannot be recorded: {eager} are not graphable.")
    notify_model_changes(backend)
    forward(backend, force=True)
    backend.is_stepping = True
    try:
        # The outer capture owns stream dependencies; Torch must submit to the same stream as Warp.
        with (
            torch.cuda.stream(wp.stream_to_torch(wp.get_stream(backend.device)))
            if wp.get_device(backend.device).is_cuda
            else contextlib.nullcontext()
        ):
            graph.record()
    finally:
        backend.is_stepping = False
    backend.transforms_may_change_on_graph_replay = True
    return graph


# ----- CUDA graph capture and scene queries ------------------------------------------


class _GraphAllocator:
    """Keep Warp scratch in Torch's graph pool, including inside CUDA conditional nodes.

    CUDA conditional nodes cannot contain allocation nodes. Torch reserves storage during recording instead,
    and this graph-owned allocator retains it without reuse until the graph and all bound arrays are released.
    """

    def __init__(self, device: wp.Device):
        self.device = wp.device_to_torch(device)
        self.buffers: list[torch.Tensor] = []

    def allocate(self, size_in_bytes: int) -> int:
        buffer = torch.empty(size_in_bytes, dtype=torch.uint8, device=self.device)
        self.buffers.append(buffer)
        return buffer.data_ptr()

    def deallocate(self, ptr: int, size_in_bytes: int) -> None:
        # Storage belongs to the recorded graph, not the temporary Warp array that requested it.
        pass


class CapturedGraph:
    """A graph with both allocators' capture lifetimes retained until its last replay."""

    def __init__(
        self, torch_graph: torch.cuda.CUDAGraph, warp_graph: wp.Graph, device: wp.Device, allocator: _GraphAllocator
    ):
        self.allocator = allocator
        self.torch_graph = torch_graph
        self.warp_graph = warp_graph
        self.device = device

    def launch(self) -> None:
        """Replay on the simulation stream, ordered with the caller's Torch work."""
        with execution_stream(self.device):
            self.torch_graph.replay()


@contextlib.contextmanager
def execution_stream(device: wp.DeviceLike):
    """Order mixed eager Torch/Warp work with the caller, using the simulation's Warp stream."""
    device = wp.get_device(device)
    with wp.ScopedDevice(device):
        if not device.is_cuda:
            yield
            return
        stream = wp.stream_to_torch(wp.get_stream(device))
        caller = torch.cuda.current_stream(stream.device)
        stream.wait_stream(caller)
        try:
            with torch.cuda.stream(stream):
                yield
        finally:
            caller.wait_stream(stream)


def capture_graph(device: wp.DeviceLike, fn: Callable[[], None], *, relaxed: bool = False) -> CapturedGraph:
    """Record Torch and Warp work on one stream without advancing the simulation.

    Torch owns capture so temporary controller tensors use its graph memory pool. Warp joins that capture and
    retains its own allocation lifetimes. Both owners must live as long as the executable graph.

    Args:
        device: CUDA device.
        fn: Work to record. Perform lazy initialization before capture.
        relaxed: Allow unrelated work on other threads, as required by Kit rendering.

    Returns:
        The captured graph and its allocation owners.
    """
    device = wp.get_device(device)
    warp_stream = wp.get_stream(device)
    stream = wp.stream_to_torch(warp_stream)
    if relaxed:
        stream = torch.cuda.Stream(device=wp.device_to_torch(device))
        warp_stream = wp.stream_from_torch(stream)
    caller = torch.cuda.current_stream(stream.device)
    stream.wait_stream(caller)
    graph = torch.cuda.CUDAGraph()
    allocator = _GraphAllocator(device)
    with _paused_gc(), torch.cuda.stream(stream), wp.ScopedStream(warp_stream), wp.ScopedAllocator(device, allocator):
        graph.capture_begin(capture_error_mode="relaxed" if relaxed else "thread_local")
        try:
            wp.capture_begin(stream=warp_stream, external=True)
            try:
                fn()
            finally:
                warp_graph = wp.capture_end(stream=warp_stream)
        finally:
            graph.capture_end()
    return CapturedGraph(graph, warp_graph, device, allocator)


@contextlib.contextmanager
def _paused_gc():
    """Keep collection-driven frees out of CUDA graph capture.

    The deferred garbage is left to the next automatic collection: an explicit collection after every capture costs
    hundreds of milliseconds on a large heap.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def run_query(
    backend: NewtonBackend,
    timestamp: int,
    query: Callable[[], None],
    graph: tuple[tuple[int, ...], CapturedGraph] | None,
    *,
    use_cuda_graph: bool = True,
) -> tuple[tuple[int, ...], CapturedGraph] | None:
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
                refit = partial(_refit_bvh, backend)
                cached.data = pointers, capture_graph(str(model.device), refit, relaxed=has_kit())
            cached.data[1].launch()
        else:
            _refit_bvh(backend)
        cached.timestamp = timestamp
    if not model.device.is_cuda or not use_cuda_graph:
        query()
        return None
    if graph is None or graph[0] != pointers:
        graph = pointers, capture_graph(str(model.device), query, relaxed=has_kit())
    graph[1].launch()
    return graph


def _refit_bvh(backend: NewtonBackend) -> None:
    model, state = backend.model, backend.state_0
    if model.shape_count:
        model.bvh_refit_shapes(state)
    if model.particle_count:
        model.bvh_refit_particles(state)
