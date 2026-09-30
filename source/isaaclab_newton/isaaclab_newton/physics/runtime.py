# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton simulation data: construction requests, world schema, and the mutable runtime."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp
from newton import CollisionPipeline, Contacts, ModelBuilder, State
from newton.sensors import SensorContact, SensorFrameTransform, SensorIMU

from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.warp.index_kernel import IndexKernelDispatcher

from .featherstone_manager_cfg import FeatherstoneSolverCfg
from .mjwarp_manager_cfg import MJWarpSolverCfg
from .step_program import StepOp, StepPhase, StepProgram, StepStage
from .xpbd_manager_cfg import XPBDSolverCfg

if TYPE_CHECKING:
    from isaaclab.actuators.newton import NewtonActuatorAdapter

    from .newton_collision_cfg import NewtonCollisionPipelineCfg
    from .newton_manager import NewtonBackend
    from .newton_manager_cfg import NewtonCfg
    from .solver_binding import NewtonSolverBinding

logger = logging.getLogger(__name__)

SiteEntry = tuple[int, None] | tuple[None, list[list[int]]]
"""Resolved site: ``(global_shape_index, None)`` for a global site, or ``(None, per_world_indices)``."""

_SENSORS_BY_STATE_ATTRIBUTE = {
    "body_qdd": "the IMU or PVA sensor",
    "body_parent_f": "the joint-wrench sensor",
}
"""Isaac Lab sensors that read each state attribute MuJoCo Warp's sensor stage fills."""


# ----- Reset-mask kernels ------------------------------------------------------


@wp.kernel(enable_backward=False)
def _mark_reset_from_mask(
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
def _mark_reset_from_ids(
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


_MARK_RESET_FROM_IDS = IndexKernelDispatcher(_mark_reset_from_ids, ("env_ids",))


@wp.kernel(enable_backward=False)
def _mark_worlds_from_mask(env_mask: wp.array(dtype=wp.bool), world_mask: wp.array(dtype=wp.bool)):
    """Flag masked worlds for a solver reset without requesting FK."""
    world = wp.tid()
    if env_mask[world]:
        world_mask[world] = True


@wp.kernel(enable_backward=False)
def _mark_worlds_from_ids(env_ids: wp.array(dtype=wp.int32), world_mask: wp.array(dtype=wp.bool)):
    """Flag selected worlds for a solver reset without requesting FK."""
    world_mask[env_ids[wp.tid()]] = True


# ----- Immutable world description -------------------------------------------


@dataclass(frozen=True)
class NewtonSchema:
    """Immutable description of the finalized Newton worlds.

    The schema is what task compilers bind against: world membership, counts, and timing. It never holds mutable
    simulation state, and a structural change (a new model or step timing) produces a new schema.
    """

    device: str
    """Simulation device."""

    world_count: int
    """Number of simulated worlds (environments)."""

    world_prototypes: np.ndarray | None
    """Clone-plan world prototype of each world, shape ``(world_count,)``, or ``None`` without a clone plan.

    Worlds that share a prototype contain the same assets in the same layout.
    """

    physics_dt: float
    """Duration of one physics step [s]."""

    num_substeps: int
    """Solver substeps per physics step."""

    collision_decimation: int
    """Solver substeps between mid-step collision passes, or ``0`` to collide once per physics step."""

    body_count: int
    """Rigid bodies across all worlds."""

    joint_dof_count: int
    """Joint degrees of freedom across all worlds."""

    articulation_count: int
    """Articulations across all worlds."""

    @property
    def solver_dt(self) -> float:
        """Duration of one solver substep [s]."""
        return self.physics_dt / self.num_substeps


# ----- Construction requests --------------------------------------------------


@dataclass
class NewtonCloneRecord:
    """Native replication outputs consumed after model finalization."""

    num_envs: int
    """Number of cloned worlds."""

    world_prototypes: np.ndarray
    """Clone-plan world prototype of each world."""

    site_index_map: dict[str, SiteEntry]
    """Resolved site indices by site label."""

    world_xforms: list[wp.transform] | None
    """Root transform of each cloned world."""

    source_builders: dict[str, ModelBuilder]
    """Per-source builders retained so single-model consumers can finalize one environment."""

    particle_ranges: dict[str, tuple[int, int]]
    """Native ``(start, count)`` particle range of each imported particle prim."""

    cable_bindings: dict[str, list[int]]
    """Native capsule shape indices of each open cable."""


@dataclass
class NewtonBuildRequests:
    """Construction requests collected before model finalization.

    Sensors request sites and extended attributes, the cloner records its outputs, and callers extend each cloned
    world. The requests persist across hard resets, which rebuild the model from the same builder.
    """

    sites: dict[tuple[str | None, bool, tuple[float, ...]], tuple[str, wp.transform]] = field(default_factory=dict)
    """Pending site requests keyed by ``(body_pattern, per_world, transform)``; consumed by :meth:`inject_sites`."""

    site_index_map: dict[str, SiteEntry] = field(default_factory=dict)
    """Resolved sites by label."""

    state_attributes: set[str] = field(default_factory=set)
    """Extended :class:`newton.State` attributes to allocate."""

    contact_attributes: set[str] = field(default_factory=set)
    """Extended :class:`newton.Contacts` attributes to allocate."""

    world_builder_hooks: list[Callable[[ModelBuilder, int, np.ndarray, np.ndarray], None]] = field(default_factory=list)
    """Callbacks extending every world built by Newton replication."""

    clone: NewtonCloneRecord | None = None
    """Outputs of the last native replication."""

    def register_site(self, body_pattern: str | None, xform: wp.transform, *, per_world: bool = False) -> str:
        """Request a site; identical requests share one label.

        Args:
            body_pattern: Regex matched against body labels, or ``None`` for a global site.
            xform: Site transform relative to the body.
            per_world: Create one bodyless site in each world's frame. Requires ``body_pattern=None``.

        Returns:
            The assigned site label.
        """
        if per_world and body_pattern is not None:
            raise ValueError("per_world site registration requires body_pattern=None.")
        key = (body_pattern, per_world, tuple(xform))
        if key not in self.sites:
            self.sites[key] = (f"ft_{len(self.sites)}", xform)
        return self.sites[key][0]

    def inject_sites(
        self, main_builder: ModelBuilder, source_builders: dict[str, ModelBuilder]
    ) -> tuple[dict[str, int], dict[int, dict[str, list[int]]], dict[str, wp.transform]]:
        """Add pending sites to source builders, or to the main builder for global and shared-asset sites.

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
        for (body_pattern, per_world, _), (label, xform) in self.sites.items():
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
        self.sites.clear()
        return global_sites, source_sites, root_sites

    def prepare_builder(self, builder: ModelBuilder) -> None:
        """Apply pending sites and attribute requests to a builder about to be finalized.

        Args:
            builder: Builder of the simulated model.
        """
        if self.sites:
            global_sites, body_sites, root_sites = self.inject_sites(builder, {})
            self.site_index_map.update((label, (index, None)) for label, index in global_sites.items())
            self.site_index_map.update(
                (label, (None, [indices])) for label, indices in body_sites.get(id(builder), {}).items()
            )
            for label, xform in root_sites.items():
                self.site_index_map[label] = (None, [[builder.add_site(body=-1, xform=xform, label=label)]])
        if self.state_attributes:
            builder.request_state_attributes(*self.state_attributes)
        builder.request_contact_attributes(*sorted(self.contact_attributes))


# ----- Sensors ----------------------------------------------------------------


@dataclass
class NewtonSensors:
    """Newton sensors updated at the end of every step program."""

    contact: dict[tuple, SensorContact] = field(default_factory=dict)
    frame_transform: list[SensorFrameTransform] = field(default_factory=list)
    imu: list[SensorIMU] = field(default_factory=list)

    def update(self, state: State, contacts: Contacts | None, solver: NewtonSolverBinding) -> None:
        """Push the latest state to every sensor.

        Args:
            state: Current state.
            contacts: Contacts of the last collision pass.
            solver: Solver that reports its contacts for contact sensors.
        """
        for sensor in self.frame_transform:
            sensor.update(state)
        for sensor in self.imu:
            sensor.update(state)
        if self.contact:
            solver.solver.update_contacts(contacts, state)
            for sensor in self.contact.values():
                sensor.update(state, contacts)


# ----- Determinism ------------------------------------------------------------


def resolve_deterministic_mode(cfg: NewtonCfg) -> wp.DeterministicMode:
    """Translate the backend-agnostic determinism request into a Warp mode.

    An explicit :attr:`NewtonCfg.deterministic_mode` wins over the generic
    :attr:`~isaaclab.physics.PhysicsCfg.deterministic` request. MuJoCo on the CPU is reproducible on its own and Warp's
    mode does not reach it, so no mode is applied there.

    Args:
        cfg: Newton physics configuration. A generic request on MJWarp disables its internal sensors.

    Returns:
        The mode to apply to the solver and collision pipeline.
    """
    solver_cfg = cfg.solver_cfg
    if getattr(solver_cfg, "use_mujoco_cpu", False):
        if cfg.deterministic or cfg.deterministic_mode != "not_guaranteed":
            logger.info("MuJoCo CPU backend is already reproducible; Newton's deterministic mode is not applied.")
        return wp.DeterministicMode.NOT_GUARANTEED
    if cfg.deterministic_mode != "not_guaranteed":
        return {
            "run_to_run": wp.DeterministicMode.RUN_TO_RUN,
            "gpu_to_gpu": wp.DeterministicMode.GPU_TO_GPU,
        }[cfg.deterministic_mode]
    if not cfg.deterministic:
        return wp.DeterministicMode.NOT_GUARANTEED
    # MuJoCo Warp cannot honor a guarantee while its sensor kernels run, so the generic request implies it.
    if isinstance(solver_cfg, MJWarpSolverCfg):
        solver_cfg.disable_sensors = True
    return wp.DeterministicMode.RUN_TO_RUN


def validate_deterministic_mode(cfg: NewtonCfg, mode: wp.DeterministicMode, state_attributes: set[str]) -> None:
    """Reject solver settings that cannot provide the requested determinism guarantee.

    Args:
        cfg: Newton physics configuration.
        mode: Requested mode.
        state_attributes: Extended state attributes requested by sensors.
    """
    if mode == wp.DeterministicMode.NOT_GUARANTEED:
        return
    solver_cfg = cfg.solver_cfg
    if not isinstance(solver_cfg, (FeatherstoneSolverCfg, MJWarpSolverCfg, XPBDSolverCfg)):
        raise ValueError(
            f"Newton deterministic mode {mode.name} is not supported by {type(solver_cfg).__name__}. "
            "Use MJWarp on the GPU, XPBD, or Featherstone, or disable deterministic mode."
        )
    if getattr(solver_cfg, "use_mujoco_cpu", False):
        raise ValueError(
            f"Newton deterministic mode {mode.name} is not supported by the MuJoCo CPU backend. "
            "Set MJWarpSolverCfg.use_mujoco_cpu=False or disable deterministic mode."
        )
    if not isinstance(solver_cfg, MJWarpSolverCfg):
        return
    if not solver_cfg.disable_sensors:
        raise ValueError(
            f"Newton deterministic mode {mode.name} is not supported while MuJoCo Warp's internal sensor "
            "computation is enabled. Set MJWarpSolverCfg.disable_sensors=True or disable deterministic mode."
        )
    blocked = state_attributes & _SENSORS_BY_STATE_ATTRIBUTE.keys()
    if blocked:
        sensors = sorted({_SENSORS_BY_STATE_ATTRIBUTE[attr] for attr in blocked})
        raise ValueError(
            f"This task does not support deterministic physics: it uses {' and '.join(sensors)},"
            f" reading {sorted(blocked)}. Those attributes come from MuJoCo's post-constraint pass,"
            " which runs inside the sensor stage that a determinism guarantee must disable, so the"
            " values would never be refreshed. Remove the sensors, or drop the determinism request"
            f" (deterministic_mode={mode.name})."
        )


# ----- Runtime ----------------------------------------------------------------


class NewtonRuntime:
    """Everything bound to one finalized Newton model.

    The runtime is created when the model is finalized and discarded on a hard reset or close, so no buffer, solver,
    or consumer stage outlives the model it points into. Consumers register stages, sensors, and actuators while
    ``PHYSICS_READY`` dispatches; :meth:`bind_solver` then constructs the solver and contacts, and :meth:`compile`
    turns the whole step into one :class:`StepProgram`.
    """

    def __init__(self, backend: NewtonBackend, schema: NewtonSchema):
        """Initialize the runtime.

        Args:
            backend: Borrowed native model, state, and control; the simulation registry owns it.
            schema: World description of the finalized model.
        """
        model = backend.model
        self.backend = backend
        self.schema = schema
        # Newton reserves the final world-mask slot for global entities in world -1. Isaac Lab resets local worlds
        # only, so that slot stays false.
        self.world_mask = wp.zeros(model.world_count + 1, dtype=wp.bool, device=schema.device)
        self.fk_mask = wp.zeros(model.articulation_count, dtype=wp.bool, device=schema.device)
        self.kinematics_dirty = False
        self.transforms_may_change_on_graph_replay = False
        """Set when state is written during an outer capture, whose replays bypass Python invalidation."""
        self.model_changes: set[int] = set()
        self._warned_model_changes: set[int] = set()

        self.stages: list[StepStage] = []
        self.sensors = NewtonSensors()
        self.actuators: NewtonActuatorAdapter | None = None
        self.owns_decimation = False
        """Whether one :meth:`step` advances the whole decimation loop; set when Newton actuators are active."""

        self.solver: NewtonSolverBinding | None = None
        self.collision_pipeline: CollisionPipeline | None = None
        self.contacts: Contacts | None = None
        self.program: StepProgram | None = None

    # ----- Solver -----------------------------------------------------------

    def bind_solver(
        self,
        binding_type: type[NewtonSolverBinding],
        cfg: NewtonCfg,
        deterministic_mode: wp.DeterministicMode,
    ) -> None:
        """Construct the solver and its contacts.

        Args:
            binding_type: Solver binding to construct.
            cfg: Newton physics configuration.
            deterministic_mode: Determinism guarantee for the solver and collision pipeline.
        """
        binding_type.validate_cfg(cfg)
        self.solver = binding_type(self.backend.model, cfg.solver_cfg, deterministic_mode)
        if self.sensors.contact and not self.solver.supports_contact_sensors:
            raise NotImplementedError(
                f"Newton contact sensors are not yet supported by {binding_type.__name__} because its contact forces"
                " live in per-entry buffers. Remove the contact sensor."
            )
        if not self.solver.single_state:
            self.solver.initialize_output_state(self.backend.state_1)
        self.allocate_contacts(cfg.collision_cfg)

    def allocate_contacts(self, collision_cfg: NewtonCollisionPipelineCfg | None) -> None:
        """Allocate contacts, and the collision pipeline when the solver does not collide internally.

        Args:
            collision_cfg: Collision pipeline configuration, or ``None`` for an explicit broad phase.
        """
        solver, model = self.solver, self.backend.model
        self.invalidate_program()
        if not solver.needs_collision_pipeline:
            self.contacts = solver.create_contacts()
        else:
            args = collision_cfg.to_pipeline_args() if collision_cfg is not None else {"broad_phase": "explicit"}
            deterministic = solver.deterministic_mode != wp.DeterministicMode.NOT_GUARANTEED
            args["deterministic"] = deterministic
            required = solver.minimum_contact_capacity()
            if self.collision_pipeline is None:
                self.collision_pipeline = CollisionPipeline(model, **args)
            self.contacts = self.collision_pipeline.contacts()
            # Solvers such as MuJoCo Warp can require more contacts than the pipeline estimates.
            if required > self.contacts.rigid_contact_max:
                if deterministic:
                    # The deterministic sort buffer is sized at construction, so rebuild the pipeline to match.
                    args["rigid_contact_max"] = required
                    self.collision_pipeline = CollisionPipeline(model, **args)
                    self.contacts = self.collision_pipeline.contacts()
                else:
                    self.contacts = Contacts(
                        rigid_contact_max=required,
                        soft_contact_max=0,
                        device=self.schema.device,
                        requested_attributes=model.get_requested_contact_attributes(),
                    )
        if self.contacts is not None:
            solver.prepare_contacts(self.contacts, self.collision_pipeline if solver.needs_collision_pipeline else None)

    # ----- Authored-state invalidation ----------------------------------------

    def invalidate_fk(
        self,
        env_mask: wp.array | None = None,
        env_ids: wp.array | None = None,
        articulation_ids: wp.array | None = None,
    ) -> None:
        """Flag articulations for forward kinematics and their worlds for a solver reset.

        View rows are mapped to worlds through the model's articulation-to-world table, so views that span a subset
        of worlds flag the right worlds.

        Args:
            env_mask: Mask over view rows.
            env_ids: Selected view rows.
            articulation_ids: Model articulation index of each ``(row, articulation)``; ``None`` flags everything.
        """
        self.kinematics_dirty = True
        model = self.backend.model
        outputs = [self.world_mask, self.fk_mask]
        if articulation_ids is not None and env_mask is not None:
            inputs = [env_mask, articulation_ids, model.articulation_world]
            wp.launch(_mark_reset_from_mask, articulation_ids.shape, inputs, outputs, device=self.schema.device)
        elif articulation_ids is not None and env_ids is not None:
            kernel = _MARK_RESET_FROM_IDS.select(env_ids)
            dim = (env_ids.shape[0], articulation_ids.shape[1])
            inputs = [env_ids, articulation_ids, model.articulation_world]
            wp.launch(kernel, dim, inputs, outputs, device=self.schema.device)
        else:
            self.world_mask[: model.world_count].fill_(True)
            self.fk_mask.fill_(True)

    def invalidate_body_state(self, env_ids: wp.array | None = None, env_mask: wp.array | None = None) -> None:
        """Flag worlds whose maximal-coordinate body state was written, without requesting FK.

        Args:
            env_ids: Selected worlds.
            env_mask: Mask over worlds.
        """
        self.kinematics_dirty = True
        if env_mask is not None:
            wp.launch(
                _mark_worlds_from_mask, env_mask.shape[0], [env_mask], [self.world_mask], device=self.schema.device
            )
        elif env_ids is not None:
            wp.launch(_mark_worlds_from_ids, env_ids.shape[0], [env_ids], [self.world_mask], device=self.schema.device)
        else:
            self.world_mask[: self.schema.world_count].fill_(True)

    def reconcile(self) -> None:
        """Reset solver internals and run FK for the flagged worlds, then clear the flags.

        Runs only when state was authored since the last boundary, or when an outer graph replay may have authored it.
        """
        if not (self.kinematics_dirty or self.transforms_may_change_on_graph_replay):
            return
        if self.solver is None:
            raise RuntimeError(
                "Newton state was authored before the solver was initialized. NewtonManager.initialize_solver() must"
                " run (via reset()) before forward() or step()."
            )
        state = self.backend.state_0
        self.solver.reset(state, self.world_mask)
        self.solver.eval_fk(state, self.world_mask, self.fk_mask)
        self.fk_mask.zero_()
        self.world_mask.zero_()
        self.kinematics_dirty = False

    def apply_model_changes(self) -> None:
        """Notify the solver of model changes authored since the last step."""
        if not self.model_changes:
            return
        with wp.ScopedDevice(self.schema.device):
            for change in self.model_changes:
                warning = self.solver.ignored_model_changes.get(change)
                if warning is not None and change not in self._warned_model_changes:
                    logger.warning(warning)
                    self._warned_model_changes.add(change)
                self.solver.notify_model_changed(change)
        self.model_changes = set()

    # ----- Step program -----------------------------------------------------

    def add_stage(self, stage: StepStage) -> StepStage:
        """Schedule a consumer stage into every subsequent step program.

        Args:
            stage: Stage to add. A stage whose operation and phase match an existing stage is not added again.

        Returns:
            The scheduled stage, for :meth:`remove_stage`.
        """
        for existing in self.stages:
            if existing.fn == stage.fn and existing.phase == stage.phase:
                return existing
        self.stages.append(stage)
        self.invalidate_program()
        return stage

    def remove_stage(self, stage: StepStage) -> None:
        """Remove a stage; removing an absent stage is a no-op.

        Args:
            stage: Stage returned by :meth:`add_stage`.
        """
        if stage in self.stages:
            self.stages.remove(stage)
            self.invalidate_program()

    def invalidate_program(self) -> None:
        """Drop the compiled program and its graphs; the next step compiles and captures again."""
        self.program = None

    def compile(self, steps: int) -> StepProgram:
        """Compile ``steps`` physics steps into one program with every buffer bound.

        Each physics step runs ``collide -> Newton actuators -> CONTROL stages -> substeps``, where every substep runs
        ``SUBSTEP stages -> solver -> clear forces`` and an optional mid-step collision. ``POST_STEP`` stages and
        sensors run once at the end. Double-buffered states alternate at compile time, and each physics step ends in
        :attr:`NewtonBackend.state_0`, so stages and bound consumer views always observe the canonical state.

        Args:
            steps: Physics steps per program.

        Returns:
            The compiled program.
        """
        backend, solver, schema = self.backend, self.solver, self.schema
        state_0, control = backend.state_0, backend.control
        spare = state_0 if solver.single_state else backend.state_1
        pipeline = self.collision_pipeline if solver.needs_collision_pipeline else None
        contacts = self.contacts if pipeline is not None else None
        collide_every = schema.collision_decimation if pipeline is not None else 0
        stages = {phase: [stage for stage in self.stages if stage.phase == phase] for phase in StepPhase}
        ops: list[StepOp] = []

        def emit(fn: Callable[[], None], name: str, graph_safe: bool = True) -> None:
            ops.append(StepOp(fn, graph_safe, name))

        actuator_steps = 0
        for _ in range(steps):
            if solver.prepares_step:
                emit(partial(solver.prepare_step, state_0), "solver.prepare_step")
            if pipeline is not None:
                emit(partial(pipeline.collide, state_0, contacts), "collide")
            if self.actuators is not None:
                self._emit_actuators(emit, state_0, control, schema.physics_dt, actuator_steps % 2)
                actuator_steps += 1
            for stage in stages[StepPhase.CONTROL]:
                emit(stage.fn, stage.name, stage.graph_safe)
            state_in, state_out = state_0, spare
            for substep in range(schema.num_substeps):
                for stage in stages[StepPhase.SUBSTEP]:
                    emit(partial(stage.fn, state_in), stage.name, stage.graph_safe)
                emit(partial(solver.step, state_in, state_out, control, contacts, schema.solver_dt), "solver.step")
                if not solver.single_state:
                    state_in, state_out = state_out, state_in
                emit(state_in.clear_forces, "clear_forces")
                if collide_every > 0 and (substep + 1) % collide_every == 0 and substep + 1 < schema.num_substeps:
                    emit(partial(pipeline.collide, state_in, contacts), "collide")
            if state_in is not state_0:
                emit(partial(state_0.assign, state_in), "state.assign")
        if actuator_steps % 2:
            # Keep actuator history in the canonical buffers so every replay starts from the same addresses.
            for actuator, current, previous in zip(self.actuators.actuators, *self.actuators.state_buffers):
                if current is not None:
                    emit(partial(current.assign, previous), "actuators.assign", actuator.is_graphable())
        for stage in stages[StepPhase.POST_STEP]:
            emit(stage.fn, stage.name, stage.graph_safe)
        if self.sensors.contact or self.sensors.frame_transform or self.sensors.imu:
            emit(partial(self.sensors.update, state_0, self.contacts, solver), "sensors")
        return StepProgram(ops, steps)

    def _emit_actuators(self, emit: Callable[..., None], state: State, control: Any, dt: float, parity: int) -> None:
        """Emit Newton actuator operations reading one history buffer and writing the other.

        Graph-safe actuators join the captured segment; others, such as TorchScript networks, run eagerly in place.
        """
        adapter = self.actuators
        states_a, states_b = adapter.state_buffers
        states_in, states_out = (states_a, states_b) if parity == 0 else (states_b, states_a)
        emit(partial(adapter.zero_outputs, control), "actuators.zero")
        for actuator, state_in, state_out in zip(adapter.actuators, states_in, states_out):
            step = partial(actuator.step, state, control, state_in, state_out, dt=dt)
            emit(step, f"actuator.{type(actuator.controller).__name__}", actuator.is_graphable())
