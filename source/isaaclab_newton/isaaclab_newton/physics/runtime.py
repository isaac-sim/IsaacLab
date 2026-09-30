# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton simulation data: construction requests, world schema, and the mutable runtime."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp
from newton import CollisionPipeline, Contacts, Model, ModelBuilder, State
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
def _mark_worlds_from_mask(
    env_mask: wp.array(dtype=wp.bool), row_worlds: wp.array(dtype=wp.int32), world_mask: wp.array(dtype=wp.bool)
):
    """Flag the worlds of masked view rows for a solver reset without requesting FK."""
    row = wp.tid()
    if env_mask[row]:
        world_mask[row_worlds[row]] = True


@wp.kernel(enable_backward=False)
def _mark_worlds_from_ids(
    env_ids: wp.array(dtype=wp.int32), row_worlds: wp.array(dtype=wp.int32), world_mask: wp.array(dtype=wp.bool)
):
    """Flag the worlds of selected view rows for a solver reset without requesting FK."""
    world_mask[row_worlds[env_ids[wp.tid()]]] = True


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

    @classmethod
    def from_model(
        cls,
        model: Model,
        *,
        physics_dt: float,
        num_substeps: int = 1,
        collision_decimation: int = 0,
        world_prototypes: np.ndarray | None = None,
    ) -> NewtonSchema:
        """Describe a finalized model.

        Args:
            model: Finalized model.
            physics_dt: Duration of one physics step [s].
            num_substeps: Solver substeps per physics step.
            collision_decimation: Substeps between mid-step collision passes, or ``0`` to collide once per step.
            world_prototypes: Clone-plan world prototype of each world, if the model was cloned.

        Returns:
            The schema.
        """
        return cls(
            device=str(model.device),
            world_count=model.world_count,
            world_prototypes=world_prototypes,
            physics_dt=physics_dt,
            num_substeps=num_substeps,
            collision_decimation=collision_decimation,
            body_count=model.body_count,
            joint_dof_count=model.joint_dof_count,
            articulation_count=model.articulation_count,
        )


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


def update_sensors(
    sensors: NewtonSensors, state: State, contacts: Contacts | None, solver: NewtonSolverBinding
) -> None:
    """Push the latest state to every sensor.

    Args:
        sensors: Sensors to update.
        state: Current state.
        contacts: Contacts of the last collision pass.
        solver: Solver that reports its contacts for contact sensors.
    """
    for sensor in sensors.frame_transform:
        sensor.update(state)
    for sensor in sensors.imu:
        sensor.update(state)
    if sensors.contact:
        solver.solver.update_contacts(contacts, state)
        for sensor in sensors.contact.values():
            sensor.update(state, contacts)


def _compile_label_pattern(expr: str | list[str] | None) -> re.Pattern[str] | None:
    """Compile selector expressions for Newton's full label matching."""
    if not expr:
        return None
    return re.compile("|".join((expr,) if isinstance(expr, str) else expr))


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


@dataclass(eq=False)
class NewtonRuntime:
    """Everything bound to one finalized Newton model, as plain data.

    The runtime has no behavior of its own: the functions in this module read and update it explicitly, for example
    ``step(runtime, steps, capture)``. Nothing in them depends on class state, so several runtimes, each bound to its
    own :class:`NewtonBackend`, can coexist. :class:`NewtonManager` drives the single runtime of the active simulation.

    A runtime is created when its model is finalized and discarded on a hard reset or close, so no buffer, solver, or
    consumer stage outlives the model it points into.
    """

    backend: NewtonBackend
    """Borrowed native model, state, and control; the simulation registry owns it."""

    schema: NewtonSchema
    """Immutable description of the finalized worlds."""

    world_mask: wp.array
    """Worlds flagged for a solver reset, shape ``(world_count + 1,)``; the final slot selects global entities."""

    fk_mask: wp.array
    """Articulations flagged for forward kinematics, shape ``(articulation_count,)``."""

    kinematics_dirty: bool = False
    """Whether state was authored since the last reconcile."""

    transforms_may_change_on_graph_replay: bool = False
    """Whether state was written during an outer capture, whose replays bypass Python invalidation."""

    model_changes: set[int] = field(default_factory=set)
    """Model changes to notify the solver of before the next step."""

    warned_model_changes: set[int] = field(default_factory=set)
    """Ignored model changes already reported."""

    stages: list[StepStage] = field(default_factory=list)
    """Consumer stages scheduled into every step program."""

    sensors: NewtonSensors = field(default_factory=NewtonSensors)
    """Newton sensors updated at the end of every step program."""

    actuators: NewtonActuatorAdapter | None = None
    """Newton actuators run inside the step program, once an articulation activates them."""

    newton_actuators_active: bool = False
    """Whether an articulation runs its explicit actuators as Newton actuators inside the step program."""

    host_physics_steps: bool = False
    """Whether a consumer does host work between physics steps (e.g. Isaac Lab actuator models), so the environment
    must drive the decimation loop one physics step at a time."""

    solver: NewtonSolverBinding | None = None
    """Solver binding, set by :func:`bind_solver`."""

    collision_cfg: NewtonCollisionPipelineCfg | None = None
    """Collision pipeline configuration, set by :func:`bind_solver`."""

    collision_pipeline: CollisionPipeline | None = None
    """Newton collision pipeline, when the solver does not collide internally."""

    contacts: Contacts | None = None
    """Contacts filled by the pipeline or reported by the solver."""

    program: StepProgram | None = None
    """Compiled step program; ``None`` until the next step compiles it."""

    applied_forces: dict[str, wp.array] | None = None
    """Snapshot buffers of the force arrays consumers author before a step, by :class:`newton.State` attribute.

    Allocated when the first program is compiled."""

    identity_rows: wp.array | None = None
    """Cached identity row-to-world map for views that span every world."""


def create_runtime(backend: NewtonBackend, schema: NewtonSchema) -> NewtonRuntime:
    """Create the runtime of a finalized model.

    Args:
        backend: Borrowed native model, state, and control.
        schema: World description of the finalized model.

    Returns:
        A runtime with cleared reset masks and no solver.
    """
    model = backend.model
    # Isaac Lab resets local worlds only, so the final (global) world-mask slot stays false.
    return NewtonRuntime(
        backend=backend,
        schema=schema,
        world_mask=wp.zeros(model.world_count + 1, dtype=wp.bool, device=schema.device),
        fk_mask=wp.zeros(model.articulation_count, dtype=wp.bool, device=schema.device),
    )


# ----- Solver ----------------------------------------------------------------


def bind_solver(
    runtime: NewtonRuntime,
    binding_type: type[NewtonSolverBinding],
    cfg: NewtonCfg,
    deterministic_mode: wp.DeterministicMode,
) -> None:
    """Construct the solver and its contacts.

    Args:
        runtime: Runtime to bind.
        binding_type: Solver binding to construct.
        cfg: Newton physics configuration.
        deterministic_mode: Determinism guarantee for the solver and collision pipeline.
    """
    binding_type.validate_cfg(cfg)
    runtime.solver = solver = binding_type(runtime.backend.model, cfg.solver_cfg, deterministic_mode)
    if runtime.sensors.contact and not solver.supports_contact_sensors:
        raise NotImplementedError(
            f"Newton contact sensors are not yet supported by {binding_type.__name__} because its contact forces"
            " live in per-entry buffers. Remove the contact sensor."
        )
    if not solver.single_state:
        solver.initialize_output_state(runtime.backend.state_1)
    runtime.collision_cfg = cfg.collision_cfg
    allocate_contacts(runtime)


def allocate_contacts(runtime: NewtonRuntime) -> None:
    """Allocate contacts, and the collision pipeline when the solver does not collide internally.

    Args:
        runtime: Runtime with a bound solver.
    """
    solver, model = runtime.solver, runtime.backend.model
    invalidate_program(runtime)
    if not solver.needs_collision_pipeline:
        runtime.contacts = solver.create_contacts()
    else:
        collision_cfg = runtime.collision_cfg
        args = collision_cfg.to_pipeline_args() if collision_cfg is not None else {"broad_phase": "explicit"}
        deterministic = solver.deterministic_mode != wp.DeterministicMode.NOT_GUARANTEED
        args["deterministic"] = deterministic
        required = solver.minimum_contact_capacity()
        if runtime.collision_pipeline is None:
            runtime.collision_pipeline = CollisionPipeline(model, **args)
        runtime.contacts = runtime.collision_pipeline.contacts()
        # Solvers such as MuJoCo Warp can require more contacts than the pipeline estimates.
        if required > runtime.contacts.rigid_contact_max:
            if deterministic:
                # The deterministic sort buffer is sized at construction, so rebuild the pipeline to match.
                args["rigid_contact_max"] = required
                runtime.collision_pipeline = CollisionPipeline(model, **args)
                runtime.contacts = runtime.collision_pipeline.contacts()
            else:
                runtime.contacts = Contacts(
                    rigid_contact_max=required,
                    soft_contact_max=0,
                    device=runtime.schema.device,
                    requested_attributes=model.get_requested_contact_attributes(),
                )
    if runtime.contacts is not None:
        pipeline = runtime.collision_pipeline if solver.needs_collision_pipeline else None
        solver.prepare_contacts(runtime.contacts, pipeline)


# ----- Authored-state invalidation -------------------------------------------


def invalidate_fk(
    runtime: NewtonRuntime,
    env_mask: wp.array | None = None,
    env_ids: wp.array | None = None,
    articulation_ids: wp.array | None = None,
) -> None:
    """Flag articulations for forward kinematics and their worlds for a solver reset, without host synchronization.

    View rows map to worlds through the model's articulation-to-world table, so views that span a subset of worlds
    flag the right worlds, and global articulations flag FK only.

    Args:
        runtime: Runtime to flag.
        env_mask: Mask over view rows.
        env_ids: Selected view rows.
        articulation_ids: Model articulation index of each ``(row, articulation)``; ``None`` flags everything.
    """
    runtime.kinematics_dirty = True
    model, device = runtime.backend.model, runtime.schema.device
    outputs = [runtime.world_mask, runtime.fk_mask]
    if articulation_ids is not None and env_mask is not None:
        inputs = [env_mask, articulation_ids, model.articulation_world]
        wp.launch(_mark_reset_from_mask, articulation_ids.shape, inputs, outputs, device=device)
    elif articulation_ids is not None and env_ids is not None:
        dim = (env_ids.shape[0], articulation_ids.shape[1])
        inputs = [env_ids, articulation_ids, model.articulation_world]
        wp.launch(_MARK_RESET_FROM_IDS.select(env_ids), dim, inputs, outputs, device=device)
    else:
        runtime.world_mask[: model.world_count].fill_(True)
        runtime.fk_mask.fill_(True)


def view_row_worlds(runtime: NewtonRuntime, articulation_ids: wp.array) -> wp.array:
    """Return the world of each row of an articulation view.

    Rows equal worlds only while every view spans all worlds; heterogeneous scenes map them through the model.

    Args:
        runtime: Runtime of the view's model.
        articulation_ids: Model articulation index of each ``(row, articulation)``.

    Returns:
        World index of each row, shape ``(num_rows,)``; ``-1`` for global articulations.
    """
    worlds = runtime.backend.model.articulation_world.numpy()[articulation_ids.numpy()[:, 0]]
    return wp.array(worlds.astype(np.int32), dtype=wp.int32, device=runtime.schema.device)


def invalidate_body_state(
    runtime: NewtonRuntime,
    env_ids: wp.array | None = None,
    env_mask: wp.array | None = None,
    row_worlds: wp.array | None = None,
) -> None:
    """Flag worlds whose maximal-coordinate body state was written, without requesting FK.

    Args:
        runtime: Runtime to flag.
        env_ids: Selected view rows.
        env_mask: Mask over view rows.
        row_worlds: World of each view row (see :func:`view_row_worlds`); ``None`` when rows are worlds.
    """
    runtime.kinematics_dirty = True
    device = runtime.schema.device
    selection = env_mask if env_mask is not None else env_ids
    if selection is None:
        runtime.world_mask[: runtime.schema.world_count].fill_(True)
        return
    if row_worlds is None:
        row_worlds = _identity_rows(runtime)
    kernel = _mark_worlds_from_mask if env_mask is not None else _mark_worlds_from_ids
    wp.launch(kernel, selection.shape[0], [selection, row_worlds], [runtime.world_mask], device=device)


def _identity_rows(runtime: NewtonRuntime) -> wp.array:
    """Return the cached identity row-to-world map of views that span every world."""
    rows = runtime.identity_rows
    if rows is None:
        rows = runtime.identity_rows = wp.array(
            np.arange(runtime.schema.world_count, dtype=np.int32), dtype=wp.int32, device=runtime.schema.device
        )
    return rows


def reconcile(runtime: NewtonRuntime) -> None:
    """Reset solver internals and run FK for the flagged worlds, then clear the flags.

    Runs only when state was authored since the last boundary, or when an outer graph replay may have authored it.

    Args:
        runtime: Runtime to reconcile.
    """
    if not (runtime.kinematics_dirty or runtime.transforms_may_change_on_graph_replay):
        return
    solver = runtime.solver
    if solver is None:
        raise RuntimeError(
            "Newton state was authored before the solver was initialized. NewtonManager.initialize_solver() must"
            " run (via reset()) before forward() or step()."
        )
    state = runtime.backend.state_0
    solver.reset(state, runtime.world_mask)
    solver.eval_fk(state, runtime.world_mask, runtime.fk_mask)
    runtime.fk_mask.zero_()
    runtime.world_mask.zero_()
    runtime.kinematics_dirty = False


def apply_model_changes(runtime: NewtonRuntime) -> None:
    """Notify the solver of model changes authored since the last step.

    Args:
        runtime: Runtime whose solver to notify.
    """
    if not runtime.model_changes:
        return
    solver = runtime.solver
    with wp.ScopedDevice(runtime.schema.device):
        for change in runtime.model_changes:
            warning = solver.ignored_model_changes.get(change)
            if warning is not None and change not in runtime.warned_model_changes:
                logger.warning(warning)
                runtime.warned_model_changes.add(change)
            solver.notify_model_changed(change)
    runtime.model_changes = set()


# ----- Consumers -------------------------------------------------------------


def add_stage(runtime: NewtonRuntime, stage: StepStage) -> StepStage:
    """Schedule a consumer stage into every subsequent step program.

    Args:
        runtime: Runtime to extend.
        stage: Stage to add. A stage whose operation and phase match an existing stage is not added again.

    Returns:
        The scheduled stage, for :func:`remove_stage`.
    """
    for existing in runtime.stages:
        if existing.fn == stage.fn and existing.phase == stage.phase:
            return existing
    runtime.stages.append(stage)
    invalidate_program(runtime)
    return stage


def remove_stage(runtime: NewtonRuntime, stage: StepStage) -> None:
    """Remove a stage; removing an absent stage is a no-op.

    Args:
        runtime: Runtime to update.
        stage: Stage returned by :func:`add_stage`.
    """
    if stage in runtime.stages:
        runtime.stages.remove(stage)
        invalidate_program(runtime)


def owns_decimation(runtime: NewtonRuntime, fold: bool | None) -> bool:
    """Return whether one step advances the whole decimation loop.

    The program folds the loop unless a consumer needs host work between physics steps. Without an explicit request
    from the environment, it folds only when Newton actuators run inside the program.

    Args:
        runtime: Runtime to query.
        fold: The environment's folding request; see :meth:`~isaaclab.physics.PhysicsManager.set_decimation`.
    """
    if runtime.host_physics_steps or fold is False:
        return False
    return fold is True or runtime.newton_actuators_active


def activate_actuators(runtime: NewtonRuntime) -> None:
    """Run the model's Newton actuators inside the step program.

    Idempotent. The adapter addresses the model's flat DOF space, and articulations bind their own views of it, so
    worlds may hold different DOF layouts.

    Args:
        runtime: Runtime to extend.
    """
    runtime.newton_actuators_active = True
    invalidate_program(runtime)
    model = runtime.backend.model
    if runtime.actuators is not None or not model.actuators:
        return
    from isaaclab.actuators.newton import NewtonActuatorAdapter  # noqa: PLC0415

    num_envs = model.num_envs
    runtime.actuators = NewtonActuatorAdapter(
        actuators=list(model.actuators),
        num_envs=num_envs,
        num_joints=model.joint_dof_count // num_envs,
        dof_offset=0,
        device=runtime.schema.device,
        dof_count=model.joint_dof_count,
    )
    runtime.actuators.finalize(runtime.backend.control)


def add_contact_sensor(
    runtime: NewtonRuntime,
    body_names_expr: str | list[str] | None = None,
    shape_names_expr: str | list[str] | None = None,
    contact_partners_body_expr: str | list[str] | None = None,
    contact_partners_shape_expr: str | list[str] | None = None,
    verbose: bool = False,
) -> SensorContact:
    """Add a contact sensor between bodies or shapes; identical requests share one sensor.

    Args:
        runtime: Runtime to extend.
        body_names_expr: Expression for body names to sense.
        shape_names_expr: Expression for shape names to sense.
        contact_partners_body_expr: Expression for contact partner body names.
        contact_partners_shape_expr: Expression for contact partner shape names.
        verbose: Print verbose information.

    Returns:
        The Newton contact sensor updated at the end of every step.
    """
    if runtime.solver is not None and not runtime.solver.supports_contact_sensors:
        raise NotImplementedError(
            "Newton contact sensors are not yet supported by the active coupled solver because its "
            "contact forces live in per-entry buffers."
        )
    if body_names_expr is None and shape_names_expr is None:
        raise ValueError("At least one of body_names_expr or shape_names_expr must be provided")
    if body_names_expr is not None and shape_names_expr is not None:
        raise ValueError("Only one of body_names_expr or shape_names_expr must be provided")
    if contact_partners_body_expr is not None and contact_partners_shape_expr is not None:
        raise ValueError("Only one of contact_partners_body_expr or contact_partners_shape_expr must be provided")

    exprs = (body_names_expr, shape_names_expr, contact_partners_body_expr, contact_partners_shape_expr)
    key = tuple(tuple(expr) if isinstance(expr, list) else expr for expr in exprs)
    sensor = runtime.sensors.contact.get(key)
    if sensor is not None:
        return sensor
    partner_filter = contact_partners_body_expr or contact_partners_shape_expr or "all bodies/shapes"
    logger.info(f"Adding contact sensor for {body_names_expr or shape_names_expr} with filter {partner_filter}")
    sensor = SensorContact(
        runtime.backend.model,
        sensing_bodies=_compile_label_pattern(body_names_expr),
        sensing_shapes=_compile_label_pattern(shape_names_expr),
        counterpart_bodies=_compile_label_pattern(contact_partners_body_expr),
        counterpart_shapes=_compile_label_pattern(contact_partners_shape_expr),
        measure_total=True,
        verbose=verbose,
    )
    runtime.sensors.contact[key] = sensor
    invalidate_program(runtime)
    # Contacts allocated before the sensor requested the force attribute must be reallocated.
    if runtime.contacts is not None and runtime.contacts.force is None:
        allocate_contacts(runtime)
    return sensor


def add_frame_transform_sensor(
    runtime: NewtonRuntime, shapes: list[int], reference_sites: list[int]
) -> SensorFrameTransform:
    """Add a frame transform sensor measuring shapes relative to reference sites.

    Args:
        runtime: Runtime to extend.
        shapes: Ordered shape indices to measure.
        reference_sites: Reference site index of each shape.

    Returns:
        The Newton sensor updated at the end of every step.
    """
    sensor = SensorFrameTransform(runtime.backend.model, shapes=shapes, reference_sites=reference_sites)
    runtime.sensors.frame_transform.append(sensor)
    invalidate_program(runtime)
    return sensor


def add_imu_sensor(runtime: NewtonRuntime, sites: list[int]) -> SensorIMU:
    """Add an IMU sensor at sites; the ``body_qdd`` state attribute must be requested before finalization.

    Args:
        runtime: Runtime to extend.
        sites: Site index of each environment.

    Returns:
        The Newton sensor updated at the end of every step.
    """
    sensor = SensorIMU(runtime.backend.model, sites=sites, request_state_attributes=False)
    runtime.sensors.imu.append(sensor)
    invalidate_program(runtime)
    return sensor


# ----- Step program ----------------------------------------------------------


def invalidate_program(runtime: NewtonRuntime) -> None:
    """Drop the compiled program and its graphs; the next step compiles and captures again.

    Args:
        runtime: Runtime to update.
    """
    runtime.program = None


def compile_program(runtime: NewtonRuntime, steps: int) -> StepProgram:
    """Compile ``steps`` physics steps into one program with every buffer bound.

    Each physics step runs ``collide -> COMMAND stages -> Newton actuators -> CONTROL stages -> substeps``, where every
    substep runs
    ``apply authored forces -> SUBSTEP stages -> solver`` and an optional mid-step collision. ``POST_STEP`` stages and
    sensors run once at the end. Double-buffered states alternate at compile time, and each physics step ends in
    :attr:`NewtonBackend.state_0`, so stages and bound consumer views always observe the canonical state.

    Forces written to ``state_0`` before the step (external wrenches, particle forces) are snapshotted once and
    re-applied before every substep of every physics step, then cleared after the program, so each write applies to
    exactly the next step regardless of substeps or who owns the decimation loop.

    Args:
        runtime: Runtime with a bound solver.
        steps: Physics steps per program.

    Returns:
        The compiled program.
    """
    backend, solver, schema = runtime.backend, runtime.solver, runtime.schema
    state_0, control = backend.state_0, backend.control
    spare = state_0 if solver.single_state else backend.state_1
    pipeline = runtime.collision_pipeline if solver.needs_collision_pipeline else None
    contacts = runtime.contacts if pipeline is not None else None
    collide_every = schema.collision_decimation if pipeline is not None else 0
    stages = {phase: [stage for stage in runtime.stages if stage.phase == phase] for phase in StepPhase}
    ops: list[StepOp] = []

    def emit(fn: Callable[[], None], name: str, graph_safe: bool = True) -> None:
        ops.append(StepOp(fn, graph_safe, name))

    if runtime.applied_forces is None:
        counts = (("body_f", state_0.body_count), ("particle_f", state_0.particle_count))
        runtime.applied_forces = {name: wp.empty_like(getattr(state_0, name)) for name, count in counts if count}
    applied = list(runtime.applied_forces.items())
    for name, buffer in applied:
        emit(partial(wp.copy, buffer, getattr(state_0, name)), f"forces.snapshot.{name}")
    actuator_steps = 0
    for _ in range(steps):
        if solver.prepares_step:
            emit(partial(solver.prepare_step, state_0), "solver.prepare_step")
        if pipeline is not None:
            emit(partial(pipeline.collide, state_0, contacts), "collide")
        for stage in stages[StepPhase.COMMAND]:
            emit(stage.fn, stage.name, stage.graph_safe)
        if runtime.actuators is not None:
            _emit_actuators(emit, runtime.actuators, state_0, control, schema.physics_dt, actuator_steps % 2)
            actuator_steps += 1
        for stage in stages[StepPhase.CONTROL]:
            emit(stage.fn, stage.name, stage.graph_safe)
        state_in, state_out = state_0, spare
        for substep in range(schema.num_substeps):
            for name, buffer in applied:
                emit(partial(wp.copy, getattr(state_in, name), buffer), f"forces.apply.{name}")
            for stage in stages[StepPhase.SUBSTEP]:
                emit(partial(stage.fn, state_in), stage.name, stage.graph_safe)
            emit(partial(solver.step, state_in, state_out, control, contacts, schema.solver_dt), "solver.step")
            if not solver.single_state:
                state_in, state_out = state_out, state_in
            if collide_every > 0 and (substep + 1) % collide_every == 0 and substep + 1 < schema.num_substeps:
                emit(partial(pipeline.collide, state_in, contacts), "collide")
        if state_in is not state_0:
            emit(partial(state_0.assign, state_in), "state.assign")
    if actuator_steps % 2:
        # Keep actuator history in the canonical buffers so every replay starts from the same addresses.
        adapter = runtime.actuators
        for actuator, current, previous in zip(adapter.actuators, *adapter.state_buffers):
            if current is not None:
                emit(partial(current.assign, previous), "actuators.assign", actuator.is_graphable())
    for stage in stages[StepPhase.POST_STEP]:
        emit(stage.fn, stage.name, stage.graph_safe)
    sensors = runtime.sensors
    if sensors.contact or sensors.frame_transform or sensors.imu:
        emit(partial(update_sensors, sensors, state_0, runtime.contacts, solver), "sensors")
    # Forces apply to one step; consumers author them again before the next.
    emit(state_0.clear_forces, "clear_forces")
    return StepProgram(ops, steps)


def _emit_actuators(
    emit: Callable[..., None], adapter: NewtonActuatorAdapter, state: State, control: Any, dt: float, parity: int
) -> None:
    """Emit Newton actuator operations reading one history buffer and writing the other.

    Graph-safe actuators join the captured segment; others, such as TorchScript networks, run eagerly in place.
    """
    states_a, states_b = adapter.state_buffers
    states_in, states_out = (states_a, states_b) if parity == 0 else (states_b, states_a)
    emit(partial(adapter.zero_outputs, control), "actuators.zero")
    for actuator, state_in, state_out in zip(adapter.actuators, states_in, states_out):
        step_actuator = partial(actuator.step, state, control, state_in, state_out, dt=dt)
        emit(step_actuator, f"actuator.{type(actuator.controller).__name__}", actuator.is_graphable())


def prepare(
    runtime: NewtonRuntime, steps: int, capture: Callable[[Callable[[], None]], wp.Graph] | None = None
) -> StepProgram:
    """Compile, and capture, the program for ``steps`` physics steps unless the current one already matches.

    Compilation and capture do all host work of a step ahead of time and never advance physics, so callers such as a
    task compiler can prepare before their own capture.

    Args:
        runtime: Runtime with a bound solver.
        steps: Physics steps per program.
        capture: Records a callable into a CUDA graph, or ``None`` to keep the program eager.

    Returns:
        The prepared program.
    """
    program = runtime.program
    if program is None or program.steps != steps:
        program = runtime.program = compile_program(runtime, steps)
        if capture is not None and runtime.solver.supports_graph_capture:
            program.capture(capture)
    return program


def step(
    runtime: NewtonRuntime, steps: int, capture: Callable[[Callable[[], None]], wp.Graph] | None = None
) -> StepProgram:
    """Advance ``steps`` physics steps.

    Notifies model changes, reconciles authored state, prepares the program (see :func:`prepare`), and runs it. Only
    preparation does host work beyond launching; a caller that owns an outer capture passes ``capture=None`` and
    records this call.

    Args:
        runtime: Runtime with a bound solver.
        steps: Physics steps to advance.
        capture: Records a callable into a CUDA graph, or ``None`` to run eagerly.

    Returns:
        The program that ran.
    """
    apply_model_changes(runtime)
    reconcile(runtime)
    program = prepare(runtime, steps, capture)
    if program.is_captured and program.graph_safe:
        # Graph replays carry their device; skip the host-side device scope.
        program.run()
    else:
        with wp.ScopedDevice(runtime.schema.device):
            program.run()
    return program
