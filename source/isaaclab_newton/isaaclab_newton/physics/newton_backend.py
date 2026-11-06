# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton simulation data bound to one finalized model, and how it is stepped."""

from __future__ import annotations

import contextlib
import gc
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache, partial
from typing import TYPE_CHECKING

import torch
import warp as wp
from newton import CollisionPipeline, Contacts, ModelBuilder, State
from newton.sensors import SensorContact, SensorIMU
from newton.solvers import SolverBase

from isaaclab.sim import SimulationContext
from isaaclab.utils import instantiate
from isaaclab.utils.buffers import TimestampedBuffer
from isaaclab.utils.timer import Timer
from isaaclab.utils.version import has_kit

from isaaclab_newton.renderers.visual_material import VisualMaterialWriter

from .newton_builder_cfg import NewtonBuilderCfg
from .newton_manager_cfg import NewtonCfg

if TYPE_CHECKING:
    from isaaclab.actuators.newton import NewtonActuatorAdapter
    from isaaclab.renderers.base_renderer import VisualMaterialBatch

    from .newton_backend_cfg import NewtonBackendCfg
    from .newton_solver import NewtonSolver

logger = logging.getLogger(__name__)

SiteEntry = tuple[int, None] | tuple[None, list[list[int]]]
"""Resolved site: ``(global_shape_index, None)`` for a global site, or ``(None, per_world_indices)``."""


@cache
def _make_flag_worlds_and_articulations(num_articulations: int, masked: bool) -> wp.Kernel:
    """Specialize the flag kernel for a view's articulation count and whether an articulation mask selects them."""

    @wp.kernel(enable_backward=False, module="unique")
    def flag_worlds_and_articulations(
        env_mask: wp.array(dtype=wp.bool),
        articulation_ids: wp.array2d(dtype=wp.int32),
        articulation_world: wp.array(dtype=wp.int32),
        articulation_mask: wp.array(dtype=wp.bool),
        consume_mask: bool,
        world_mask: wp.array(dtype=wp.bool),
        fk_mask: wp.array(dtype=wp.bool),
    ):
        """Flag the masked envs' articulations for FK and their worlds for a solver reset.

        Global articulations (world ``-1``) get FK without resetting the global world. Optionally consumes the env mask.
        """
        env = wp.tid()
        if not env_mask[env]:
            return
        for view_articulation in range(wp.static(num_articulations)):
            if wp.static(masked):
                if not articulation_mask[view_articulation]:
                    continue
            articulation = articulation_ids[env, view_articulation]
            fk_mask[articulation] = True
            world = articulation_world[articulation]
            if world >= 0:
                world_mask[world] = True
        if consume_mask:
            env_mask[env] = False

    return flag_worlds_and_articulations


@wp.kernel(enable_backward=False)
def _flag_worlds(env_mask: wp.array(dtype=wp.bool), consume_mask: bool, world_mask: wp.array(dtype=wp.bool)):
    """Flag reset work for the masked worlds without requesting FK; optionally consume the env mask."""
    world = wp.tid()
    if not env_mask[world]:
        return
    world_mask[world] = True
    if consume_mask:
        env_mask[world] = False


@dataclass
class NewtonCloneRecord:
    """Native replication outputs, kept across hard resets."""

    world_xforms: list[wp.transform] | None
    """Root transform of each cloned world, or ``None`` without replication."""
    source_builders: dict[str, ModelBuilder]
    """Per-source builders, keyed by clone-plan source path."""
    particle_ranges: dict[str, tuple[int, int]]
    """``(start, count)`` particle range per replicated deformable prim path."""
    cable_bindings: dict[str, list[int]]
    """Body indices per replicated cable prim path."""
    geometry_batches: list
    """Shape batches the scene data backend publishes as geometry."""


class NewtonBackend:
    """Everything bound to one finalized Newton model, and the operations that initialize, step and query it.

    A render-only backend (non-Newton physics) holds only the model and :attr:`state_0`.
    """

    def __init__(self, cfg: NewtonBackendCfg, builder: ModelBuilder | None = None, *, dt: float | None = None):
        """Finalize a builder and allocate the buffers bound to its model.

        Args:
            cfg: Physics settings and device. :class:`NewtonCfg` physics makes this a simulation backend; any other
                physics makes it render-only.
            builder: Builder to finalize. Defaults to the simulation's shared builder for ``cfg.physics_cfg``.
            dt: Duration of one physics step [s]. Required for a simulation backend.
        """
        if builder is None:
            builder = SimulationContext.instance().get_or_create_backend(NewtonBuilderCfg(physics_cfg=cfg.physics_cfg))
        self.model = builder.finalize(device=cfg.device)
        """Finalized Newton model; :attr:`model.num_envs` is its world count."""
        self.model.num_envs = self.model.world_count
        self.device: wp.Device = self.model.device
        self.state_version = 0
        """Advanced whenever the step or FK changes the state; asset data recomputes derived values cached before it."""
        self.particle_ranges: dict[str, tuple[int, int]] = {}
        """``(start, count)`` particle range per replicated deformable prim path."""
        self.deformable_ranges = {
            label: (start, end - start, kind)
            for kind, labels, starts, ends in (
                ("surface", builder.surface_label, builder._surface_particle_start, builder._surface_particle_end),
                ("volume", builder.volume_label, builder._volume_particle_start, builder._volume_particle_end),
            )
            for label, start, end in zip(labels, starts, ends, strict=True)
        }
        """``(start, count, kind)`` particle range per deformable label; ``kind`` is ``"surface"`` or ``"volume"``."""
        physics_cfg = cfg.physics_cfg
        simulation = isinstance(physics_cfg, NewtonCfg)
        soft_contact = physics_cfg.soft_contact_cfg if simulation else None
        if soft_contact is not None:
            self.model.soft_contact_ke = float(soft_contact.soft_contact_ke)
            self.model.soft_contact_kd = float(soft_contact.soft_contact_kd)
            self.model.soft_contact_mu = float(soft_contact.soft_contact_mu)
        self.state_0: State = self.model.state()
        """Current state; the solver reads it and, for double-buffered solvers, writes :attr:`state_1`."""
        self.geometry_offsets = (
            {} if simulation else {path: bounds[0] for path, bounds in self.deformable_ranges.items()}
        )
        """First particle index per deformable label on a render-only backend; empty on a simulation backend."""
        self.bvh_refit = TimestampedBuffer()
        """Timestamp of the last BVH refit, shared by ray casters and cameras."""
        self.cfg: NewtonCfg | None = physics_cfg if simulation else None
        """Newton physics settings, or ``None`` for a render-only backend."""
        if not simulation:
            self.state_1 = self.control = None
            return
        if dt is None:
            raise ValueError("A simulation NewtonBackend requires the physics step duration dt.")

        self.dt = dt
        """Duration of one physics step [s]."""
        self.num_substeps = physics_cfg.num_substeps
        """Solver substeps per physics step."""
        self.solver_dt = dt / self.num_substeps
        """Duration of one solver substep [s]."""
        self.deterministic_mode = _resolve_deterministic_mode(physics_cfg)
        """Warp determinism mode of the solver and collision pipeline."""
        self.state_1: State = self.model.state()
        """Next state written by the solver."""
        self.control = self.model.control()
        """Joint targets and applied forces read by the solver."""
        self.solver: SolverBase | None = None
        """Newton solver, built by :meth:`init_solver`."""
        self.solver_ops: NewtonSolver | None = None
        """Isaac Lab driver of :attr:`solver`, built from :attr:`NewtonCfg.solver_cfg` by :meth:`init_solver`."""
        self.collision_pipeline: CollisionPipeline | None = None
        """Collision pipeline, or ``None`` when the solver collides internally."""
        self.contacts: Contacts | None = None
        """Contacts produced by the collision pipeline or the solver."""
        self.contact_observables: SolverBase.Observables | None = None
        """Solver-owned contact forces when registered sensors require them."""
        self.staged_body_f: wp.array | None = None
        """Body forces [N, N·m] written before the step, re-applied before every substep."""
        self.world_mask = wp.zeros(self.model.world_count + 1, dtype=wp.bool, device=self.device)
        """Worlds to reset; the final slot is Newton's global world, which Isaac Lab never resets."""
        self.fk_mask = wp.zeros(self.model.articulation_count, dtype=wp.bool, device=self.device)
        """Articulations whose forward kinematics must be evaluated."""
        self.kinematics_dirty = False
        """Whether joint coordinates changed since the last forward kinematics."""
        self.transforms_may_change_on_graph_replay = False
        """Whether state was written inside an outer CUDA graph capture; :meth:`forward` then always runs."""
        self.model_changes: set[int] = set()
        """Model-property categories to notify the solver of at the next step."""
        self.warned_model_changes: set[int] = set()
        """Ignored model-property categories already warned about."""
        self.contact_sensors: dict[tuple, SensorContact] = {}
        """Newton contact sensors keyed by their request."""
        self.imu_sensors: list[SensorIMU] = []
        """Newton IMU sensors, updated after each physics step."""
        self.actuators: NewtonActuatorAdapter | None = None
        """Newton actuators run inside the step, or ``None``."""
        self.actuators_active = False
        """Whether an articulation runs its Newton actuators in the step (:meth:`activate_actuators`)."""
        self.pre_actuator_callbacks: list[Callable[[], None]] = []
        """Run before the Newton actuators of every physics step."""
        self.post_actuator_callbacks: list[Callable[[], None]] = []
        """Run after the Newton actuators of every physics step, before its solver substeps."""
        self.state_force_callbacks: list[Callable[[State], None]] = []
        """Add forces [N, N·m] to the input state of every solver substep."""
        self.post_step_callbacks: list[Callable[[], None]] = []
        """Run once after the last solver substep of a :meth:`step`."""
        self.solver_init_callbacks: list[Callable[[], None]] = []
        """Run once by :meth:`init_solver`, after the solver exists and before the initial FK."""
        self.decimation = 1
        """Physics steps per environment step."""
        self.capture: Callable[[Callable[[], None]], wp.Graph] | None = None
        """Records the step as a CUDA graph, or ``None`` when the step runs eagerly; set by :meth:`init_solver`."""
        self.graphs: dict[str | tuple[str, int], wp.Graph] = {}
        """Captured parts of the step, keyed by part; empty until the first step is captured."""

    def create_visual_material_writer(self, batches: tuple[VisualMaterialBatch, ...]) -> VisualMaterialWriter:
        """Bind material writes to this resource's native shape-color buffer."""
        return VisualMaterialWriter(self.model, batches)

    def close(self) -> None:
        """Drop native handles after consumers release their bindings."""
        self.model = self.state_0 = self.state_1 = self.control = None
        self.particle_ranges, self.deformable_ranges, self.geometry_offsets = {}, {}, {}
        self.bvh_refit = TimestampedBuffer()
        if self.cfg is not None:
            self.solver = self.solver_ops = self.collision_pipeline = self.contacts = self.contact_observables = None
            self.actuators = None
            self.staged_body_f = self.capture = None
            self.graphs = {}
            self.contact_sensors, self.imu_sensors = {}, []
            self.pre_actuator_callbacks, self.post_actuator_callbacks = [], []
            self.state_force_callbacks, self.post_step_callbacks, self.solver_init_callbacks = [], [], []
            self.world_mask = self.fk_mask = None

    def init_solver(
        self, env_template: str | None = None, *, relaxed_capture: Callable[[], bool] | None = None
    ) -> None:
        """Build the solver and its contacts, run the solver-init callbacks, and evaluate the initial FK.

        Args:
            env_template: Environment prim path template that expands ``{ENV_REGEX_NS}`` in solver selectors.
            relaxed_capture: Returns, when the step is captured, whether to capture in relaxed mode; see
                :func:`capture_graph`. Defaults to strict capture.
        """
        solver_cfg, mode = self.cfg.solver_cfg, self.deterministic_mode
        if mode != wp.DeterministicMode.NOT_GUARANTEED and not solver_cfg.class_type.deterministic:
            raise ValueError(
                f"Newton deterministic mode {mode.name} is not supported by {type(solver_cfg).__name__}."
                " Use MJWarp on the GPU, XPBD, or Featherstone, or disable deterministic mode."
            )
        ops = self.solver_ops = instantiate(solver_cfg, self.model, mode, env_template)
        solver = self.solver = ops.native
        if not ops.single_state:
            ops.init_state(self.state_1)
        self._allocate_contacts()
        self.contact_observables = None
        if self.contact_sensors:
            self.contact_observables = solver.observables(SensorContact.solver_observable_kinds, contacts=self.contacts)
        state_0 = self.state_0
        self.staged_body_f = wp.zeros_like(state_0.body_f) if state_0.body_count else None
        callbacks, self.solver_init_callbacks = self.solver_init_callbacks, []
        for callback in callbacks:
            callback()
        ops.eval_fk(state_0, None, None)
        self.capture = None
        if self.cfg.use_cuda_graph and self.device.is_cuda:
            if ops.supports_capture():
                self.capture = lambda fn: capture_graph(
                    self.device, fn, relaxed=relaxed_capture is not None and relaxed_capture()
                )
            else:
                logger.warning(
                    "%s cannot be captured in its current configuration; using eager execution.", type(solver).__name__
                )
        self.invalidate_graph()

    def invalidate_fk(
        self,
        env_mask: wp.array | None = None,
        articulation_ids: wp.array | None = None,
        *,
        articulation_mask: wp.array | None = None,
        consume_mask: bool = False,
    ) -> None:
        """Schedule solver reset and FK for the masked worlds and articulations.

        Args:
            env_mask: Mask over envs, which map to worlds through the model's articulation-to-world table.
            articulation_ids: Model articulation index of each ``(env, view-local articulation)``, from
                ``ArticulationView.articulation_ids``. With either argument None, every world and articulation is
                flagged, as for native state writes.
            articulation_mask: View-local articulations to flag. None selects every articulation of the view.
            consume_mask: Clear the flagged env_mask entries after reading them. Caller masks are otherwise unchanged.
        """
        self.kinematics_dirty = True
        if env_mask is None or articulation_ids is None:
            self.world_mask[: self.model.world_count].fill_(True)
            self.fk_mask.fill_(True)
            return
        wp.launch(
            _make_flag_worlds_and_articulations(articulation_ids.shape[1], articulation_mask is not None),
            dim=articulation_ids.shape[0],
            inputs=[env_mask, articulation_ids, self.model.articulation_world, articulation_mask, consume_mask],
            outputs=[self.world_mask, self.fk_mask],
            device=self.device,
        )

    def invalidate_body_state(self, env_mask: wp.array | None = None, *, consume_mask: bool = False) -> None:
        """Schedule solver reset for directly authored body state, without requesting FK.

        Args:
            env_mask: Environments to flag, shaped (world_count,). None selects every world.
            consume_mask: Clear the flagged env_mask entries after reading them.
        """
        self.kinematics_dirty = True
        if env_mask is None:
            self.world_mask[: self.model.world_count].fill_(True)
            return
        wp.launch(
            _flag_worlds,
            dim=env_mask.shape[0],
            inputs=[env_mask, consume_mask],
            outputs=[self.world_mask],
            device=self.device,
        )

    def forward(self) -> None:
        """Reset solver internals and evaluate FK for the flagged worlds and articulations, then clear the flags."""
        self.notify_model_changes()
        if not (self.kinematics_dirty or self.transforms_may_change_on_graph_replay):
            return
        ops, state = self.solver_ops, self.state_0
        if ops is None:
            raise RuntimeError("Newton state was authored before the solver was initialized (init_solver).")
        ops.reset(state, self.world_mask)
        ops.eval_fk(state, self.world_mask, self.fk_mask)
        self.fk_mask.zero_()
        self.world_mask.zero_()
        self.kinematics_dirty = False
        self.state_version += 1

    def notify_model_changes(self) -> None:
        """Notify the solver of the queued model-property changes once, combining their categories.

        Inside an enclosing CUDA graph capture, the refresh is recorded and runs on every replay.
        """
        if not self.model_changes:
            return
        ignored = self.solver_ops.ignored_model_changes
        flags = 0
        for change in self.model_changes:
            if change in ignored and change not in self.warned_model_changes:
                logger.warning(ignored[change])
                self.warned_model_changes.add(change)
            flags |= change
        with wp.ScopedDevice(self.device):
            self.solver.notify_model_changed(flags)
        self.model_changes.clear()

    def activate_actuators(self) -> None:
        """Run the model's Newton actuators inside the step. Idempotent."""
        self.actuators_active = True
        model = self.model
        if self.actuators is not None or not model.actuators:
            return
        from isaaclab.actuators.newton import NewtonActuatorAdapter  # noqa: PLC0415

        num_envs = model.world_count
        self.actuators = NewtonActuatorAdapter(
            actuators=list(model.actuators),
            num_envs=num_envs,
            num_joints=model.joint_dof_count // num_envs,
            dof_offset=0,
            device=str(self.device),
        )
        self.actuators.finalize(self.control)
        self.invalidate_graph()

    def add_contact_sensor(
        self,
        body_names_expr: str | list[str] | None = None,
        shape_names_expr: str | list[str] | None = None,
        contact_partners_body_expr: str | list[str] | None = None,
        contact_partners_shape_expr: str | list[str] | None = None,
        verbose: bool = False,
    ) -> SensorContact:
        """Add a contact sensor between bodies or shapes; identical requests share one sensor.

        Args:
            body_names_expr: Expression for body names to sense.
            shape_names_expr: Expression for shape names to sense.
            contact_partners_body_expr: Expression for contact partner body names.
            contact_partners_shape_expr: Expression for contact partner shape names.
            verbose: Print verbose information.

        Returns:
            The Newton contact sensor, updated at the end of every step.
        """
        solver_cfg = self.cfg.solver_cfg
        if not solver_cfg.class_type.contact_sensors:
            raise NotImplementedError(
                f"Newton contact sensors require CONTACT_F observables, which {type(solver_cfg).__name__}"
                " does not support."
            )
        if (body_names_expr is None) == (shape_names_expr is None):
            raise ValueError("Exactly one of body_names_expr or shape_names_expr must be provided.")
        if contact_partners_body_expr is not None and contact_partners_shape_expr is not None:
            raise ValueError("Only one of contact_partners_body_expr or contact_partners_shape_expr must be provided.")
        exprs = (body_names_expr, shape_names_expr, contact_partners_body_expr, contact_partners_shape_expr)
        key = tuple(tuple(expr) if isinstance(expr, list) else expr for expr in exprs)
        sensor = self.contact_sensors.get(key)
        if sensor is not None:
            return sensor
        with Timer(name="newton_contact_sensor", msg="Contact sensor construction took:", synchronize="both"):
            sensor = SensorContact(
                self.model,
                sensing_bodies=_compile_label_pattern(body_names_expr),
                sensing_shapes=_compile_label_pattern(shape_names_expr),
                counterpart_bodies=_compile_label_pattern(contact_partners_body_expr),
                counterpart_shapes=_compile_label_pattern(contact_partners_shape_expr),
                measure_total=True,
                verbose=verbose,
                request_contact_attributes=False,
            )
        self.contact_sensors[key] = sensor
        if self.solver is not None:
            if self.contacts is None:
                self._allocate_contacts()
            if self.contact_observables is None:
                self.contact_observables = self.solver.observables(
                    SensorContact.solver_observable_kinds, contacts=self.contacts
                )
            self.invalidate_graph()
        return sensor

    def add_imu_sensor(self, sites: list[int]) -> SensorIMU:
        """Add an IMU sensor at sites; the ``body_qdd`` state attribute must be requested before finalization.

        Args:
            sites: Site index of each environment.

        Returns:
            The Newton sensor, updated at the end of every step.
        """
        sensor = SensorIMU(self.model, sites=sites, request_state_attributes=False)
        self.imu_sensors.append(sensor)
        return sensor

    def handles_decimation(self) -> bool:
        """Whether one :meth:`step` runs the whole decimation loop: once an articulation runs Newton actuators in it."""
        return self.actuators_active

    def set_decimation(self, decimation: int) -> None:
        """Set the physics steps per environment step."""
        self.decimation = max(1, decimation)

    def invalidate_graph(self) -> None:
        """Drop the captured step; with capture enabled, the next step runs eagerly and is captured again."""
        self.graphs = {}

    def step(self) -> bool:
        """Advance one physics step, or the whole decimation loop when :meth:`handles_decimation`.

        Each physics step runs :meth:`_before_actuators`, the Newton actuators, then :meth:`_after_actuators`, and
        the step ends with :meth:`_finish_step`. With capture-safe actuators the whole loop replays from one CUDA
        graph; otherwise the actuators run eagerly between the captured :meth:`_before_actuators` and
        :meth:`_after_actuators` graphs. Each part runs eagerly the first time and is then captured. Inside an
        enclosing capture the step records into it instead. The solver then checks the step for asynchronous failures
        (:meth:`NewtonSolver.check_step`).

        Returns:
            Whether the step replays captured graphs.
        """
        steps = self.decimation if self.handles_decimation() else 1
        actuators = self.actuators
        with wp.ScopedDevice(self.device):
            self.forward()
            if actuators is None or actuators.is_all_graphable:
                self._launch(("simulate", steps), partial(self._simulate, steps))
            else:
                if self.device.is_capturing:
                    raise RuntimeError("Newton actuators that cannot be captured prevent recording the step.")
                self._stage_body_forces()
                for i in range(steps):
                    self._launch("before_actuators", self._before_actuators)
                    actuators.step(self.state_0, self.control, self.dt)
                    # Only the final physics step supplies forces read by the sensors.
                    observables = self.contact_observables if i == steps - 1 else None
                    self._launch(
                        ("after_actuators", observables is not None), partial(self._after_actuators, observables)
                    )
                self._launch("finish_step", self._finish_step)
        # Graph replays skip the per-substep increments.
        self.state_version += 1
        captured = bool(self.graphs)
        if not self.device.is_capturing:
            self.solver_ops.check_step(captured, self.cfg.debug_mode)
        return captured

    def state_pointers(self) -> tuple[int, int]:
        """Addresses of :attr:`state_0`'s body and particle arrays, ``0`` for an absent array.

        Captured queries read these arrays, so they are recaptured when an address changes.
        """
        state = self.state_0
        return tuple(0 if array is None else array.ptr for array in (state.body_q, state.particle_q))

    def refit_bvh(self, timestamp: int, *, use_cuda_graph: bool = True) -> None:
        """Refit the model's shape and particle BVHs to :attr:`state_0` once per state ``timestamp``.

        Args:
            timestamp: Monotonic version of the state, e.g. the sum of the scene-data transform and geometry timestamps.
            use_cuda_graph: Whether to replay the refit from a CUDA graph on CUDA devices.
        """
        cache = self.bvh_refit
        if self.device.is_capturing:
            # Outer graph replays cannot consult the host timestamp, so each replay must refit.
            self._refit_bvh()
            cache.timestamp = -1
            return
        if cache.timestamp == timestamp:
            return
        if not (use_cuda_graph and self.device.is_cuda):
            self._refit_bvh()
            cache.timestamp = timestamp
            return
        pointers = self.state_pointers()
        if cache.data is None or cache.data[0] != pointers:
            cache.data = pointers, capture_graph(self.device, self._refit_bvh, relaxed=has_kit())
        wp.capture_launch(cache.data[1])
        cache.timestamp = timestamp

    def _allocate_contacts(self) -> None:
        """Allocate contact storage; solver observables belong to initialization or sensor registration."""
        self.invalidate_graph()
        model, solver, ops = self.model, self.solver, self.solver_ops
        collision_cfg = self.cfg.collision_cfg
        if not ops.uses_collision_pipeline():
            if collision_cfg is not None:
                raise ValueError(
                    f"NewtonCfg.collision_cfg cannot be set while {type(solver).__name__} does not use Newton's"
                    " collision pipeline. Remove collision_cfg, or configure the solver to use the pipeline (e.g."
                    " use_mujoco_contacts=False)."
                )
            self.contacts = ops.create_contacts()
            if self.contact_sensors and self.contacts is None:
                # Native detectors need output storage when a sensor requests contact forces.
                self.contacts = Contacts(
                    rigid_contact_max=model.rigid_contact_max or 0,
                    soft_contact_max=0,
                    device=self.device,
                    requested_attributes=model.get_requested_contact_attributes(),
                )
        else:
            args = collision_cfg.to_pipeline_args() if collision_cfg is not None else {"broad_phase": "explicit"}
            args["deterministic"] = self.deterministic_mode != wp.DeterministicMode.NOT_GUARANTEED
            required = getattr(solver, "get_max_contact_count", lambda: 0)()
            if self.collision_pipeline is None:
                self.collision_pipeline = ops.create_collision_pipeline(**args)
            self.contacts = self.collision_pipeline.contacts()
            # MuJoCo Warp can require more contacts than the pipeline estimates.
            if required > self.contacts.rigid_contact_max:
                if args["deterministic"]:
                    # The deterministic sort buffer is sized at construction, so rebuild the pipeline to match.
                    args["rigid_contact_max"] = required
                    self.collision_pipeline = ops.create_collision_pipeline(**args)
                    self.contacts = self.collision_pipeline.contacts()
                else:
                    self.contacts = Contacts(
                        rigid_contact_max=required,
                        soft_contact_max=0,
                        device=self.device,
                        requested_attributes=model.get_requested_contact_attributes(),
                    )
        if self.contacts is not None:
            ops.prepare_contacts(self.contacts, self.collision_pipeline)

    def _launch(self, key: str | tuple[str, int], fn: Callable[[], None]) -> None:
        """Replay ``fn`` from its captured graph, or run it and capture it; inside an enclosing capture, record it."""
        if self.device.is_capturing:
            fn()
            return
        graph = self.graphs.get(key)
        if graph is not None:
            wp.capture_launch(graph)
            return
        fn()
        if self.capture is not None:
            with Timer(name="newton_cuda_graph", msg="CUDA graph took:"):
                self.graphs[key] = self.capture(fn)

    def _simulate(self, steps: int) -> None:
        """Run ``steps`` physics steps with the Newton actuators, then finish the step."""
        actuators = self.actuators
        self._stage_body_forces()
        for i in range(steps):
            self._before_actuators()
            if actuators is not None:
                # An odd step count copies the last actuator output back, keeping buffer addresses stable.
                swap_state = steps % 2 == 0 or i < steps - 1
                actuators.step(self.state_0, self.control, self.dt, swap_state=swap_state)
            self._after_actuators(self.contact_observables if i == steps - 1 else None)
        self._finish_step()

    def _before_actuators(self) -> None:
        """Prepare the solver, collide, and run the pre-actuator callbacks of one physics step."""
        self.solver_ops.pre_step(self.state_0)
        if self.collision_pipeline is not None:
            self.collision_pipeline.collide(self.state_0, self.contacts)
        for callback in self.pre_actuator_callbacks:
            callback()

    def _after_actuators(self, observables: SolverBase.Observables | None) -> None:
        """Run the post-actuator callbacks and the solver substeps of one physics step."""
        for callback in self.post_actuator_callbacks:
            callback()
        contacts = self.contacts if self.collision_pipeline is not None or observables is not None else None
        self._run_solver_substeps(contacts, observables=observables)

    def _finish_step(self) -> None:
        """Run the post-step callbacks and update the sensors once after the last physics step."""
        for callback in self.post_step_callbacks:
            callback()
        self._update_sensors()

    def _stage_body_forces(self) -> None:
        """Copy the body forces written to ``state_0`` before the step, so that every solver substep applies them."""
        if self.staged_body_f is not None:
            wp.copy(self.staged_body_f, self.state_0.body_f)

    def _run_solver_substeps(self, contacts: Contacts | None, *, observables: SolverBase.Observables | None) -> None:
        """Run the solver substeps of one physics step, ending in :attr:`state_0`.

        Each substep restores the staged body forces, then state-force callbacks add to them.
        """
        ops = self.solver_ops
        num_substeps, control, dt = self.num_substeps, self.control, self.solver_dt
        collide_every = self.cfg.collision_decimation
        # The last substep is skipped: its contacts would only feed the next physics step's collision.
        collide_mid_loop = collide_every > 0 and self.collision_pipeline is not None and contacts is not None
        for i in range(num_substeps):
            state = self.state_0
            if self.staged_body_f is not None:
                wp.copy(state.body_f, self.staged_body_f)
            for callback in self.state_force_callbacks:
                callback(state)
            state_out = state if ops.single_state else self.state_1
            # Only final-substep contact forces are read by sensors.
            step_observables = observables if i == num_substeps - 1 else None
            ops.step(state, state_out, control, contacts, dt, observables=step_observables)
            if not ops.single_state:
                if num_substeps % 2 == 1 and i == num_substeps - 1:
                    state.assign(self.state_1)
                else:
                    self.state_0, self.state_1 = self.state_1, self.state_0
            self.state_0.clear_forces()
            # Derived data read by the next substep's state-force callbacks must see this substep's state.
            self.state_version += 1
            if collide_mid_loop and (i + 1) % collide_every == 0 and i + 1 < num_substeps:
                self.collision_pipeline.collide(self.state_0, contacts)
        ops.post_step(self.state_0)

    def _update_sensors(self) -> None:
        """Push the current state to IMU and contact sensors."""
        for sensor in self.imu_sensors:
            sensor.update(self.state_0)
        if self.contact_sensors:
            for sensor in self.contact_sensors.values():
                sensor.update(self.state_0, self.contacts, observables=self.contact_observables)

    def _refit_bvh(self) -> None:
        model, state = self.model, self.state_0
        if model.shape_count:
            model.bvh_refit_shapes(state)
        if model.particle_count:
            model.bvh_refit_particles(state)


def capture_graph(device: wp.DeviceLike, fn: Callable[[], None], *, relaxed: bool = False) -> wp.Graph:
    """Record ``fn`` into a CUDA graph without running it.

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


def _resolve_deterministic_mode(cfg: NewtonCfg) -> wp.DeterministicMode:
    """Translate the determinism request of a Newton configuration into a Warp mode.

    An explicit :attr:`NewtonCfg.deterministic_mode` wins over the generic
    :attr:`~isaaclab.physics.PhysicsCfg.deterministic` request; the selected solver then decides which mode applies
    (:meth:`NewtonSolver.applied_deterministic_mode`).

    Args:
        cfg: Newton physics configuration.

    Returns:
        The mode for the solver and collision pipeline.
    """
    if cfg.deterministic_mode != "not_guaranteed":
        mode = {"run_to_run": wp.DeterministicMode.RUN_TO_RUN, "gpu_to_gpu": wp.DeterministicMode.GPU_TO_GPU}[
            cfg.deterministic_mode
        ]
    else:
        mode = wp.DeterministicMode.RUN_TO_RUN if cfg.deterministic else wp.DeterministicMode.NOT_GUARANTEED
    return cfg.solver_cfg.class_type.applied_deterministic_mode(cfg.solver_cfg, mode)


def _compile_label_pattern(expr: str | list[str] | None) -> re.Pattern[str] | None:
    """Compile Isaac Lab selector expressions for Newton's full label matching."""
    if not expr:
        return None
    return re.compile("|".join((expr,) if isinstance(expr, str) else expr))


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
