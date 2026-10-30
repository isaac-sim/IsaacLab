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
from typing import TYPE_CHECKING, Any

import torch
import warp as wp
from newton import CollisionPipeline, Contacts, ModelBuilder, State
from newton.sensors import SensorContact, SensorIMU
from newton.solvers import SolverBase

from isaaclab.sim import SimulationContext
from isaaclab.utils.buffers import TimestampedBuffer
from isaaclab.utils.timer import Timer
from isaaclab.utils.version import has_kit
from isaaclab.utils.warp.index_kernel import IndexKernelDispatcher

from isaaclab_newton.renderers.visual_material import VisualMaterialWriter

from .newton_builder_cfg import NewtonBuilderCfg
from .newton_manager_cfg import NewtonCfg

if TYPE_CHECKING:
    from isaaclab.actuators.newton import NewtonActuatorAdapter
    from isaaclab.renderers.base_renderer import VisualMaterialBatch

    from .newton_backend_cfg import NewtonBackendCfg

logger = logging.getLogger(__name__)

SiteEntry = tuple[int, None] | tuple[None, list[list[int]]]
"""Resolved site: ``(global_shape_index, None)`` for a global site, or ``(None, per_world_indices)``."""


# ----- Reset-mask kernels ----------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _flag_worlds_and_articulations_masked(
    env_mask: wp.array(dtype=wp.bool),
    articulation_ids: wp.array2d(dtype=wp.int32),
    clear_env_mask: bool,
    world_mask: wp.array(dtype=wp.bool),
    fk_mask: wp.array(dtype=wp.bool),
):
    """Flag each masked world for a solver reset and its articulations for FK, optionally clearing the input."""
    world = wp.tid()
    if env_mask[world]:
        world_mask[world] = True
        for articulation in range(articulation_ids.shape[1]):
            fk_mask[articulation_ids[world, articulation]] = True
        if clear_env_mask:
            env_mask[world] = False


@wp.kernel(enable_backward=False)
def _flag_worlds_masked(env_mask: wp.array(dtype=wp.bool), clear_env_mask: bool, world_mask: wp.array(dtype=wp.bool)):
    """Flag each masked world for a solver reset, optionally clearing the input."""
    world = wp.tid()
    if env_mask[world]:
        world_mask[world] = True
        if clear_env_mask:
            env_mask[world] = False


@wp.kernel(enable_backward=False)
def _mask_from_ids(env_ids: wp.array(dtype=Any), env_mask: wp.array(dtype=wp.bool)):
    """Set the mask entry of each selected environment."""
    env_mask[wp.int32(env_ids[wp.tid()])] = True


_MASK_FROM_IDS = IndexKernelDispatcher(_mask_from_ids, ("env_ids",))


# ----- Data ------------------------------------------------------------------------------


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
        self.collision_pipeline: CollisionPipeline | None = None
        """Collision pipeline, or ``None`` when the solver collides internally."""
        self.contacts: Contacts | None = None
        """Contacts produced by the collision pipeline or the solver."""
        self.staged_body_f: wp.array | None = None
        """Body forces [N, N·m] written before the step, re-applied before every substep."""
        self.world_mask = wp.zeros(self.model.world_count + 1, dtype=wp.bool, device=self.device)
        """Worlds to reset; the final slot is Newton's global world, which Isaac Lab never resets."""
        self.fk_mask = wp.zeros(self.model.articulation_count, dtype=wp.bool, device=self.device)
        """Articulations whose forward kinematics must be evaluated."""
        self.kinematics_dirty = False
        """Whether joint coordinates changed since the last forward kinematics."""
        self.ids_mask = wp.zeros(self.model.world_count, dtype=wp.bool, device=self.device)
        """Environment mask written by :meth:`mask_from_ids`; the invalidation that consumes it clears it again."""
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
        """Run before the actuators of every physics step."""
        self.post_actuator_callbacks: list[Callable[[], None]] = []
        """Run after the actuators of every physics step."""
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
        self.graph: wp.Graph | None = None
        """Captured step, or ``None`` until the next step is captured."""

    def create_visual_material_writer(self, batches: tuple[VisualMaterialBatch, ...]) -> VisualMaterialWriter:
        """Bind material writes to this resource's native shape-color buffer."""
        return VisualMaterialWriter(self.model, batches)

    def close(self) -> None:
        """Drop native handles after consumers release their bindings."""
        self.model = self.state_0 = self.state_1 = self.control = None
        self.particle_ranges, self.deformable_ranges, self.geometry_offsets = {}, {}, {}
        self.bvh_refit = TimestampedBuffer()
        if self.cfg is not None:
            self.solver = self.collision_pipeline = self.contacts = self.actuators = None
            self.staged_body_f = self.capture = self.graph = None
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
        solver_cls = solver_cfg.class_type
        if mode != wp.DeterministicMode.NOT_GUARANTEED and not solver_cls.deterministic:
            raise ValueError(
                f"Newton deterministic mode {mode.name} is not supported by {type(solver_cfg).__name__}."
                " Use MJWarp on the GPU, XPBD, or Featherstone, or disable deterministic mode."
            )
        solver = self.solver = solver_cls.build(solver_cfg, self.model, mode, env_template)
        if not solver_cls.single_state:
            solver_cls.init_state(solver_cfg, solver, self.state_1)
        self._allocate_contacts()
        state_0 = self.state_0
        self.staged_body_f = wp.zeros_like(state_0.body_f) if state_0.body_count else None
        callbacks, self.solver_init_callbacks = self.solver_init_callbacks, []
        for callback in callbacks:
            callback()
        solver_cls.eval_fk(solver_cfg, solver, state_0, None, None)
        self.capture = None
        if self.cfg.use_cuda_graph and self.device.is_cuda:
            if solver_cls.supports_capture(solver_cfg, solver):
                self.capture = lambda fn: capture_graph(
                    self.device, fn, relaxed=relaxed_capture is not None and relaxed_capture()
                )
            else:
                logger.warning(
                    "%s cannot be captured in its current configuration; using eager execution.", type(solver).__name__
                )
        self.invalidate_graph()

    def mask_from_ids(self, env_ids: wp.array | torch.Tensor) -> wp.array:
        """Return a mask selecting ``env_ids``, written into :attr:`ids_mask`.

        Pass the mask to :meth:`invalidate_fk` or :meth:`invalidate_body_state`, which clear it again for the next call.

        Args:
            env_ids: Environment indices, int32 or int64.

        Returns:
            :attr:`ids_mask`.
        """
        if isinstance(env_ids, torch.Tensor):
            env_ids = wp.from_torch(env_ids)
        wp.launch(
            _MASK_FROM_IDS.select(env_ids),
            dim=env_ids.shape[0],
            inputs=[env_ids],
            outputs=[self.ids_mask],
            device=self.device,
        )
        return self.ids_mask

    def invalidate_fk(self, env_mask: wp.array | None = None, articulation_ids: wp.array | None = None) -> None:
        """Flag articulations for forward kinematics and their worlds for a solver reset, without host synchronization.

        Args:
            env_mask: Mask over environments. Shape ``(num_envs,)``. ``None`` flags every environment.
            articulation_ids: Model articulation index of each ``(world, articulation)``, from
                ``ArticulationView.articulation_ids``. Shape ``(world_count, count_per_world)``. ``None`` flags every
                articulation.
        """
        self.kinematics_dirty = True
        if env_mask is None or articulation_ids is None:
            self.world_mask[: self.model.world_count].fill_(True)
            self.fk_mask.fill_(True)
            if env_mask is not None and env_mask is self.ids_mask:
                self.ids_mask.zero_()
            return
        wp.launch(
            _flag_worlds_and_articulations_masked,
            dim=articulation_ids.shape[0],
            inputs=[env_mask, articulation_ids, env_mask is self.ids_mask],
            outputs=[self.world_mask, self.fk_mask],
            device=self.device,
        )

    def invalidate_body_state(self, env_mask: wp.array | None = None) -> None:
        """Flag worlds whose maximal-coordinate body state was written for a solver reset, without requesting FK.

        Args:
            env_mask: Mask over environments. ``None`` flags every environment.
        """
        self.kinematics_dirty = True
        if env_mask is None:
            self.world_mask[: self.model.world_count].fill_(True)
            return
        wp.launch(
            _flag_worlds_masked,
            dim=env_mask.shape[0],
            inputs=[env_mask, env_mask is self.ids_mask],
            outputs=[self.world_mask],
            device=self.device,
        )

    def forward(self) -> None:
        """Reset solver internals and evaluate FK for the flagged worlds and articulations, then clear the flags."""
        if not (self.kinematics_dirty or self.transforms_may_change_on_graph_replay):
            return
        if self.solver is None:
            raise RuntimeError("Newton state was authored before the solver was initialized (init_solver).")
        solver_cfg, solver, state = self.cfg.solver_cfg, self.solver, self.state_0
        solver_cfg.class_type.reset(solver_cfg, solver, state, self.world_mask)
        solver_cfg.class_type.eval_fk(solver_cfg, solver, state, self.world_mask, self.fk_mask)
        self.fk_mask.zero_()
        self.world_mask.zero_()
        self.kinematics_dirty = False

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
                f"Newton contact sensors are not yet supported by {type(solver_cfg).__name__}:"
                " contact forces live in per-entry buffers."
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
            )
        self.contact_sensors[key] = sensor
        # Contacts allocated before the sensor requested the force attribute must be reallocated.
        if self.solver is not None and self.contacts is not None and self.contacts.force is None:
            self._allocate_contacts()
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
        """Whether one :meth:`step` runs the whole decimation loop: Newton actuators are active and all graph-safe."""
        return self.actuators_active and (self.actuators is None or self.actuators.is_all_graphable)

    def set_decimation(self, decimation: int) -> None:
        """Set the physics steps per environment step."""
        self.decimation = max(1, decimation)
        if self.handles_decimation():
            self.invalidate_graph()

    def invalidate_graph(self) -> None:
        """Drop the captured step; with capture enabled, the next step runs eagerly and is captured again."""
        self.graph = None

    def step(self) -> bool:
        """Advance one physics step, or the whole decimation loop when :meth:`handles_decimation`.

        Without the decimation loop, Newton actuators and their callbacks run eagerly before the step. The first step
        after :meth:`invalidate_graph` runs eagerly and is then captured for later steps. The solver then checks the
        step for asynchronous failures (:meth:`NewtonSolver.check_step`).

        Returns:
            Whether the step has a captured graph.
        """
        self._notify_model_changes()
        self.forward()
        if self.handles_decimation():
            simulate = self._simulate_full
        else:
            for callback in self.pre_actuator_callbacks:
                callback()
            if self.actuators is not None:
                self.actuators.step(self.state_0, self.control, self.dt)
            for callback in self.post_actuator_callbacks:
                callback()
            simulate = self._simulate_physics_only
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            with wp.ScopedDevice(self.device):
                simulate()
            if self.capture is not None:
                with Timer(name="newton_cuda_graph", msg="CUDA graph took:"):
                    self.graph = self.capture(simulate)
        captured = self.graph is not None
        solver_cfg = self.cfg.solver_cfg
        solver_cfg.class_type.check_step(solver_cfg, self.solver, captured, self.cfg.debug_mode)
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
        if cache.timestamp == timestamp:
            return
        cache.timestamp = timestamp
        if not (use_cuda_graph and self.device.is_cuda):
            self._refit_bvh()
            return
        pointers = self.state_pointers()
        if cache.data is None or cache.data[0] != pointers:
            cache.data = pointers, capture_graph(self.device, self._refit_bvh, relaxed=has_kit())
        wp.capture_launch(cache.data[1])

    def _allocate_contacts(self) -> None:
        """Allocate contacts from Newton's collision pipeline, or from the solver when it detects contacts itself."""
        model, solver, solver_cfg = self.model, self.solver, self.cfg.solver_cfg
        solver_cls = solver_cfg.class_type
        collision_cfg = self.cfg.collision_cfg
        if not solver_cls.uses_collision_pipeline(solver_cfg, solver):
            if collision_cfg is not None:
                raise ValueError(
                    f"NewtonCfg.collision_cfg cannot be set while {type(solver).__name__} does not use Newton's"
                    " collision pipeline. Remove collision_cfg, or configure the solver to use the pipeline (e.g."
                    " use_mujoco_contacts=False)."
                )
            self.contacts = solver_cls.create_contacts(solver_cfg, solver)
        else:
            args = collision_cfg.to_pipeline_args() if collision_cfg is not None else {"broad_phase": "explicit"}
            args["deterministic"] = self.deterministic_mode != wp.DeterministicMode.NOT_GUARANTEED
            required = getattr(solver, "get_max_contact_count", lambda: 0)()
            if self.collision_pipeline is None:
                self.collision_pipeline = solver_cls.create_collision_pipeline(solver_cfg, solver, **args)
            self.contacts = self.collision_pipeline.contacts()
            # MuJoCo Warp can require more contacts than the pipeline estimates.
            if required > self.contacts.rigid_contact_max:
                if args["deterministic"]:
                    # The deterministic sort buffer is sized at construction, so rebuild the pipeline to match.
                    args["rigid_contact_max"] = required
                    self.collision_pipeline = solver_cls.create_collision_pipeline(solver_cfg, solver, **args)
                    self.contacts = self.collision_pipeline.contacts()
                else:
                    self.contacts = Contacts(
                        rigid_contact_max=required,
                        soft_contact_max=0,
                        device=self.device,
                        requested_attributes=model.get_requested_contact_attributes(),
                    )
        if self.contacts is not None:
            solver_cls.prepare_contacts(solver_cfg, solver, self.contacts, self.collision_pipeline)

    def _notify_model_changes(self) -> None:
        """Notify the solver of the queued model-property changes once, combining their categories."""
        if not self.model_changes:
            return
        ignored = self.cfg.solver_cfg.class_type.ignored_model_changes
        flags = 0
        for change in self.model_changes:
            if change in ignored and change not in self.warned_model_changes:
                logger.warning(ignored[change])
                self.warned_model_changes.add(change)
            flags |= change
        with wp.ScopedDevice(self.device):
            self.solver.notify_model_changed(flags)
        self.model_changes.clear()

    def _simulate_full(self) -> None:
        """Run ``decimation x (collide, actuators, substeps)``, then post-step callbacks and sensors."""
        solver_cfg, pipeline, actuators = self.cfg.solver_cfg, self.collision_pipeline, self.actuators
        contacts = self.contacts if pipeline is not None else None
        decimation = self.decimation
        self._stage_body_forces()
        for i in range(decimation):
            solver_cfg.class_type.pre_step(solver_cfg, self.solver, self.state_0)
            if pipeline is not None:
                pipeline.collide(self.state_0, contacts)
            for callback in self.pre_actuator_callbacks:
                callback()
            if actuators is not None:
                # An odd decimation copies the last actuator output back, keeping buffer addresses stable.
                swap_state = decimation % 2 == 0 or i < decimation - 1
                actuators.step(self.state_0, self.control, self.dt, swap_state=swap_state)
            for callback in self.post_actuator_callbacks:
                callback()
            self._run_solver_substeps(contacts)
        for callback in self.post_step_callbacks:
            callback()
        self._update_sensors(contacts)

    def _simulate_physics_only(self) -> None:
        """Run collision, solver substeps, post-step callbacks and sensors, without actuators."""
        solver_cfg, pipeline = self.cfg.solver_cfg, self.collision_pipeline
        contacts = self.contacts if pipeline is not None else None
        self._stage_body_forces()
        solver_cfg.class_type.pre_step(solver_cfg, self.solver, self.state_0)
        if pipeline is not None:
            pipeline.collide(self.state_0, contacts)
        self._run_solver_substeps(contacts)
        for callback in self.post_step_callbacks:
            callback()
        self._update_sensors(contacts)

    def _stage_body_forces(self) -> None:
        """Copy the body forces written to ``state_0`` before the step, so that every solver substep applies them."""
        if self.staged_body_f is not None:
            wp.copy(self.staged_body_f, self.state_0.body_f)

    def _run_solver_substeps(self, contacts: Contacts | None) -> None:
        """Run the solver substeps of one physics step, ending in :attr:`state_0`.

        Each substep restores the staged body forces, then state-force callbacks add to them.
        """
        solver_cfg, solver = self.cfg.solver_cfg, self.solver
        solver_cls = solver_cfg.class_type
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
            if solver_cls.single_state:
                solver_cls.step(solver_cfg, solver, state, state, control, contacts, dt)
            else:
                solver_cls.step(solver_cfg, solver, state, self.state_1, control, contacts, dt)
                if num_substeps % 2 == 1 and i == num_substeps - 1:
                    state.assign(self.state_1)
                else:
                    self.state_0, self.state_1 = self.state_1, self.state_0
            self.state_0.clear_forces()
            if collide_mid_loop and (i + 1) % collide_every == 0 and i + 1 < num_substeps:
                self.collision_pipeline.collide(self.state_0, contacts)
        solver_cls.post_step(solver_cfg, solver, self.state_0)

    def _update_sensors(self, contacts: Contacts | None) -> None:
        """Push the current state to IMU and contact sensors."""
        for sensor in self.imu_sensors:
            sensor.update(self.state_0)
        if self.contact_sensors:
            eval_contacts = contacts if contacts is not None else self.contacts
            self.solver.update_contacts(eval_contacts, self.state_0)
            for sensor in self.contact_sensors.values():
                sensor.update(self.state_0, eval_contacts)

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
