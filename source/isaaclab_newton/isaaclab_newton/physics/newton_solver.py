# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""How Isaac Lab builds and drives a Newton solver."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

import warp as wp
from newton import CollisionPipeline, Contacts, Control, Model, ModelBuilder, ModelFlags, State, eval_fk
from newton.solvers import SolverBase

from isaaclab.utils import to_dict

if TYPE_CHECKING:
    from .newton_manager_cfg import NewtonSolverCfg


def solver_kwargs(solver_cls: type, cfg: Any, mode: wp.DeterministicMode) -> dict:
    """Return the configuration fields matching ``solver_cls``'s constructor, plus ``deterministic`` if accepted.

    A constructor forwarding ``**kwargs`` also accepts its base class's parameters, e.g. a solver subclass.
    """
    valid = set()
    for cls in solver_cls.__mro__:
        parameters = inspect.signature(cls.__init__).parameters.values()
        valid.update(p.name for p in parameters if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD))
        if not any(p.kind is p.VAR_KEYWORD for p in parameters):
            break
    valid -= {"self", "model"}
    kwargs = {key: value for key, value in to_dict(cfg).items() if key in valid}
    if "deterministic" in valid:
        kwargs["deterministic"] = mode
    return kwargs


@wp.kernel(enable_backward=False)
def _flag_non_finite_particles(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    worlds: wp.array(dtype=wp.int32),
    faults: wp.array(dtype=wp.bool),
):
    index = wp.tid()
    world = worlds[index]
    if world >= 0 and not (wp.isfinite(positions[index]) and wp.isfinite(velocities[index])):
        faults[world] = True


@wp.kernel(enable_backward=False)
def _flag_non_finite_bodies(
    poses: wp.array(dtype=wp.transform),
    velocities: wp.array(dtype=wp.spatial_vector),
    worlds: wp.array(dtype=wp.int32),
    faults: wp.array(dtype=wp.bool),
):
    index = wp.tid()
    world = worlds[index]
    pose = poses[index]
    if world >= 0 and not (
        wp.isfinite(wp.transform_get_translation(pose))
        and wp.isfinite(wp.transform_get_rotation(pose))
        and wp.isfinite(velocities[index])
    ):
        faults[world] = True


@wp.kernel(enable_backward=False)
def _flag_non_finite_joints(
    q: wp.array(dtype=wp.float32),
    qd: wp.array(dtype=wp.float32),
    q_start: wp.array(dtype=wp.int32),
    qd_start: wp.array(dtype=wp.int32),
    worlds: wp.array(dtype=wp.int32),
    faults: wp.array(dtype=wp.bool),
):
    joint = wp.tid()
    world = worlds[joint]
    if world >= 0:
        for i in range(q_start[joint], q_start[joint + 1]):
            if not wp.isfinite(q[i]):
                faults[world] = True
        for i in range(qd_start[joint], qd_start[joint + 1]):
            if not wp.isfinite(qd[i]):
                faults[world] = True


class NewtonSolver:
    """How Isaac Lab builds and drives a Newton solver.

    A solver configuration selects the subclass through :attr:`~NewtonSolverCfg.class_type`, which builds Newton's
    solver on construction and drives it afterwards. Classmethods run before the model exists and take only the
    configuration. The generic methods below follow Newton's :class:`~newton.solvers.SolverBase` API; a subclass
    overrides only where its solver differs.
    """

    solver_class: ClassVar[type[SolverBase] | None] = None
    """Newton solver class that :meth:`build` constructs; ``None`` describes geometry only."""

    single_state: ClassVar[bool] = False
    """Whether the solver steps in place on one :class:`newton.State`."""

    max_deterministic_mode: ClassVar[wp.DeterministicMode] = wp.DeterministicMode.NOT_GUARANTEED
    """Strongest Warp determinism guarantee the solver can honor."""

    contact_sensors: ClassVar[bool] = True
    """Whether the Newton solver can provide CONTACT_F observables to contact sensors."""

    ignored_model_changes: ClassVar[Mapping[ModelFlags, str]] = {}
    """Model changes the solver does not apply after construction, mapped to the warning logged once."""

    # ----- Before the model exists -----

    @classmethod
    def solver_classes(cls, cfg: NewtonSolverCfg) -> tuple[type[SolverBase], ...]:
        """Newton solvers whose custom builder attributes, and USD schemas, the builder needs before import."""
        return () if cls.solver_class is None else (cls.solver_class,)

    @classmethod
    def prepare_builder(cls, cfg: NewtonSolverCfg, builder: ModelBuilder) -> None:
        """Normalize the complete builder before it is finalized."""

    @classmethod
    def applied_deterministic_mode(cls, cfg: NewtonSolverCfg, mode: wp.DeterministicMode) -> wp.DeterministicMode:
        """The Warp determinism mode the solver and collision pipeline run with for a requested ``mode``."""
        return mode

    @classmethod
    def shares_materials(cls, cfg: NewtonSolverCfg) -> bool:
        """Whether contacts read one material table shared by all environments instead of per-shape values."""
        return False

    @classmethod
    def coupling_error(cls, cfg: NewtonSolverCfg, in_place: bool) -> str | None:
        """Why the solver cannot run as a coupled-solver entry (stepping in place or not), or ``None``."""
        return None

    # ----- Construction -----

    def __init__(self, cfg: NewtonSolverCfg, model: Model, mode: wp.DeterministicMode, env_template: str | None = None):
        """Build Newton's solver.

        Args:
            cfg: Solver configuration.
            model: Model, or coupled-entry model view, the solver advances.
            mode: Warp determinism mode.
            env_template: Environment path template of the clone plan, e.g. ``"/World/envs/env_{}"``.
        """
        self.cfg = cfg
        """Solver configuration."""
        self.native: SolverBase = self.build(model, mode, env_template)
        """Newton's solver."""

    def build(self, model: Model, mode: wp.DeterministicMode, env_template: str | None) -> SolverBase:
        """Construct Newton's solver from the configuration fields matching its constructor."""
        if self.solver_class is None:
            raise NotImplementedError(f"{type(self).__name__} describes geometry only and builds no solver.")
        return self.solver_class(model, **solver_kwargs(self.solver_class, self.cfg, mode))

    def init_state(self, state: State) -> None:
        """Initialize solver-owned buffers of a double-buffered solver's output state."""

    # ----- Capabilities -----

    def applies_body_forces(self) -> bool:
        """Whether the solver consumes applied rigid-body forces from :attr:`newton.State.body_f`."""
        return True

    def supports_capture(self) -> bool:
        """Whether the solver's step can be recorded into a graph and replayed on the solver's device.

        On the CPU, Warp's graph capture records only Warp kernel launches and memory operations. Other work in the
        step, such as :meth:`warp.Mesh.refit`, a nested :func:`warp.capture_launch`, torch operations, or host
        reads, runs once at capture time and never on replay, so a solver whose step does such work on the CPU must
        return False there.
        """
        return True

    # ----- Contacts -----

    def uses_collision_pipeline(self) -> bool:
        """Whether contacts come from Newton's :class:`~newton.CollisionPipeline` instead of the solver."""
        return True

    def create_collision_pipeline(self, **kwargs) -> CollisionPipeline:
        """Construct the collision pipeline over the solver's model; ``kwargs`` are the pipeline arguments."""
        return CollisionPipeline(self.native.model, **kwargs)

    def create_contacts(self) -> Contacts | None:
        """Allocate the contacts a solver reports itself when it does not use the collision pipeline."""
        return None

    def prepare_contacts(self, contacts: Contacts, pipeline: CollisionPipeline | None) -> None:
        """Bind newly allocated contacts to solver-owned buffers before CUDA graph capture."""
        if hasattr(self.native, "prepare_contacts"):
            self.native.prepare_contacts(contacts)

    # ----- Stepping and resets -----

    def pre_step(self, state: State) -> None:
        """Refresh solver structures once per physics step, before collision."""

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts | None,
        dt: float,
        *,
        observables: SolverBase.Observables | None = None,
    ) -> None:
        """Advance one substep of ``dt`` [s]."""
        if observables is None:
            self.native.step(state_in, state_out, control, contacts, dt)
        else:
            self.native.step(state_in, state_out, control, contacts, dt, observables=observables)

    def post_step(self, state: State) -> None:
        """Publish derived state once per physics step, after the substeps."""

    def mark_non_finite_worlds(self, state: State, faults: wp.array) -> None:
        """Add worlds with nonfinite joint, body or particle state to faults, excluding the global world.

        Solver integrations extend this check for persistent state outside Newton's State.
        """
        model = self.native.model
        if model.body_count:
            wp.launch(
                _flag_non_finite_bodies, model.body_count, [state.body_q, state.body_qd, model.body_world, faults]
            )
        if model.joint_count:
            wp.launch(
                _flag_non_finite_joints,
                model.joint_count,
                [state.joint_q, state.joint_qd, model.joint_q_start, model.joint_qd_start, model.joint_world, faults],
            )
        if model.particle_count:
            wp.launch(
                _flag_non_finite_particles,
                model.particle_count,
                [state.particle_q, state.particle_qd, model.particle_world, faults],
            )

    def reset(self, state: State, world_mask: wp.array) -> None:
        """Clear solver-owned history for the worlds of a ``(world_count + 1,)`` mask, keeping authored joint state."""
        self.native.reset(state, world_mask=world_mask, flags=0)

    def eval_fk(self, state: State, world_mask: wp.array | None, fk_mask: wp.array | None) -> None:
        """Update body state from the joint coordinates of the masked articulations; ``None`` selects everything."""
        eval_fk(self.native.model, state.joint_q, state.joint_qd, state, fk_mask)

    def check_step(self, captured: bool, debug: bool) -> None:
        """Raise asynchronous solver failures after a step launch (``captured``: a CUDA graph replay), and log
        diagnostics when ``debug`` is set."""

    # ----- Consumers -----

    @classmethod
    def fixed_tendon_control(cls, cfg: NewtonSolverCfg, articulation: Any, model: Model) -> Any:
        """Build the fixed-tendon command adapter of an articulation, or ``None`` when no tendon is driven."""
        raise NotImplementedError(
            "The configured solver does not drive fixed tendons. Fixed-tendon targets require a solver that transmits"
            " to tendons, such as MJWarp."
        )
