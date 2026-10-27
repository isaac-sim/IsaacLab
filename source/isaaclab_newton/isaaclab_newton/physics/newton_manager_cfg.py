# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton physics manager."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from isaaclab.physics import PhysicsCfg
from isaaclab.utils import configclass, to_dict
from isaaclab.utils.string import string_to_callable

from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg

if TYPE_CHECKING:
    import warp as wp
    from newton import CollisionPipeline, Contacts, Control, Model, ModelBuilder, State
    from newton.solvers import SolverBase

    from isaaclab_newton.physics import NewtonManager

logger = logging.getLogger(__name__)


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


@configclass
class NewtonSolverCfg:
    """Configuration of a Newton solver, and how Isaac Lab drives it.

    Every solver runs through one generic path written against Newton's :class:`~newton.solvers.SolverBase` API. A
    solver configuration overrides the capabilities and methods below only where its solver differs; customizing a
    solver means subclassing its configuration. For the solver parameters, see the `Newton documentation`_.

    .. _Newton documentation: https://newton.readthedocs.io/en/latest/
    """

    solver_class: type[SolverBase] | str | None = None
    """Newton solver class that :meth:`build` constructs, or a ``"module:Class"`` reference to it. ``None`` describes
    geometry only."""

    solver_type: str = "None"
    """Solver type metadata (deprecated).

    .. deprecated::
        Solver dispatch is driven by the configuration class; this field is retained as metadata for logging and
        debugging only. Do not branch on ``solver_type`` in new code.
    """

    single_state: ClassVar[bool] = False
    """Whether the solver steps in place on one :class:`newton.State`."""

    deterministic: ClassVar[bool] = False
    """Whether the solver can honor a Warp determinism guarantee."""

    contact_sensors: ClassVar[bool] = True
    """Whether Newton contact sensors can read the solver's contacts."""

    ignored_model_changes: ClassVar[Mapping[str, str]] = {}
    """Names of :class:`newton.ModelFlags` the solver does not apply after construction, mapped to the warning logged
    once."""

    # ----- Before the model exists -----

    def solver_classes(self) -> tuple[type[SolverBase], ...]:
        """Newton solvers whose custom builder attributes, and USD schemas, the builder needs before import."""
        return () if self.solver_class is None else (string_to_callable(self.solver_class),)

    def prepare_builder(self, builder: ModelBuilder) -> None:
        """Normalize the complete builder before it is finalized."""

    # ----- Construction -----

    def resolve(self, model: Model) -> NewtonSolverCfg:
        """Return this configuration with model-dependent defaults resolved; the backend keeps the result."""
        return self

    def build(self, model: Model, mode: wp.DeterministicMode, env_template: str | None = None) -> SolverBase:
        """Construct Newton's solver from the configuration fields matching its constructor.

        Args:
            model: Model, or coupled-entry model view, the solver advances.
            mode: Warp determinism mode.
            env_template: Environment path template of the clone plan, e.g. ``"/World/envs/env_{}"``.
        """
        if self.solver_class is None:
            raise NotImplementedError(f"{type(self).__name__} describes geometry only and builds no solver.")
        solver_cls = string_to_callable(self.solver_class)
        return solver_cls(model, **solver_kwargs(solver_cls, self, mode))

    def init_state(self, solver: SolverBase, state: State) -> None:
        """Initialize solver-owned buffers of a double-buffered solver's output state."""

    # ----- Capabilities -----

    def applies_body_forces(self) -> bool:
        """Whether the solver consumes applied rigid-body forces from :attr:`newton.State.body_f`."""
        return True

    def supports_capture(self, solver: SolverBase) -> bool:
        """Whether the constructed solver can be recorded into a CUDA graph."""
        return True

    def coupling_error(self, in_place: bool) -> str | None:
        """Why the solver cannot run as a coupled-solver entry (stepping in place or not), or ``None``."""
        return None

    # ----- Contacts -----

    def uses_collision_pipeline(self, solver: SolverBase) -> bool:
        """Whether contacts come from Newton's :class:`~newton.CollisionPipeline` instead of the solver."""
        return True

    def create_collision_pipeline(self, model: Model, solver: SolverBase, **kwargs) -> CollisionPipeline:
        """Construct the collision pipeline over the model; ``kwargs`` are the pipeline arguments."""
        from newton import CollisionPipeline

        return CollisionPipeline(model, **kwargs)

    def create_contacts(self, solver: SolverBase) -> Contacts | None:
        """Allocate the contacts a solver reports itself when it does not use the collision pipeline."""
        return None

    def prepare_contacts(self, solver: SolverBase, contacts: Contacts, pipeline: CollisionPipeline | None) -> None:
        """Bind newly allocated contacts to solver-owned buffers before CUDA graph capture."""
        if hasattr(solver, "prepare_contacts"):
            solver.prepare_contacts(contacts)

    # ----- Stepping and resets -----

    def pre_step(self, solver: SolverBase, state: State) -> None:
        """Refresh solver structures once per physics step, before collision."""

    def step(
        self,
        solver: SolverBase,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance one substep of ``dt`` [s]."""
        solver.step(state_in, state_out, control, contacts, dt)

    def post_step(self, solver: SolverBase, state: State) -> None:
        """Publish derived state once per physics step, after the substeps."""

    def reset(self, solver: SolverBase, state: State, world_mask: wp.array) -> None:
        """Clear solver-owned history for the worlds of a ``(world_count + 1,)`` mask, keeping authored joint state."""
        solver.reset(state, world_mask=world_mask, flags=0)

    def eval_fk(self, solver: SolverBase, state: State, world_mask: wp.array | None, fk_mask: wp.array | None) -> None:
        """Update body state from the joint coordinates of the masked articulations; ``None`` selects everything."""
        from newton import eval_fk

        eval_fk(solver.model, state.joint_q, state.joint_qd, state, fk_mask)

    def check_step(self, solver: SolverBase, captured: bool, debug: bool) -> None:
        """Raise asynchronous solver failures after a step launch (``captured``: a CUDA graph replay), and log
        diagnostics when ``debug`` is set."""

    # ----- Consumers -----

    def fixed_tendon_control(self, articulation: Any, model: Model) -> Any:
        """Build the fixed-tendon command adapter of an articulation, or ``None`` when no tendon is driven."""
        raise NotImplementedError(
            "The configured solver does not drive fixed tendons. Fixed-tendon targets require a solver that transmits"
            " to tendons, such as MJWarp."
        )


@configclass
class NewtonSoftContactCfg:
    """Global soft-contact parameters applied to the finalized Newton model."""

    soft_contact_ke: float = 1.0e3
    """Body-particle and particle self-contact stiffness [N/m].

    Effective body-particle stiffness is ``0.5 * (soft_contact_ke + shape_ke)``,
    where ``shape_ke`` is the rigid shape's material stiffness.
    """

    soft_contact_kd: float = 10.0
    """Body-particle contact damping [N*s/m]."""

    soft_contact_mu: float = 0.5
    """Body-particle contact friction coefficient [dimensionless].

    Effective body-particle friction is ``sqrt(soft_contact_mu * shape_mu)``,
    where ``shape_mu`` is the rigid shape's material friction coefficient.
    """


@configclass
class NewtonShapeCfg:
    """Default per-shape collision properties applied to all shapes in a Newton scene.

    Mirrors Newton's :attr:`ModelBuilder.default_shape_cfg`. Fields that Isaac
    Lab overrides or exposes for user overrides are declared here; fields not
    represented keep Newton's upstream defaults. The struct is forwarded onto
    Newton's upstream ``ShapeConfig`` via
    :func:`~isaaclab.utils.checked_apply` at builder construction.
    """

    margin: float = 0.0
    """Default per-shape collision margin [m].

    A nonzero margin (e.g. ``0.01``) is required for stable contact on
    triangle-mesh terrain — without it, lightweight robots fail to learn
    rough-terrain locomotion on Newton. Newton's upstream default is ``0.0``.
    """

    gap: float = 0.01
    """Default per-shape contact gap [m]. Newton's upstream default is ``None``."""

    # Defaults mirror Newton's ShapeConfig defaults so an unspecified field is a no-op.
    ke: float = 2.5e3
    """Default per-shape normal contact stiffness [N/m].

    Applied to shapes that lack an explicit material; per-asset materials
    override it. Mirrors Newton's ``ShapeConfig.ke`` default.
    """

    kd: float = 100.0
    """Default per-shape normal contact damping [N*s/m].

    Applied to shapes that lack an explicit material; per-asset materials
    override it. Mirrors Newton's ``ShapeConfig.kd`` default.
    """

    mu: float = 1.0
    """Default per-shape friction coefficient [dimensionless].

    Applied to shapes that lack an explicit material; per-asset materials
    override it. Mirrors Newton's ``ShapeConfig.mu`` default.
    """


@configclass
class NewtonCfg(PhysicsCfg):
    """Configuration for Newton physics manager.

    This configuration includes Newton-specific simulation settings and solver configuration.

    ``class_type`` selects the integration manager independently of the solver, which :attr:`solver_cfg` describes.
    Downstream code can subclass :class:`NewtonManager` and set this field.
    """

    class_type: type[NewtonManager] | str = "{DIR}.newton_manager:NewtonManager"
    """Manager class, constructed by SimulationContext as ``instantiate(cfg)``."""

    num_substeps: int = 1
    """Number of substeps to use for the solver."""

    collision_decimation: int = 0
    """Re-collide every N solver substeps within a physics tick (``0`` = once per tick)."""

    debug_mode: bool = False
    """Whether to enable debug mode for the solver."""

    use_cuda_graph: bool = True
    """Whether to use CUDA graphing when simulating.

    Graphs are captured immediately after the first eager physics step, after reset and decimation
    setup. Capture does not advance physics. Kit/RTX uses a nonblocking stream with relaxed
    capture mode; kitless simulation uses Warp's standard capture mode.

    If set to False, the simulation performance will be severely degraded.
    """

    deterministic_mode: Literal["not_guaranteed", "run_to_run", "gpu_to_gpu"] = "not_guaranteed"
    """Determinism guarantee applied to the Newton solver and collision pipeline.

    The values ``"not_guaranteed"``, ``"run_to_run"``, and ``"gpu_to_gpu"``
    map to the corresponding ``warp.DeterministicMode`` values. Deterministic
    execution increases memory use and can reduce simulation performance.

    .. warning::

       Deterministic contact ordering adds sorting work and allocates buffers
       sized for the configured maximum contact count. Runtime and memory
       overhead therefore grow with contact capacity. Enable this mode only
       when its reproducibility guarantee is required.

    MJWarp on the GPU with
    :attr:`~isaaclab_newton.physics.MJWarpSolverCfg.disable_sensors` set to
    ``True``, XPBD, and Featherstone support this setting. Newton raises an
    error during solver initialization for unsupported solvers rather than
    silently running them without the requested guarantee.
    """

    solver_cfg: NewtonSolverCfg | None = None
    """Solver configuration. If None (default), MJWarpSolverCfg is used by default."""

    soft_contact_cfg: NewtonSoftContactCfg | None = None
    """Global soft-contact parameters applied after model finalization.

    If ``None``, Newton model defaults are preserved.
    """

    collision_cfg: NewtonCollisionPipelineCfg | None = None
    """Newton collision pipeline configuration.

    Controls how Newton's :class:`CollisionPipeline` is configured when it is active.
    The pipeline is active when the solver delegates collision detection to Newton:

    - :class:`MJWarpSolverCfg` with ``use_mujoco_contacts=False``,
    - :class:`KaminoPADMMSolverCfg` or :class:`KaminoDVISolverCfg` with
      ``use_collision_detector=False``,
    - :class:`XPBDSolverCfg` (always),
    - :class:`VBDSolverCfg` (always),
    - :class:`FeatherstoneSolverCfg` (always).

    :class:`~isaaclab_newton.physics.MPMSolverCfg` does not use this pipeline;
    implicit MPM treats rigid geometry as colliders internally.

    If ``None`` (default), a pipeline with ``broad_phase="explicit"`` is created
    automatically.  Set this to a :class:`NewtonCollisionPipelineCfg` to customize
    parameters such as broad phase algorithm, contact limits, or hydroelastic mode.

    .. note::
        Setting this while ``MJWarpSolverCfg.use_mujoco_contacts=True`` raises
        :class:`ValueError`.  When a Kamino solver config has ``use_collision_detector=True``,
        the field is ignored because Kamino's internal detector handles contacts.
    """

    default_shape_cfg: NewtonShapeCfg = NewtonShapeCfg()
    """Default per-shape collision properties applied to every shape in the scene.

    Forwarded to Newton's :attr:`ModelBuilder.default_shape_cfg` at builder
    construction via :func:`~isaaclab.utils.checked_apply`. See
    :class:`NewtonShapeCfg` for the declared fields.
    """

    load_visual_shapes: bool | None = None
    """Whether Newton replication imports visual-only geometry from USD.

    ``None`` imports it only when a viewer, an offscreen ``rgb_array`` capture, or a
    camera sensor is active, so headless training does not pay the USD parse time and
    memory for shapes nothing draws. Set to ``True`` to always import it, which is
    needed when a ray-cast sensor must hit geometry that carries no collider.
    """

    bvh_constructor_geometry: Literal["lbvh", "sah", "cubql"] = "cubql"
    """BVH construction algorithm for mesh geometry colliders.

    Selects the bounding-volume-hierarchy builder Newton uses for the triangle
    meshes of collision geometry, forwarded to :attr:`ModelBuilder.BvhConfig`.
    Trades build time against query (traversal) quality:

    - ``"lbvh"``: linear BVH; fastest to build, lowest-quality tree.
    - ``"sah"``: surface-area-heuristic BVH; slower build, tighter tree with
      faster ray/overlap queries.
    - ``"cubql"``: cuBQL GPU builder; balances fast construction with good tree
      quality on the GPU (default).
    """

    bvh_constructor_scene: Literal["lbvh", "sah"] = "sah"
    """BVH construction algorithm for the top-level scene (broad-phase) hierarchy.

    Selects the builder for the BVH over all colliders used during broad-phase
    culling, forwarded to :attr:`ModelBuilder.BvhConfig`. See
    :attr:`bvh_constructor_geometry` for the ``"lbvh"`` / ``"sah"`` trade-off;
    ``"cubql"`` is not available for the scene hierarchy.
    """

    bvh_constructor_gaussian: Literal["lbvh", "sah", "cubql"] = "cubql"
    """BVH construction algorithm for Gaussian-splat primitives.

    Selects the builder for the BVH over 3D Gaussian primitives (used by the
    Gaussian renderer/collision path), forwarded to
    :attr:`ModelBuilder.BvhConfig`. See :attr:`bvh_constructor_geometry` for the
    ``"lbvh"`` / ``"sah"`` / ``"cubql"`` trade-off.
    """

    def __post_init__(self):
        if self.deterministic_mode not in ("not_guaranteed", "run_to_run", "gpu_to_gpu"):
            raise ValueError(
                "NewtonCfg.deterministic_mode must be 'not_guaranteed', 'run_to_run', or 'gpu_to_gpu', "
                f"got {self.deterministic_mode!r}."
            )
        if self.solver_cfg is None:
            from isaaclab_newton.physics.mjwarp_solver_cfg import MJWarpSolverCfg

            self.solver_cfg = MJWarpSolverCfg()

        # Mid-tick re-collide is silently disabled when collision_decimation >= num_substeps.
        if self.collision_decimation > 0 and self.collision_decimation >= self.num_substeps:
            logger.warning(
                "NewtonCfg.collision_decimation=%d is >= num_substeps=%d; mid-tick re-collide is disabled. "
                "Set 0 < collision_decimation < num_substeps to enable.",
                self.collision_decimation,
                self.num_substeps,
            )
