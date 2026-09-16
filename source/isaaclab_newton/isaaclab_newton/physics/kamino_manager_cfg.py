# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton Kamino physics manager."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from isaaclab.utils import config_field, config_to_dict, validate_config

from .newton_manager_cfg import NewtonSolverCfg

if TYPE_CHECKING:
    from newton.solvers import SolverKamino

    from isaaclab_newton.physics import NewtonManager


def _non_none_kwargs(cfg: Any) -> dict[str, Any]:
    """Return ``cfg.to_dict()`` entries with ``None`` values omitted."""
    return {key: value for key, value in config_to_dict(cfg).items() if value is not None}


def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
    """Return a configuration mapping with a type-checker-friendly interface."""
    return config_to_dict(cfg)


@dataclass
class KaminoPADMMCfg:
    """P-ADMM forward-dynamics solver parameters for Kamino."""

    max_iterations: int = config_field(100)
    """Maximum number of P-ADMM solver iterations."""

    primal_tolerance: float = config_field(1e-4)
    """Primal residual convergence tolerance."""

    dual_tolerance: float = config_field(1e-4)
    """Dual residual convergence tolerance."""

    compl_tolerance: float = config_field(1e-4)
    """Complementarity residual convergence tolerance."""

    restart_tolerance: float = config_field(0.999)
    """Combined primal-dual residual tolerance for acceleration restarts."""

    rho_0: float = config_field(0.05)
    """Initial penalty parameter."""

    rho_min: float = config_field(1e-5)
    """Lower bound on the penalty parameter."""

    a_0: float = config_field(1.0)
    """Initial acceleration parameter."""

    alpha: float = config_field(10.0)
    """Primal-dual residual threshold for penalty updates."""

    tau: float = config_field(1.5)
    """Penalty increase/decrease factor."""

    eta: float = config_field(1e-5)
    """Proximal regularization parameter. Must be greater than zero."""

    penalty_update_freq: int = config_field(1)
    """Frequency of penalty updates. Zero disables updates."""

    penalty_update_method: Literal["fixed", "balanced"] = config_field("fixed")
    """Penalty update method."""

    linear_solver_tolerance: float = config_field(0.0)
    """Absolute tolerance for the iterative linear solver. Zero leaves it unchanged."""

    linear_solver_tolerance_ratio: float = config_field(0.0)
    """Ratio adapting the linear solver tolerance from the ADMM primal residual."""

    use_acceleration: bool = config_field(True)
    """Whether to use Nesterov-type acceleration (APADMM)."""

    use_graph_conditionals: bool = config_field(False)
    """Whether to use CUDA graph conditional nodes in the iterative solver."""

    warmstart_mode: Literal["none", "internal", "containers"] = config_field("containers")
    """Warmstart mode."""

    contact_warmstart_method: Literal[
        "key_and_position",
        "geom_pair_net_force",
        "geom_pair_net_wrench",
        "key_and_position_with_net_force_backup",
        "key_and_position_with_net_wrench_backup",
    ] = config_field("geom_pair_net_force")
    """Contact warm-start method."""


@dataclass
class KaminoDVICfg:
    """DVI forward-dynamics solver parameters for Kamino."""

    tolerance: float = config_field(1e-5)
    """Convergence tolerance on the projected update size."""

    regularization: float = config_field(1e-6)
    """Diagonal regularization added to each projected update denominator."""

    omega: float = config_field(1.0)
    """Relaxation factor applied to projected Gauss-Seidel updates."""

    max_alternating_iterations: int = config_field(20)
    """Maximum outer DVI iterations."""

    inequality_sweeps_per_iteration: int = config_field(1)
    """Projected Gauss-Seidel sweeps per DVI iteration."""

    bilateral_solve_interval: int = config_field(1)
    """DVI iterations between repeated direct bilateral solves."""

    bilateral_solver_type: Literal["LLTB", "LLTBRCM"] = config_field("LLTB")
    """Direct linear solver for the bilateral constraint block."""

    bilateral_solver_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments for the bilateral linear solver."""

    warmstart_mode: Literal["none", "internal", "containers"] = config_field("containers")
    """Warmstart mode."""

    contact_warmstart_method: Literal[
        "key_and_position",
        "geom_pair_net_force",
        "key_and_position_with_net_force_backup",
    ] = config_field("key_and_position_with_net_force_backup")
    """Contact warm-start method when ``warmstart_mode`` is ``containers``."""


@dataclass
class KaminoDynamicsCfg:
    """Constrained forward-dynamics problem parameters for Kamino."""

    preconditioning: bool = config_field(True)
    """Whether to precondition the dual problem. Must be ``False`` when using DVI."""

    linear_solver_type: Literal["LLTB", "LLTBRCM", "CR", "CRF"] = config_field("LLTB")
    """Linear solver for the dynamics problem."""

    linear_solver_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments for the linear solver."""


@dataclass
class KaminoConstraintsCfg:
    """Global constraint stabilization parameters for Kamino."""

    alpha: float = config_field(0.1)
    """Baumgarte stabilization for bilateral joint constraints. Valid range is [0, 1]."""

    beta: float = config_field(0.01)
    """Baumgarte stabilization for unilateral joint-limit constraints. Valid range is [0, 1]."""

    gamma: float = config_field(0.01)
    """Baumgarte stabilization for unilateral contact constraints. Valid range is [0, 1]."""

    delta: float = config_field(1.0e-6)
    """Contact penetration margin [m]."""


@dataclass
class KaminoFKCfg:
    """Forward-kinematics reset solver parameters for Kamino."""

    use_regularization: bool = config_field(True)
    """Whether to regularize the FK reset solve (Tikhonov term on body poses)."""

    regularization_weight: float = config_field(1e-5)
    """Weight of the FK reset regularizer when :attr:`use_regularization` is ``True``."""

    tolerance: float = config_field(1e-5)
    """Convergence tolerance of the FK reset solve."""


@dataclass
class KaminoCollisionDetectorCfg:
    """Internal Kamino collision-detector parameters."""

    pipeline: Literal["primitive", "unified"] | None = config_field(None)
    """Collision-detection pipeline. ``None`` uses Newton's default (``unified``)."""

    broadphase: Literal["nxn", "sap", "explicit"] | None = config_field(None)
    """Broad-phase algorithm. ``None`` uses Newton's default."""

    bvtype: Literal["aabb", "bs"] | None = config_field(None)
    """Bounding-volume type. ``None`` uses Newton's default."""

    max_contacts: int | None = config_field(None)
    """Model-wide contact buffer capacity cap."""

    max_contacts_per_world: int | None = config_field(None)
    """Per-world contact buffer capacity override."""

    max_contacts_per_pair: int | None = config_field(None)
    """Maximum contacts generated per candidate geometry pair."""

    max_triangle_pairs: int | None = config_field(None)
    """Maximum triangle-primitive shape pairs in narrow phase."""

    default_gap: float | None = config_field(None)
    """Default detection gap [m] applied as a floor to per-geometry gaps."""


@dataclass
class KaminoMaterialsCfg:
    """Material mixing parameters for Kamino contacts."""

    friction_mix_mode: Literal["average", "multiply", "max", "min"] = config_field("average")
    """How friction coefficients are mixed for a contact pair."""

    restitution_mix_mode: Literal["average", "multiply", "max", "min"] = config_field("min")
    """How restitution coefficients are mixed for a contact pair."""


@dataclass
class _KaminoSolverCfgBase(NewtonSolverCfg):
    """Common configuration for Kamino solver-related parameters.

    Kamino simulates constrained rigid multi-body systems in maximal coordinates with
    hard frictional contacts.

    .. note::

        This solver is currently in **Beta**. Its API and behavior may change in future releases.

    For more information, see the `Newton Kamino documentation`_.

    .. _Newton Kamino documentation: https://newton-physics.github.io/newton/latest/
    """

    class_type: type[NewtonManager] | str = config_field("{DIR}.kamino_manager:NewtonKaminoManager")
    """Manager class for the Kamino solver."""

    solver_type: str = config_field("kamino")
    """Solver type. Can be "kamino"."""

    integrator: Literal["euler", "moreau"] = config_field("moreau")
    """Integrator type."""

    use_collision_detector: bool = config_field(False)
    """Whether to use Kamino's internal collision detector instead of Newton's pipeline."""

    use_fk_solver: bool | None = config_field(None)
    """Whether to enable the forward kinematics solver for state resets.

    When ``None``, Kamino will automatically determine whether to use the FK solver based on the model's
    articulation structure. If the model has loop-closing joints, the FK solver will be used.

    When ``True``, :meth:`NewtonKaminoManager._eval_fk_impl` reconciles body state via
    :meth:`SolverKamino.reset` with :class:`SolverKamino.ResetConfig.from_joints`. Kamino's FK
    solver computes consistent body poses/velocities from the joint coordinates (including the
    base joint for floating bases), resolves passive / loop-closure joints, and writes back a
    consistent full joint state. Environment resets only need to write actuated DOFs in
    ``joint_q``; passive values are filled in by FK. This is required for closed-loop systems.

    When ``False``, Newton's articulated ``eval_fk`` is used instead over the full
    ``joint_q`` / ``joint_qd``. It is then up to the user to specify constraint-consistent
    values. This is the faster option for purely articulated (tree-structured) systems.
    """

    sparse_jacobian: bool | None = config_field(None)
    """Whether to use sparse Jacobian computation. ``None`` lets Newton pick per backend."""

    sparse_dynamics: bool = config_field(False)
    """Whether to use sparse dynamics computation."""

    rotation_correction: Literal["twopi", "continuous", "none"] = config_field("twopi")
    """Rotation correction mode."""

    angular_velocity_damping: float = config_field(0.0)
    """Angular velocity damping factor. Valid range is [0.0, 1.0]."""

    collect_solver_info: bool = config_field(False)
    """Whether to collect solver convergence and performance info at each step.

    .. warning::

        Enabling this significantly increases solver runtime and should only be used for debugging.
    """

    compute_solution_metrics: bool = config_field(False)
    """Whether to compute solution metrics at each step.

    .. warning::

        Enabling this significantly increases solver runtime and should only be used for debugging.
    """

    dynamics: KaminoDynamicsCfg | None = config_field(None)
    """Constrained dynamics problem parameters.

    When ``None``, Newton selects defaults appropriate to the selected dynamics solver and
    sparsity settings.
    """

    constraints: KaminoConstraintsCfg = field(default_factory=KaminoConstraintsCfg)
    """Constraint stabilization parameters."""

    fk: KaminoFKCfg = field(default_factory=KaminoFKCfg)
    """Forward-kinematics reset solver parameters."""

    collision_detector: KaminoCollisionDetectorCfg = field(default_factory=KaminoCollisionDetectorCfg)
    """Internal collision-detector parameters."""

    materials: KaminoMaterialsCfg = field(default_factory=KaminoMaterialsCfg)
    """Material mixing parameters."""

    max_contacts_per_world: int | None = config_field(None)
    """Cap the per-world contact pre-allocation handed to Kamino.

    When ``None``, Kamino falls back to ``geoms.world_minimum_contacts`` derived from the
    collision pipeline, which over-allocates dramatically for contact-rich assets. Set this
    to bound GPU memory for multi-env training of contact-heavy tasks (e.g. legged
    locomotion or manipulation). The total ``model.rigid_contact_max`` is computed as
    ``max_contacts_per_world * model.world_count`` before solver construction.

    This field is applied by :class:`NewtonKaminoManager` and is not forwarded to Newton.
    """

    def _get_dynamics_solver_config(self) -> tuple[Literal["padmm", "dvi"], dict[str, Any]]:
        """Return the selected Newton solver name and its configuration keyword arguments."""
        raise NotImplementedError

    def to_solver_config(self) -> SolverKamino.Config:
        """Build a :class:`SolverKamino.Config` from this configuration.

        Returns:
            A ``SolverKamino.Config`` instance ready for solver construction.
        """
        from newton._src.solvers.kamino.config import (
            CollisionDetectorConfig,
            ConstrainedDynamicsConfig,
            ConstraintStabilizationConfig,
            DVISolverConfig,
            ForwardKinematicsSolverConfig,
            MaterialManagerConfig,
            PADMMSolverConfig,
        )
        from newton.solvers import SolverKamino

        # Kamino Manager will set the automatic value before calling this method.
        # This is a fallback to true if that mechanism was bypassed.
        use_fk_solver = self.use_fk_solver
        if use_fk_solver is None:
            use_fk_solver = True

        collision_detector = None
        if self.use_collision_detector:
            collision_detector = CollisionDetectorConfig(**_non_none_kwargs(self.collision_detector))

        # Initialize all solver configs with default values.
        solver_config_types = {
            "padmm": PADMMSolverConfig,
            "dvi": DVISolverConfig,
        }
        solver_configs = {name: config_type() for name, config_type in solver_config_types.items()}

        # Overwrite the selected dynamics solver's config with the user-provided config.
        dynamics_solver, active_solver_kwargs = self._get_dynamics_solver_config()
        solver_configs[dynamics_solver] = solver_config_types[dynamics_solver](**active_solver_kwargs)

        # Build the final solver config.
        config = SolverKamino.Config(
            dynamics_solver=dynamics_solver,
            integrator=self.integrator,
            use_collision_detector=self.use_collision_detector,
            use_fk_solver=use_fk_solver,
            sparse_jacobian=self.sparse_jacobian,
            sparse_dynamics=self.sparse_dynamics,
            rotation_correction=self.rotation_correction,
            angular_velocity_damping=self.angular_velocity_damping,
            collect_solver_info=self.collect_solver_info,
            compute_solution_metrics=self.compute_solution_metrics,
            collision_detector=collision_detector,
            fk=ForwardKinematicsSolverConfig(**_cfg_to_dict(self.fk)),
            constraints=ConstraintStabilizationConfig(**_cfg_to_dict(self.constraints)),
            dynamics=None if self.dynamics is None else ConstrainedDynamicsConfig(**_cfg_to_dict(self.dynamics)),
            materials=MaterialManagerConfig(**_cfg_to_dict(self.materials)),
            **solver_configs,
        )
        validate_config(config)
        return config


@dataclass
class KaminoPADMMSolverCfg(_KaminoSolverCfgBase):
    """Configuration for Kamino with the P-ADMM forward-dynamics solver."""

    dynamics_solver_cfg: KaminoPADMMCfg = field(default_factory=KaminoPADMMCfg)
    """P-ADMM forward-dynamics solver parameters."""

    def _get_dynamics_solver_config(self) -> tuple[Literal["padmm"], dict[str, Any]]:
        """Return P-ADMM and its configuration keyword arguments."""
        return "padmm", _cfg_to_dict(self.dynamics_solver_cfg)


@dataclass
class KaminoDVISolverCfg(_KaminoSolverCfgBase):
    """Configuration for Kamino with the DVI forward-dynamics solver."""

    dynamics_solver_cfg: KaminoDVICfg = field(default_factory=KaminoDVICfg)
    """DVI forward-dynamics solver parameters."""

    def _get_dynamics_solver_config(self) -> tuple[Literal["dvi"], dict[str, Any]]:
        """Return DVI and its configuration keyword arguments."""
        return "dvi", _cfg_to_dict(self.dynamics_solver_cfg)
