# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton physics manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.physics import PhysicsCfg
from isaaclab.utils import configclass

from .newton_collision_cfg import NewtonCollisionPipelineCfg

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager


@configclass
class NewtonSolverCfg:
    """Configuration for Newton solver-related parameters.

    These parameters are used to configure the Newton solver. For more information, see the `Newton documentation`_.

    .. _Newton documentation: https://newton.readthedocs.io/en/latest/
    """

    solver_type: str = "None"
    """Solver type.

    Used to select the right solver class.
    """


@configclass
class MJWarpSolverCfg(NewtonSolverCfg):
    """Configuration for MuJoCo Warp solver-related parameters.

    These parameters are used to configure the MuJoCo Warp solver. For more information, see the
    `MuJoCo Warp documentation`_.

    .. _MuJoCo Warp documentation: https://github.com/google-deepmind/mujoco_warp
    """

    solver_type: str = "mujoco_warp"
    """Solver type. Can be "mujoco_warp"."""

    njmax: int = 300
    """Number of constraints per environment (world)."""

    nconmax: int | None = None
    """Number of contact points per environment (world)."""

    iterations: int = 100
    """Number of solver iterations."""

    ls_iterations: int = 50
    """Number of line search iterations for the solver."""

    solver: str = "newton"
    """Solver type. Can be "cg" or "newton", or their corresponding MuJoCo integer constants."""

    integrator: str = "euler"
    """Integrator type. Can be "euler", "rk4", or "implicitfast", or their corresponding MuJoCo integer constants."""

    use_mujoco_cpu: bool = False
    """Whether to use the pure MuJoCo backend instead of `mujoco_warp`."""

    disable_contacts: bool = False
    """Whether to disable contact computation in MuJoCo."""

    default_actuator_gear: float | None = None
    """Default gear ratio for all actuators."""

    actuator_gears: dict[str, float] | None = None
    """Dictionary mapping joint names to specific gear ratios, overriding the `default_actuator_gear`."""

    update_data_interval: int = 1
    """Frequency (in simulation steps) at which to update the MuJoCo Data object from the Newton state.

    If 0, Data is never updated after initialization.
    """

    save_to_mjcf: str | None = None
    """Optional path to save the generated MJCF model file.

    If None, the MJCF model is not saved.
    """

    impratio: float = 1.0
    """Frictional-to-normal constraint impedance ratio."""

    cone: str = "pyramidal"
    """The type of contact friction cone. Can be "pyramidal" or "elliptic"."""

    ccd_iterations: int = 35
    """Maximum iterations for convex collision detection (GJK/EPA).

    Increase this if you see warnings about ``opt.ccd_iterations`` needing to be increased,
    which typically occurs with complex collision geometries (e.g. multi-finger hands).
    """

    ls_parallel: bool = False
    """Whether to use parallel line search."""

    use_mujoco_contacts: bool = True
    """Whether to use MuJoCo's internal contact solver.

    If ``True`` (default), MuJoCo handles collision detection and contact resolution internally.
    If ``False``, Newton's :class:`CollisionPipeline` is used instead.  A default pipeline
    (``broad_phase="explicit"``) is created automatically when :attr:`NewtonCfg.collision_cfg`
    is ``None``.  Set :attr:`NewtonCfg.collision_cfg` to a :class:`NewtonCollisionPipelineCfg`
    to customize pipeline parameters (broad phase, contact limits, hydroelastic, etc.).

    .. note::
        Setting ``collision_cfg`` while ``use_mujoco_contacts=True`` raises
        :class:`ValueError` because the two collision modes are mutually exclusive.
    """

    tolerance: float = 1e-6
    """Solver convergence tolerance for the constraint residual.

    The solver iterates until the residual drops below this threshold or
    ``iterations`` is reached.  Lower values give more precise constraint
    satisfaction at the cost of more iterations.  MuJoCo default is ``1e-8``;
    Newton default is ``1e-6``.
    """


@configclass
class XPBDSolverCfg(NewtonSolverCfg):
    """An implicit integrator using eXtended Position-Based Dynamics (XPBD) for rigid and soft body simulation.

    References:
        - Miles Macklin, Matthias Müller, and Nuttapong Chentanez. 2016. XPBD: position-based simulation of compliant
          constrained dynamics. In Proceedings of the 9th International Conference on Motion in Games (MIG '16).
          Association for Computing Machinery, New York, NY, USA, 49-54. https://doi.org/10.1145/2994258.2994272
        - Matthias Müller, Miles Macklin, Nuttapong Chentanez, Stefan Jeschke, and Tae-Yong Kim. 2020. Detailed rigid
          body simulation with extended position based dynamics. In Proceedings of the ACM SIGGRAPH/Eurographics
          Symposium on Computer Animation (SCA '20). Eurographics Association, Goslar, DEU,
          Article 10, 1-12. https://doi.org/10.1111/cgf.14105

    """

    solver_type: str = "xpbd"
    """Solver type. Can be "xpbd"."""

    iterations: int = 2
    """Number of solver iterations."""

    soft_body_relaxation: float = 0.9
    """Relaxation parameter for soft body simulation."""

    soft_contact_relaxation: float = 0.9
    """Relaxation parameter for soft contact simulation."""

    joint_linear_relaxation: float = 0.7
    """Relaxation parameter for joint linear simulation."""

    joint_angular_relaxation: float = 0.4
    """Relaxation parameter for joint angular simulation."""

    joint_linear_compliance: float = 0.0
    """Compliance parameter for joint linear simulation."""

    joint_angular_compliance: float = 0.0
    """Compliance parameter for joint angular simulation."""

    rigid_contact_relaxation: float = 0.8
    """Relaxation parameter for rigid contact simulation."""

    rigid_contact_con_weighting: bool = True
    """Whether to use contact constraint weighting for rigid contact simulation."""

    angular_damping: float = 0.0
    """Angular damping parameter for rigid contact simulation."""

    enable_restitution: bool = False
    """Whether to enable restitution for rigid contact simulation."""


@configclass
class FeatherstoneSolverCfg(NewtonSolverCfg):
    """A semi-implicit integrator using symplectic Euler.

    It operates on reduced (also called generalized) coordinates to simulate articulated rigid body dynamics
    based on Featherstone's composite rigid body algorithm (CRBA).

    See: Featherstone, Roy. Rigid Body Dynamics Algorithms. Springer US, 2014.

    Semi-implicit time integration is a variational integrator that
    preserves energy, however it not unconditionally stable, and requires a time-step
    small enough to support the required stiffness and damping forces.

    See: https://en.wikipedia.org/wiki/Semi-implicit_Euler_method
    """

    solver_type: str = "featherstone"
    """Solver type. Can be "featherstone"."""

    angular_damping: float = 0.05
    """Angular damping parameter for rigid contact simulation."""

    update_mass_matrix_interval: int = 1
    """Frequency (in simulation steps) at which to update the mass matrix."""

    friction_smoothing: float = 1.0
    """Friction smoothing parameter."""

    use_tile_gemm: bool = False
    """Whether to use tile-based GEMM for the mass matrix."""

    fuse_cholesky: bool = True
    """Whether to fuse the Cholesky decomposition."""


@configclass
class KaminoSolverCfg(NewtonSolverCfg):
    """Configuration for Kamino solver-related parameters.

    Kamino is a Proximal Alternating Direction Method of Multipliers (P-ADMM) based solver for
    constrained multi-body dynamics. It operates in maximal coordinates and supports rigid bodies
    and articulations with hard frictional contacts.

    .. note::

        This solver is currently in **Beta**. Its API and behavior may change in future releases.

    For more information, see the `Newton Kamino documentation`_.

    .. _Newton Kamino documentation: https://newton.readthedocs.io/en/latest/
    """

    solver_type: str = "kamino"
    """Solver type. Can be "kamino"."""

    integrator: str = "euler"
    """Integrator type. Can be "euler" or "moreau"."""

    use_collision_detector: bool = False
    """Whether to use Kamino's internal collision detector instead of Newton's pipeline."""

    use_fk_solver: bool = True
    """Whether to enable the forward kinematics solver for state resets.

    Required for proper environment resets. The FK solver computes consistent body poses
    from joint angles after state writes, which is essential for maximal-coordinate solvers.
    """

    sparse_jacobian: bool = False
    """Whether to use sparse Jacobian computation."""

    sparse_dynamics: bool = False
    """Whether to use sparse dynamics computation."""

    rotation_correction: str = "twopi"
    """Rotation correction mode. Can be "twopi", "continuous", or "none"."""

    angular_velocity_damping: float = 0.0
    """Angular velocity damping factor. Valid range is [0.0, 1.0]."""

    constraints_alpha: float = 0.01
    """Baumgarte stabilization parameter for bilateral joint constraints. Valid range is [0, 1]."""

    constraints_beta: float = 0.01
    """Baumgarte stabilization parameter for unilateral joint-limit constraints. Valid range is [0, 1]."""

    constraints_gamma: float = 0.01
    """Baumgarte stabilization parameter for unilateral contact constraints. Valid range is [0, 1]."""

    constraints_delta: float = 1.0e-6
    """Contact penetration margin [m]."""

    padmm_max_iterations: int = 200
    """Maximum number of P-ADMM solver iterations."""

    padmm_primal_tolerance: float = 1e-6
    """Primal residual convergence tolerance for P-ADMM."""

    padmm_dual_tolerance: float = 1e-6
    """Dual residual convergence tolerance for P-ADMM."""

    padmm_compl_tolerance: float = 1e-6
    """Complementarity residual convergence tolerance for P-ADMM."""

    padmm_rho_0: float = 1.0
    """Initial penalty parameter for P-ADMM."""

    padmm_use_acceleration: bool = True
    """Whether to use acceleration in the P-ADMM solver."""

    padmm_warmstart_mode: str = "containers"
    """Warmstart mode for P-ADMM. Can be "none", "internal", or "containers"."""

    padmm_eta: float = 1e-5
    """Proximal regularization parameter for P-ADMM. Must be greater than zero."""

    padmm_contact_warmstart_method: str = "key_and_position"
    """Contact warm-start method for P-ADMM.

    Can be "key_and_position", "geom_pair_net_force", "geom_pair_net_wrench",
    "key_and_position_with_net_force_backup", or "key_and_position_with_net_wrench_backup".
    """

    padmm_use_graph_conditionals: bool = True
    """Whether to use CUDA graph conditional nodes in the P-ADMM iterative solver.

    When ``False``, replaces ``wp.capture_while`` with unrolled for-loops over max iterations.
    """

    collect_solver_info: bool = False
    """Whether to collect solver convergence and performance info at each step.

    .. warning::

        Enabling this significantly increases solver runtime and should only be used for debugging.
    """

    compute_solution_metrics: bool = False
    """Whether to compute solution metrics at each step.

    .. warning::

        Enabling this significantly increases solver runtime and should only be used for debugging.
    """

    collision_detector_pipeline: str | None = None
    """Collision detection pipeline type. Can be "primitive" or "unified".

    Only used when :attr:`use_collision_detector` is ``True``. If ``None``, Newton's default
    (``"unified"``) is used.
    """

    collision_detector_max_contacts_per_pair: int | None = None
    """Maximum number of contacts to generate per candidate geometry pair.

    Only used when :attr:`use_collision_detector` is ``True``. If ``None``, Newton's default is used.
    """

    dynamics_preconditioning: bool = True
    """Whether to use preconditioning in the constrained dynamics solver.

    Preconditioning improves convergence of the PADMM solver by rescaling the
    problem. Disabling may be useful for debugging or profiling solver behavior.
    """

    def to_solver_config(self):
        """Build a :class:`SolverKamino.Config` from this configuration.

        Converts the flat field layout of :class:`KaminoSolverCfg` into the
        nested dataclass hierarchy expected by :class:`SolverKamino`.

        Returns:
            A ``SolverKamino.Config`` instance ready for solver construction.
        """
        from newton._src.solvers.kamino.config import (
            CollisionDetectorConfig,
            ConstrainedDynamicsConfig,
            ConstraintStabilizationConfig,
            PADMMSolverConfig,
        )
        from newton.solvers import SolverKamino

        # Build collision detector config if using Kamino's internal detector
        collision_detector = None
        if self.use_collision_detector:
            cd_kwargs: dict = {}
            if self.collision_detector_pipeline is not None:
                cd_kwargs["pipeline"] = self.collision_detector_pipeline
            if self.collision_detector_max_contacts_per_pair is not None:
                cd_kwargs["max_contacts_per_pair"] = self.collision_detector_max_contacts_per_pair
            collision_detector = CollisionDetectorConfig(**cd_kwargs)

        return SolverKamino.Config(
            integrator=self.integrator,
            use_collision_detector=self.use_collision_detector,
            use_fk_solver=self.use_fk_solver,
            sparse_jacobian=self.sparse_jacobian,
            sparse_dynamics=self.sparse_dynamics,
            rotation_correction=self.rotation_correction,
            angular_velocity_damping=self.angular_velocity_damping,
            collect_solver_info=self.collect_solver_info,
            compute_solution_metrics=self.compute_solution_metrics,
            collision_detector=collision_detector,
            constraints=ConstraintStabilizationConfig(
                alpha=self.constraints_alpha,
                beta=self.constraints_beta,
                gamma=self.constraints_gamma,
                delta=self.constraints_delta,
            ),
            dynamics=ConstrainedDynamicsConfig(
                preconditioning=self.dynamics_preconditioning,
            ),
            padmm=PADMMSolverConfig(
                max_iterations=self.padmm_max_iterations,
                primal_tolerance=self.padmm_primal_tolerance,
                dual_tolerance=self.padmm_dual_tolerance,
                compl_tolerance=self.padmm_compl_tolerance,
                rho_0=self.padmm_rho_0,
                eta=self.padmm_eta,
                use_acceleration=self.padmm_use_acceleration,
                use_graph_conditionals=self.padmm_use_graph_conditionals,
                warmstart_mode=self.padmm_warmstart_mode,
                contact_warmstart_method=self.padmm_contact_warmstart_method,
            ),
        )


@configclass
class NewtonShapeCfg:
    """Default per-shape collision properties applied to all shapes in a Newton scene.

    Mirrors Newton's :attr:`ModelBuilder.default_shape_cfg`. Only fields Isaac
    Lab actually overrides are declared here; unspecified fields keep Newton's
    upstream default. The struct is forwarded onto Newton's upstream
    ``ShapeConfig`` via :func:`~isaaclab.utils.checked_apply` at builder
    construction.
    """

    margin: float = 0.0
    """Default per-shape collision margin [m].

    A nonzero margin (e.g. ``0.01``) is required for stable contact on
    triangle-mesh terrain — without it, lightweight robots fail to learn
    rough-terrain locomotion on Newton. Newton's upstream default is ``0.0``.
    """

    gap: float = 0.01
    """Default per-shape contact gap [m]. Newton's upstream default is ``None``."""


@configclass
class NewtonCfg(PhysicsCfg):
    """Configuration for Newton physics manager.

    This configuration includes Newton-specific simulation settings and solver configuration.
    """

    class_type: type[NewtonManager] | str = "{DIR}.newton_manager:NewtonManager"
    """The class type of the NewtonManager."""

    num_substeps: int = 1
    """Number of substeps to use for the solver."""

    debug_mode: bool = False
    """Whether to enable debug mode for the solver."""

    use_cuda_graph: bool = True
    """Whether to use CUDA graphing when simulating.

    If set to False, the simulation performance will be severely degraded.
    """

    solver_cfg: NewtonSolverCfg = MJWarpSolverCfg()
    """Solver configuration. Default is MJWarpSolverCfg()."""

    collision_cfg: NewtonCollisionPipelineCfg | None = None
    """Newton collision pipeline configuration.

    Controls how Newton's :class:`CollisionPipeline` is configured when it is active.
    The pipeline is active when:

    - :class:`MJWarpSolverCfg` with ``use_mujoco_contacts=False``, or
    - any non-MuJoCo solver (:class:`XPBDSolverCfg`, :class:`FeatherstoneSolverCfg`).

    If ``None`` (default), a pipeline with ``broad_phase="explicit"`` is created
    automatically.  Set this to a :class:`NewtonCollisionPipelineCfg` to customize
    parameters such as broad phase algorithm, contact limits, or hydroelastic mode.

    .. note::
        Must not be set when ``use_mujoco_contacts=True`` (raises :class:`ValueError`).
    """

    default_shape_cfg: NewtonShapeCfg = NewtonShapeCfg()
    """Default per-shape collision properties applied to every shape in the scene.

    Forwarded to Newton's :attr:`ModelBuilder.default_shape_cfg` at builder
    construction via :func:`~isaaclab.utils.checked_apply`. See
    :class:`NewtonShapeCfg` for the declared fields.
    """
