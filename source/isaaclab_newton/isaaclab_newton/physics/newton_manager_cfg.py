# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton physics manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.physics import PhysicsCfg
from isaaclab.utils import configclass

from .newton_manager import NewtonManager

if TYPE_CHECKING:
    from isaaclab.physics import PhysicsManager


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

    ls_parallel: bool = False
    """Whether to use parallel line search."""

    use_mujoco_contacts: bool = True
    """Whether to use MuJoCo's contact solver."""


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
class NewtonCfg(PhysicsCfg):
    """Configuration for Newton physics manager.

    This configuration includes Newton-specific simulation settings and solver configuration.
    """

    class_type: type[PhysicsManager] = NewtonManager
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
