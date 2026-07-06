# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Newton FeatherPGS physics manager."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from isaaclab.utils.configclass import configclass

from .newton_manager_cfg import NewtonSolverCfg

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager


@configclass
class FeatherPGSSolverCfg(NewtonSolverCfg):
    """Configuration for Newton's reduced-coordinate FeatherPGS solver.

    FeatherPGS evaluates articulated-body dynamics in generalized coordinates
    and resolves contacts and enabled joint constraints with projected
    Gauss-Seidel iterations. It always uses Newton's collision pipeline.
    """

    class_type: type[NewtonManager] | str = "{DIR}.feather_pgs_manager:NewtonFeatherPGSManager"
    """Manager class for the FeatherPGS solver."""

    solver_type: str = "feather_pgs"
    """Solver type metadata."""

    angular_damping: float = 0.05
    """Angular velocity damping rate [s^-1]."""

    update_mass_matrix_interval: int = 1
    """Number of simulation steps between mass-matrix updates."""

    friction_smoothing: float = 1.0
    """Huber delta for normalizing tangential contact velocity [m/s]."""

    enable_contact_friction: bool = True
    """Whether to resolve Coulomb friction in the PGS solve."""

    enable_joint_limits: bool = False
    """Whether to enforce joint position limits as unilateral constraints."""

    enable_joint_velocity_limits: bool = False
    """Whether to enforce per-DOF joint velocity limits.

    This option currently requires :attr:`pgs_mode` to be ``"matrix_free"``.
    """

    pgs_iterations: int = 12
    """Number of projected Gauss-Seidel iterations per solver step."""

    pgs_beta: float = 0.2
    """Dimensionless error-reduction factor for position correction."""

    pgs_cfm: float = 1.0e-6
    """Regularization added to Newton's impulse-space Delassus diagonal.

    The solver-space units depend on the generalized coordinate represented by
    each constraint row.
    """

    dense_contact_compliance: float = 0.0
    """Normal compliance for dense articulated contact rows [m/N]."""

    pgs_omega: float = 1.0
    """Dimensionless successive over-relaxation factor."""

    dense_max_constraints: int = 32
    """Maximum dense articulated-contact rows stored per world."""

    pgs_warmstart: bool = False
    """Whether to initialize persistent contacts from the previous step's impulses."""

    pgs_mode: Literal["dense", "split", "matrix_free"] = "split"
    """Constraint solve layout.

    ``"dense"`` builds the full Delassus matrix, ``"split"`` uses the dense
    path for articulations and a matrix-free path for free rigid bodies, and
    ``"matrix_free"`` avoids constructing the dense matrix.
    """

    friction_mode: Literal["current", "bisection", "bisection_desaxce", "coulomb_newton"] = "current"
    """Coulomb friction strategy for matrix-free constraints.

    Values other than ``"current"`` require :attr:`pgs_mode` to be
    ``"matrix_free"``.
    """

    mf_max_constraints: int = 512
    """Maximum matrix-free constraint rows stored per world."""

    cholesky_kernel: Literal["tiled", "loop", "auto"] = "auto"
    """Kernel used for Cholesky factorization."""

    trisolve_kernel: Literal["tiled", "loop", "auto"] = "auto"
    """Kernel used for triangular solves."""

    hinv_jt_kernel: Literal["tiled", "par_row", "auto"] = "auto"
    """Kernel used to compute the articulated response ``H^-1 J^T``."""

    delassus_kernel: Literal["tiled", "par_row_col", "auto"] = "auto"
    """Kernel used to accumulate the Delassus matrix."""

    pgs_kernel: Literal["loop", "tiled_row", "tiled_contact", "streaming"] = "tiled_row"
    """Kernel used for the dense projected Gauss-Seidel sweep."""

    delassus_chunk_size: int | None = None
    """Constraint-row chunk size for the streaming Delassus kernel.

    A value of ``None`` lets Newton select the size automatically.
    """

    pgs_chunk_size: int | None = None
    """Contact chunk size for the streaming PGS kernel.

    A value of ``None`` uses Newton's default.
    """

    small_dof_threshold: int = 12
    """DOF count above which automatic kernel selection chooses tiled kernels."""

    use_parallel_streams: bool = True
    """Whether to dispatch articulation-size groups on separate CUDA streams."""

    double_buffer: bool = True
    """Whether to double-buffer solver workspaces between steps."""

    nvtx: bool = False
    """Whether to emit NVTX ranges for solver stages."""

    pgs_debug: bool = False
    """Whether to collect PGS convergence diagnostics."""

    effort_limit_mode: Literal["actuator"] = "actuator"
    """Effort-limit policy.

    FeatherPGS currently supports actuator-drive clamping only.
    """
