# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton FeatherPGS physics manager."""

from __future__ import annotations

import math
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
    """How often to update the mass matrix, in simulation steps."""

    friction_smoothing: float = 1.0
    """Huber delta for normalizing tangential contact velocity [m/s]."""

    contact_friction_gap_threshold: float = math.inf
    """Gap threshold [m] for enabling tangential contact friction rows."""

    contact_friction_position_iterations: int = -1
    """Number of position iterations used for contact-friction anchoring."""

    contact_friction_shared_anchor: bool = False
    """Whether contact-friction rows use shared anchors."""

    contact_friction_anchor_limit: int = 0
    """Maximum stored contact-friction anchors per world."""

    contact_friction_scale: float = 1.0
    """Dimensionless scale factor applied to contact friction limits."""

    contact_shared_anchor: bool = False
    """Whether normal contact rows use shared anchors."""

    enable_contact_friction: bool = True
    """Whether to enable Coulomb contact friction in PGS."""

    enable_joint_limits: bool = False
    """Whether to enforce joint position limits as unilateral PGS constraints."""

    joint_limit_activation_gap: float = math.inf
    """Distance [m or rad, depending on joint type] from a joint limit at which rows activate."""

    enable_joint_velocity_limits: bool = False
    """Whether to enforce per-DOF joint velocity limits as PGS constraints.

    This option requires :attr:`pgs_mode` to be ``"matrix_free"``.
    """

    velocity_limit_activation_fraction: float = 0.0
    """Dimensionless fraction of each joint velocity limit at which velocity-limit rows activate."""

    pgs_iterations: int = 12
    """Number of Gauss-Seidel iterations per simulation step."""

    pgs_velocity_iterations: int = 0
    """Number of velocity-level Gauss-Seidel iterations per simulation step."""

    pgs_beta: float = 0.2
    """Dimensionless ERP-style position correction factor."""

    pgs_cfm: float = 1.0e-6
    """Regularization added to Newton's impulse-space Delassus diagonal.

    Units follow the constraint row: [kg^-1] for contact and prismatic rows, and [(kg·m²)^-1] for revolute and angular
    rows.
    """

    dense_contact_compliance: float = 0.0
    """Normal contact compliance for dense articulated contact rows [m/N]."""

    speculative_dense_contact_compliance: float = 0.0
    """Additional normal contact compliance for speculative dense contact rows [m/N]."""

    pgs_omega: float = 1.0
    """Dimensionless successive over-relaxation factor for PGS."""

    pgs_velocity_omega: float | None = None
    """Dimensionless velocity-level relaxation factor. ``None`` uses :attr:`pgs_omega`."""

    pgs_velocity_drive_mode: Literal["active", "freeze"] = "freeze"
    """Drive-row handling during velocity-only PGS: ``"active"`` updates rows and ``"freeze"`` holds impulses."""

    dense_max_constraints: int = 32
    """Maximum articulation-contact constraints per world."""

    pgs_warmstart: bool = False
    """Whether to warm-start impulses from the previous frame."""

    pgs_mode: Literal["dense", "split", "matrix_free"] = "split"
    """Constraint solve layout.

    ``"dense"`` builds the full Delassus matrix, ``"split"`` uses dense articulated rows and matrix-free free-body
    rows, and ``"matrix_free"`` avoids constructing the dense matrix.
    """

    pgs_schedule: Literal["interleaved", "contact_then_internal", "physx_grasp"] = "interleaved"
    """Ordering schedule for matrix-free contact and internal rows.

    Values other than ``"interleaved"`` require :attr:`pgs_mode` to be ``"matrix_free"``.
    """

    friction_mode: Literal["current", "bisection", "bisection_desaxce", "coulomb_newton"] = "current"
    """Per-row Coulomb friction strategy.

    Values other than ``"current"`` require :attr:`pgs_mode` to be ``"matrix_free"``.
    """

    mf_max_constraints: int = 512
    """Maximum matrix-free constraints per world."""

    cholesky_kernel: Literal["tiled", "loop", "auto"] = "auto"
    """Cholesky kernel: ``"tiled"``, ``"loop"``, or ``"auto"``."""

    trisolve_kernel: Literal["tiled", "loop", "auto"] = "auto"
    """Tri-solve kernel: ``"tiled"``, ``"loop"``, or ``"auto"``."""

    hinv_jt_kernel: Literal["tiled", "par_row", "auto"] = "auto"
    """H^-1 J^T kernel: ``"tiled"``, ``"par_row"``, or ``"auto"``."""

    delassus_kernel: Literal["tiled", "par_row_col", "auto"] = "auto"
    """Delassus accumulation kernel: ``"tiled"``, ``"par_row_col"``, or ``"auto"``."""

    pgs_kernel: Literal["loop", "tiled_row", "tiled_contact", "streaming"] = "tiled_row"
    """PGS kernel: ``"loop"``, ``"tiled_row"``, ``"tiled_contact"``, or ``"streaming"``."""

    delassus_chunk_size: int | None = None
    """Chunk size for streaming Delassus kernels. ``None`` selects automatically."""

    pgs_chunk_size: int | None = None
    """Chunk size for streaming PGS kernels. ``None`` selects default behavior."""

    small_dof_threshold: int = 12
    """DOF threshold for automatic kernel selection."""

    use_parallel_streams: bool = True
    """Whether to dispatch size groups on separate CUDA streams."""

    double_buffer: bool = True
    """Whether to use double buffering for solver internals."""

    nvtx: bool = False
    """Whether to enable NVTX profiling ranges."""

    pgs_debug: bool = False
    """Whether to enable PGS debug diagnostics."""

    drive_mode: Literal["augmented", "physx_pgs"] = "augmented"
    """Joint drive model.

    ``"augmented"`` uses implicit PD, while ``"physx_pgs"`` uses PhysX-style rows and requires
    :attr:`pgs_mode` to be ``"matrix_free"``.
    """

    effort_limit_mode: Literal["actuator"] = "actuator"
    """Effort-limit policy. FeatherPGS supports actuator-drive clamping only."""

    serial_kernel_block_dim: int = 256
    """Block dimension for serial matrix-free kernels."""

    tile_threads: int = 64
    """Thread count for tiled kernels."""

    row_watermark: bool = False
    """Whether to enable row high-water watermark diagnostics/allocation path."""
