# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton FeatherPGS physics manager."""

from __future__ import annotations

import math

from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass

from .newton_manager_cfg import NewtonSolverCfg

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager


@configclass
class FeatherPGSSolverCfg(NewtonSolverCfg):
    """Configuration for Newton's FeatherPGS reduced-coordinate solver."""

    class_type: type[NewtonManager] | str = "{DIR}.feather_pgs_manager:NewtonFeatherPGSManager"
    """Manager class for the FeatherPGS solver."""

    solver_type: str = "feather_pgs"
    """Solver type metadata."""

    angular_damping: float = 0.05
    """Angular damping factor."""

    update_mass_matrix_interval: int = 1
    """How often to update the mass matrix, in simulation steps."""

    friction_smoothing: float = 1.0
    """Huber smoothing value for friction normalization."""

    contact_friction_gap_threshold: float = math.inf
    """Gap threshold [m] for enabling tangential contact friction rows."""

    contact_friction_position_iterations: int = -1
    """Number of position iterations used for contact-friction anchoring."""

    contact_friction_shared_anchor: bool = False
    """Whether contact-friction rows use shared anchors."""

    contact_friction_anchor_limit: int = 0
    """Maximum stored contact-friction anchors per world."""

    contact_friction_scale: float = 1.0
    """Scale factor applied to contact friction limits."""

    contact_shared_anchor: bool = False
    """Whether normal contact rows use shared anchors."""

    enable_contact_friction: bool = True
    """Whether to enable Coulomb contact friction in PGS."""

    enable_joint_limits: bool = False
    """Whether to enforce joint position limits as unilateral PGS constraints."""

    joint_limit_activation_gap: float = math.inf
    """Distance [m or rad, depending on joint type] from a joint limit at which rows activate."""

    enable_joint_velocity_limits: bool = False
    """Whether to enforce per-DOF joint velocity limits as PGS constraints."""

    velocity_limit_activation_fraction: float = 0.0
    """Fraction of each joint velocity limit at which velocity-limit rows activate."""

    pgs_iterations: int = 12
    """Number of Gauss-Seidel iterations per simulation step."""

    pgs_velocity_iterations: int = 0
    """Number of velocity-level Gauss-Seidel iterations per simulation step."""

    pgs_beta: float = 0.2
    """ERP-style position correction factor."""

    pgs_cfm: float = 1.0e-6
    """Compliance/regularization on the Delassus diagonal."""

    dense_contact_compliance: float = 0.0
    """Normal contact compliance for dense articulated contact rows."""

    speculative_dense_contact_compliance: float = 0.0
    """Normal compliance for speculative dense contact rows."""

    pgs_omega: float = 1.0
    """Successive over-relaxation factor for PGS."""

    pgs_velocity_omega: float | None = None
    """Velocity-level successive over-relaxation factor. ``None`` uses the solver default."""

    pgs_velocity_drive_mode: str = "freeze"
    """Velocity-drive handling mode for velocity-level PGS rows."""

    dense_max_constraints: int = 32
    """Maximum articulation-contact constraints per world."""

    pgs_warmstart: bool = False
    """Whether to warm-start impulses from the previous frame."""

    pgs_mode: str = "split"
    """PGS mode: ``"dense"``, ``"split"``, or ``"matrix_free"``."""

    pgs_schedule: str = "interleaved"
    """Ordering schedule for contact/internal PGS rows."""

    friction_mode: str = "current"
    """Per-row Coulomb friction strategy for the matrix-free PGS path."""

    mf_max_constraints: int = 512
    """Maximum matrix-free constraints per world."""

    cholesky_kernel: str = "auto"
    """Cholesky kernel: ``"tiled"``, ``"loop"``, or ``"auto"``."""

    trisolve_kernel: str = "auto"
    """Tri-solve kernel: ``"tiled"``, ``"loop"``, or ``"auto"``."""

    hinv_jt_kernel: str = "auto"
    """H^-1 J^T kernel: ``"tiled"``, ``"par_row"``, or ``"auto"``."""

    delassus_kernel: str = "auto"
    """Delassus accumulation kernel: ``"tiled"``, ``"par_row_col"``, or ``"auto"``."""

    pgs_kernel: str = "tiled_row"
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

    drive_mode: str = "augmented"
    """Drive model used by the FeatherPGS solver."""

    effort_limit_mode: str = "actuator"
    """Effort-limit clamp mode forwarded to Newton's FeatherPGS solver."""

    serial_kernel_block_dim: int = 256
    """Block dimension for serial matrix-free kernels."""

    tile_threads: int = 64
    """Thread count for tiled kernels."""

    row_watermark: bool = False
    """Whether to enable row high-water watermark diagnostics/allocation path."""
