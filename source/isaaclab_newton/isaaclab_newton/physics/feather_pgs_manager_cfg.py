# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Newton FeatherPGS solver."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from isaaclab.utils import configclass

from .newton_manager_cfg import NewtonSolverCfg

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager


@configclass
class FeatherPGSSolverCfg(NewtonSolverCfg):
    """Configuration for Newton's experimental FeatherPGS solver.

    FeatherPGS integrates articulations in reduced coordinates and resolves contacts, joint limits and,
    optionally, joint drives with projected Gauss-Seidel iterations. It uses Newton's collision pipeline,
    configured through :attr:`NewtonCfg.collision_cfg`. Defaults match the solver's constructor; see
    :class:`newton.solvers.SolverFeatherPGS` for the full description of each option.
    """

    class_type: type[NewtonManager] | str = "{DIR}.feather_pgs_manager:NewtonFeatherPGSManager"
    """Manager class for the FeatherPGS solver."""

    solver_type: str = "feather_pgs"
    """Solver type. Can be "feather_pgs"."""

    pgs_mode: Literal["matrix_free", "split"] = "matrix_free"
    """Constraint solve.

    ``"matrix_free"`` sweeps every row against the articulated-body response and requires CUDA.
    ``"split"`` assembles the articulated rows of each world into a dense system and runs on CPU and CUDA.
    """

    pgs_iterations: int = 12
    """Number of PGS iterations per step."""

    pgs_beta: float = 0.2
    """Position-error correction factor (Baumgarte) of contact and limit rows."""

    pgs_cfm: float = 1.0e-6
    """Constraint-force mixing added to the diagonal of every row."""

    pgs_omega: float = 1.0
    """Successive over-relaxation factor of the PGS sweep."""

    update_mass_matrix_interval: int = 1
    """Number of steps between mass-matrix updates."""

    drive_mode: Literal["augmented", "physx_pgs"] = "augmented"
    """Joint drive formulation.

    ``"augmented"`` integrates drives implicitly in the mass matrix. ``"physx_pgs"`` solves each driven DOF
    as a constraint row (``pgs_mode="matrix_free"`` only).
    """

    enable_joint_limits: bool = False
    """Whether to enforce joint position limits as unilateral constraint rows."""

    joint_limit_activation_gap: float = float("inf")
    """Distance [m or rad, depending on joint type] from a finite joint limit at which its row is activated.

    ``float("inf")`` keeps a row for every finite limit on every step.
    """

    enable_joint_velocity_limits: bool = False
    """Whether to enforce joint velocity limits as constraint rows (``pgs_mode="matrix_free"`` only)."""

    velocity_limit_activation_fraction: float = 0.0
    """Fraction of a finite velocity limit at which its rows are activated; ``0.0`` keeps them always active."""

    fuse_joint_velocity_limits: bool = True
    """Whether velocity limits of driven joints are fused into the drive rows when ``drive_mode="physx_pgs"``."""

    dense_max_constraints: int = 32
    """Capacity of constraint rows involving articulated bodies, per world."""

    mf_max_constraints: int = 512
    """Capacity of free-body contact rows, per world."""

    warn_constraint_overflow: bool = True
    """Whether to print a device-side warning the first time a world exceeds a row capacity."""

    raise_on_constraint_overflow: bool = False
    """Whether to raise after any step that dropped constraint rows or contacts.

    The check reads the solver's overflow status on the host after every step.
    """

    pgs_velocity_iterations: int = 0
    """Velocity-only iterations after the position solve (``pgs_mode="matrix_free"`` only)."""

    pgs_velocity_drive_mode: Literal["freeze", "active"] = "freeze"
    """Drive-row handling in the velocity-only iterations."""

    pgs_schedule: Literal["interleaved", "contact_then_internal", "physx_grasp"] = "interleaved"
    """Row order of the matrix-free sweeps."""

    friction_mode: Literal["current", "bisection", "bisection_desaxce", "coulomb_newton"] = "current"
    """Coulomb friction projection of the matrix-free solve."""

    pgs_warmstart: bool = False
    """Whether to start each step from the previous step's impulses (``pgs_mode="matrix_free"`` only)."""

    pgs_warmstart_decay: float = 1.0
    """Scale applied to warm-start impulses."""

    friction_anchor_beta: float | None = None
    """Stabilization gain of persistent friction anchors; ``0.0`` selects point friction.

    ``None`` selects ``0.2`` with ``pgs_mode="matrix_free"`` and point friction with ``pgs_mode="split"``.
    """

    pgs_contact_regularization: float = 0.0
    """Proximal regularization of matrix-free normal contact rows; ``0.0`` keeps the rigid contact law."""

    restitution_velocity_threshold: float = 0.5
    """Minimum incident normal speed [m/s] for a rebound."""

    contact_speculative_scale: float = 1.0
    """Fraction of a positive contact gap that the contact may close during the step."""

    contact_gap_gate: float = 0.0
    """Contacts with a separation [m] above this gate get no constraint row; ``0.0`` disables the gate."""

    contact_friction_gap_threshold: float = float("inf")
    """Separation [m] above which a contact gets a normal row only; ``inf`` gives every contact friction."""

    contact_shared_anchor: bool = False
    """Whether to apply every contact row at the midpoint of the two witness points.

    Friction patch rows keep their patch anchors, so with patches this moves only the normal rows.
    """

    contact_friction_shared_anchor: bool = False
    """Whether to apply point-friction rows at the midpoint of the two witness points."""

    contact_torsion_radius: float = 0.0
    """Effective spin radius [m] of contact torsion on articulated bodies; ``0.0`` disables torsion.

    Torsion requires a CUDA device and ``pgs_warmstart=False``.
    """

    contact_torsion_shape_patterns: tuple[str, ...] | None = None
    """Regular expressions full-matched against shape labels to select the contacts that get torsion.

    ``None`` selects every shape. A pattern that matches no shape raises at solver construction.
    """

    contact_torsion_device: bool = False
    """Whether to prepare the torsion rows on the device.

    Host preparation reads the contacts back every step, so the manager steps eagerly instead of capturing a CUDA
    graph. Device preparation supports CUDA graph capture.
    """

    enable_sleeping: bool = False
    """Whether supported, quiet islands of articulations sleep and freeze their published state until woken.

    Requires ``pgs_mode="matrix_free"``, ``pgs_warmstart=False`` and ``pgs_velocity_iterations=0``.
    """

    sleep_linear_threshold: float = 0.05
    """Body center-of-mass speed [m/s] below which a body is quiet."""

    sleep_angular_threshold: float = 0.15
    """Body angular speed [rad/s] below which a body is quiet."""

    sleep_quiet_time: float = 0.5
    """Quiet, supported interval [s] after which an island sleeps."""

    sleep_skip_constraints: bool = True
    """Whether to skip the constraint rows and articulated dynamics of sleeping islands."""

    articulated_contact_response: Literal["immediate", "propagation", "propagation-fused", "propagation-colored"] = (
        "immediate"
    )
    """How articulated contact rows are resolved against the articulated-body response.

    The ``propagation`` modes require ``pgs_mode="matrix_free"``.
    """

    parallel_tree: bool = False
    """Whether to traverse independent branches of each articulation tree in parallel."""

    use_parallel_streams: bool = False
    """Whether to dispatch per-size articulation groups on separate CUDA streams."""

    double_buffer: bool = False
    """Whether to alternate two sets of grouped mass-matrix and Jacobian buffers across steps on CUDA."""

    speculative_contact_gap_max: float | None = None
    """Maximum velocity-based extension [m] of the rigid-contact search, for predictive contacts.

    The collision pipeline then predicts contacts over the full physics step. ``None`` disables predictive
    contacts.
    """
