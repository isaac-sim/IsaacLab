# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton's implicit MPM solver."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from isaaclab.utils.configclass import configclass

from .newton_manager_cfg import NewtonSolverCfg

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager


@configclass
class MPMSolverCfg(NewtonSolverCfg):
    """Configuration for Newton's implicit Material Point Method (MPM) solver.

    The implicit MPM solver advances particle materials and treats rigid geometry
    as colliders. It is not a rigid-body or articulation dynamics solver.
    """

    class_type: type[NewtonManager] | str = "{DIR}.mpm_manager:NewtonMPMManager"
    """Manager class for the implicit MPM solver."""

    solver_type: str = "implicit_mpm"
    """Solver type. Can be "implicit_mpm"."""

    # numerics
    max_iterations: int = 250
    """Maximum number of iterations for the rheology solver."""

    tolerance: float = 1.0e-4
    """Tolerance for the rheology solver."""

    solver: str | tuple[str, ...] = "auto"
    """Rheology solver, or an ordered warm-start sequence of solvers.

    ``"auto"`` lets Newton pick the solver from the velocity basis (``"gs"`` for
    ``Q1``, ``"gs-batched"`` for ``B2``/``B3``). Other accepted values include
    ``"gauss-seidel"``, ``"jacobi"``, ``"cg"``, ``"cr"``, and ``"gmres"``; pass a
    tuple such as ``("cr", "gs")`` to warm-start solvers left-to-right.
    """

    warmstart_mode: Literal["none", "auto", "particles", "grid", "smoothed"] = "auto"
    """Warm-start mode for the rheology solver."""

    collider_velocity_mode: Literal["forward", "backward", "instantaneous", "finite_difference"] = "forward"
    """Collider velocity computation mode."""

    # grid
    voxel_size: float = 0.1
    """Size of the MPM grid voxels [m]."""

    grid_type: Literal["sparse", "dense", "fixed"] = "sparse"
    """Type of grid to use."""

    grid_padding: int = 0
    """Number of empty cells to add around particles when allocating the grid."""

    max_active_cell_count: int = -1
    """Maximum active cell count for dense-grid active subsets. ``-1`` means unlimited."""

    transfer_scheme: Literal["apic", "pic"] = "apic"
    """Particle-grid transfer scheme."""

    integration_scheme: Literal["pic", "gimp"] = "pic"
    """Integration scheme controlling shape-function support."""

    # material / background
    critical_fraction: float = 0.0
    """Dimensionless fraction under which the yield surface collapses."""

    air_drag: float = 1.0
    """Numerical drag for background air."""

    # experimental
    collider_normal_from_sdf_gradient: bool = False
    """Whether collider normals are computed from SDF gradients rather than closest points."""

    collider_basis: str = "S2"
    """Collider basis function, such as ``"S2"`` or ``"Q1"``."""

    strain_basis: str = "P0"
    """Strain basis function, such as ``"P0"``, ``"P1d"``, ``"Q1"``, or ``"Q1d"``."""

    velocity_basis: str = "Q1"
    """Velocity basis function, such as ``"Q1"``, ``"B2"``, or ``"B3"``."""

    # collision handling (applied by the Isaac Lab manager, not the Newton solver config)
    project_outside_colliders: bool = False
    """Whether to hard-project particles out of collider interiors after each substep.

    When ``True``, :class:`~isaaclab_newton.physics.NewtonMPMManager` calls
    :meth:`SolverImplicitMPM.project_outside` immediately after every solver
    substep: it applies a Coulomb response and pushes particles that drifted into
    a collider back onto its surface. The implicit solve already resolves
    colliders at the grid level; this is the particle-level correction that stops
    material from slowly settling inside colliders, mirroring Newton's MPM
    examples. Leave it ``False`` for collider-free scenes to skip a per-substep
    projection pass over every particle.

    This is a manager-level stepping option and is intentionally **not** part of
    ``SolverImplicitMPM.Config``.
    """
