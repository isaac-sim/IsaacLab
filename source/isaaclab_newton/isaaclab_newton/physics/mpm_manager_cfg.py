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
    from newton.solvers import SolverImplicitMPM

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
    """Solver type metadata. Can be ``"implicit_mpm"``."""

    # numerics
    max_iterations: int = 250
    """Maximum number of iterations for the rheology solver."""

    tolerance: float = 1.0e-4
    """Tolerance for the rheology solver."""

    solver: str | tuple[str, ...] = "gauss-seidel"
    """Solver to use for the rheology solve, or an ordered warm-start sequence.

    Newton's upstream default is ``"auto"``, but Isaac Lab uses the explicit
    ``"gauss-seidel"`` spelling so environments with older Newton 1.2.0.dev0
    installs can step MPM without relying on Newton's newer auto-resolution helper.
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

    def to_solver_config(self) -> SolverImplicitMPM.Config:
        """Build a :class:`SolverImplicitMPM.Config` from this configuration.

        Returns:
            A ``SolverImplicitMPM.Config`` instance ready for solver construction.
        """
        from newton.solvers import SolverImplicitMPM

        cfg = SolverImplicitMPM.Config()
        for key, value in self.to_dict().items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg
