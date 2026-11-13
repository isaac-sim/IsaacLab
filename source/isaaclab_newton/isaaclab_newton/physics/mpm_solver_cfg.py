# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton's implicit MPM solver."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .newton_manager_cfg import NewtonSolverCfg


@configclass
class MPMSleepingCfg:
    """Native GPU particle activity settings, applied after scene preparation.

    Requires :attr:`MPMSolverCfg.reuse_grid_topology` to enable guarded sparse execution.
    Thresholds describe approximate equilibrium, not a guarantee of force equivalence to full MPM.
    Newton retains particle identity and history while settled regions sleep. Configure with
    :func:`~isaaclab_newton.physics.configure_mpm_sleeping` outside graph capture;
    all subsequent activity planning runs on GPU.
    """

    enabled: bool = True
    """Enable sleeping, or restore eligible particles when false."""

    patch_size: float | None = None
    """Activity patch edge length [m]; None selects Newton's two-voxel default."""

    velocity_threshold: float = 0.01
    """Maximum quiet particle speed [m/s]."""

    strain_threshold: float | None = None
    """Maximum quiet velocity-gradient norm [1/s]; None derives it from speed and voxel size."""

    stress_threshold: float = 0.05
    """Maximum relative quiet stress change per step."""

    stress_filter_time: float = 0.0
    """Signed stress averaging time [s]; zero disables filtering."""

    stress_min_strain_rate: float = 0.02
    """Minimum strain rate [1/s] for filtered stress changes to wake a patch."""

    sleep_time: float = 0.5
    """Continuous quiet time before sleeping [s]."""

    collider_margin: float | None = None
    """Additional collider wake distance [m]; None selects two voxels."""

    collider_lookahead: float = 0.1
    """Collider and particle motion prediction interval [s]."""

    planning_interval: int = 1
    """Substeps between full planning passes; urgent motion and forces are checked every substep."""

    min_moving_particles: int = 1
    """Ordinary translating particles required to keep a patch awake."""

    spill_velocity_threshold: float = 0.3
    """Speed [m/s] above which particles request a swept wake every substep."""


@configclass
class MPMWorkingSetCfg:
    """Bounded native evaluation buffers, applied after settling and activity classification.

    All particle states remain on GPU. ``page_size=0`` compacts individual particle IDs;
    positive values cache original-ID pages. Paging does not provide out-of-core storage.
    Requires :attr:`MPMSolverCfg.reuse_grid_topology` for guarded sparse execution.
    Configure with :func:`~isaaclab_newton.physics.configure_mpm_working_set` using the current solver input state.
    """

    particle_capacity: int = MISSING
    """Maximum evaluation slots, including inactive page lanes."""

    page_size: int = 0
    """Original particle IDs per page; zero selects individual-ID compaction."""

    max_active_cell_count: int | None = None
    """Optional replacement sparse-grid cell capacity."""

    max_leaf_node_count: int | None = None
    """Optional replacement sparse-grid leaf-node capacity."""


@configclass
class MPMSolverCfg(NewtonSolverCfg):
    """Configuration for Newton's implicit Material Point Method (MPM) solver.

    The implicit MPM solver advances particle materials and treats rigid geometry
    as colliders. It is not a rigid-body or articulation dynamics solver.
    """

    class_type: type | str = "{DIR}.mpm_solver:MPMSolver"

    solver_type: str = "implicit_mpm"
    """Solver type. Can be "implicit_mpm"."""

    sleeping: MPMSleepingCfg | None = None
    """Deferred native activity settings; None leaves sleeping unconfigured.

    Call :func:`~isaaclab_newton.physics.configure_mpm_sleeping` after scene preparation. These runtime settings are
    intentionally not applied by solver construction or authored to USD.
    """

    working_set: MPMWorkingSetCfg | None = None
    """Deferred bounded evaluation settings; None leaves the solver unchanged.

    Call :func:`~isaaclab_newton.physics.configure_mpm_working_set` after classification and before final capture.
    A new solver starts with Newton's full evaluation pool.
    Initial settling and resets can wake more particles than the configured capacity.
    """

    reuse_grid_topology: bool = False
    """Enable guarded sparse execution and reuse unchanged grid structure where supported.

    Required for sleeping and bounded working sets. Numerical values still update each step; point bases
    retain device guards but rebuild structure. Requires the companion Newton and Warp feature builds.
    Opt-out remains compatible with the pinned dependency.
    """

    # numerics
    max_iterations: int = 250
    """Maximum number of iterations for the rheology solver."""

    tolerance: float = 1.0e-4
    """Tolerance for the rheology solver."""

    solver: str | tuple[str, ...] = "auto"
    """Rheology solver, or an ordered warm-start sequence of solvers.

    Values use the canonical tokens defined by ``NewtonMPMSceneAPI``. ``"auto"``
    lets Newton pick the solver from the velocity basis. Pass a tuple such as
    ``("conjugate-residual", "gauss-seidel")`` to warm-start solvers left-to-right.
    """

    warmstart_mode: Literal["none", "auto", "particles", "grid", "smoothed"] = "auto"
    """Warm-start mode for the rheology solver."""

    collider_velocity_mode: Literal["forward", "backward", "instantaneous", "finite_difference"] = "forward"
    """Collider velocity computation mode.

    ``"instantaneous"`` is deprecated in favor of ``"forward"``, and
    ``"finite_difference"`` is deprecated in favor of ``"backward"``.
    """

    # grid
    voxel_size: float = 0.1
    """Size of the MPM grid voxels [m]."""

    grid_type: Literal["sparse", "dense", "fixed"] = "sparse"
    """Type of grid to use."""

    grid_padding: int = 0
    """Number of empty cells to add around particles when allocating the grid."""

    max_active_cell_count: int = -1
    """Maximum active grid-cell count shared by all worlds.

    A positive value reserves persistent capacity for rebuildable sparse grids and bounds active
    subsets of dense and fixed grids. ``-1`` keeps sparse allocation dynamic and uses exact active
    counts for the other grid types.
    """

    max_leaf_node_count: int = -1
    """Maximum sparse-grid leaf-node count shared by all worlds.

    ``-1`` lets Newton derive the capacity from :attr:`max_active_cell_count`.
    """

    max_lower_node_count: int = -1
    """Maximum sparse-grid lower internal-node count shared by all worlds.

    ``-1`` lets Newton derive the capacity from the initial topology.
    """

    max_upper_node_count: int = -1
    """Maximum sparse-grid upper internal-node count shared by all worlds.

    ``-1`` lets Newton derive the capacity from the initial topology.
    """

    separate_worlds: bool = False
    """Whether each Newton world uses an independent local MPM grid environment."""

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

    When ``True``, :meth:`step` calls
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
