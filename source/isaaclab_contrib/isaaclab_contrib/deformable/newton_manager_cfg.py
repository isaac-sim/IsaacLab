# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for VBD, coupled solver, and global Newton model parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from isaaclab_newton.physics import FeatherstoneSolverCfg, MJWarpSolverCfg, NewtonSolverCfg

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager


@configclass
class NewtonModelCfg:
    """Global Newton model parameters applied after builder finalization.

    These control model-level contact behavior shared across all objects.
    """

    soft_contact_ke: float = 1.0e3
    """Body-particle and particle self-contact stiffness [N/m].

    Effective per-contact stiffness is the average of this value and the rigid
    shape's material stiffness.
    """

    soft_contact_kd: float = 1.0e-2
    """Body-particle contact damping [N*s/m]."""

    soft_contact_mu: float = 0.5
    """Body-particle contact friction coefficient [dimensionless].

    Effective per-contact friction is ``sqrt(soft_contact_mu * shape_mu)``, where
    ``shape_mu`` is the rigid shape's own friction coefficient (from its per-asset
    material or :attr:`~isaaclab_newton.physics.NewtonShapeCfg.mu` default), not
    set by this config.
    """


@configclass
class NewtonModelSolverCfg(NewtonSolverCfg):
    """Base for solver configs whose manager applies :class:`NewtonModelCfg` to the finalized model.

    TODO: Temporary. This base only exists because :class:`NewtonModelCfg` lives in
    ``isaaclab_contrib`` while :class:`NewtonSolverCfg` is in ``isaaclab_newton`` core.
    Once these model params move into core, ``model_cfg`` should live on
    :class:`NewtonSolverCfg` (or ``NewtonCfg``) directly and this class can be removed.
    """

    model_cfg: NewtonModelCfg | None = None
    """Global Newton model parameters applied after builder finalization."""


@configclass
class VBDSolverCfg(NewtonModelSolverCfg):
    """Configuration for the Vertex Block Descent (VBD) solver.

    Supports cloth, soft bodies, and coupled rigid-body systems. Requires
    ``ModelBuilder.color()`` before ``finalize()`` to build the vertex coloring.
    """

    class_type: type[NewtonManager] | str = "{DIR}.vbd_manager:NewtonVBDManager"
    """Manager class for the VBD solver."""

    iterations: int = 10
    """Number of VBD iterations per substep."""

    integrate_with_external_rigid_solver: bool = False
    """Whether rigid bodies are integrated by an external solver (one-way coupling).

    Set to ``True`` when coupling cloth with a separate rigid-body solver so VBD
    only integrates the cloth particles.
    """

    particle_enable_self_contact: bool = False
    """Whether to enable VBD deformable's self-contact."""

    particle_self_contact_radius: float = 0.005
    """Particle radius used for self-contact detection [m]."""

    particle_self_contact_margin: float = 0.005
    """Self-contact detection margin [m]. Should be >= particle_self_contact_radius."""

    particle_collision_detection_interval: int = -1
    """How often particle self-contact detection is applied.

    ``< 0``: once before initialization. ``0``: once before and once after
    initialization. ``k >= 1``: before every ``k`` VBD iterations.
    """

    particle_vertex_contact_buffer_size: int = 32
    """Preallocation size for each vertex's vertex-triangle collision buffer."""

    particle_edge_contact_buffer_size: int = 64
    """Preallocation size for each edge's edge-edge collision buffer."""

    particle_topological_contact_filter_threshold: int = 2
    """Maximum topological distance (in rings) below which self-contacts are discarded.

    Only used when ``particle_enable_self_contact`` is ``True``. Values > 3
    significantly increase computation time.
    """

    particle_rest_shape_contact_exclusion_radius: float = 0.0
    """Rest-configuration separation threshold for filtering close primitives [m].

    Only used when ``particle_enable_self_contact`` is ``True``.
    """

    rigid_contact_k_start: float = 1.0e2
    """Initial stiffness seed for all rigid body contacts [N/m]."""


@configclass
class CoupledMJWarpVBDSolverCfg(NewtonModelSolverCfg):
    """Configuration for the coupled MJWarp + VBD solver.

    Alternates a rigid-body solver (:class:`MJWarpSolverCfg`) and VBD per substep.
    The coupling direction is controlled by :attr:`coupling_mode`.
    """

    class_type: type[NewtonManager] | str = "{DIR}.coupled_mjwarp_vbd_manager:NewtonCoupledMJWarpVBDManager"
    """Manager class for the coupled MJWarp + VBD solver."""

    rigid_solver_cfg: MJWarpSolverCfg = MJWarpSolverCfg()
    """Rigid-body sub-solver configuration."""

    soft_solver_cfg: VBDSolverCfg = VBDSolverCfg(integrate_with_external_rigid_solver=True)
    """VBD sub-solver configuration for cloth/particle dynamics."""

    coupling_mode: Literal["one_way", "two_way"] = "two_way"
    """Coupling direction between the rigid and VBD solvers.

    - ``"one_way"``: Rigid -> soft only.
    - ``"two_way"``: Same-substep two-way coupling with normal + Coulomb friction.
    """


@configclass
class CoupledFeatherstoneVBDSolverCfg(NewtonModelSolverCfg):
    """Configuration for the coupled Featherstone + VBD solver.

    Alternates a rigid-body solver (:class:`FeatherstoneSolverCfg`) and VBD per
    substep. The coupling direction is controlled by :attr:`coupling_mode`.
    """

    class_type: type[NewtonManager] | str = "{DIR}.coupled_featherstone_vbd_manager:NewtonCoupledFeatherstoneVBDManager"
    """Manager class for the coupled Featherstone + VBD solver."""

    rigid_solver_cfg: FeatherstoneSolverCfg = FeatherstoneSolverCfg()
    """Rigid-body sub-solver configuration."""

    soft_solver_cfg: VBDSolverCfg = VBDSolverCfg(integrate_with_external_rigid_solver=True)
    """VBD sub-solver configuration for cloth/particle dynamics."""

    coupling_mode: Literal["one_way", "two_way", "kinematic"] = "kinematic"
    """Coupling direction between the rigid and VBD solvers.

    Accepts the same values as :attr:`CoupledMJWarpVBDSolverCfg.coupling_mode`,
    plus ``"kinematic"`` (rigid -> soft only, rigid bodies kinematically updated).
    """
