# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for Newton coupled solvers.

Defines a generic :class:`CoupledSolverCfg` base and two algorithm-specific
subclasses, :class:`CoupledProxySolverCfg` (lagged-impulse virtual-proxy
coupling) and :class:`CoupledAdmmSolverCfg` (linearized ADMM coupling).
Both are consumed by :class:`~isaaclab_contrib.coupling.coupled_manager.NewtonCoupledSolverManager`,
which dispatches to the matching Newton experimental coupled solver.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab_newton.physics import NewtonSolverCfg

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager


@configclass
class CoupledSolverCfg(NewtonSolverCfg):
    """Base configuration for any Newton experimental coupled solver.

    Wraps a pair of sub-solvers, partitioning the Newton model's bodies (and
    derived joints/shapes) between a source (``src``) and destination (``dst``)
    entry. All particles are routed to the destination entry.

    Body selectors are either :class:`~isaaclab.managers.SceneEntityCfg`
    (scoped by the asset's ``prim_path``, optionally narrowed by ``body_names``
    full-matched against body short names) or raw prim-path regex strings
    (e.g. ``"/World/envs/env_.*/MyCube"``) matched against ``model.body_label``
    via ``^<string>(/|$)``.

    The concrete coupling algorithm is selected by subclassing — use
    :class:`CoupledProxySolverCfg` for :class:`newton.solvers.experimental.coupled.SolverCoupledProxy`
    and :class:`CoupledAdmmSolverCfg` for :class:`newton.solvers.experimental.coupled.SolverCoupledAdmm`.
    """

    class_type: type[NewtonManager] | str = "{DIR}.coupled_manager:NewtonCoupledSolverManager"
    """Manager class for the coupled solver."""

    requires_graph_coloring: bool = True
    """VBD-style graph coloring is built when either sub-solver needs it (kept as a default)."""

    src_solver_cfg: NewtonSolverCfg = MISSING
    """Source sub-solver configuration (e.g. :class:`~isaaclab_newton.physics.MJWarpSolverCfg`)."""

    dst_solver_cfg: NewtonSolverCfg = MISSING
    """Destination sub-solver configuration (e.g. :class:`~isaaclab_contrib.deformable.VBDSolverCfg`)."""

    src_bodies: list[SceneEntityCfg | str] = []
    """Selectors whose bodies/joints/shapes go to the source entry.

    Joints inherit their child body's owner; shapes inherit their body's
    owner; static shapes (``body == -1``) always go to the destination entry.
    """

    dst_bodies: list[SceneEntityCfg | str] = []
    """Selectors routed to the destination entry (see :attr:`src_bodies`)."""


@configclass
class CoupledProxySolverCfg(CoupledSolverCfg):
    """Configuration for the lagged-impulse virtual-proxy coupled solver.

    Wraps Newton's :class:`newton.solvers.experimental.coupled.SolverCoupledProxy`.
    Selected source bodies are exposed as proxy bodies in the destination view
    so the destination solver detects contacts against them and returns
    feedback wrenches to the source via lagged impulses.
    """

    proxy_bodies: list[SceneEntityCfg | str] = []
    """Selectors naming source bodies to expose as proxies in the destination view.

    For :class:`SceneEntityCfg` entries, ``body_names`` is **required**
    (proxies are a subset, not the whole asset); raw strings are accepted
    as-is. Matched bodies are filtered to those owning at least one
    :attr:`newton.ShapeFlags.COLLIDE_SHAPES` shape. Empty list = no proxies.
    """

    proxy_mode: str = "lagged"
    """Proxy transfer mode passed to :class:`newton.solvers.experimental.coupled.SolverCoupledProxy.Proxy`.

    - ``"lagged"``: syncs source begin poses and end velocities, then rewinds
      lagged feedback before the destination solve.
    - ``"staggered"``: syncs source end poses and end velocities directly.
    """

    proxy_iterations: int = 1
    """Number of relaxation iterations per coupled substep."""

    proxy_collide_interval: int = 1
    """Collision-detection refresh interval (in proxy passes)."""

    proxy_mass_scale: float = 1.0
    """Mass / inertia scale applied to destination proxy bodies (virtual inertia) [dimensionless]."""


@configclass
class CoupledAdmmSolverCfg(CoupledSolverCfg):
    """Configuration for the linearized ADMM coupled solver.

    Wraps Newton's :class:`newton.solvers.experimental.coupled.SolverCoupledAdmm`,
    which enforces inter-solver constraints via a penalty method with explicit
    contact pairs between the source and destination entries.
    """

    iterations: int = 5
    """Number of ADMM dual iterations per coupled substep."""

    rho: float = 1.0
    """ADMM penalty parameter [dimensionless]. Larger values stiffen the constraint enforcement."""

    gamma: float = 0.0
    """Proximal mass scaling parameter [dimensionless]. Adds virtual mass to the sub-solvers."""

    baumgarte: float = 0.0
    """Position-error correction fraction [dimensionless], in ``[0, 1)``. Stabilization bias."""

    joint_stiffness: float = 1.0e4
    """Translational joint-attachment stiffness [N/m]."""

    joint_damping: float = 0.0
    """Translational joint-attachment damping [N*s/m]."""

    joint_angular_stiffness: float = 1.0e4
    """Angular joint-attachment stiffness [N*m/rad]."""

    joint_angular_damping: float = 0.0
    """Angular joint-attachment damping [N*m*s/rad]."""

    enable_contacts: bool = True
    """Whether to register a contact pair between the source and destination entries.

    When ``True``, a single :class:`newton.solvers.experimental.coupled.SolverCoupledAdmm.ContactPair`
    is added with the configured :attr:`contact_distance` / :attr:`detection_margin`.
    """

    contact_distance: float | None = None
    """Per-pair contact distance override [m]. ``None`` uses Newton's default."""

    detection_margin: float | None = None
    """Per-pair detection margin override [m]. ``None`` uses Newton's default."""
