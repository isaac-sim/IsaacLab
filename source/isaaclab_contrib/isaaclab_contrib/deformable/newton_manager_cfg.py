# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for VBD, coupled solver, and global Newton model parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_newton.physics import FeatherstoneSolverCfg, MJWarpSolverCfg, NewtonCfg, NewtonSolverCfg

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager

    from isaaclab.scene import InteractiveSceneCfg


@configclass
class VBDSolverCfg(NewtonSolverCfg):
    """Configuration for the Vertex Block Descent (VBD) solver.

    Supports particle simulation (cloth, soft bodies) and coupled rigid-body systems.
    Requires ``ModelBuilder.color()`` to be called before ``finalize()`` to build
    the parallel vertex colouring needed by the solver.
    """

    class_type: type[NewtonManager] | str = "{DIR}.vbd_manager:NewtonVBDManager"
    """Manager class for the VBD solver."""

    solver_type: str = "vbd"

    iterations: int = 10
    """Number of VBD iterations per substep."""

    integrate_with_external_rigid_solver: bool = False
    """Whether rigid bodies are integrated by an external solver (one-way coupling).

    Set to ``True`` when coupling cloth with a separate rigid-body solver
    (e.g. ``SolverFeatherstone``) so that VBD only integrates the cloth particles.
    """

    particle_enable_self_contact: bool = False
    """Whether to enable VBD deformable's self-contact."""

    particle_self_contact_radius: float = 0.005
    """Particle radius used for self-contact detection [m]."""

    particle_self_contact_margin: float = 0.005
    """Self-contact detection margin [m]. Should be >= particle_self_contact_radius."""

    particle_collision_detection_interval: int = -1
    """Controls how frequently particle self-contact detection is applied.

    If set to a value < 0, collision detection is only performed once before the
    initialization step. If set to 0, collision detection is applied twice: once
    before and once immediately after initialization. If set to a value ``k`` >= 1,
    collision detection is applied before every ``k`` VBD iterations.
    """

    particle_vertex_contact_buffer_size: int = 32
    """Preallocation size for each vertex's vertex-triangle collision buffer."""

    particle_edge_contact_buffer_size: int = 64
    """Preallocation size for each edge's edge-edge collision buffer."""

    particle_topological_contact_filter_threshold: int = 2
    """Maximum topological distance (in rings) below which self-contacts are discarded.

    Only used when ``particle_enable_self_contact`` is ``True``.
    Increase to suppress contacts between closely connected mesh elements.
    Values > 3 significantly increase computation time.
    """

    particle_rest_shape_contact_exclusion_radius: float = 0.0
    """World-space distance threshold for filtering topologically close primitives [m].

    Candidate self-contacts whose rest-configuration separation is shorter than
    this value are ignored. Only used when ``particle_enable_self_contact`` is ``True``.
    """

    rigid_contact_k_start: float = 1.0e2
    """Initial stiffness seed for all rigid body contacts (body-body and body-particle) [N/m].

    Used by the AVBD rigid contact solver. Increase to make rigid contacts stiffer.
    """


@configclass
class CoupledMJWarpVBDSolverCfg(NewtonSolverCfg):
    """Configuration for the coupled rigid-body MJWarp + VBD solver.

    Alternates a rigid-body solver (:class:`MJWarpSolverCfg`) and a soft-body solver (:class:`SolverVBD`) per
    substep. The coupling direction is controlled by :attr:`coupling_mode`:

    - ``"one_way"`` (default): Rigid solver advances first, then VBD reads
      the updated body poses. The rigid solver does not feel particle contacts.
    - ``"two_way"``: Same-substep two-way coupling with normal + Coulomb
      friction. Contact detection runs first, reaction forces are injected
      into ``body_f``, then the rigid solver reads ``body_f`` and feels
      resistance from the deformable object. The friction reaction lets
      actuators carry the object against gravity during a lift.
    """

    class_type: type[NewtonManager] | str = "{DIR}.coupled_mjwarp_vbd_manager:NewtonCoupledMJWarpVBDManager"
    """Manager class for the VBD solver."""

    solver_type: str = "coupledmjwarpvbd"

    rigid_solver_cfg: MJWarpSolverCfg = MJWarpSolverCfg()
    """Rigid-body sub-solver configuration for :class:`MJWarpSolverCfg`."""

    soft_solver_cfg: VBDSolverCfg = VBDSolverCfg(integrate_with_external_rigid_solver=True)
    """VBD sub-solver configuration for cloth/particle dynamics."""

    coupling_mode: str = "two_way"
    """Coupling direction between the rigid and VBD solvers.

    - ``"one_way"``: Rigid -> soft only (default, existing behavior).
    - ``"two_way"``: Same-substep two-way coupling with normal + Coulomb friction.
    """


@configclass
class ProxyCoupledMJWarpVBDSolverCfg(NewtonSolverCfg):
    """Configuration for the proxy-coupled MJWarp + VBD solver.

    Wraps Newton's :class:`newton.solvers.SolverProxyCoupled` (lagged-impulse
    virtual-proxy coupling) with MuJoCo Warp as the rigid sub-solver and VBD as
    the soft sub-solver. Selected source (MuJoCo) bodies are exposed as proxy
    bodies in the VBD view so that VBD detects contacts against them and
    returns the harvested feedback wrenches to MuJoCo via lagged impulses.

    Selection of proxy bodies is driven by :attr:`proxy_bodies`: each entry is
    a :class:`~isaaclab.managers.SceneEntityCfg` naming the source asset and a
    list of body-name regexes to match against the asset's bodies. Matched
    bodies that own at least one shape flagged ``COLLIDE_SHAPES`` are promoted
    to proxies. Mirrors the asset-scoped naming convention used by
    :class:`~isaaclab.envs.mdp.BinaryJointPositionActionCfg.joint_names`.
    """

    class_type: type[NewtonManager] | str = "{DIR}.proxy_coupled_mjwarp_vbd_manager:NewtonProxyCoupledMJWarpVBDManager"
    """Manager class for the proxy-coupled MJWarp + VBD solver."""

    solver_type: str = "proxycoupledmjwarpvbd"

    requires_graph_coloring: bool = True

    mjwarp_cfg: MJWarpSolverCfg = MJWarpSolverCfg()
    """MuJoCo Warp sub-solver configuration."""

    vbd_cfg: VBDSolverCfg = VBDSolverCfg(integrate_with_external_rigid_solver=True)
    """VBD sub-solver configuration. ``integrate_with_external_rigid_solver``
    defaults to ``True`` because the rigid bodies live in the MuJoCo entry."""

    mjwarp_bodies: list[SceneEntityCfg] = []
    """Scene-entity specs whose bodies/joints/shapes go to the MuJoCo entry.

    Each entry is a :class:`~isaaclab.managers.SceneEntityCfg` whose
    :attr:`~isaaclab.managers.SceneEntityCfg.name` references a scene-
    registered asset; the asset's :attr:`prim_path` template scopes the
    body-label match. Optionally
    :attr:`~isaaclab.managers.SceneEntityCfg.body_names` narrows the match to
    a list of body-short-name regexes (``re.fullmatch``, same convention as
    :class:`~isaaclab.envs.mdp.BinaryJointPositionActionCfg.joint_names`);
    leave it unset to claim every body under the asset's prim_path.

    Joints inherit their child body's owner; shapes inherit their body's
    owner; static shapes (``body == -1``) always go to the VBD entry.
    """

    vbd_bodies: list[SceneEntityCfg] = []
    """Scene-entity specs whose bodies/joints/shapes/particles go to the VBD
    entry. Same conventions as :attr:`mjwarp_bodies`."""

    proxy_bodies: list[SceneEntityCfg] = []
    """Scene-entity specs naming the bodies to expose as proxies in the VBD view.

    Same shape as :attr:`mjwarp_bodies` / :attr:`vbd_bodies`, but
    :attr:`~isaaclab.managers.SceneEntityCfg.body_names` is **required** —
    proxies are a subset, not "every body under the asset". Matched bodies
    that also own at least one shape flagged
    ``newton.ShapeFlags.COLLIDE_SHAPES`` are promoted to proxies. Empty list
    means no proxies (the solver still runs but rigid bodies are invisible
    to VBD).
    """

    proxy_mode: str = "lagged"
    """Proxy transfer mode passed to :class:`newton.solvers.SolverProxyCoupled.Proxy`.

    ``"lagged"`` syncs source begin poses and end velocities, then rewinds
    lagged feedback before the destination solve to avoid double-counting.
    ``"staggered"`` syncs source end poses and end velocities directly.
    """

    proxy_iterations: int = 1
    """Number of relaxation iterations per coupled substep
    (:attr:`newton.solvers.SolverProxyCoupled.Config.iterations`)."""

    proxy_collide_interval: int = 1
    """Collision-detection refresh interval (in proxy passes) for the proxy
    collision pipeline."""

    proxy_mass_scale: float = 1.0
    """Mass / inertia scale applied to destination proxy bodies (virtual
    inertia)."""


@configclass
class CoupledFeatherstoneVBDSolverCfg(NewtonSolverCfg):
    """Configuration for the coupled rigid-body Featherstone + VBD solver.

    Alternates a rigid-body solver (:class:`FeatherstoneSolverCfg`) and a soft-body solver (:class:`SolverVBD`) per
    substep. The coupling direction is controlled by :attr:`coupling_mode`:

    - ``"kinematic"`` (default): Rigid -> soft only. Rigid bodies are kinematically updated by the rigid solver,
      then VBD reads the updated body poses and reacts to them. The rigid solver does not feel particle contacts.
    - ``"one_way"``: Rigid solver advances first, then VBD reads
      the updated body poses. The rigid solver does not feel particle contacts.
    - ``"two_way"``: Same-substep two-way coupling with normal + Coulomb
      friction. Contact detection runs first, reaction forces are injected
      into ``body_f``, then the rigid solver reads ``body_f`` and feels
      resistance from the deformable object. The friction reaction lets
      actuators carry the object against gravity during a lift.
    """

    class_type: type[NewtonManager] | str = "{DIR}.coupled_featherstone_vbd_manager:NewtonCoupledFeatherstoneVBDManager"
    """Manager class for the VBD solver."""

    solver_type: str = "coupledfeatherstonevbd"

    rigid_solver_cfg: FeatherstoneSolverCfg = FeatherstoneSolverCfg()
    """Rigid-body sub-solver configuration for :class:`FeatherstoneSolverCfg`."""

    soft_solver_cfg: VBDSolverCfg = VBDSolverCfg(integrate_with_external_rigid_solver=True)
    """VBD sub-solver configuration for cloth/particle dynamics."""

    coupling_mode: str = "kinematic"
    """Coupling direction between the rigid and VBD solvers.

    - ``"kinematic"``: Rigid -> soft only (default)
    - ``"one_way"``: Rigid -> soft only (existing behavior).
    - ``"two_way"``: Same-substep two-way coupling with normal + Coulomb friction.
    """


@configclass
class NewtonModelCfg:
    """Global Newton model parameters.

    These parameters are applied to the ``newton.Model`` after finalization.
    They control model-level contact behavior shared across all objects.
    """

    soft_contact_ke: float = 1.0e3
    """Body-particle contact stiffness [N/m].

    Controls the stiffness of the penalty force of contacts between cloth/soft-body particles
    and rigid body shapes, and self-contacts of cloth/soft-body particles. The effective stiffness per contact is the
    average of this value and the rigid shape's material stiffness.
    """

    soft_contact_kd: float = 1.0e-2
    """Body-particle contact damping [N*s/m]."""

    soft_contact_mu: float = 0.5
    """Body-particle contact friction coefficient.

    The effective friction per contact is ``sqrt(soft_contact_mu * shape_material_mu)``.
    Increase for better grip (e.g. gripper picking up cloth).
    """

    shape_material_ke: float | None = None
    """Per-shape contact stiffness override [N/m].

    When set, all collision shapes in the model will have their contact
    stiffness overwritten to this value.  If ``None`` (default), the
    per-shape values parsed from USD/MJCF are kept.
    """

    shape_material_kd: float | None = None
    """Per-shape contact damping override [N*s/m].

    When set, all collision shapes in the model will have their contact
    damping overwritten to this value.  If ``None`` (default), the
    per-shape values parsed from USD/MJCF are kept.
    """

    shape_material_mu: float | None = None
    """Per-shape friction coefficient override [dimensionless].

    When set, all collision shapes in the model will have their friction
    coefficient overwritten to this value.  If ``None`` (default), the
    per-shape values parsed from USD/MJCF are kept.
    """


@configclass
class CoupledNewtonCfg(NewtonCfg):
    """:class:`NewtonCfg` extended for coupled-solver setups.

    Adds :attr:`model_cfg` (mirrors what :class:`DeformableNewtonCfg` carries
    for global model parameters) and :attr:`scene_cfg`, which lets the manager
    resolve :class:`~isaaclab.managers.SceneEntityCfg`-based selectors (e.g.
    :attr:`ProxyCoupledMJWarpVBDSolverCfg.proxy_bodies`) against the scene at
    solver-build time, without the env having to monkey-patch a private
    attribute onto the solver cfg.

    Uses a distinct class name so :func:`_is_kitless_physics` does not match
    it, ensuring Kit is launched for USD deformable/coupled spawning.
    """

    model_cfg: NewtonModelCfg | None = None
    """Global Newton model parameters applied after builder finalization."""

    scene_cfg: InteractiveSceneCfg | None = None
    """Scene cfg used by coupled solvers to resolve scene-entity selectors.

    Set this to ``self.scene`` from the env's ``__post_init__``. The
    :class:`NewtonProxyCoupledMJWarpVBDManager` reads it via
    :attr:`PhysicsManager._cfg.scene_cfg` to look up each
    :class:`~isaaclab.managers.SceneEntityCfg` in
    :attr:`ProxyCoupledMJWarpVBDSolverCfg.proxy_bodies` and resolve the
    asset's :attr:`prim_path` template.
    """
