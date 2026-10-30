# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton physics manager."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from isaaclab.physics import PhysicsCfg
from isaaclab.utils import configclass

from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager, NewtonSolver

logger = logging.getLogger(__name__)


@configclass
class NewtonSolverCfg:
    """Configuration of a Newton solver.

    :attr:`class_type` selects the :class:`~isaaclab_newton.physics.NewtonSolver` that builds and drives the
    solver; customizing a solver means subclassing it. For the solver parameters, see the
    `Newton documentation`_.

    .. _Newton documentation: https://newton.readthedocs.io/en/latest/
    """

    class_type: type[NewtonSolver] | str = "{DIR}.newton_solver:NewtonSolver"
    """Solver class whose class methods build and drive Newton's solver, e.g. ``class_type.build(cfg, model, mode)``."""

    solver_type: str = "None"
    """Solver type metadata (deprecated).

    .. deprecated::
        Solver dispatch is driven by the configuration class; this field is retained as metadata for logging and
        debugging only. Do not branch on ``solver_type`` in new code.
    """


@configclass
class NewtonSoftContactCfg:
    """Global soft-contact parameters applied to the finalized Newton model."""

    soft_contact_ke: float = 1.0e3
    """Body-particle and particle self-contact stiffness [N/m].

    Effective body-particle stiffness is ``0.5 * (soft_contact_ke + shape_ke)``,
    where ``shape_ke`` is the rigid shape's material stiffness.
    """

    soft_contact_kd: float = 10.0
    """Body-particle contact damping [N*s/m]."""

    soft_contact_mu: float = 0.5
    """Body-particle contact friction coefficient [dimensionless].

    Effective body-particle friction is ``sqrt(soft_contact_mu * shape_mu)``,
    where ``shape_mu`` is the rigid shape's material friction coefficient.
    """


@configclass
class NewtonShapeCfg:
    """Default per-shape collision properties applied to all shapes in a Newton scene.

    Mirrors Newton's :attr:`ModelBuilder.default_shape_cfg`. Fields that Isaac
    Lab overrides or exposes for user overrides are declared here; fields not
    represented keep Newton's upstream defaults. The struct is forwarded onto
    Newton's upstream ``ShapeConfig`` via
    :func:`~isaaclab.utils.checked_apply` at builder construction.
    """

    margin: float = 0.0
    """Default per-shape collision margin [m].

    A nonzero margin (e.g. ``0.01``) is required for stable contact on
    triangle-mesh terrain — without it, lightweight robots fail to learn
    rough-terrain locomotion on Newton. Newton's upstream default is ``0.0``.
    """

    gap: float = 0.01
    """Default per-shape contact gap [m]. Newton's upstream default is ``None``."""

    # Defaults mirror Newton's ShapeConfig defaults so an unspecified field is a no-op.
    ke: float = 2.5e3
    """Default per-shape normal contact stiffness [N/m].

    Applied to shapes that lack an explicit material; per-asset materials
    override it. Mirrors Newton's ``ShapeConfig.ke`` default.
    """

    kd: float = 100.0
    """Default per-shape normal contact damping [N*s/m].

    Applied to shapes that lack an explicit material; per-asset materials
    override it. Mirrors Newton's ``ShapeConfig.kd`` default.
    """

    mu: float = 1.0
    """Default per-shape friction coefficient [dimensionless].

    Applied to shapes that lack an explicit material; per-asset materials
    override it. Mirrors Newton's ``ShapeConfig.mu`` default.
    """


@configclass
class NewtonCfg(PhysicsCfg):
    """Configuration for Newton physics manager.

    This configuration includes Newton-specific simulation settings and solver configuration.

    ``class_type`` selects the integration manager independently of the solver, which :attr:`solver_cfg` describes.
    Downstream code can subclass :class:`NewtonManager` and set this field.
    """

    class_type: type[NewtonManager] | str = "{DIR}.newton_manager:NewtonManager"
    """Manager class, constructed by SimulationContext as ``instantiate(cfg)``."""

    num_substeps: int = 1
    """Number of substeps to use for the solver."""

    collision_decimation: int = 0
    """Re-collide every N solver substeps within a physics tick (``0`` = once per tick)."""

    debug_mode: bool = False
    """Whether to enable debug mode for the solver."""

    use_cuda_graph: bool = True
    """Whether to use CUDA graphing when simulating.

    Graphs are captured immediately after the first eager physics step, after reset and decimation
    setup. Capture does not advance physics. Kit/RTX uses a nonblocking stream with relaxed
    capture mode; kitless simulation uses Warp's standard capture mode.

    If set to False, the simulation performance will be severely degraded.
    """

    deterministic_mode: Literal["not_guaranteed", "run_to_run", "gpu_to_gpu"] = "not_guaranteed"
    """Determinism guarantee applied to the Newton solver and collision pipeline.

    The values ``"not_guaranteed"``, ``"run_to_run"``, and ``"gpu_to_gpu"``
    map to the corresponding ``warp.DeterministicMode`` values. Deterministic
    execution increases memory use and can reduce simulation performance.

    .. warning::

       Deterministic contact ordering adds sorting work and allocates buffers
       sized for the configured maximum contact count. Runtime and memory
       overhead therefore grow with contact capacity. Enable this mode only
       when its reproducibility guarantee is required.

    MJWarp on the GPU (built with its sensor stage disabled), XPBD, and
    Featherstone support this setting. Newton raises an error during solver
    initialization for unsupported solvers rather than silently running them
    without the requested guarantee.
    """

    solver_cfg: NewtonSolverCfg | None = None
    """Solver configuration. If None (default), MJWarpSolverCfg is used by default."""

    soft_contact_cfg: NewtonSoftContactCfg | None = None
    """Global soft-contact parameters applied after model finalization.

    If ``None``, Newton model defaults are preserved.
    """

    collision_cfg: NewtonCollisionPipelineCfg | None = None
    """Newton collision pipeline configuration.

    Controls how Newton's :class:`CollisionPipeline` is configured when it is active.
    The pipeline is active when the solver delegates collision detection to Newton:

    - :class:`MJWarpSolverCfg` with ``use_mujoco_contacts=False``,
    - :class:`KaminoPADMMSolverCfg` or :class:`KaminoDVISolverCfg` with
      ``use_collision_detector=False``,
    - :class:`XPBDSolverCfg` (always),
    - :class:`VBDSolverCfg` (always),
    - :class:`FeatherstoneSolverCfg` (always).

    :class:`~isaaclab_newton.physics.MPMSolverCfg` does not use this pipeline;
    implicit MPM treats rigid geometry as colliders internally.

    If ``None`` (default), a pipeline with ``broad_phase="explicit"`` is created
    automatically.  Set this to a :class:`NewtonCollisionPipelineCfg` to customize
    parameters such as broad phase algorithm, contact limits, or hydroelastic mode.

    .. note::
        Setting this while ``MJWarpSolverCfg.use_mujoco_contacts=True`` raises
        :class:`ValueError`.  When a Kamino solver config has ``use_collision_detector=True``,
        the field is ignored because Kamino's internal detector handles contacts.
    """

    default_shape_cfg: NewtonShapeCfg = NewtonShapeCfg()
    """Default per-shape collision properties applied to every shape in the scene.

    Forwarded to Newton's :attr:`ModelBuilder.default_shape_cfg` at builder
    construction via :func:`~isaaclab.utils.checked_apply`. See
    :class:`NewtonShapeCfg` for the declared fields.
    """

    load_visual_shapes: bool | None = None
    """Whether Newton replication imports visual-only geometry from USD.

    ``None`` imports it only when a viewer, an offscreen ``rgb_array`` capture, or a
    camera sensor is active, so headless training does not pay the USD parse time and
    memory for shapes nothing draws. Set to ``True`` to always import it, which is
    needed when a ray-cast sensor must hit geometry that carries no collider.
    """

    bvh_constructor_geometry: Literal["lbvh", "sah", "cubql"] = "cubql"
    """BVH construction algorithm for mesh geometry colliders.

    Selects the bounding-volume-hierarchy builder Newton uses for the triangle
    meshes of collision geometry, forwarded to :attr:`ModelBuilder.BvhConfig`.
    Trades build time against query (traversal) quality:

    - ``"lbvh"``: linear BVH; fastest to build, lowest-quality tree.
    - ``"sah"``: surface-area-heuristic BVH; slower build, tighter tree with
      faster ray/overlap queries.
    - ``"cubql"``: cuBQL GPU builder; balances fast construction with good tree
      quality on the GPU (default).
    """

    bvh_constructor_scene: Literal["lbvh", "sah"] = "sah"
    """BVH construction algorithm for the top-level scene (broad-phase) hierarchy.

    Selects the builder for the BVH over all colliders used during broad-phase
    culling, forwarded to :attr:`ModelBuilder.BvhConfig`. See
    :attr:`bvh_constructor_geometry` for the ``"lbvh"`` / ``"sah"`` trade-off;
    ``"cubql"`` is not available for the scene hierarchy.
    """

    bvh_constructor_gaussian: Literal["lbvh", "sah", "cubql"] = "cubql"
    """BVH construction algorithm for Gaussian-splat primitives.

    Selects the builder for the BVH over 3D Gaussian primitives (used by the
    Gaussian renderer/collision path), forwarded to
    :attr:`ModelBuilder.BvhConfig`. See :attr:`bvh_constructor_geometry` for the
    ``"lbvh"`` / ``"sah"`` / ``"cubql"`` trade-off.
    """

    def __post_init__(self):
        if self.class_type is None:
            raise ValueError("NewtonCfg.class_type must select a NewtonManager class; omit it for the default.")
        if self.deterministic_mode not in ("not_guaranteed", "run_to_run", "gpu_to_gpu"):
            raise ValueError(
                "NewtonCfg.deterministic_mode must be 'not_guaranteed', 'run_to_run', or 'gpu_to_gpu', "
                f"got {self.deterministic_mode!r}."
            )
        if self.solver_cfg is None:
            from isaaclab_newton.physics.mjwarp_solver_cfg import MJWarpSolverCfg

            self.solver_cfg = MJWarpSolverCfg()

        # Mid-tick re-collide is silently disabled when collision_decimation >= num_substeps.
        if self.collision_decimation > 0 and self.collision_decimation >= self.num_substeps:
            logger.warning(
                "NewtonCfg.collision_decimation=%d is >= num_substeps=%d; mid-tick re-collide is disabled. "
                "Set 0 < collision_decimation < num_substeps to enable.",
                self.collision_decimation,
                self.num_substeps,
            )
