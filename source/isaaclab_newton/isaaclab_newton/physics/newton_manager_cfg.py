# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton physics manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.physics import PhysicsCfg
from isaaclab.utils import configclass

from .newton_collision_cfg import NewtonCollisionPipelineCfg

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager


@configclass
class NewtonSolverCfg:
    """Configuration for Newton solver-related parameters.

    These parameters are used to configure the Newton solver. For more information, see the `Newton documentation`_.

    Subclasses set :attr:`class_type` to their matching :class:`NewtonManager`
    subclass; :class:`NewtonCfg` propagates that to its own
    :attr:`NewtonCfg.class_type` in :meth:`NewtonCfg.__post_init__` so that
    ``SimulationContext`` resolves the correct manager via the existing
    dispatch path.

    .. _Newton documentation: https://newton.readthedocs.io/en/latest/
    """

    class_type: type[NewtonManager] | str = "{DIR}.newton_manager:NewtonManager"
    """Manager class for this solver.

    Default points at the abstract :class:`NewtonManager`; concrete subclasses
    override it.
    """

    solver_type: str = "None"
    """Solver type metadata (deprecated).

    .. deprecated::
        Manager dispatch is now driven by :attr:`class_type`; this field is
        retained as metadata for logging and debugging only.  Do not branch on
        ``solver_type`` in new code.
    """


@configclass
class NewtonShapeCfg:
    """Default per-shape collision properties applied to all shapes in a Newton scene.

    Mirrors Newton's :attr:`ModelBuilder.default_shape_cfg`. Only fields Isaac
    Lab actually overrides are declared here; unspecified fields keep Newton's
    upstream default. The struct is forwarded onto Newton's upstream
    ``ShapeConfig`` via :func:`~isaaclab.utils.checked_apply` at builder
    construction.
    """

    margin: float = 0.0
    """Default per-shape collision margin [m].

    A nonzero margin (e.g. ``0.01``) is required for stable contact on
    triangle-mesh terrain — without it, lightweight robots fail to learn
    rough-terrain locomotion on Newton. Newton's upstream default is ``0.0``.
    """

    gap: float = 0.01
    """Default per-shape contact gap [m]. Newton's upstream default is ``None``."""


@configclass
class NewtonCfg(PhysicsCfg):
    """Configuration for Newton physics manager.

    This configuration includes Newton-specific simulation settings and solver configuration.

    The active :class:`NewtonManager` subclass is determined by
    :attr:`solver_cfg.class_type`, which :meth:`__post_init__` propagates to
    :attr:`class_type` so that ``SimulationContext`` resolves the right
    manager subclass automatically.  User code keeps the existing two-level
    shape ``NewtonCfg(solver_cfg=...)`` and does not need to set
    :attr:`class_type` explicitly.
    """

    class_type: type[NewtonManager] | str | None = None
    """The class type of the :class:`NewtonManager`.

    Auto-set in :meth:`__post_init__` from :attr:`solver_cfg.class_type`.
    Users normally do not set this directly.
    """

    num_substeps: int = 1
    """Number of substeps to use for the solver."""

    debug_mode: bool = False
    """Whether to enable debug mode for the solver."""

    use_cuda_graph: bool = True
    """Whether to use CUDA graphing when simulating.

    If set to False, the simulation performance will be severely degraded.
    """

    solver_cfg: NewtonSolverCfg | None = None
    """Solver configuration. If None (default), MJWarpSolverCfg is used by default."""

    collision_cfg: NewtonCollisionPipelineCfg | None = None
    """Newton collision pipeline configuration.

    Controls how Newton's :class:`CollisionPipeline` is configured when it is active.
    The pipeline is active when the solver delegates collision detection to Newton:

    - :class:`MJWarpSolverCfg` with ``use_mujoco_contacts=False``,
    - :class:`KaminoSolverCfg` with ``use_collision_detector=False``,
    - :class:`XPBDSolverCfg` (always),
    - :class:`FeatherstoneSolverCfg` (always).

    If ``None`` (default), a pipeline with ``broad_phase="explicit"`` is created
    automatically.  Set this to a :class:`NewtonCollisionPipelineCfg` to customize
    parameters such as broad phase algorithm, contact limits, or hydroelastic mode.

    .. note::
        Setting this while ``MJWarpSolverCfg.use_mujoco_contacts=True`` raises
        :class:`ValueError`.  When ``KaminoSolverCfg.use_collision_detector=True``,
        the field is ignored because Kamino's internal detector handles contacts.
    """

    default_shape_cfg: NewtonShapeCfg = NewtonShapeCfg()
    """Default per-shape collision properties applied to every shape in the scene.

    Forwarded to Newton's :attr:`ModelBuilder.default_shape_cfg` at builder
    construction via :func:`~isaaclab.utils.checked_apply`. See
    :class:`NewtonShapeCfg` for the declared fields.
    """

    def __post_init__(self):
        # NewtonCfg.class_type is auto-derived from solver_cfg.class_type.
        # Refuse a user-set value: setting both is ambiguous and was
        # previously silently overwritten.
        if self.class_type is not None:
            raise TypeError("Cannot manually set NewtonCfg.class_type; it is auto-derived from solver_cfg.class_type.")
        if self.solver_cfg is None:
            from .mjwarp_manager_cfg import MJWarpSolverCfg

            self.solver_cfg = MJWarpSolverCfg()
        self.class_type = self.solver_cfg.class_type
