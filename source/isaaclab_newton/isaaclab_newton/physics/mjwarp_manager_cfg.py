# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton physics manager."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass

from .newton_manager_cfg import NewtonSolverCfg

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager


@configclass
class MJWarpSolverCfg(NewtonSolverCfg):
    """Configuration for MuJoCo Warp solver-related parameters.

    These parameters are used to configure the MuJoCo Warp solver. For more information, see the
    `MuJoCo Warp documentation`_.

    .. _MuJoCo Warp documentation: https://github.com/google-deepmind/mujoco_warp
    """

    class_type: type[NewtonManager] | str = "{DIR}.mjwarp_manager:NewtonMJWarpManager"
    """Manager class for the MuJoCo Warp solver."""

    solver_type: str = "mujoco_warp"
    """Solver type. Can be "mujoco_warp"."""

    njmax: int = 300
    """Number of constraints per environment (world)."""

    nconmax: int | None = None
    """Number of contact points per environment (world)."""

    iterations: int = 100
    """Number of solver iterations."""

    ls_iterations: int = 50
    """Number of line search iterations for the solver."""

    solver: str = "newton"
    """Solver type. Can be "cg" or "newton", or their corresponding MuJoCo integer constants."""

    integrator: str = "euler"
    """Integrator type. Can be "euler", "rk4", or "implicitfast", or their corresponding MuJoCo integer constants."""

    use_mujoco_cpu: bool = False
    """Whether to use the pure MuJoCo backend instead of `mujoco_warp`.

    The CPU backend requires the deprecated internal contact pipeline and cannot consume contacts from Newton's
    :class:`CollisionPipeline`.
    """

    disable_contacts: bool = False
    """Whether to disable contact computation in MuJoCo."""

    default_actuator_gear: float | None = None
    """Default gear ratio for all actuators."""

    actuator_gears: dict[str, float] | None = None
    """Dictionary mapping joint names to specific gear ratios, overriding the `default_actuator_gear`."""

    update_data_interval: int = 1
    """Frequency (in simulation steps) at which to update the MuJoCo Data object from the Newton state.

    If 0, Data is never updated after initialization.
    """

    save_to_mjcf: str | None = None
    """Optional path to save the generated MJCF model file.

    If None, the MJCF model is not saved.
    """

    impratio: float = 1.0
    """Frictional-to-normal constraint impedance ratio."""

    cone: str = "pyramidal"
    """The type of contact friction cone. Can be "pyramidal" or "elliptic"."""

    ccd_iterations: int = 35
    """Maximum iterations for the deprecated internal convex collision path (GJK/EPA).

    This field has no effect when ``use_mujoco_contacts=False``. On the legacy path, increase it if you see warnings
    about ``opt.ccd_iterations`` needing to be increased, which typically occurs with complex collision geometries
    such as multi-finger hands.
    """

    ls_parallel: bool = False
    """Deprecated parallel line search option.

    Setting this to ``True`` emits a :class:`DeprecationWarning` and is ignored.
    MuJoCo Warp is dropping support for parallel line search; Isaac Lab uses
    iterative line search for performance.
    """

    use_mujoco_contacts: bool = True
    """Whether to use MuJoCo's deprecated internal contact pipeline.

    If ``False``, Newton's :class:`CollisionPipeline` generates contacts for MJWarp. A default pipeline
    (``broad_phase="explicit"``) is created automatically when :attr:`NewtonCfg.collision_cfg`
    is ``None``. Set :attr:`NewtonCfg.collision_cfg` to a :class:`NewtonCollisionPipelineCfg`
    to customize pipeline parameters (broad phase, contact limits, hydroelastic, etc.).

    If ``True``, MuJoCo handles collision detection internally. The default remains ``True`` only for
    compatibility during the deprecation window; first-party configurations explicitly use ``False``.

    .. deprecated::
        MuJoCo's internal contact pipeline is deprecated. Set ``use_mujoco_contacts=False`` and configure
        :attr:`NewtonCfg.collision_cfg` instead. The legacy path will be removed in a future release.

    .. note::
        Setting ``collision_cfg`` while ``use_mujoco_contacts=True`` raises
        :class:`ValueError` because the two collision modes are mutually exclusive.
    """

    tolerance: float = 1e-6
    """Solver convergence tolerance for the constraint residual.

    The solver iterates until the residual drops below this threshold or
    ``iterations`` is reached.  Lower values give more precise constraint
    satisfaction at the cost of more iterations.  MuJoCo default is ``1e-8``;
    Newton default is ``1e-6``.
    """

    def validate_contact_mode(self) -> None:
        """Validate the selected contact pipeline.

        Warns:
            DeprecationWarning: When MuJoCo's internal contact pipeline is selected.

        Raises:
            ValueError: When the MuJoCo CPU backend is combined with Newton-generated contacts.
        """
        if self.use_mujoco_contacts:
            warnings.warn(
                "MJWarpSolverCfg.use_mujoco_contacts=True selects MuJoCo's internal contact pipeline, which is "
                "deprecated. Set use_mujoco_contacts=False and configure NewtonCfg.collision_cfg instead.",
                DeprecationWarning,
                stacklevel=6,
            )
        if self.use_mujoco_cpu and not self.use_mujoco_contacts:
            raise ValueError(
                "MJWarpSolverCfg.use_mujoco_cpu=True cannot consume Newton CollisionPipeline contacts; "
                "set use_mujoco_contacts=True for the legacy CPU path."
            )

    def __post_init__(self):
        if self.ls_parallel:
            warnings.warn(
                "MJWarpSolverCfg.ls_parallel is deprecated and ignored. "
                "Isaac Lab uses iterative line search for performance because "
                "MuJoCo Warp is dropping parallel line search support. Tune "
                "MJWarpSolverCfg.ls_iterations instead.",
                DeprecationWarning,
                stacklevel=5,
            )
            self.ls_parallel = False
