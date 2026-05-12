# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton physics manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

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
    """Whether to use the pure MuJoCo backend instead of `mujoco_warp`."""

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
    """Maximum iterations for convex collision detection (GJK/EPA).

    Increase this if you see warnings about ``opt.ccd_iterations`` needing to be increased,
    which typically occurs with complex collision geometries (e.g. multi-finger hands).
    """

    ls_parallel: bool = False
    """Whether to use parallel line search."""

    use_mujoco_contacts: bool = True
    """Whether to use MuJoCo's internal contact solver.

    If ``True`` (default), MuJoCo handles collision detection and contact resolution internally.
    If ``False``, Newton's :class:`CollisionPipeline` is used instead.  A default pipeline
    (``broad_phase="explicit"``) is created automatically when :attr:`NewtonCfg.collision_cfg`
    is ``None``.  Set :attr:`NewtonCfg.collision_cfg` to a :class:`NewtonCollisionPipelineCfg`
    to customize pipeline parameters (broad phase, contact limits, hydroelastic, etc.).

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
