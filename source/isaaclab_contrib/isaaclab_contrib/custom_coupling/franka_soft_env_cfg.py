# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka soft lifting environment using the custom coupling manager."""

from dataclasses import dataclass, field
from typing import Any

from isaaclab_newton.physics import MJWarpSolverCfg, VBDSolverCfg

from isaaclab.utils import replace_config

from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import FrankaSoftEnvCfg
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import PhysicsCfg as CorePhysicsCfg

from .newton_manager_cfg import CoupledMJWarpVBDSolverCfg


@dataclass
class PhysicsCfg(CorePhysicsCfg):
    """Adds the manual MJWarp and VBD coupling preset on top of the core proxy presets."""

    newton_mjwarp_vbd: Any = field(
        default_factory=lambda: replace_config(
            CorePhysicsCfg().newton_mjwarp_vbd_proxy,
            class_type=None,
            solver_cfg=CoupledMJWarpVBDSolverCfg(
                rigid_solver_cfg=MJWarpSolverCfg(
                    njmax=40,
                    nconmax=20,
                    ls_iterations=20,
                    integrator="implicitfast",
                    ccd_iterations=100,
                ),
                soft_solver_cfg=VBDSolverCfg(
                    integrate_with_external_rigid_solver=True,
                ),
            ),
        )
    )

    default: Any = field(
        default_factory=lambda: replace_config(
            CorePhysicsCfg().newton_mjwarp_vbd_proxy,
            class_type=None,
            solver_cfg=CoupledMJWarpVBDSolverCfg(
                rigid_solver_cfg=MJWarpSolverCfg(
                    njmax=40,
                    nconmax=20,
                    ls_iterations=20,
                    integrator="implicitfast",
                    ccd_iterations=100,
                ),
                soft_solver_cfg=VBDSolverCfg(
                    integrate_with_external_rigid_solver=True,
                ),
            ),
        )
    )


@dataclass
class FrankaSoftCustomCouplingEnvCfg(FrankaSoftEnvCfg):
    """Franka soft lifting with manual MJWarp and VBD coupling."""

    def __post_init__(self) -> None:
        if parent_post_init := getattr(super(), "__post_init__", None):
            parent_post_init()
        self.sim.physics = PhysicsCfg()
