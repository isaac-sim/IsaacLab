# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka soft lifting environment using the custom coupling manager."""

from isaaclab_newton.physics import MJWarpSolverCfg

from isaaclab.utils.configclass import configclass

from isaaclab_contrib.deformable.newton_manager_cfg import VBDSolverCfg

from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import FrankaSoftEnvCfg
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import PhysicsCfg as CorePhysicsCfg

from .newton_manager_cfg import CoupledMJWarpVBDSolverCfg


@configclass
class PhysicsCfg(CorePhysicsCfg):
    """Adds the manual MJWarp and VBD coupling preset on top of the core proxy presets."""

    newton_mjwarp_vbd = CorePhysicsCfg().newton_mjwarp_vbd_proxy.replace(
        # Required: ``NewtonCfg.__post_init__`` rejects a preset class_type and re-derives it.
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
            model_cfg=CorePhysicsCfg().newton_mjwarp_vbd_proxy.solver_cfg.model_cfg,
        ),
    )

    default = newton_mjwarp_vbd


@configclass
class FrankaSoftCustomCouplingEnvCfg(FrankaSoftEnvCfg):
    """Franka soft lifting with manual MJWarp and VBD coupling."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim.physics = PhysicsCfg()
