# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.envs.mdp.actions.newton_ik_actions_cfg import NewtonInverseKinematicsActionCfg
from isaaclab_newton.ik.newton_ik_objectives_cfg import NewtonIKJointLimitObjectiveCfg, NewtonIKPoseObjectiveCfg
from isaaclab_newton.ik.newton_ik_solver_cfg import NewtonIKSolverCfg
from isaaclab_newton.physics import NewtonCfg

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.deploy.gear_assembly.gear_assembly_env_cfg import GearAssemblyPhysicsCfg

from . import joint_pos_env_cfg


@configclass
class Rizon4sGearAssemblyIKNewtonPhysicsCfg(GearAssemblyPhysicsCfg):
    """Gear-assembly physics presets with Newton MJWarp as the fallback."""

    default: NewtonCfg = GearAssemblyPhysicsCfg().newton_mjwarp


@configclass
class Rizon4sGearAssemblyIKNewtonEnvCfg(joint_pos_env_cfg.Rizon4sGearAssemblyEnvCfg):
    """Gear assembly with a Newton inverse-kinematics action for the Rizon 4s arm.

    The task defaults to Newton MJWarp and accepts ``newton_sdf`` and
    ``newton_hydroelastic`` presets. Newton IK is incompatible with PhysX presets.
    """

    _newton_default = True

    def validate_config(self) -> None:
        """Validate the selected controller and physics backend."""
        super().validate_config()
        if isinstance(self.actions.arm_action, NewtonInverseKinematicsActionCfg) and not isinstance(
            self.sim.physics, NewtonCfg
        ):
            raise ValueError("Newton inverse-kinematics actions require a Newton physics preset.")

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = Rizon4sGearAssemblyIKNewtonPhysicsCfg()

        # Command the physical flange frame used by the real Flexiv Cartesian controller.
        self.actions.arm_action = NewtonInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
            controller=NewtonIKSolverCfg(optimizer="lm", jacobian_mode="analytic", iterations=24),
            clip={".*": (-0.5, 0.5)},
            objectives=[
                NewtonIKPoseObjectiveCfg(
                    body_name="flange",
                    command_type="pose",
                    use_relative_mode=True,
                    scale=0.025,
                ),
                NewtonIKJointLimitObjectiveCfg(weight=0.1),
            ],
        )
