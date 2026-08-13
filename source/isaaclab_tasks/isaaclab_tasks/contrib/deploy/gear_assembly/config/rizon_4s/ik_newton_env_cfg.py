# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.envs.mdp.actions.newton_ik_actions_cfg import NewtonInverseKinematicsActionCfg
from isaaclab_newton.ik.newton_ik_objectives_cfg import NewtonIKJointLimitObjectiveCfg, NewtonIKPoseObjectiveCfg
from isaaclab_newton.ik.newton_ik_solver_cfg import NewtonIKSolverCfg

from isaaclab.utils.configclass import configclass

from . import joint_pos_env_cfg


@configclass
class Rizon4sGearAssemblyIKNewtonEnvCfg(joint_pos_env_cfg.Rizon4sGearAssemblyEnvCfg):
    """Gear-assembly with a Newton inverse-kinematics (task-space) action for the Rizon 4s arm.

    Replaces the joint-space :class:`~isaaclab.envs.mdp.RelativeJointPositionActionCfg` with a
    Newton-solved relative end-effector pose action, so the policy commands end-effector motion
    directly. This provides a six-dimensional task-space alternative to the seven-dimensional
    joint-space action. The gripper remains a fixed mimic mechanism (not commanded by the policy).

    Note:
        Newton IK consumes the replicated robot prototype, so this variant requires a Newton preset
        (``presets=newton_mjwarp``, ``presets=newton_sdf``, or
        ``presets=newton_hydroelastic``); it is not compatible with ``presets=physx``.
    """

    def __post_init__(self):
        super().__post_init__()

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
