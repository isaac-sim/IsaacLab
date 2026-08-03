# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Reach environment configuration."""

import math

from isaaclab_newton.envs.mdp.actions.newton_ik_actions_cfg import NewtonInverseKinematicsActionCfg
from isaaclab_newton.ik.newton_ik_objectives_cfg import NewtonIKJointLimitObjectiveCfg, NewtonIKPoseObjectiveCfg
from isaaclab_newton.ik.newton_ik_solver_cfg import NewtonIKSolverCfg
from isaaclab_newton.physics import NewtonCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as mdp
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.reach.reach_env_cfg import ReachEnvCfg
from isaaclab_tasks.utils import PresetCfg, preset

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_MENAGERIE_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class FrankaArmActionCfg(PresetCfg):
    """Arm-controller presets for Franka Reach."""

    joint_pos: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
    )
    diffik: DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
            ik_params={"lambda_val": 0.01},
        ),
        scale=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )
    diffik_abs: DifferentialInverseKinematicsActionCfg = diffik.replace(
        controller=diffik.controller.replace(use_relative_mode=False),
        scale=1.0,
    )
    newton_ik: NewtonInverseKinematicsActionCfg = NewtonInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        controller=NewtonIKSolverCfg(optimizer="lm", jacobian_mode="analytic", iterations=4),
        objectives=[
            NewtonIKPoseObjectiveCfg(
                body_name="panda_hand",
                body_offset_pos=(0.0, 0.0, 0.107),
                command_type="pose",
                use_relative_mode=True,
                scale=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
                rotation_weight=2.0,
            ),
            NewtonIKJointLimitObjectiveCfg(weight=0.1),
        ],
    )
    default: mdp.JointPositionActionCfg = joint_pos


@configclass
class FrankaReachEnvCfg(ReachEnvCfg):
    """Franka Reach configuration with selectable arm and physics presets."""

    def validate_config(self) -> None:
        """Validate the selected controller and physics backend."""

        if isinstance(self.actions.arm_action, NewtonInverseKinematicsActionCfg) and not isinstance(
            self.sim.physics, NewtonCfg
        ):
            raise ValueError("The 'newton_ik' action preset requires a Newton physics preset.")
        if (
            isinstance(self.actions.arm_action, DifferentialInverseKinematicsActionCfg)
            and not self.actions.arm_action.controller.use_relative_mode
            and not isinstance(self.sim.physics, PhysxCfg)
        ):
            raise ValueError("The 'diffik_abs' action preset requires the 'isaacsim_physx' physics preset.")

    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = FRANKA_PANDA_MENAGERIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.rigid_props.disable_gravity = preset(default=False, diffik_abs=True)
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.joint_vel.params["asset_cfg"].joint_names = ["panda_joint.*"]

        # override actions
        self.actions.arm_action = FrankaArmActionCfg()
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = "panda_hand"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)
