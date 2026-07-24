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

import isaaclab.envs.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.reach.reach_env_cfg import ReachEnvCfg
from isaaclab_tasks.utils import PresetCfg

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip


_FRANKA_PANDA_REACH_CFG = FRANKA_PANDA_CFG.copy()
_FRANKA_PANDA_REACH_CFG.spawn.usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/franka_panda.usda"
_FRANKA_PANDA_REACH_CFG.actuators = {
    "panda_arm": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-7]"],
        # Override the converter-authored angular drives with SI gains until the USD is corrected.
        velocity_limit_sim={"panda_joint[1-4]": 20.0, "panda_joint[5-7]": 25.0},
        stiffness={"panda_joint[1-2]": 1000.0, "panda_joint[3-4]": 750.0, "panda_joint[5-7]": 300.0},
        damping={"panda_joint[1-2]": 20.0, "panda_joint[3-4]": 4.0, "panda_joint[5-7]": 2.0},
    ),
    "panda_hand": ImplicitActuatorCfg(
        joint_names_expr=["panda_finger_joint.*"],
        stiffness=None,
        damping=None,
    ),
}

_IK_ACTION_SCALE = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5)


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
        scale=_IK_ACTION_SCALE,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
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
                scale=_IK_ACTION_SCALE,
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

    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = _FRANKA_PANDA_REACH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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
