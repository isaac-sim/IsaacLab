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
from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

import isaaclab.envs.mdp as mdp
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices import DevicesCfg
from isaaclab.devices.gamepad import Se3GamepadCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sim.schemas import UsdPhysicsCollisionCfg
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
        controller=diffik.controller.replace(
            use_relative_mode=False,
            ik_params={"lambda_val": 0.45},
        ),
        body_offset=None,
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
                scale=(0.05, 0.05, 0.05, 0.25, 0.25, 0.25),
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
        self.scene.robot = FRANKA_PANDA_MENAGERIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # IK targets need backend-native gravity control to hold steady between commands.
        self.scene.robot.spawn.rigid_props = [
            PhysxRigidBodyCfg(
                disable_gravity=preset(default=False, diffik=True, diffik_abs=True, newton_ik=True),
                max_depenetration_velocity=5.0,
            ),
            MujocoRigidBodyCfg(gravcomp=preset(default=None, diffik=1.0, diffik_abs=1.0, newton_ik=1.0)),
        ]
        # The Menagerie asset ships its designated convex link-collision meshes disabled.
        physx_collision_props = {"/Geometry/.*_c.*": [UsdPhysicsCollisionCfg(collision_enabled=True)]}
        self.scene.robot.spawn.make_uninstanceable = preset(
            default=False, isaacsim_physx=True, physx=True, ovphysx=True
        )
        self.scene.robot.spawn.collision_props = preset(
            default=None,
            isaacsim_physx=physx_collision_props,
            physx=physx_collision_props,
            ovphysx=physx_collision_props,
        )
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.joint_vel.params["asset_cfg"].joint_names = ["panda_joint.*"]
        self.rewards.action_magnitude.weight = preset(
            default=self.rewards.action_magnitude.weight,
            diffik_abs=0.0,
        )

        # override actions
        self.actions.arm_action = FrankaArmActionCfg()
        # Native SE(3) devices match the 6D relative differential and Newton IK actions.
        relative_ik_teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(gripper_term=False, sim_device=self.sim.device),
                "gamepad": Se3GamepadCfg(gripper_term=False, sim_device=self.sim.device),
                "spacemouse": Se3SpaceMouseCfg(gripper_term=False, sim_device=self.sim.device),
            }
        )
        self.teleop_devices = preset(
            default=DevicesCfg(),
            diffik=relative_ik_teleop_devices,
            newton_ik=relative_ik_teleop_devices,
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = "panda_hand"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)
