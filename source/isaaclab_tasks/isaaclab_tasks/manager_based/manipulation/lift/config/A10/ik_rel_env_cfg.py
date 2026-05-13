# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.gamepad import Se3GamepadCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from A10_Single.robot.a10_single_cfg import A10_SINGLE_CFG  # isort: skip

@configclass
class A10CubeLiftEnvCfg(joint_pos_env_cfg.A10CubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set A10 as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = A10_SINGLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (a10)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            body_name="link6",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.4,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.1034]),
        )
        # Use one scalar action for the gripper and map it to both finger joints.
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper_.*_joint"],
            open_command_expr={"gripper_.*_joint": 0.05},
            close_command_expr={"gripper_.*_joint": 0.0},
        )

        # On reset, clear previous joint targets as well to avoid posture sag/slow recovery.
        if self.events.reset_all.params is None:
            self.events.reset_all.params = {}
        self.events.reset_all.params["reset_joint_targets"] = True

        # Explicit arm-only teleop setup: output 6D command without gripper term.
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    gripper_term=True,
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.04,
                    sim_device=self.sim.device,
                ),
                "gamepad": Se3GamepadCfg(
                    gripper_term=True,
                    pos_sensitivity=0.10,
                    rot_sensitivity=0.12,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    gripper_term=True,
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.04,
                    sim_device=self.sim.device,
                ),
            }
        )


@configclass
class A10CubeLiftEnvCfg_PLAY(A10CubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
