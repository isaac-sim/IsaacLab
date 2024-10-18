# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from omni.isaac.lab.assets import DeformableObjectCfg
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg
from omni.isaac.lab.sim.spawners import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.math import quat_from_euler_xyz

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from omni.isaac.lab_assets.kuka import KUKA_VICTOR_HIGH_PD_CFG, KUKA_VICTOR_LEFT_HIGH_PD_CFG  # isort: skip


@configclass
class KukaCubeLiftEnvCfg(joint_pos_env_cfg.KukaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = KUKA_VICTOR_LEFT_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["victor_left_arm_joint.*"],
            body_name="victor_left_tool0",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        )


@configclass
class KukaCubeLiftEnvCfg_PLAY(KukaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


@configclass
class KukaTeddyBearLiftEnvCfg(KukaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.object = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=DeformableObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.05], rot=[0.707, 0, 0, 0.707]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Teddy_Bear/teddy_bear.usd",
                scale=(0.01, 0.01, 0.01),
            ),
        )

        # Make the end effector less stiff to not hurt the poor teddy bear
        self.scene.robot.actuators["victor_left_gripper"].effort_limit = 50.0
        self.scene.robot.actuators["victor_left_gripper"].stiffness = 40.0
        self.scene.robot.actuators["victor_left_gripper"].damping = 10.0

        # Remove all the terms for the lift_teddy_bear_sm demo
        self.terminations.object_dropping = None
        self.rewards.reaching_object = None
        self.rewards.lifting_object = None
        self.rewards.object_goal_tracking = None
        self.rewards.object_goal_tracking_fine_grained = None
        self.events.reset_object_position = None
        self.observations.policy.object_position = None
