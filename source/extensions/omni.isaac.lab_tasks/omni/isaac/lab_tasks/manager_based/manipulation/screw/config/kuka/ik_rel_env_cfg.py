# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.utils import configclass
import omni.isaac.lab_tasks.manager_based.manipulation.screw.mdp as mdp
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab_tasks.manager_based.manipulation.screw.screw_env_cfg import BaseNutTightenEnvCfg, BaseNutThreadEnvCfg
import omni.isaac.lab.sim as sim_utils

##
# Pre-defined configs
from omni.isaac.lab_assets.kuka import KUKA_VICTOR_LEFT_HIGH_PD_CFG, KUKA_VICTOR_LEFT_CFG  # isort: skip


@configclass
class IKRelKukaNutTightenEnvCfg(BaseNutTightenEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.act_lows = [-0.001, -0.001, -0.001, -0.5, -0.5, -0.5]
        self.act_highs = [0.001, 0.001, 0.001, 0.5, 0.5, 0.5]
        # Set Kuka as robot
        
        self.scene.robot = KUKA_VICTOR_LEFT_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = [-0.25, -0.2, -0.8]
        
        # # override actions
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["victor_left_arm_joint.*"],
            body_name="victor_left_tool0",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        )
        self.actions.gripper_action = mdp.Robotiq3FingerActionCfg(
            asset_name="robot",
            side="left",
            lows=self.act_lows,
            highs=self.act_highs,
        )
        
        
        # self.scene.nut.spawn.activate_contact_sensors = True
        
        # self.scene.bolt.spawn.activate_contact_sensors = True
        # self.scene.contact_sensor = ContactSensorCfg(
        #     prim_path="{ENV_REGEX_NS}/Nut/factory_nut",
        #     filter_prim_paths_expr= ["{ENV_REGEX_NS}/Bolt/factory_bolt"],
        #     update_period=0.0,
        # )
        # self.rewards.contact_force_penalty = RewTerm(
        #     func=mdp.contact_forces,
        #     params={"threshold":0, "sensor_cfg": SceneEntityCfg(name="contact_sensor")},
        #     weight=0.01)


@configclass
class IKRelKukaNutThreadEnv(BaseNutThreadEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.act_lows = [-0.001, -0.001, -0.001, -0.5, -0.5, -0.5]
        self.act_highs = [0.001, 0.001, 0.001, 0.5, 0.5, 0.5]
        self.scene.robot = KUKA_VICTOR_LEFT_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = [-0.15, -0.5, -0.8]
        # self.scene.nut.spawn.rigid_props.disable_gravity = False
        # self.scene.nut.init_state.pos = (0.5, 0, 0)
        # override actions
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["victor_left_arm_joint.*"],
            body_name="victor_left_tool0",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        )
        
        arm_joint_angles = [ 1.5202e+00, -3.5097e-01,  2.2632e+00,  1.5006e+00, -2.0696e+00,
          1.0335e+00,  3.7628e-01,  1.1594e-01, -1.1594e-01,  7.3437e-01,
          7.3437e-01,  7.3437e-01,  2.3933e-11, -3.4038e-12,  7.6929e-11,
         -7.3437e-01, -7.3437e-01, -7.3437e-01]
        ori_init_joints = self.scene.robot.init_state.joint_pos
        for key, value in zip(ori_init_joints.keys(), arm_joint_angles):
            if "arm" in key:
                ori_init_joints[key] = value
        self.scene.robot.init_state.joint_pos = ori_init_joints
        self.actions.gripper_action = mdp.Robotiq3FingerActionCfg(
            asset_name="robot",
            side="left",
            lows=self.act_lows,
            highs=self.act_highs,
            use_relative_mode=True,
            is_accumulate_action=True
        )

        # self.scene.bolt.spawn.activate_contact_sensors = True
        # self.scene.nut.spawn.activate_contact_sensors = True
        # self.scene.contact_sensor = ContactSensorCfg(
        #     prim_path="{ENV_REGEX_NS}/Nut/factory_nut",
        #     filter_prim_paths_expr= ["{ENV_REGEX_NS}/Bolt/factory_bolt"],
        #     update_period=0.0,
        # )
        # self.rewards.contact_force_penalty = RewTerm(
        #     func=mdp.contact_forces,
        #     params={"threshold":0, "sensor_cfg": SceneEntityCfg(name="contact_sensor")},
        #     weight=0.01)

   

# @configclass
# class RelFloatNutTightenEnvCfg_PLAY(RelFloatNutTightenEnvCfg):
#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()
#         # make a smaller scene for play
#         self.scene.num_envs = 50
#         self.scene.env_spacing = 2.5
#         # disable randomization for play
#         self.observations.policy.enable_corruption = False
        



