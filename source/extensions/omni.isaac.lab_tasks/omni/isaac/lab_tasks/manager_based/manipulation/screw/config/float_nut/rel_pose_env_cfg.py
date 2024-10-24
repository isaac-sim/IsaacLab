# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from re import T
import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.manipulation.screw.mdp as mdp
from omni.isaac.lab_tasks.manager_based.manipulation.screw.screw_env_cfg import (
    BaseNutThreadEnvCfg,
    BaseNutTightenEnvCfg,
)

from . import abs_pose_env_cfg

##
# Pre-defined configs


@configclass
class RelFloatNutTightenEnvCfg(abs_pose_env_cfg.AbsFloatNutTightenEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        screw_dict = self.scene.screw_dict
        self.act_lows = [-0.001, -0.001, -0.001, -0.5, -0.5, -0.5]
        self.act_highs = [0.001, 0.001, 0.001, 0.5, 0.5, 0.5]
        self.scene.nut.spawn.rigid_props.sleep_threshold = 0.0
        self.scene.nut.spawn.rigid_props.stabilization_threshold = 0.0

        # override actions
        self.actions.nut_action = mdp.RigidObjectPoseActionTermCfg(
            asset_name="nut",
            command_type="pose",
            use_relative_mode=True,
            is_accumulate_action=False,
            p_gain=screw_dict["float_gain"],
            d_gain=screw_dict["float_damp"],
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
class RelFloatNutThreadEnv(BaseNutThreadEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        screw_dict = self.scene.screw_dict
        self.scene.robot = None
        # self.act_lows = [-0.0001, -0.0001, -0.005, -0.0001, -0.0001, -0.5]
        # self.act_highs = [0.0010, 0.0001, 0.005, 0.0001, 0.0001, 0.5]
        self.act_lows = [-0.0001, -0.0001, -0.05, -0.01, -0.01, -0.6]
        self.act_highs = [0.0001, 0.0001, 0.05, 0.01, 0.01, 0.]
        scale = [0.0001, 0.0001, 0.01, 0.01, 0.01, 0.6]
        # override actions
        self.actions.nut_action = mdp.RigidObjectPoseActionTermCfg(
            asset_name="nut",
            command_type="pose",
            use_relative_mode=True,
            p_gain=screw_dict["float_gain"],
            d_gain=screw_dict["float_damp"],
            is_accumulate_action=False,
            lows=self.act_lows,
            highs=self.act_highs,
            scale=scale,
        )

        # self.scene.bolt.spawn.activate_contact_sensors = True
        self.scene.nut.spawn.activate_contact_sensors = True
        self.scene.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Nut/factory_nut",
            filter_prim_paths_expr= ["{ENV_REGEX_NS}/Bolt/factory_bolt"],
            update_period=0.0,
        )
        self.rewards.contact_force_penalty = RewTerm(
            func=mdp.contact_forces,
            params={"threshold":0.0004, "sensor_cfg": SceneEntityCfg(name="contact_sensor")},
            weight=0.00001)


@configclass
class RelFloatNutTightenEnvCfg_PLAY(RelFloatNutTightenEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
