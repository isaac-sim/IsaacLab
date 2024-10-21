# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import pickle
from einops import repeat
from force_tool.utils.data_utils import SmartDict
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.sim.spawners import materials
from omni.isaac.lab.utils import configclass
import omni.isaac.lab.utils.math as math_utils
import omni.isaac.lab_tasks.manager_based.manipulation.screw.mdp as mdp
from omni.isaac.lab_tasks.manager_based.manipulation.screw.screw_env_cfg import (
    BaseNutThreadEnvCfg,
    BaseNutTightenEnvCfg,
)

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


def reset_scene_with_grasping(env: ManagerBasedEnv, env_ids: torch.Tensor):
    # standard reset
    
    # set friction
    robot = env.unwrapped.scene["robot"]
    robot_material = robot.root_physx_view.get_material_properties()
    robot_material[..., 0] = 2
    robot_material[..., 1] = 2
    robot.root_physx_view.set_material_properties(robot_material, torch.arange(env.scene.num_envs, device="cpu"))
    mdp.reset_scene_to_default(env, env_ids)

    cached_env_state = SmartDict(pickle.load(open("data/kuka_nut_thread_pre_grasp.pkl", "rb")))
    # nut_eulers = torch.zeros(1, 3, device=env.device)
    # nut_eulers[0, 2] = 0.3
    # nut_quat = math_utils.quat_from_euler_xyz(nut_eulers[:, 0], nut_eulers[:, 1], nut_eulers[:, 2])
    # cached_env_state["nut"]["root_state"][0, 3:7] = nut_quat
    cached_env_state.to_tensor(device=env.device)
    new_env_state = cached_env_state.apply(lambda x: repeat(x, "1 ... -> n ...", n=env.num_envs).clone())
    env.unwrapped.write_state(new_env_state)



@configclass
class EventCfg:
    """Configuration for events."""
    reset_default = EventTerm(
        func=reset_scene_with_grasping,
        mode="reset",
    )


@configclass
class IKRelKukaNutThreadEnv(BaseNutThreadEnvCfg):
    """Configuration for the IK-based relative Kuka nut threading environment."""
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.events = EventCfg()
        self.act_lows = [-0.001, -0.001, -0.001, -0.5, -0.5, -0.5]
        self.act_highs = [0.001, 0.001, 0.001, 0.5, 0.5, 0.5]
        self.scene.robot = KUKA_VICTOR_LEFT_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = [-0.15, -0.5, -0.8]
        self.sim.dt = 1/120
        
        # self.scene.robot.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
        #     contact_offset=0.002, rest_offset=0.001)
        scene_params = self.scene.scene_params
        self.scene.robot.spawn.collision_props.contact_offset = scene_params.get("contact_offset", 0.002)
        self.scene.robot.spawn.collision_props.rest_offset = scene_params.get("rest_offset", 0.001)

        self.scene.robot.spawn.rigid_props.max_depenetration_velocity = scene_params.get("max_depenetration_velocity", 0.5)
        self.scene.robot.spawn.rigid_props.sleep_threshold = scene_params.get("sleep_threshold", None)
        self.scene.robot.spawn.rigid_props.stabilization_threshold = scene_params.get("stabilization_threshold", None)
        
        self.scene.robot.spawn.physics_material = materials.RigidBodyMaterialCfg(
            static_friction=2, dynamic_friction=2, restitution=0.5
        )
        
        # self.scene.nut.spawn.rigid_props.max_depenetration_velocity = 0.2
        # self.scene.nut.spawn.rigid_props.sleep_threshold = 0.0025
        # self.scene.nut.spawn.rigid_props.stabilization_threshold = 0.0025
        # self.scene.nut.spawn.rigid_props.linear_damping = 0.1
        # self.scene.nut.spawn.rigid_props.angular_damping = 0.
        # self.scene.nut.spawn.collision_props = sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.00)
        # self.scene.nut.spawn.rigid_props.disable_gravity = False
        # self.scene.nut.init_state.pos = (0.5, 0, 0)
        # override actions
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["victor_left_arm_joint.*"],
            body_name="victor_left_tool0",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        )
    #     arm_joint_angles = [
    #   1.4693e+00, -4.3030e-01,  2.2680e+00,  1.5199e+00, -2.1248e+00,
    #       1.0958e+00,  3.9552e-01,
    #     ]
        arm_joint_angles = [
            1.4464e00,
            -4.6657e-01,
            2.2600e00,
            1.5216e00,
            -2.1492e00,
            1.1364e00,
            4.0521e-01,
            8.8714e-02,
            -8.8715e-02,
            7.3445e-01,
            7.3446e-01,
            7.3445e-01,
            -9.1062e-12,
            1.3638e-10,
            2.5490e-10,
            -7.3443e-01,
            -7.3443e-01,
            -7.3443e-01,
        ]
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
            is_accumulate_action=True,
        )
        self.viewer.eye = (0.4, 0, 0.2)
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
