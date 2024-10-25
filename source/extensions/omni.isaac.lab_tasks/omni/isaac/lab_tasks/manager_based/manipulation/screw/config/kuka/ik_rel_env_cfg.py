# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
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
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
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
from omegaconf import OmegaConf
##
# Pre-defined configs
from omni.isaac.lab_assets.kuka import KUKA_VICTOR_LEFT_HIGH_PD_CFG, KUKA_VICTOR_LEFT_CFG  # isort: skip


@configclass
class IKRelKukaNutTightenEnvCfg(BaseNutTightenEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.act_lows = [-0.001, -0.001, -0.001, -0.1, -0.1, -0.1]
        self.act_highs = [0.001, 0.001, 0.001, 0.1, 0.1, 0.1]
        # Set Kuka as robot

        self.scene.robot = KUKA_VICTOR_LEFT_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = [-0.25, -0.2, -0.8]
        scale = [0.001, 0.001, 0.001, 0.01, 0.01, 0.8]
        
        # # override actions
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["victor_left_arm_joint.*"],
            body_name="victor_left_tool0",
            controller=DifferentialIKControllerCfg(command_type="pose",
                                                   use_relative_mode=True, ik_method="dls"),
            scale=scale,
        )
        self.actions.gripper_action = mdp.Robotiq3FingerActionCfg(
            asset_name="robot",
            side="left",
            lows=self.act_lows,
            highs=self.act_highs,
        )

from omni.isaac.lab.managers import EventTermCfg, ManagerTermBase
class reset_scene_to_grasp_state(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        screw_type = self._env.cfg.scene.screw_type
        subdir = self._env.cfg.env_params.scene.robot.collision_approximation  
        cached_pre_grasp_state = pickle.load(open(f"cached/{subdir}/kuka_{screw_type}_pre_grasp.pkl", "rb"))
        cached_pre_grasp_state = SmartDict(cached_pre_grasp_state).to_tensor(device=env.device)
        self.cached_pre_grasp_state = cached_pre_grasp_state.apply(lambda x: repeat(x, "1 ... -> n ...", n=env.num_envs).clone())
        if os.path.exists(f"cached/{subdir}/kuka_{screw_type}_grasp.pkl"):
            cached_grasp_state = pickle.load(open(f"cached/{subdir}/kuka_{screw_type}_grasp.pkl", "rb"))
            cached_grasp_state = SmartDict(cached_grasp_state).to_tensor(device=env.device)
            self.cached_grasp_state = cached_grasp_state.apply(lambda x: repeat(x, "1 ... -> n ...", n=env.num_envs).clone())

    def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor):
        robot = env.unwrapped.scene["robot"]
        robot_material = robot.root_physx_view.get_material_properties()
        robot_material[..., 0] = 2
        robot_material[..., 1] = 2
        robot.root_physx_view.set_material_properties(robot_material, torch.arange(env.scene.num_envs, device="cpu"))

        env.unwrapped.write_state(self.cached_pre_grasp_state[env_ids].clone(), env_ids)
        # env.unwrapped.write_state(self.cached_grasp_state[env_ids].clone(), env_ids)

@configclass
class EventCfg:
    """Configuration for events."""
    reset_default = EventTerm(
        # func=reset_scene_with_grasping,
        func=reset_scene_to_grasp_state,
        mode="reset",
    )
    
def robot_tool_pose(env: ManagerBasedEnv):
    return env.unwrapped.scene["robot"].read_body_state_w("victor_left_tool0")[:, 0]


def terminate_if_nut_fallen(env):
    # relative pose between gripper and nut
    nut_root_pose = env.unwrapped.scene["nut"].read_root_state_from_sim()
    gripper_state_w = robot_tool_pose(env)
    relative_pos = nut_root_pose[:, :3] - gripper_state_w[:, :3]
    relative_pos, relative_quat = math_utils.subtract_frame_transforms(
        nut_root_pose[:, :3], nut_root_pose[:, 3:7],
        gripper_state_w[:, :3], gripper_state_w[:, 3:7]
    )
    ideal_relative_pose = torch.tensor([[ 0.0018,  0.9668, -0.2547, -0.0181]], device=env.device)
    ideal_relative_pose = ideal_relative_pose.repeat(relative_pos.shape[0], 1)
    quat_dis = math_utils.quat_error_magnitude(relative_quat, ideal_relative_pose)
    dis = torch.norm(relative_pos, dim=-1)
    return torch.logical_or(dis > 0.03 , quat_dis > 0.3)

def terminate_if_far_from_nut(env):
    diff = mdp.rel_nut_bolt_tip_distance(env)
    return torch.norm(diff, dim=-1) > 0.05

@configclass
class IKRelKukaNutThreadEnv(BaseNutThreadEnvCfg):
    """Configuration for the IK-based relative Kuka nut threading environment."""

    def get_default_env_params(self):
        super().get_default_env_params()
        self.env_params.sim.dt = self.env_params.sim.get("dt", 1.0 / 120.0)
        self.env_params.scene.robot = self.env_params.scene.get("robot", OmegaConf.create())
        # self.pre_grasp_path
        robot_params = self.env_params.scene.robot
        robot_params["collision_approximation"] = robot_params.get("collision_approximation", "convexHull")
        robot_params["contact_offset"] = robot_params.get("contact_offset", 0.002)
        robot_params["rest_offset"] = robot_params.get("rest_offset", 0.001)
        robot_params["max_depenetration_velocity"] = robot_params.get("max_depenetration_velocity", 0.5)
        robot_params["sleep_threshold"] = robot_params.get("sleep_threshold", None)
        robot_params["stabilization_threshold"] = robot_params.get("stabilization_threshold", None)
        robot_params["static_friction"] = robot_params.get("static_friction", 2)
        robot_params["dynamic_friction"] = robot_params.get("dynamic_friction", 2)
        robot_params["compliant_contact_stiffness"] = robot_params.get("compliant_contact_stiffness", 0.)
        robot_params["compliant_contact_damping"] = robot_params.get("compliant_contact_damping", 0.)

        # By default use the default params in USD
        nut_params = self.env_params.scene.nut
        nut_params["max_depenetration_velocity"] = nut_params.get("max_depenetration_velocity", None)
        nut_params["sleep_threshold"] = nut_params.get("sleep_threshold", None)
        nut_params["stabilization_threshold"] = nut_params.get("stabilization_threshold", None)
        nut_params["linear_damping"] = nut_params.get("linear_damping", None)
        nut_params["angular_damping"] = nut_params.get("angular_damping", None)

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.events = EventCfg()
        robot_params = self.env_params.scene.robot
        self.scene.robot = KUKA_VICTOR_LEFT_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if robot_params.collision_approximation == "convexHull":
            self.scene.robot.spawn.usd_path = "assets/victor/victor_left_arm_with_gripper_v2/victor_left_arm_with_gripper_v2.usd"
        elif robot_params.collision_approximation == "convexHull2":
            self.scene.robot.spawn.usd_path = "assets/victor/victor_left_arm/victor_left_arm.usd"
        self.scene.robot.init_state.pos = [-0.15, -0.5, -0.8]

        self.scene.robot.spawn.collision_props.contact_offset = robot_params.contact_offset
        self.scene.robot.spawn.collision_props.rest_offset = robot_params.rest_offset
        self.scene.robot.spawn.rigid_props.max_depenetration_velocity = robot_params.max_depenetration_velocity
        self.scene.robot.spawn.rigid_props.sleep_threshold = robot_params.sleep_threshold
        self.scene.robot.spawn.rigid_props.stabilization_threshold = robot_params.stabilization_threshold

        self.scene.robot.spawn.physics_material = materials.RigidBodyMaterialCfg(
            static_friction=robot_params.static_friction,
            dynamic_friction=robot_params.dynamic_friction,
            compliant_contact_stiffness=robot_params.compliant_contact_stiffness,
            compliant_contact_damping=robot_params.compliant_contact_damping,
        )

        nut_params = self.env_params.scene.nut
        self.scene.nut.spawn.rigid_props.max_depenetration_velocity = nut_params.max_depenetration_velocity
        self.scene.nut.spawn.rigid_props.sleep_threshold = nut_params.sleep_threshold
        self.scene.nut.spawn.rigid_props.stabilization_threshold = nut_params.stabilization_threshold
        self.scene.nut.spawn.rigid_props.linear_damping = nut_params.linear_damping
        self.scene.nut.spawn.rigid_props.angular_damping = nut_params.angular_damping

        # override actions
        
        self.scene.robot.actuators["victor_left_arm"].stiffness = 300.0
        self.scene.robot.actuators["victor_left_arm"].damping = 100.0
        self.scene.robot.actuators["victor_left_gripper"].velocity_limit = 1
        self.act_lows = [-0.0001, -0.0001, -0.015, -0.01, -0.01, -0.8]
        self.act_highs = [0.0001, 0.0001, 0.015, 0.01, 0.01, 0.]
        scale = [0.001, 0.001, 0.01, 0.01, 0.01, 0.8]
        # self.act_lows = [-0.003, -0.003, -0.01, -0.01, -0.01, -0.2]
        # self.act_highs = [0.003, 0.003, 0.01, 0.01, 0.01, 0.]
        # scale = [0.003, 0.003, 0.01, 0.01, 0.01, 0.2]

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["victor_left_arm_joint.*"],
            body_name="victor_left_tool0",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, 
                                                   ik_method="dls", ik_params={"lambda_val": 0.1}),
            scale=scale,
        )

        self.gripper_act_lows = [-0.005, -0.005]
        self.gripper_act_highs = [0.005, 0.005]
        self.actions.gripper_action = mdp.Robotiq3FingerActionCfg(
            asset_name="robot",
            side="left",
            lows=self.gripper_act_lows,
            highs=self.gripper_act_highs,
            use_relative_mode=True,
            is_accumulate_action=True,
            keep_grasp_state=True
        )
        self.viewer.eye = (0.3, 0, 0.15)
        
    
        # additional observations
        self.observations.policy.wrist_wrench = ObsTerm(
            func=mdp.body_incoming_wrench,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["victor_left_arm_flange"])},
            scale=1
        )
        self.observations.policy.tool_pose = ObsTerm(
            func=robot_tool_pose,
        )
        
        # terminations
        # self.terminations.nut_fallen = DoneTerm(func=terminate_if_nut_fallen)
        self.terminations.far_from_nut = DoneTerm(func=terminate_if_far_from_nut)
        self.scene.nut.spawn.activate_contact_sensors = True

        # self.scene.contact_sensor = ContactSensorCfg(
        #     prim_path="{ENV_REGEX_NS}/Nut/factory_nut",
        #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*finger.*_link_3"],
        #     update_period=0.0,
        #     max_contact_data_count=512,
        # )
        self.scene.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Nut/factory_nut",
            filter_prim_paths_expr= ["{ENV_REGEX_NS}/Bolt/factory_bolt"],
            update_period=0.0,
        )
        self.rewards.contact_force_penalty = RewTerm(
            func=mdp.contact_forces,
            params={"threshold":1, "sensor_cfg": SceneEntityCfg(name="contact_sensor")},
            weight=0.0000001)
        self.rewards.action_rate.weight =-0.00000001