# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from regex import F
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
from omni.isaac.lab.managers import EventTermCfg, ManagerTermBase

from omni.isaac.lab.assets import Articulation, RigidObject
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
from omni.isaac.lab_assets.kuka import KUKA_VICTOR_LEFT_HIGH_PD_CFG  # isort: skip


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


class reset_scene_to_grasp_state(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv, ):
        super().__init__(cfg, env)
        screw_type = self._env.cfg.scene.screw_type
        col_approx = env.cfg.params.scene.robot.collision_approximation  
        cached_pre_grasp_state = pickle.load(open(f"cached/{col_approx}/{screw_type}/kuka_pre_grasp.pkl", "rb"))
        cached_pre_grasp_state = SmartDict(cached_pre_grasp_state).to_tensor(device=env.device)
        self.cached_pre_grasp_state = cached_pre_grasp_state.apply(lambda x: repeat(x, "1 ... -> n ...", n=env.num_envs).clone())
        if os.path.exists(f"cached/{col_approx}/{screw_type}/kuka_grasp.pkl"):
            cached_grasp_state = pickle.load(open(f"cached/{col_approx}/{screw_type}/kuka_grasp.pkl", "rb"))
            cached_grasp_state = SmartDict(cached_grasp_state).to_tensor(device=env.device)
            self.cached_grasp_state = cached_grasp_state.apply(lambda x: repeat(x, "1 ... -> n ...", n=env.num_envs).clone())

    def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor, 
                 static_friction: float, dynamic_friction: float,
                 reset_target: str):
        # return
        # robot = env.unwrapped.scene["robot"]
        # robot_material = robot.root_physx_view.get_material_properties()
        # robot_material[..., 0] = static_friction
        # robot_material[..., 1] = dynamic_friction
        # robot.root_physx_view.set_material_properties(robot_material, torch.arange(env.scene.num_envs, device="cpu"))
        if reset_target == "pre_grasp":
            env.unwrapped.write_state(self.cached_pre_grasp_state[env_ids].clone(), env_ids)
        elif reset_target == "grasp":
            env.unwrapped.write_state(self.cached_grasp_state[env_ids].clone(), env_ids)
        # env.unwrapped.write_state(self.cached_grasp_state[env_ids], env_ids)

@configclass
class EventCfg:
    """Configuration for events."""

    
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

def terminate_if_far_from_bolt(env):
    diff = mdp.rel_nut_bolt_tip_distance(env)
    return torch.norm(diff, dim=-1) > 0.05

def initialize_contact_properties(env: ManagerBasedEnv, 
                                env_ids: torch.Tensor | None,
                                asset_cfg: SceneEntityCfg,
                                contact_offset: float, rest_offset: float):
    asset: RigidObject | Articulation = env.unwrapped.scene[asset_cfg.name]
    
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()
    # Note: shape of contact_offset and bodies are not matched
    # since there're several virtual body in kuka.
    # if asset_cfg.body_ids == slice(None):
    #     body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    # else:
    #     body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")
    cur_contact_offset = asset.root_physx_view.get_contact_offsets()
    cur_contact_offset[env_ids] = contact_offset
    asset.root_physx_view.set_contact_offsets(cur_contact_offset, env_ids)
    cur_rest_offset = asset.root_physx_view.get_rest_offset()
    cur_rest_offset[env_ids] = rest_offset
    asset.root_physx_view.set_rest_offset(cur_rest_offset, env_ids)

@configclass
class IKRelKukaNutThreadEnv(BaseNutThreadEnvCfg):
    """Configuration for the IK-based relative Kuka nut threading environment."""

    def get_default_env_params(self):
        super().get_default_env_params()
        self.params.sim.dt = self.params.sim.get("dt", 1.0 / 120.0)
        self.params.scene.robot = self.params.scene.get("robot", OmegaConf.create())
        # self.pre_grasp_path
        robot_params = self.params.scene.robot
        robot_params.collision_approximation = robot_params.get("collision_approximation", "convexHull")
        robot_params.contact_offset = robot_params.get("contact_offset", 0.002)
        robot_params.rest_offset = robot_params.get("rest_offset", 0.001)
        robot_params.max_depenetration_velocity = robot_params.get("max_depenetration_velocity", 0.5)
        robot_params.sleep_threshold = robot_params.get("sleep_threshold", None)
        robot_params.stabilization_threshold = robot_params.get("stabilization_threshold", None)
        robot_params.static_friction = robot_params.get("static_friction", 2)
        robot_params.dynamic_friction = robot_params.get("dynamic_friction", 2)
        robot_params.compliant_contact_stiffness = robot_params.get("compliant_contact_stiffness", 0.)
        robot_params.compliant_contact_damping = robot_params.get("compliant_contact_damping", 0.)
        robot_params.arm_stiffness = robot_params.get("arm_stiffness", 300.0)
        robot_params.arm_damping = robot_params.get("arm_damping", 100.0)
        robot_params.gripper_stiffness = robot_params.get("gripper_stiffness", 2e2)
        robot_params.gripper_damping = robot_params.get("gripper_damping", 1e2)
        robot_params.gripper_effort_limit = robot_params.get("gripper_effort_limit", 200.0)
        
        action_params = self.params.actions
        action_params.ik_lambda = action_params.get("ik_lambda", 0.1)
        action_params.keep_grasp_state = action_params.get("keep_grasp_state", False)
        
        rewards_params = self.params.rewards
        rewards_params.coarse_nut_w = rewards_params.get("coarse_nut_w", 0.5)
        rewards_params.fine_nut_w = rewards_params.get("fine_nut_w", 2.0)
        rewards_params.upright_reward_w = rewards_params.get("upright_reward_w", 1)
        rewards_params.task_success_w = rewards_params.get("task_success_w", 2.0)
        rewards_params.action_rate_w = rewards_params.get("action_rate_w", -0.000001)
        rewards_params.contact_force_penalty_w = rewards_params.get("contact_force_penalty_w", -0.0000001)
        
        termination_params = self.params.terminations
        termination_params.far_from_bolt = termination_params.get("far_from_bolt", True)
        termination_params.nut_fallen = termination_params.get("nut_fallen", False)
        
        events_params = self.params.events
        events_params.reset_target = events_params.get("reset_target", "grasp")
        
        
    def __post_init__(self):
        super().__post_init__()
        
        # robot
        self.scene.robot = KUKA_VICTOR_LEFT_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        robot = self.scene.robot
        robot_params = self.params.scene.robot
        if robot_params.collision_approximation == "convexHull":
            robot.spawn.usd_path = "assets/victor/victor_left_arm_with_gripper_v2/victor_left_arm_with_gripper_v2.usd"
        elif robot_params.collision_approximation == "convexHull2":
            robot.spawn.usd_path = "assets/victor/victor_left_arm/victor_left_arm.usd"
            # robot.spawn.usd_path = "assets/victor/victor_left_arm_v2/victor_left_arm_v2.usd"
        robot.init_state.pos = [-0.15, -0.5, -0.8]

        robot.spawn.collision_props.contact_offset = robot_params.contact_offset
        robot.spawn.collision_props.rest_offset = robot_params.rest_offset
        robot.spawn.rigid_props.max_depenetration_velocity = robot_params.max_depenetration_velocity
        robot.spawn.rigid_props.sleep_threshold = robot_params.sleep_threshold
        robot.spawn.rigid_props.stabilization_threshold = robot_params.stabilization_threshold

        # robot.spawn.physics_material = materials.RigidBodyMaterialCfg(
        #     static_friction=robot_params.static_friction,
        #     dynamic_friction=robot_params.dynamic_friction,
        #     compliant_contact_stiffness=robot_params.compliant_contact_stiffness,
        #     compliant_contact_damping=robot_params.compliant_contact_damping,
        # )
        robot.actuators["victor_left_arm"].stiffness = robot_params.arm_stiffness
        robot.actuators["victor_left_arm"].damping = robot_params.arm_damping
        robot.actuators["victor_left_gripper"].velocity_limit = 1
        robot.actuators["victor_left_gripper"].effort_limit = robot_params.gripper_effort_limit
        robot.actuators["victor_left_gripper"].stiffness = robot_params.gripper_stiffness
        robot.actuators["victor_left_gripper"].damping = robot_params.gripper_damping
        # action
        action_params = self.params.actions
        arm_lows = [-0.001, -0.001, -0.01, -0.001, -0.001, -0.8]
        arm_highs = [0.001, 0.001, 0.01, 0.001, 0.001, 0.0]
        scale = [0.001, 0.001, 0.01, 0.001, 0.001, 0.8]
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["victor_left_arm_joint.*"],
            body_name="victor_left_tool0",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, 
                ik_method="dls", ik_params={"lambda_val":action_params.ik_lambda}),
            lows=arm_lows,
            highs=arm_highs,
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
            keep_grasp_state=action_params.keep_grasp_state
        )
        
        # observations
        self.observations.policy.wrist_wrench = ObsTerm(
            func=mdp.body_incoming_wrench,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["victor_left_arm_flange"])},
            scale=1
        )
        self.observations.policy.tool_pose = ObsTerm(
            func=robot_tool_pose,
        )
        
        # events
        self.events = EventCfg()
        self.events.set_robot_collision = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (robot_params.static_friction, robot_params.static_friction),
                    "dynamic_friction_range": (robot_params.dynamic_friction, robot_params.dynamic_friction),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 1},
        )
        # self.events.set_robot_properties = EventTerm(
        #     func=initialize_contact_properties,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names=".*finger.*"),
        #         "contact_offset": robot_params.contact_offset,
        #         "rest_offset": robot_params.rest_offset},
        #     mode="startup",
        # )
        self.events.reset_default = EventTerm(
            func=reset_scene_to_grasp_state,
            params={"reset_target": self.params.events.reset_target,
                    "static_friction": robot_params.static_friction,
                    "dynamic_friction": robot_params.dynamic_friction},
            mode="reset",
        )
        
        # terminations
        termination_params = self.params.terminations
        if termination_params.nut_fallen:
            self.terminations.nut_fallen = DoneTerm(func=terminate_if_nut_fallen)
        if termination_params.far_from_bolt:
            self.terminations.far_from_bolt = DoneTerm(func=terminate_if_far_from_bolt)
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
        
        # rewards
        rewards_params = self.params.rewards
        self.rewards.coarse_nut.weight = rewards_params.coarse_nut_w
        self.rewards.fine_nut.weight = rewards_params.fine_nut_w
        self.rewards.upright_reward.weight = rewards_params.upright_reward_w
        self.rewards.task_success.weight = rewards_params.task_success_w
        self.rewards.action_rate.weight = rewards_params.action_rate_w
        self.rewards.contact_force_penalty = RewTerm(
            func=mdp.contact_forces,
            params={"threshold":1, "sensor_cfg": SceneEntityCfg(name="contact_sensor")},
            weight=rewards_params.contact_force_penalty_w)
        self.viewer.eye = (0.3, 0, 0.15)