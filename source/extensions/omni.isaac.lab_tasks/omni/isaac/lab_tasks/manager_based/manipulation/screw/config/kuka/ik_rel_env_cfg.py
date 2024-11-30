# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import pickle
from tracemalloc import start
import torch
from typing import Literal, Sequence

from numba.core import event
import omni.isaac.core.utils.stage as stage_utils
import omni.physx.scripts.utils as physx_utils
from einops import repeat
from force_tool.utils.data_utils import SmartDict, read_h5_dict
from force_tool.utils.curobo_utils import CuRoboArm
from omegaconf import OmegaConf
from pxr import Usd, UsdGeom
from regex import F

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ManagerTermBase
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.utils import  configclass

import omni.isaac.lab_tasks.manager_based.manipulation.screw.mdp as mdp
from omni.isaac.lab_tasks.manager_based.manipulation.screw.screw_env_cfg import (
    BaseNutThreadEnvCfg,
    BaseNutTightenEnvCfg,
)
from omni.isaac.lab.utils.noise import GaussianNoiseCfg
from omni.isaac.lab.utils.modifiers import NoiseModifierCfg
from curobo.types.math import Pose
from curobo.types.base import TensorDeviceType
import time
import numpy as np
##
# Pre-defined configs
from omni.isaac.lab_assets.kuka import KUKA_VICTOR_LEFT_HIGH_PD_CFG



@configclass
class IKRelKukaNutTightenEnvCfg(BaseNutTightenEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.act_lows = [-0.001, -0.001, -0.001, -0.1, -0.1, -0.1]
        self.act_highs = [0.001, 0.001, 0.001, 0.1, 0.1, 0.1]

        self.scene.robot = KUKA_VICTOR_LEFT_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = [-0.25, -0.2, -0.8]
        scale = [0.001, 0.001, 0.001, 0.01, 0.01, 0.8]

        # # override actions
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["victor_left_arm_joint.*"],
            body_name="victor_left_tool0",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=scale,
        )
        self.actions.gripper_action = mdp.Robotiq3FingerActionCfg(
            asset_name="robot",
            side="left",
            lows=self.act_lows,
            highs=self.act_highs,
        )


class GraspResetEventTermCfg(EventTerm):
    def __init__(
        self,
        reset_target: Literal["pre_grasp", "grasp", "mate", "rigid_grasp", "rigid_grasp_open_align"] = "grasp",
        reset_range_scale: float = 1.0,
        reset_joint_std: float = 0.0,
        reset_randomize_mode: Literal["task", "joint", None] = "task",
        reset_use_adr: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reset_target = reset_target
        self.reset_range_scale = reset_range_scale
        self.reset_joint_std = reset_joint_std
        self.reset_randomize_mode = reset_randomize_mode
        self.reset_use_adr = reset_use_adr


class reset_scene_to_grasp_state(ManagerTermBase):
    def __init__(self, cfg: GraspResetEventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        screw_type = env.cfg.scene.screw_type
        col_approx = env.cfg.params.scene.robot.collision_approximation
        cached_state = pickle.load(open(f"cached/{col_approx}/{screw_type}/kuka_{cfg.reset_target}.pkl", "rb"))
        cached_state = SmartDict(cached_state).to_tensor(device=env.device)
        self.cached_state = cached_state.apply(lambda x: repeat(x, "1 ... -> n ...", n=env.num_envs).clone())
        
        # randomization parameters
        tensor_args = TensorDeviceType(device=env.device)
        self.robot_base_pose = Pose.from_list([-0.15, -0.5, -0.8, 1, 0, 0, 0], tensor_args)
        self.curobo_arm = CuRoboArm("victor_left.yml", 
                                    external_asset_path="/home/zixuanh/force_tool/assets/victor",
                                    base_pose=self.robot_base_pose, num_ik_seeds=10, device=env.device,
                                    )
        self.curobo_arm.update_world()
        self.reset_randomize_mode = cfg.reset_randomize_mode
        self.reset_trans_low = torch.tensor([-0.03, -0.03, -0.0], device=env.device) * cfg.reset_range_scale
        self.reset_trans_high = torch.tensor([0.03, 0.03, 0.04], device=env.device) * cfg.reset_range_scale
        # TODO: add rotation noise
        self.reset_rot_std = 0.5
        self.reset_joint_std = cfg.reset_joint_std
        self.reset_use_adr = cfg.reset_use_adr
        self.rand_init_configurations = None
        self.num_buckets = int(5e3)
        self.bucket_update_freq = 4
        # self.update_random_initializations(env)

    def update_random_initializations(self, env:ManagerBasedEnv):
        cached_state = self.cached_state[0:1].clone()
        B = self.num_buckets
        noise_scale = 1.
        if self.reset_use_adr:
            # step a: activate noise
            # step b: maximize noise
            raise NotImplementedError
        
        if self.reset_randomize_mode == "task":
            arm_state = cached_state["robot"]["joint_state"]["position"][:, :7]
            default_tool_pose = self.curobo_arm.forward_kinematics(arm_state).ee_pose
            default_tool_pose = default_tool_pose.repeat(B)
            delta_trans = torch.rand((B, 3), device=env.device) * (self.reset_trans_high - self.reset_trans_low) + self.reset_trans_low
            delta_trans *= noise_scale
            delta_quat = torch.zeros((B, 4), device=env.device)
            delta_quat[:, 0] = 1
            delta_pose = Pose(position=delta_trans, quaternion=delta_quat)
            randomized_tool_pose = delta_pose.multiply(default_tool_pose)
            ik_result = self.curobo_arm.compute_ik(randomized_tool_pose)
            randomized_joint_state = ik_result.solution.squeeze(1)
        elif self.reset_randomize_mode == "joint":
            randomized_joint_state = torch.randn_like(arm_state) * self.reset_joint_std + arm_state
        else:
            arm_state = cached_state["robot"]["joint_state"]["position"][:, :7].repeat(B, 1)
            randomized_joint_state = arm_state
        self.rand_init_configurations = randomized_joint_state.detach().cpu().numpy()
        
    def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor):
        cached_state = self.cached_state[env_ids].clone()
        global_step = env._sim_step_counter // env.cfg.decimation
        if global_step % int(self.bucket_update_freq * env.unwrapped.max_episode_length) == 0:
            with torch.inference_mode(False):
                self.update_random_initializations(env)
        if self.reset_randomize_mode is not None:
            # draw random initializations
            # select = torch.randint(0, self.num_buckets, (env_ids.shape[0],), device=env.device)
            # randomized_joint_state = self.rand_init_configurations[select].clone()
            select = np.random.choice(self.num_buckets, env_ids.shape[0], replace=True)
            randomized_joint_state = self.rand_init_configurations[select].copy()
            randomized_joint_state = torch.tensor(randomized_joint_state, device=env.device)
            cached_state["robot"]["joint_state"]["position"][:, :7] = randomized_joint_state
            cached_state["robot"]["joint_state"]["position_target"][:, :7] = randomized_joint_state
        
        env.unwrapped.write_state(cached_state, env_ids)

class DTWReferenceTrajRewardCfg(RewTerm):
    def __init__(self, his_traj_len: int = 10, soft_dtw_gamma: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.his_traj_len = his_traj_len
        self.soft_dtw_gamma = soft_dtw_gamma


class DTWReferenceTrajReward(ManagerTermBase):
    def __init__(self, cfg: DTWReferenceTrajRewardCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        screw_type = env.cfg.scene.screw_type
        col_approx = env.cfg.params.scene.robot.collision_approximation
        ref_traj_path = f"cached/{col_approx}/{screw_type}/kuka_disassembly.h5"
        ref_traj = read_h5_dict(ref_traj_path)
        self.nut_ref_pos_traj = torch.tensor(ref_traj["pos"], device=env.device).flip(1)
        self.nut_ref_quat_traj = torch.tensor(ref_traj["quat"], device=env.device).flip(1)
        self.nut_traj_his = torch.zeros((env.num_envs, cfg.his_traj_len, 3), device=env.device)
        self.soft_dtw_criterion = mdp.SoftDTW(use_cuda=True, gamma=cfg.soft_dtw_gamma)

    def reset(self, env_ids: torch.Tensor):
        scene = self._env.unwrapped.scene
        nut_frame = scene["nut_frame"]
        nut_cur_pos = nut_frame.data.target_pos_w - scene.env_origins[:, None]
        self.nut_traj_his[env_ids] = nut_cur_pos[env_ids]

    def __call__(self, env: ManagerBasedEnv):
        nut_frame = env.unwrapped.scene["nut_frame"]
        cur_nut_pos = nut_frame.data.target_pos_w[:, 0] - env.unwrapped.scene.env_origins

        imitation_rwd, new_nut_traj_his = mdp.get_imitation_reward_from_dtw(
            self.nut_ref_pos_traj, cur_nut_pos, self.nut_traj_his, self.soft_dtw_criterion, env.device
        )
        # imitation_rwd2,_ = mdp.get_imitation_reward_from_dtw_v2(self.nut_ref_pos_traj, cur_nut_pos,
        #                                                 self.nut_traj_his,
        #                                                 self.soft_dtw_criterion,
        #                                                 env.device)
        # print(torch.norm(imitation_rwd - imitation_rwd2, dim=-1))
        self.nut_traj_his = new_nut_traj_his
        return imitation_rwd


def spawn_nut_with_rigid_grasp(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    stage = stage_utils.get_current_stage()
    tool_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot/victor_left_tool0")
    xfCache = UsdGeom.XformCache()
    tool_pose = xfCache.GetLocalToWorldTransform(tool_prim)
    tool_pos = tuple(tool_pose.ExtractTranslation())
    tool_pos = torch.tensor(tool_pos)[None]
    tool_quat = tool_pose.ExtractRotationQuat()
    tool_quat = [tool_quat.real, tool_quat.imaginary[0], tool_quat.imaginary[1], tool_quat.imaginary[2]]
    tool_quat = torch.tensor(tool_quat)[None]

    # grasp_rel_pos = torch.tensor([[-0.0007,  0.0001,  0.0177]])
    # grasp_rel_quat = torch.tensor([[-0.0020,  0.9663, -0.2569, -0.0170]])

    grasp_rel_pos = torch.tensor([[0, 0.000, 0.02]])
    grasp_rel_quat = torch.tensor([[0, 1, 0, 0]])

    nut_pos, nut_quat = math_utils.combine_frame_transforms(tool_pos, tool_quat, grasp_rel_pos, grasp_rel_quat)

    nut_prim = sim_utils.spawn_from_usd(prim_path, cfg, nut_pos[0], nut_quat[0])
    return nut_prim


def create_fixed_joint(env: ManagerBasedEnv, env_ids: torch.Tensor):
    stage = stage_utils.get_current_stage()
    for i in range(env.num_envs):
        child_prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot/victor_left_tool0")
        parent_prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Nut/factory_nut")
        physx_utils.createJoint(stage, "Fixed", child_prim, parent_prim)


@configclass
class EventCfg:
    """Configuration for events."""


def robot_tool_pose(env: ManagerBasedEnv):
    tool_w = env.unwrapped.scene["robot"].read_body_state_w("victor_left_tool0")[:, 0, :7]
    tool_w[:, :3] = tool_w[:, :3] - env.unwrapped.scene.env_origins
    return tool_w


def terminate_if_nut_fallen(env):
    # relative pose between gripper and nut
    nut_root_pose = env.unwrapped.scene["nut"].read_root_state_from_sim()
    gripper_state_w = robot_tool_pose(env)
    # relative_pos = nut_root_pose[:, :3] - gripper_state_w[:, :3]
    relative_pos, relative_quat = math_utils.subtract_frame_transforms(
        nut_root_pose[:, :3], nut_root_pose[:, 3:7], gripper_state_w[:, :3], gripper_state_w[:, 3:7]
    )
    ideal_relative_pose = torch.tensor([[0.0018, 0.9668, -0.2547, -0.0181]], device=env.device)
    ideal_relative_pose = ideal_relative_pose.repeat(relative_pos.shape[0], 1)
    quat_dis = math_utils.quat_error_magnitude(relative_quat, ideal_relative_pose)
    dis = torch.norm(relative_pos, dim=-1)
    return torch.logical_or(dis > 0.03, quat_dis > 0.3)


def terminate_if_far_from_bolt(env):
    diff = mdp.rel_nut_bolt_tip_distance(env)
    return torch.norm(diff, dim=-1) > 0.05


def initialize_contact_properties(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    contact_offset: float,
    rest_offset: float,
):
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

# curriculum
def modify_noise_scale(
    env: ManagerBasedEnv, 
    env_ids: torch.Tensor,
    begin_steps: int, end_steps: int
):
    modifier = env.observation_manager.get_term_cfg("policy", "nut_pos").modifiers[0].func
    # linear
    scale = max(0., env.common_step_counter-begin_steps)/(begin_steps-end_steps)
    scale = min(1., scale)
    modifier.noise_scale = scale

@configclass
class IKRelKukaNutThreadEnvCfg(BaseNutThreadEnvCfg):
    """Configuration for the IK-based relative Kuka nut threading environment."""

    def get_default_env_params(self):
        super().get_default_env_params()
        self.params.sim.dt = self.params.sim.get("dt", 1.0 / 120.0)
        self.params.scene.robot = self.params.scene.get("robot", OmegaConf.create())
        # self.pre_grasp_path
        robot_params = self.params.scene.robot
        robot_params.collision_approximation = robot_params.get("collision_approximation", "convexHull2")
        robot_params.contact_offset = robot_params.get("contact_offset", 0.002)
        robot_params.rest_offset = robot_params.get("rest_offset", 0.001)
        robot_params.max_depenetration_velocity = robot_params.get("max_depenetration_velocity", 0.5)
        robot_params.sleep_threshold = robot_params.get("sleep_threshold", None)
        robot_params.stabilization_threshold = robot_params.get("stabilization_threshold", None)
        robot_params.static_friction = robot_params.get("static_friction", 3)
        robot_params.dynamic_friction = robot_params.get("dynamic_friction", 3)
        robot_params.compliant_contact_stiffness = robot_params.get("compliant_contact_stiffness", 0.0)
        robot_params.compliant_contact_damping = robot_params.get("compliant_contact_damping", 0.0)
        robot_params.arm_stiffness = robot_params.get("arm_stiffness", 300.0)
        robot_params.arm_damping = robot_params.get("arm_damping", 100.0)
        robot_params.gripper_stiffness = robot_params.get("gripper_stiffness", 100)
        robot_params.gripper_damping = robot_params.get("gripper_damping", 1e2)
        robot_params.gripper_effort_limit = robot_params.get("gripper_effort_limit", 200.0)

        nut_params = self.params.scene.nut
        nut_params.rigid_grasp = nut_params.get("rigid_grasp", True)

        action_params = self.params.actions
        action_params.ik_lambda = action_params.get("ik_lambda", 0.1)
        action_params.keep_grasp_state = action_params.get("keep_grasp_state", False)
        action_params.uni_rotate = action_params.get("uni_rotate", False)

        obs_params = self.params.observations
        obs_params.hist_len = obs_params.get("hist_len", 1)
        obs_params.include_action = obs_params.get("include_action", True)
        obs_params.include_wrench = obs_params.get("include_wrench", True)
        obs_params.include_tool = obs_params.get("include_tool", False)
        obs_params.nut_pos = obs_params.get("nut_pos", OmegaConf.create())
        obs_params.nut_pos.noise_std = obs_params.nut_pos.get("noise_std", 0.0)
        obs_params.nut_pos.bias_std = obs_params.nut_pos.get("bias_std", 0.0)

        rewards_params = self.params.rewards
        rewards_params.dtw_ref_traj_w = rewards_params.get("dtw_ref_traj_w", 0.0)
        rewards_params.coarse_nut_w = rewards_params.get("coarse_nut_w", 1)
        rewards_params.fine_nut_w = rewards_params.get("fine_nut_w", 2.0)
        rewards_params.upright_reward_w = rewards_params.get("upright_reward_w", 0.3)
        rewards_params.success_w = rewards_params.get("success_w", 1.0)
        rewards_params.action_rate_w = rewards_params.get("action_rate_w", -0.0)
        rewards_params.contact_force_penalty_w = rewards_params.get("contact_force_penalty_w", -0.01)

        termination_params = self.params.terminations
        termination_params.far_from_bolt = termination_params.get("far_from_bolt", False)
        termination_params.nut_fallen = termination_params.get("nut_fallen", False)

        events_params = self.params.events
        events_params.reset_target = events_params.get("reset_target", "grasp")
        events_params.reset_range_scale = events_params.get("reset_range_scale", 1.0)
        events_params.reset_randomize_mode = events_params.get("reset_randomize_mode", None)
        events_params.reset_joint_std = events_params.get("reset_joint_std", 0.0)
        events_params.reset_use_adr = events_params.get("reset_use_adr", False)
        
        curri_params = self.params.curriculum
        curri_params.use_obs_noise_curri = curri_params.get("use_obs_noise_curri", False)
        curri_params.use_contact_force_curri = curri_params.get("use_contact_force_curri", False)
        
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
        # arm_lows = [-0.002, -0.002, -0.002, -0.0005, -0.0005, -0.5]
        # arm_highs = [0.002, 0.002, 0.002, 0.0005, 0.0005, 0.5]
        # scale = [0.002, 0.002, 0.002, 0.0005, 0.0005, 0.5]
        # arm_lows = [-0.004, -0.004, -0.002, -0.5, -0.5, -0.5]
        # arm_highs = [0.004, 0.004, 0.002, 0.5, 0.5, 0.5]
        # scale = [0.004, 0.004, 0.002, 0.5, 0.5, 0.5]
        
        # arm_lows = [-0.002, -0.002, -0.002, -0.01, -0.01, -0.5]
        # arm_highs = [0.002, 0.002, 0.002, 0.01, 0.01, 0.5]
        # scale = [0.002, 0.002, 0.002, 0.01, 0.01, 0.5]
        
        arm_lows = [-0.004, -0.004, -0.004, -0.01, -0.01, -0.5]
        arm_highs = [0.004, 0.004, 0.004, 0.01, 0.01, 0.5]
        scale = [0.004, 0.004, 0.004, 0.01, 0.01, 0.5]


        if self.params.events.reset_target == "rigid_grasp_open_tilt" or \
                self.params.events.reset_joint_std > 0:
            arm_lows = [-0.002, -0.002, -0.002, -0.01, -0.01, -0.5]
            arm_highs = [0.002, 0.002, 0.002, 0.01, 0.01, 0.5]
            scale = [0.002, 0.002, 0.002, 0.01, 0.01, 0.5]

        if action_params.uni_rotate:
            arm_highs[5] = 0.0

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["victor_left_arm_joint.*"],
            body_name="victor_left_tool0",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
                ik_params={"lambda_val": action_params.ik_lambda},
            ),
            lows=arm_lows,
            highs=arm_highs,
            scale=scale,
        )

        # self.gripper_act_lows = [-0.005, -0.005]
        # self.gripper_act_highs = [0.005, 0.005]
        # self.actions.gripper_action = mdp.Robotiq3FingerActionCfg(
        #     asset_name="robot",
        #     side="left",
        #     lows=self.gripper_act_lows,
        #     highs=self.gripper_act_highs,
        #     use_relative_mode=True,
        #     is_accumulate_action=True,
        #     keep_grasp_state=action_params.keep_grasp_state
        # )

        nut_params = self.params.scene.nut
        if nut_params.rigid_grasp:
            self.scene.nut.spawn.func = spawn_nut_with_rigid_grasp

        # observations
        obs_params = self.params.observations
        if obs_params.include_wrench:
            self.observations.policy.wrist_wrench = ObsTerm(
                func=mdp.body_incoming_wrench,
                params={"asset_cfg": SceneEntityCfg("robot", body_names=["victor_left_arm_flange"])},
                scale=1,
            )
        if obs_params.include_tool:
            self.observations.policy.tool_pose = ObsTerm(
                func=robot_tool_pose)
        if obs_params.include_action:
            self.observations.policy.last_action = ObsTerm(
                func=mdp.last_action,
                params={"action_name": "arm_action"},
                scale=1,
            )
            
        self.observations.policy.nut_pos.modifiers = [NoiseModifierCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=obs_params.nut_pos.noise_std, operation="add"),
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=obs_params.nut_pos.bias_std, operation="abs"),
        )]
            
        for term in self.observations.policy.__dict__.values():
            if isinstance(term, ObsTerm):
                term.hist_len = obs_params.hist_len

        # events
        event_params = self.params.events
        self.events = EventCfg()
        self.events.set_robot_collision = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (robot_params.static_friction, robot_params.static_friction),
                "dynamic_friction_range": (robot_params.dynamic_friction, robot_params.dynamic_friction),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 1,
            },
        )
        if nut_params.rigid_grasp:
            self.events.set_robot_properties = EventTerm(
                func=create_fixed_joint,
                mode="startup",
            )
        # self.events.set_robot_properties = EventTerm(
        #     func=initialize_contact_properties,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names=".*finger.*"),
        #         "contact_offset": robot_params.contact_offset,
        #         "rest_offset": robot_params.rest_offset},
        #     mode="startup",
        # )
        self.events.reset_default = GraspResetEventTermCfg(
            func=reset_scene_to_grasp_state,
            mode="reset",
            reset_target=event_params.reset_target,
            reset_range_scale=event_params.reset_range_scale,
            reset_randomize_mode=event_params.reset_randomize_mode,
            reset_joint_std=event_params.reset_joint_std,
            reset_use_adr=event_params.reset_use_adr,
        )

        # terminations
        termination_params = self.params.terminations
        if termination_params.nut_fallen:
            self.terminations.nut_fallen = DoneTerm(func=terminate_if_nut_fallen)
        if termination_params.far_from_bolt:
            self.terminations.far_from_bolt = DoneTerm(func=terminate_if_far_from_bolt)
        self.scene.nut.spawn.activate_contact_sensors = True


        self.scene.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Nut/factory_nut",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Bolt/factory_bolt"],
            update_period=0.0,
        )

        # rewards
        rewards_params = self.params.rewards
        self.rewards.coarse_nut.weight = rewards_params.coarse_nut_w
        self.rewards.fine_nut.weight = rewards_params.fine_nut_w
        self.rewards.upright_reward.weight = rewards_params.upright_reward_w
        self.rewards.success.weight = rewards_params.success_w
        self.rewards.action_rate.weight = rewards_params.action_rate_w
        if rewards_params.dtw_ref_traj_w > 0:
            self.rewards.dtw_ref_traj = DTWReferenceTrajRewardCfg(
                his_traj_len=10,
                func=DTWReferenceTrajReward,
                weight=rewards_params.dtw_ref_traj_w,
            )
        self.rewards.contact_force_penalty = RewTerm(
            func=mdp.contact_forces,
            params={"threshold": 0, "sensor_cfg": SceneEntityCfg(name="contact_sensor")},
            weight=rewards_params.contact_force_penalty_w,
        )
        self.viewer.eye = (0.3, 0, 0.15)

        # curriculum
        curri_params = self.params.curriculum
        if curri_params.use_obs_noise_curri:
            self.curriculum.modify_nut_pos_noise = CurrTerm(
                func=modify_noise_scale,
                params={"begin_steps": 500*32, "end_steps": 2000*32},
            )
        if curri_params.use_contact_force_curri:
            self.curriculum.modify_contact_force_penalty = CurrTerm(
                func=mdp.modify_reward_weight,
                params={"term_name": "contact_force_penalty", "weight": -10, "num_steps": 800*32},
            )