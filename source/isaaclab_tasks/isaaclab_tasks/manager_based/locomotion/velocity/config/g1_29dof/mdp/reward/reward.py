# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import os
import torch
from typing import TYPE_CHECKING, Tuple

from isaaclab.envs import mdp
import isaaclab.utils.math as math_utils
from isaaclab.utils.string import resolve_matching_names_values
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat, euler_xyz_from_quat
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
base rewards
"""

def body_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Reward based on the L2 norm of the body orientation deviation from upright.
    
    Returns:
        sum of L2 norm of the body orientation deviation from upright for each environment.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)

"""
foot rewards
"""

def _feet_rpy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    )->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute roll, pitch, yaw angles of feet.

    Returns:
        roll, pitch, yaw angles of feet in radians.
    """
    # Get the robot entity
    entity = env.scene[asset_cfg.name]
    
    # Get the body IDs to use
    feet_quat = entity.data.body_quat_w[:, asset_cfg.body_ids, :]
    original_shape = feet_quat.shape
    roll, pitch, yaw = euler_xyz_from_quat(feet_quat.reshape(-1, 4))

    roll = (roll + torch.pi) % (2*torch.pi) - torch.pi
    pitch = (pitch + torch.pi) % (2*torch.pi) - torch.pi

    return roll.reshape(original_shape[0], -1), \
                pitch.reshape(original_shape[0], -1), \
                    yaw.reshape(original_shape[0], -1)

def _base_rpy(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_index: list[int] = [0],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute roll, pitch, yaw angles of base.

    Returns:
        Roll, pitch, yaw angles of base in radians.
    """
    # Get the robot entity
    entity = env.scene[asset_cfg.name]
    
    # Get the body IDs to use
    body_quat = entity.data.body_quat_w[:, base_index, :]
    original_shape = body_quat.shape
    roll, pitch, yaw = euler_xyz_from_quat(body_quat.reshape(-1, 4))

    return roll.reshape(original_shape[0]), \
                pitch.reshape(original_shape[0]), \
                    yaw.reshape(original_shape[0])

def reward_feet_roll(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """
    Reward based on the roll angles of the feet.
    
    Returns:
        sum of squared roll angles of the feet for each environment.
    """
    asset = env.scene[asset_cfg.name]
    
    # Calculate roll angles from quaternions for the feet
    feet_roll, _, _ = _feet_rpy(
        env, 
        asset_cfg=asset_cfg, 
    )
    
    return torch.sum(torch.square(feet_roll), dim=-1)

def reward_feet_roll_diff(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """Reward minimizing the difference between feet roll angles.
    
    This function rewards the agent for having similar roll angles for all feet,
    which encourages a more stable and coordinated gait.
    
    Returns:
        The absolute difference between the roll angles of the feet for each environment.
    """

    asset = env.scene[asset_cfg.name]
    
    # Calculate pitch angles from quaternions for the feet
    feet_roll, _, _ = _feet_rpy(
        env, 
        asset_cfg=asset_cfg, 
    )
    roll_rel_diff = torch.abs((feet_roll[:, 1] - feet_roll[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi)
    return roll_rel_diff

def reward_feet_pitch(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """
    Reward based on the pitch angles of the feet.
    
    Returns:
        sum of squared pitch angles of the feet for each environment.
    """

    asset = env.scene[asset_cfg.name]
    
    # Calculate roll angles from quaternions for the feet
    _, feet_pitch, _ = _feet_rpy(
        env, 
        asset_cfg=asset_cfg, 
    )
    return torch.sum(torch.square(feet_pitch), dim=-1)

def reward_feet_pitch_diff(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """Reward minimizing the difference between feet pitch angles.
    
    This function rewards the agent for having similar pitch angles for all feet,
    which encourages a more stable and coordinated gait.

    Returns:
        The absolute difference between the pitch angles of the feet for each environment.
    """

    asset = env.scene[asset_cfg.name]
    
    # Calculate pitch angles from quaternions for the feet
    _, feet_pitch, _ = _feet_rpy(
        env, 
        asset_cfg=asset_cfg, 
    )
    pitch_rel_diff = torch.abs((feet_pitch[:, 1] - feet_pitch[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi)
    return pitch_rel_diff

def reward_feet_yaw_diff(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward minimizing the difference between feet yaw angles.
    
    This function rewards the agent for having similar yaw angles for all feet,
    which encourages a more stable and coordinated gait.
    
    Returns:
        The absolute difference between the yaw angles of the feet for each environment.
    """

    asset = env.scene[asset_cfg.name]
    
    # Calculate yaw angles from quaternions for the feet
    _, _, feet_yaw = _feet_rpy(
        env, 
        asset_cfg=asset_cfg, 
    )
    yaw_rel_diff = torch.abs((feet_yaw[:, 1] - feet_yaw[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi)
    return yaw_rel_diff

def reward_feet_yaw_mean(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """
    Reward minimizing the difference between feet yaw mean and base yaw.
    
    This function rewards the agent for having the mean yaw angle of all feet
    close to the base yaw angle, which encourages better alignment between the feet and the body.
    
    Returns:
        The absolute difference between the mean yaw angle of the feet and the base yaw angle for each environment.
    """

    # Get the entity
    entity = env.scene[asset_cfg.name]
    
    # Calculate yaw angles from quaternions for the feet
    _, _, feet_yaw = _feet_rpy(
        env,
        asset_cfg=asset_cfg, 
    )
    
    _, _, base_yaw = _base_rpy(
        env, asset_cfg=asset_cfg, base_index=[0]
    )
    mean_yaw = feet_yaw.mean(dim=-1) + torch.pi * (torch.abs(feet_yaw[:, 1] - feet_yaw[:, 0]) > torch.pi)
    
    yaw_diff =  torch.abs((base_yaw - mean_yaw + torch.pi) % (2 * torch.pi) - torch.pi)
    
    return yaw_diff


"""
joint regularization
"""

def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Reward based on the energy consumption of the robot.
    
    Returns:
        The energy consumption for each environment.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward

class variable_posture(ManagerTermBase):
    """
    Compute L2 reward to regularize robot's whole body posture for each gait.
    This reward is modified based on mjlab's gaussian kernel reward. 
    
    The class takes in three different weight dictionaries for standing, walking, and running gaits.
    Example configuration:
        "weight_standing": {
            "joint0": 0.2, 
            "joint1": 0.1,
            ...
        },  
        "weight_walking": {
            "joint0": 0.02, 
            "joint1": 0.15,
            ...
        },
        "weight_running": {
            "joint0": 0.02, 
            "joint1": 0.15,
            ...
        },
    
    Returns: 
        The L2 posture error for each environment.
    """
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        asset = env.scene[cfg.params["asset_cfg"].name]
        self.default_joint_pos = asset.data.default_joint_pos 

        _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

        _, _, weight_standing = resolve_matching_names_values(
        data=cfg.params["weight_standing"],
        list_of_strings=joint_names,
        )
        self.weight_standing = torch.tensor(
        weight_standing, device=env.device, dtype=torch.float32
        )

        _, _, weight_walking = resolve_matching_names_values(
        data=cfg.params["weight_walking"],
        list_of_strings=joint_names,
        )
        self.weight_walking = torch.tensor(weight_walking, device=env.device, dtype=torch.float32)

        _, _, weight_running = resolve_matching_names_values(
        data=cfg.params["weight_running"],
        list_of_strings=joint_names,
        )
        self.weight_running = torch.tensor(weight_running, device=env.device, dtype=torch.float32)

        
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        weight_standing: dict, 
        weight_walking: dict,
        weight_running: dict,
        walking_threshold: float = 0.5,
        running_threshold: float = 1.5,
    ) -> torch.Tensor:
        
        asset = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)

        linear_speed = torch.norm(command[:, :2], dim=-1)
        angular_speed = torch.abs(command[:, 2])
        total_speed = linear_speed + angular_speed

        standing_mask = (total_speed < walking_threshold).float()
        walking_mask = (
        (total_speed >= walking_threshold) & (total_speed < running_threshold)
        ).float()
        running_mask = (total_speed >= running_threshold).float()

        weight = (
        self.weight_standing * standing_mask.unsqueeze(1) + 
        self.weight_walking * walking_mask.unsqueeze(1) + 
        self.weight_running * running_mask.unsqueeze(1)
        )

        current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
        error = torch.abs(current_joint_pos - desired_joint_pos)
        return (weight * error).sum(dim=1)

"""
gait
"""

def reward_feet_swing(    
    env: ManagerBasedRLEnv,
    swing_period: float,
    sensor_cfg: SceneEntityCfg,
    cmd_threshold: float = 0.05,
    command_name:str = None,
    gait_command_name:str = None
    ) -> torch.Tensor:
    """
    Reward based on the similarity between predifined swing phase and measured contact state of the feet.
    Swing period is defined as the portion of the gait cycle where the foot is off the ground.
    
    Example swing period where locomotion phase is 0 ~ 1 
    where SS stands for single support and DS stands for double support:
    swing_period=0.2 -> |0.0-0.15 DS| |0.15-0.35 SS| |0.35-0.65 DS| |0.65-0.85 SS| |0.85-1.0 DS|
    swing_period=0.3 -> |0.0-0.1  DS| |0.1-0.4   SS| |0.4-0.6   DS| |0.6-0.9   SS| |0.9-1.0  DS|
    swing period=0.4 -> |0.0-0.05 DS| |0.05-0.45 SS| |0.45-0.55 DS| |0.55-0.95 SS| |0.95-1.0 DS|

    Returns:
        The swing phase reward for each environment.
    """
    gait_generator = env.command_manager.get_term(gait_command_name)
    freq = 1 / gait_generator.phase_dt
    phase = gait_generator.phase

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # type: ignore
        .norm(dim=-1)
        > 1.0
    )
    
    left_swing = (torch.abs(phase - 0.25) < 0.5 * swing_period) & (freq > 1.0e-8)
    right_swing = (torch.abs(phase - 0.75) < 0.5 * swing_period) & (freq > 1.0e-8)
    reward = (left_swing & ~contacts[:, 0]).float() + (right_swing & ~contacts[:, 1]).float()

    # weight by command magnitude
    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > cmd_threshold

    return reward

def fly(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Reward encourages one feet to be in the air at all times.
    
    Returns:
        mask indicating whether at least one foot is in the air for each environment.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5

"""
contact foot penalties
"""

def reward_foot_distance(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, ref_dist: float
) -> torch.Tensor:
    """
    Calculates the reward based on the distance between the feet. 
    Penalize feet get close to each other or too far away.
    
    Returns:
        The reward based on the distance between the feet for each environment.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]
    foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)

    reward = torch.clip(ref_dist - foot_dist, min=0.0, max=0.1)
    
    return reward

def feet_force(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    """
    Penalize the vertical contact forces on the feet to reduce impact force.
    
    Returns: 
        The penalty based on the vertical contact forces on the feet for each environment.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward

"""
swing foot penalties
"""

def foot_clearance_reward(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    target_height: float, 
    std: float, 
    tanh_mult: float, 
    standing_position_foot_z: float = 0.039,
) -> torch.Tensor:
    """
    Reward the swinging feet for clearing a specified height off the ground, weighted by foot velocity.
    
    target_height is the desired max foot clearance. 
    standing_position_foot_z is the foot link (ankle roll for G1) height when the robot is standing still on flat ground.
    
    Returns:
        The foot clearance reward for each environment.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - (target_height + standing_position_foot_z))
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)

"""
action rate
"""

def action_rate_l2(env: ManagerBasedRLEnv, joint_idx:list[int]) -> torch.Tensor:
    """
    Penalize the rate of change of the actions using L2 squared kernel.
    This is different from IsaacLab's built-in action rate penalty and lets you choose specific joints indedices.
    
    Returns:
        The L2 squared action rate penalty for each environment.
    """
    return torch.sum(
        torch.square(env.action_manager.action[:, joint_idx] - env.action_manager.prev_action[:, joint_idx]), 
        dim=1)