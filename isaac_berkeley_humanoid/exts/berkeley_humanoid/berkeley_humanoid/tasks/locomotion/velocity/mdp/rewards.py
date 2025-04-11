# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold_min: float,
                  threshold_max: float) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # negative reward for small steps
    air_time = (last_air_time - threshold_min) * first_contact
    # no reward for large steps
    air_time = torch.clamp(air_time, max=threshold_max - threshold_min)
    reward = torch.sum(air_time, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold_min: float, threshold_max: float,
                                 sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold_max)
    # no reward for small steps
    reward *= reward > threshold_min
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


# def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Penalize feet sliding"""
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
#     asset = env.scene[asset_cfg.name]
#     body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
#     reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
#     return reward
def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward feet sliding motion.

    This function rewards the robot for increasing foot sliding motion. The reward is calculated
    based on the body velocity during foot contacts.
    """
    # Access the contact sensor and its data
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Determine contact status (True if foot is in contact with ground and forces are applied)
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    
    # Access the robot's asset data for body velocity
    asset = env.scene[asset_cfg.name]
    # Linear velocity of the body parts in contact
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    
    # Compute sliding velocity during contacts
    sliding_velocity = body_vel.norm(dim=-1) * contacts
    
    # Reward for sliding (higher sliding velocity yields higher reward)
    reward = torch.sum(sliding_velocity, dim=1)
    
    return reward

def foot_contact_sequence(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Reward specific foot contact sequences: left and right foot contact patterns.

    Args:
        env (ManagerBasedRLEnv): The environment containing sensors and scene data.
        sensor_cfg (SceneEntityCfg): Configuration for the contact force sensor.
        asset_cfg (SceneEntityCfg): Configuration for the robot's foot asset.

    Returns:
        torch.Tensor: Reward values for each environment.
    """
    # Get contact sensor data
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history

    # Extract body IDs for robot feet
    body_ids = sensor_cfg.body_ids

    # Calculate contact status for each foot
    contact_status = torch.norm(net_contact_forces[:, :, body_ids], dim=-1) > 0.0  # Shape: (num_envs, time_steps, num_feet)

    # Define foot indices (assuming 2 feet: 0=left, 1=right)
    left_foot, right_foot = 0, 1

    # Check desired contact patterns
    alternating_contact = (
        (contact_status[:, :, left_foot] & ~contact_status[:, :, right_foot]) |
        (~contact_status[:, :, left_foot] & contact_status[:, :, right_foot])
    )

    # Reward is the sum of alternating contact patterns over time for each environment
    rewards = torch.sum(alternating_contact, dim=1)  # Sum over time_steps

    return rewards