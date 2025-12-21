# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera, ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_body_names,
    record_dtype,
    record_joint_names,
    record_joint_pos_offsets,
    record_joint_vel_offsets,
    record_shape,
)

"""
body kinematics
"""

@generic_io_descriptor(observation_type="BodyState", on_inspect=[record_shape, record_dtype, record_body_names])
def foot_pos_w(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The flattened body poses of the asset w.r.t the env.scene.origin.

    Note: Only the bodies configured in :attr:`asset_cfg.body_ids` will have their poses returned.

    Returns:
        The position of bodies in articulation [num_env, 3 * num_bodies].
        Output is stacked horizontally per body.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # access the body poses in world frame
    pose = asset.data.body_pose_w[:, asset_cfg.body_ids, :7]
    pose[..., :3] = pose[..., :3] - env.scene.env_origins.unsqueeze(1)

    pos = pose[..., :3] # (num_envs, num_bodies, 3)
    quat = pose[..., 3:7] # (num_envs, num_bodies, 4)
    rot = math_utils.matrix_from_quat(quat) # (num_envs, num_bodies, 3, 3)

    local_pos = torch.tensor([0.0, 0.0, -0.039], device=pos.device).reshape(1, 1, 3) # (1, 1, 3)
    pos_foot = pos + (rot @ local_pos.unsqueeze(-1)).squeeze(-1) # (num_envs, num_bodies, 3)

    return pos_foot.reshape(env.num_envs, -1)


"""
gait
"""

def clock(
    env: ManagerBasedRLEnv, 
    cmd_threshold: float = 0.05,
    command_name:str = "base_velocity",
    gait_command_name:str = "locomotion_gait", 
    ) -> torch.Tensor:
    """
    Clock time using sin and cos from the phase of the simulation.
    When using this reward, gait generator command must exist in command cfg. 
    
    Returns:
        Gait clock with shape (num_envs, 2)
    """
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    gait_generator = env.command_manager.get_term(gait_command_name)
    
    phase = gait_generator.phase
    phase *= (cmd_norm > cmd_threshold).float()
    return torch.cat(
        [
            torch.sin(2 * torch.pi * phase).unsqueeze(1),
            torch.cos(2 * torch.pi * phase).unsqueeze(1),
        ],
        dim=1,
    ).to(env.device)

"""
foot state
"""

def foot_height(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Get the height of the foot links wrt global frame.
    
    Returns:
        Foot heights with shape (num_envs, num_foot_links)
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # access the body poses in world frame
    pose = asset.data.body_pose_w[:, asset_cfg.body_ids, :7]
    pose[..., :3] = pose[..., :3] - env.scene.env_origins.unsqueeze(1)
    return pose[..., 2].reshape(env.num_envs, -1)

def foot_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """
    Get the air time of the foot links.
    
    Returns:
        Foot air times with shape (num_envs, num_foot_links)
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    return air_time

def foot_contact(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    ) -> torch.Tensor:
    """
    Get the contact state of the foot links.
    
    Returns:
        Foot contact states with shape (num_envs, num_foot_links)
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :] # (num_envs, num_body_ids, 3)
    contact = (torch.norm(contact_forces, dim=-1) > 1.0).float()
    return contact

def foot_contact_forces(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    ) -> torch.Tensor:
    """
    Get the contact forces of the foot links.
    
    Returns:
        Foot contact forces with shape (num_envs, num_foot_links * 3)
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :] # (num_envs, num_body_ids, 3)
    forces_flat = contact_forces.reshape(env.num_envs, -1)
    return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))