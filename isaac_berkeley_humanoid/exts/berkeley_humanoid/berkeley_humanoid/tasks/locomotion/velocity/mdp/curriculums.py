# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
import numpy as np
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import RLTaskEnv


def terrain_levels_vel(
    env: RLTaskEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked

    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5

    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())



def modify_reward_weight(env: RLTaskEnv, env_ids: Sequence[int], term_name: str, weight: float, num_steps: int):
    """Curriculum that modifies a reward weight after some number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """

    
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)

def modify_push_force( 
        env: RLTaskEnv, 
        env_ids: Sequence[int], 
        term_name: str, 
        max_velocity: Sequence[float], 
        interval: int, 
        starting_step: float = 0.0
        ):
    """Curriculum that modifies the maximum push (perturbation) velocity over some intervals. 

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        max_velocity: The maximum velocity of the push.
        interval: The number of steps after which the condition is checked again
        starting_step: The number of steps after which the curriculum is applied.
    """
    try:
        term_cfg = env.event_manager.get_term_cfg('push_robot')
    except:
        # print("No push_robot term found in the event manager")
        return 0.0
    curr_setting = term_cfg.params['velocity_range']['x'][1]
    if env.common_step_counter < starting_step:
        return curr_setting
    if env.common_step_counter % interval == 0:

        
        if torch.sum(env.termination_manager._term_dones["base_contact"]) < torch.sum(env.termination_manager._term_dones["time_out"]) * 2:
            # obtain term settings
            term_cfg = env.event_manager.get_term_cfg('push_robot')
            # update term settings
            curr_setting = term_cfg.params['velocity_range']['x'][1]
            curr_setting = np.clip(curr_setting * 1.5, 0.0, max_velocity[0])
            term_cfg.params['velocity_range']['x'] = (-curr_setting, curr_setting)
            curr_setting = term_cfg.params['velocity_range']['y'][1]
            curr_setting = np.clip(curr_setting * 1.5, 0.0, max_velocity[1])
            term_cfg.params['velocity_range']['y'] = (-curr_setting, curr_setting)
            env.event_manager.set_term_cfg('push_robot', term_cfg)
        

        if torch.sum(env.termination_manager._term_dones["base_contact"]) > torch.sum(env.termination_manager._term_dones["time_out"]) / 2:
            # obtain term settings
            term_cfg = env.event_manager.get_term_cfg('push_robot')
            # update term settings
            curr_setting = term_cfg.params['velocity_range']['x'][1]
            curr_setting = np.clip(curr_setting - 0.2, 0.0, max_velocity[0])
            term_cfg.params['velocity_range']['x'] = (-curr_setting, curr_setting)
            curr_setting = term_cfg.params['velocity_range']['y'][1]
            curr_setting = np.clip(curr_setting - 0.2, 0.0, max_velocity[1])
            term_cfg.params['velocity_range']['y'] = (-curr_setting, curr_setting)
            env.event_manager.set_term_cfg('push_robot', term_cfg)

    return curr_setting

def modify_reward_weight_vel(env: RLTaskEnv, env_ids: Sequence[int], term_name: str, weight: float, target_vel: float, cmd : str):
    command_cfg = env.command_manager.get_term(cmd).cfg
    curr_lin_vel_x = command_cfg.ranges.lin_vel_x

    if curr_lin_vel_x[1] == target_vel:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)

def modify_command_leg_pitch(
    env: RLTaskEnv,
    env_ids: Sequence[int],
    cmd: str,
    term_name: str,
    max_pitch: Sequence[float],
    interval: int,
):
    command_cfg = env.command_manager.get_term(cmd).cfg
    #print(command_cfg.ranges.pitch)

    if env.common_step_counter % interval == 0:
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        rew = env.reward_manager._episode_sums[term_name][env_ids]
        
        if torch.mean(rew) / env.max_episode_length > 0.4 * term_cfg.weight * env.step_dt:
            curr_leg_pitch = command_cfg.ranges.pitch[1] + 0.02
            command_cfg.ranges.pitch = (command_cfg.ranges.pitch[0], curr_leg_pitch)
            print("다음 커리큘럼", command_cfg.ranges.pitch)






def modify_command_velocity_x(
    env: RLTaskEnv,
    env_ids: Sequence[int],
    cmd : str,
    term_name: str,
    max_velocity: Sequence[float],
    interval: int,
    starting_step: float = 0.0
):
    
    command_cfg = env.command_manager.get_term(cmd).cfg
    curr_lin_vel_x = command_cfg.ranges.lin_vel_x  # 현재 X 방향 속도 범위
    
    if env.common_step_counter < starting_step:
        return curr_lin_vel_x[1]  # 초기에는 현재 속도 유지
    
    if env.common_step_counter % interval == 0:
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        rew = env.reward_manager._episode_sums[term_name][env_ids]
        
        # X 방향 속도만 평가
        if torch.mean(rew) / env.max_episode_length > 0.8 * term_cfg.weight * env.step_dt:
            curr_lin_vel_x = (
                np.clip(curr_lin_vel_x[0] - 0.25, max_velocity[0], 0.), 
                np.clip(curr_lin_vel_x[1] + 0.25, 0., max_velocity[1])
            )
            command_cfg.ranges.lin_vel_x = curr_lin_vel_x  # X 방향 속도 업데이트

    return curr_lin_vel_x[1]  # 최종 속도 상한 반환

def modify_command_velocity_z(
    env: RLTaskEnv,
    env_ids: Sequence[int],
    cmd : str,
    term_name: str,
    max_velocity: Sequence[float],
    interval: int,
    starting_step: float = 0.0
):
    

    command_cfg = env.command_manager.get_term(cmd).cfg
    curr_ang_vel_z = command_cfg.ranges.ang_vel_z  # 현재 Z 방향 속도 범위
    
    if env.common_step_counter < starting_step:
        return curr_ang_vel_z[1]  # 초기에는 현재 속도 유지
    
    if env.common_step_counter % interval == 0:
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        rew = env.reward_manager._episode_sums[term_name][env_ids]
        # Z 방향 속도만 평가
        if torch.mean(rew) / env.max_episode_length > 0.6 * term_cfg.weight * env.step_dt:
            curr_ang_vel_z = (
                np.clip(curr_ang_vel_z[0] - 0.2, max_velocity[0], 0.), 
                np.clip(curr_ang_vel_z[1] + 0.2, 0., max_velocity[1])
            )
            command_cfg.ranges.ang_vel_z = curr_ang_vel_z  # Z 방향 속도 업데이트

    return curr_ang_vel_z[1]  # 최종 속도 상한 반환