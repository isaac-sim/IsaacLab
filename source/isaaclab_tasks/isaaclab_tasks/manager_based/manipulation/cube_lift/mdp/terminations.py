# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg

from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms
from isaaclab.utils.logging_helper import LoggingHelper, ErrorType, LogType
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    loghelper : LoggingHelper = LoggingHelper()
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
   # print(f"For DEBUG : DISTANCE TO GOAL : {distance}")
    # if(distance.item() < threshold):
    #     loghelper.logsubtask(LogType.FINISH)
    test = distance < threshold
    if (torch.any(test)):
        print("debug : terminations : ", torch.any(test))
    return distance < threshold


def placed(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg=SceneEntityCfg("ee_frame"),
    command_name: str = "object_pose",
    threshold: torch.tensor = torch.tensor([0.05], device="cuda:0"),
    gripper_open_val= torch.tensor([0.038], device="cuda:0"), #open
    atol=0.0001,
    rtol=0.0001,
    gripper_threshold: float = 0.005,
):
    robot: Articulation = env.scene[robot_cfg.name]
    
    command = env.command_manager.get_command(command_name)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name],
    object: RigidObject = env.scene[object_cfg.name]

    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    
    object_pos = object.data.root_pos_w
    #end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
   # pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)
    distance = torch.norm(des_pos_w - object_pos[:, :3], dim=1)
    object_position = distance<threshold
  #  print("object in position ? : ", object_position[0].item())

    
    #print("Gripper state : ",robot.data.joint_pos[:, -1] >  gripper_open_val)
    gripper_open = robot.data.joint_pos[:, -1] >  gripper_open_val
    state = torch.logical_and(object_position, gripper_open)
    return state