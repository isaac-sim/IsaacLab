# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_done_pick_place(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    right_wrist_max_x: float = 0.26,
    min_x: float = 0.40,
    max_x: float = 0.85,
    min_y: float = 0.35,
    max_y: float = 0.60,
    max_height: float = 1.10,
    min_vel: float = 0.20,
) -> torch.Tensor:
    """Determine if the object placement task is complete.

    This function checks whether all success conditions for the task have been met:
    1. object is within the target x/y range
    2. object is below a minimum height
    3. object velocity is below threshold
    4. Right robot wrist is retracted back towards body (past a given x pos threshold)

    Args:
        env: The RL environment instance.
        object_cfg: Configuration for the object entity.
        right_wrist_max_x: Maximum x position of the right wrist for task completion.
        min_x: Minimum x position of the object for task completion.
        max_x: Maximum x position of the object for task completion.
        min_y: Minimum y position of the object for task completion.
        max_y: Maximum y position of the object for task completion.
        max_height: Maximum height (z position) of the object for task completion.
        min_vel: Minimum velocity magnitude of the object for task completion.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    # Get object entity from the scene
    object: RigidObject = env.scene[object_cfg.name]

    # Extract wheel position relative to environment origin
    object_x = object.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    object_y = object.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    object_height = object.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    object_vel = torch.abs(object.data.root_vel_w)

    # Get right wrist position relative to environment origin
    robot_body_pos_w = env.scene["robot"].data.body_pos_w
    right_eef_idx = env.scene["robot"].data.body_names.index("right_hand_roll_link")
    right_wrist_x = robot_body_pos_w[:, right_eef_idx, 0] - env.scene.env_origins[:, 0]

    # Check all success conditions and combine with logical AND
    done = object_x < max_x
    done = torch.logical_and(done, object_x > min_x)
    done = torch.logical_and(done, object_y < max_y)
    done = torch.logical_and(done, object_y > min_y)
    done = torch.logical_and(done, object_height < max_height)
    done = torch.logical_and(done, right_wrist_x < right_wrist_max_x)
    done = torch.logical_and(done, object_vel[:, 0] < min_vel)
    done = torch.logical_and(done, object_vel[:, 1] < min_vel)
    done = torch.logical_and(done, object_vel[:, 2] < min_vel)

    return done


def task_done_nut_pour(
    env: ManagerBasedRLEnv,
    sorting_scale_cfg: SceneEntityCfg = SceneEntityCfg("sorting_scale"),
    sorting_bowl_cfg: SceneEntityCfg = SceneEntityCfg("sorting_bowl"),
    sorting_beaker_cfg: SceneEntityCfg = SceneEntityCfg("sorting_beaker"),
    factory_nut_cfg: SceneEntityCfg = SceneEntityCfg("factory_nut"),
    sorting_bin_cfg: SceneEntityCfg = SceneEntityCfg("black_sorting_bin"),
    max_bowl_to_scale_x: float = 0.055,
    max_bowl_to_scale_y: float = 0.055,
    max_bowl_to_scale_z: float = 0.025,
    max_nut_to_bowl_x: float = 0.050,
    max_nut_to_bowl_y: float = 0.050,
    max_nut_to_bowl_z: float = 0.019,
    max_beaker_to_bin_x: float = 0.08,
    max_beaker_to_bin_y: float = 0.12,
    max_beaker_to_bin_z: float = 0.07,
) -> torch.Tensor:
    """Determine if the nut pouring task is complete.

    This function checks whether all success conditions for the task have been met:
    1. The factory nut is in the sorting bowl
    2. The sorting beaker is in the sorting bin
    3. The sorting bowl is placed on the sorting scale

    Args:
        env: The RL environment instance.
        sorting_scale_cfg: Configuration for the sorting scale entity.
        sorting_bowl_cfg: Configuration for the sorting bowl entity.
        sorting_beaker_cfg: Configuration for the sorting beaker entity.
        factory_nut_cfg: Configuration for the factory nut entity.
        sorting_bin_cfg: Configuration for the sorting bin entity.
        max_bowl_to_scale_x: Maximum x position of the sorting bowl relative to the sorting scale for task completion.
        max_bowl_to_scale_y: Maximum y position of the sorting bowl relative to the sorting scale for task completion.
        max_bowl_to_scale_z: Maximum z position of the sorting bowl relative to the sorting scale for task completion.
        max_nut_to_bowl_x: Maximum x position of the factory nut relative to the sorting bowl for task completion.
        max_nut_to_bowl_y: Maximum y position of the factory nut relative to the sorting bowl for task completion.
        max_nut_to_bowl_z: Maximum z position of the factory nut relative to the sorting bowl for task completion.
        max_beaker_to_bin_x: Maximum x position of the sorting beaker relative to the sorting bin for task completion.
        max_beaker_to_bin_y: Maximum y position of the sorting beaker relative to the sorting bin for task completion.
        max_beaker_to_bin_z: Maximum z position of the sorting beaker relative to the sorting bin for task completion.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    # Get object entities from the scene
    sorting_scale: RigidObject = env.scene[sorting_scale_cfg.name]
    sorting_bowl: RigidObject = env.scene[sorting_bowl_cfg.name]
    factory_nut: RigidObject = env.scene[factory_nut_cfg.name]
    sorting_beaker: RigidObject = env.scene[sorting_beaker_cfg.name]
    sorting_bin: RigidObject = env.scene[sorting_bin_cfg.name]

    # Get positions relative to environment origin
    scale_pos = sorting_scale.data.root_pos_w - env.scene.env_origins
    bowl_pos = sorting_bowl.data.root_pos_w - env.scene.env_origins
    sorting_beaker_pos = sorting_beaker.data.root_pos_w - env.scene.env_origins
    nut_pos = factory_nut.data.root_pos_w - env.scene.env_origins
    bin_pos = sorting_bin.data.root_pos_w - env.scene.env_origins

    # nut to bowl
    nut_to_bowl_x = torch.abs(nut_pos[:, 0] - bowl_pos[:, 0])
    nut_to_bowl_y = torch.abs(nut_pos[:, 1] - bowl_pos[:, 1])
    nut_to_bowl_z = nut_pos[:, 2] - bowl_pos[:, 2]

    # bowl to scale
    bowl_to_scale_x = torch.abs(bowl_pos[:, 0] - scale_pos[:, 0])
    bowl_to_scale_y = torch.abs(bowl_pos[:, 1] - scale_pos[:, 1])
    bowl_to_scale_z = bowl_pos[:, 2] - scale_pos[:, 2]

    # beaker to bin
    beaker_to_bin_x = torch.abs(sorting_beaker_pos[:, 0] - bin_pos[:, 0])
    beaker_to_bin_y = torch.abs(sorting_beaker_pos[:, 1] - bin_pos[:, 1])
    beaker_to_bin_z = sorting_beaker_pos[:, 2] - bin_pos[:, 2]

    done = nut_to_bowl_x < max_nut_to_bowl_x
    done = torch.logical_and(done, nut_to_bowl_y < max_nut_to_bowl_y)
    done = torch.logical_and(done, nut_to_bowl_z < max_nut_to_bowl_z)
    done = torch.logical_and(done, bowl_to_scale_x < max_bowl_to_scale_x)
    done = torch.logical_and(done, bowl_to_scale_y < max_bowl_to_scale_y)
    done = torch.logical_and(done, bowl_to_scale_z < max_bowl_to_scale_z)
    done = torch.logical_and(done, beaker_to_bin_x < max_beaker_to_bin_x)
    done = torch.logical_and(done, beaker_to_bin_y < max_beaker_to_bin_y)
    done = torch.logical_and(done, beaker_to_bin_z < max_beaker_to_bin_z)

    return done


def task_done_exhaust_pipe(
    env: ManagerBasedRLEnv,
    blue_exhaust_pipe_cfg: SceneEntityCfg = SceneEntityCfg("blue_exhaust_pipe"),
    blue_sorting_bin_cfg: SceneEntityCfg = SceneEntityCfg("blue_sorting_bin"),
    max_blue_exhaust_to_bin_x: float = 0.085,
    max_blue_exhaust_to_bin_y: float = 0.200,
    min_blue_exhaust_to_bin_y: float = -0.090,
    max_blue_exhaust_to_bin_z: float = 0.070,
) -> torch.Tensor:
    """Determine if the exhaust pipe task is complete.

    This function checks whether all success conditions for the task have been met:
    1. The blue exhaust pipe is placed in the correct position

    Args:
        env: The RL environment instance.
        blue_exhaust_pipe_cfg: Configuration for the blue exhaust pipe entity.
        blue_sorting_bin_cfg: Configuration for the blue sorting bin entity.
        max_blue_exhaust_to_bin_x: Maximum x position of the blue exhaust pipe relative to the blue sorting bin for task completion.
        max_blue_exhaust_to_bin_y: Maximum y position of the blue exhaust pipe relative to the blue sorting bin for task completion.
        max_blue_exhaust_to_bin_z: Maximum z position of the blue exhaust pipe relative to the blue sorting bin for task completion.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    # Get object entities from the scene
    blue_exhaust_pipe: RigidObject = env.scene[blue_exhaust_pipe_cfg.name]
    blue_sorting_bin: RigidObject = env.scene[blue_sorting_bin_cfg.name]

    # Get positions relative to environment origin
    blue_exhaust_pipe_pos = blue_exhaust_pipe.data.root_pos_w - env.scene.env_origins
    blue_sorting_bin_pos = blue_sorting_bin.data.root_pos_w - env.scene.env_origins

    # blue exhaust to bin
    blue_exhaust_to_bin_x = torch.abs(blue_exhaust_pipe_pos[:, 0] - blue_sorting_bin_pos[:, 0])
    blue_exhaust_to_bin_y = blue_exhaust_pipe_pos[:, 1] - blue_sorting_bin_pos[:, 1]
    blue_exhaust_to_bin_z = blue_exhaust_pipe_pos[:, 2] - blue_sorting_bin_pos[:, 2]

    done = blue_exhaust_to_bin_x < max_blue_exhaust_to_bin_x
    done = torch.logical_and(done, blue_exhaust_to_bin_y < max_blue_exhaust_to_bin_y)
    done = torch.logical_and(done, blue_exhaust_to_bin_y > min_blue_exhaust_to_bin_y)
    done = torch.logical_and(done, blue_exhaust_to_bin_z < max_blue_exhaust_to_bin_z)

    return done
