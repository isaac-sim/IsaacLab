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

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cubes_stacked(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    xy_threshold: float = 0.05,
    height_threshold: float = 0.005,
    height_diff: float = 0.0233,
    gripper_open_val: torch.tensor = torch.tensor([0.5]),
    atol=0.0001,
    rtol=0.0001,
):
    robot: Articulation = env.scene[robot_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]

    pos_diff_c12 = cube_1.data.root_pos_w - cube_2.data.root_pos_w

    # Compute cube position difference in x-y plane
    xy_dist_c12 = torch.norm(pos_diff_c12[:, :2], dim=1)

    # Compute cube height difference
    h_dist_c12 = torch.norm(pos_diff_c12[:, 2:], dim=1)

    # Check cube positions
    stacked = xy_dist_c12 < xy_threshold
    print("stacked xy", stacked)
    print("xy_dist_c12", xy_dist_c12)
    print("xy_threshold", xy_threshold)
    # stacked = torch.logical_and(height_diff - h_dist_c12  < height_threshold, stacked)
    stacked = torch.logical_and(height_diff - h_dist_c12  < height_threshold, stacked)
    print("stacked h", stacked)
    print("h_dist_c12", h_dist_c12)
    print("height_diff", height_diff)
    print("height_threshold", height_threshold)
    
    # Check gripper positions
    stacked = torch.logical_and(
        torch.isclose(robot.data.joint_pos[:, -1], gripper_open_val.to(env.device), atol=0.0001, rtol=0.0001), stacked
    )
    print("stacked gripper", stacked)
    print("stacked", stacked)

    return stacked
