# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate terminations for the stack task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def success_after_minimum_horizon(
    env: ManagerBasedRLEnv,
    context_term_name: str = "progress_context",
    minimum_episode_length_s: float = 5.0,
) -> torch.Tensor:
    """Terminate successful episodes after a configurable minimum horizon.

    The success context owns the physical stability hold. This separate gate
    exists only to prevent success inherited directly from a reset state; it
    should not keep a policy in an already solved scene.
    """
    if minimum_episode_length_s <= 0.0:
        raise ValueError("minimum_episode_length_s must be positive.")
    context = env.termination_manager.get_term_cfg(context_term_name).func
    minimum_steps = math.ceil(minimum_episode_length_s / env.step_dt)
    return context.ever_success & (env.episode_length_buf >= minimum_steps)


def cubes_stacked(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg | None = SceneEntityCfg("cube_3"),
    xy_threshold: float = 0.04,
    height_threshold: float = 0.005,
    height_diff: float = 0.0468,
    atol: float = 0.0001,
    rtol: float = 0.0001,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]

    pos_diff_c12 = cube_1.data.root_pos_w.torch - cube_2.data.root_pos_w.torch

    # Compute cube position difference in x-y plane
    xy_dist_c12 = torch.linalg.norm(pos_diff_c12[:, :2], dim=1)

    # Compute cube height difference
    h_dist_c12 = torch.linalg.norm(pos_diff_c12[:, 2:], dim=1)

    # Check cube positions
    stacked = xy_dist_c12 < xy_threshold
    stacked = torch.logical_and(h_dist_c12 - height_diff < height_threshold, stacked)
    stacked = torch.logical_and(pos_diff_c12[:, 2] < 0.0, stacked)

    if cube_3_cfg is not None:
        cube_3: RigidObject = env.scene[cube_3_cfg.name]
        pos_diff_c23 = cube_2.data.root_pos_w.torch - cube_3.data.root_pos_w.torch

        # Compute cube position difference in x-y plane
        xy_dist_c23 = torch.linalg.norm(pos_diff_c23[:, :2], dim=1)

        # Compute cube height difference
        h_dist_c23 = torch.linalg.norm(pos_diff_c23[:, 2:], dim=1)

        # Check cube positions
        stacked = torch.logical_and(xy_dist_c23 < xy_threshold, stacked)
        stacked = torch.logical_and(h_dist_c23 - height_diff < height_threshold, stacked)
        stacked = torch.logical_and(pos_diff_c23[:, 2] < 0.0, stacked)

    # Check gripper positions
    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = wp.to_torch(surface_gripper.state).view(-1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_open = (suction_cup_status == -1).to(torch.float32)
        stacked = torch.logical_and(suction_cup_is_open, stacked)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) >= 1, "Terminations require at least one gripper joint"
            # Success also requires the gripper to be released (every jaw back at the open value).
            open_val = torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
            for joint_id in gripper_joint_ids:
                stacked = torch.logical_and(
                    torch.isclose(
                        robot.data.joint_pos.torch[:, joint_id],
                        open_val,
                        atol=atol,
                        rtol=rtol,
                    ),
                    stacked,
                )
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return stacked


def nonfinite_robot_state(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Detect non-finite robot joint state."""
    robot: Articulation = env.scene[robot_cfg.name]
    joint_position = robot.data.joint_pos.torch[:, robot_cfg.joint_ids]
    joint_velocity = robot.data.joint_vel.torch[:, robot_cfg.joint_ids]
    return ~torch.isfinite(joint_position).all(dim=1) | ~torch.isfinite(joint_velocity).all(dim=1)


def nonfinite_cube_state(
    env: ManagerBasedRLEnv,
    cube_cfgs: tuple[SceneEntityCfg, ...] = (
        SceneEntityCfg("cube_1"),
        SceneEntityCfg("cube_2"),
        SceneEntityCfg("cube_3"),
    ),
) -> torch.Tensor:
    """Detect non-finite cube pose or velocity state."""
    invalid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for cube_cfg in cube_cfgs:
        cube: RigidObject = env.scene[cube_cfg.name]
        invalid |= ~torch.isfinite(cube.data.root_pose_w.torch).all(dim=1)
        invalid |= ~torch.isfinite(cube.data.root_vel_w.torch).all(dim=1)
    return invalid


def cube_out_of_workspace(
    env: ManagerBasedRLEnv,
    cube_cfgs: tuple[SceneEntityCfg, ...] = (
        SceneEntityCfg("cube_1"),
        SceneEntityCfg("cube_2"),
        SceneEntityCfg("cube_3"),
    ),
    workspace_lower: tuple[float, float, float] = (-0.2, -0.7, -0.1),
    workspace_upper: tuple[float, float, float] = (1.2, 0.7, 1.0),
) -> torch.Tensor:
    """Detect cubes outside the local task workspace [m]."""
    lower = torch.tensor(workspace_lower, dtype=torch.float32, device=env.device)
    upper = torch.tensor(workspace_upper, dtype=torch.float32, device=env.device)
    invalid = torch.zeros(env.scene.env_origins.shape[0], dtype=torch.bool, device=env.device)
    for cube_cfg in cube_cfgs:
        cube: RigidObject = env.scene[cube_cfg.name]
        position = cube.data.root_pos_w.torch - env.scene.env_origins
        invalid |= ((position < lower) | (position > upper)).any(dim=1)
    return invalid
