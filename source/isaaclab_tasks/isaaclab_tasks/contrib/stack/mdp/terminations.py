# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the cube stack task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


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
    max_lin_vel: float | None = 0.05,
    symmetric_height_tolerance: bool = False,
) -> torch.Tensor:
    """Whether the cubes are stacked, released, and at rest.

    Args:
        env: The RL environment instance.
        robot_cfg: Scene entity configuration for the robot whose gripper is checked.
        cube_1_cfg: Scene entity configuration for the bottom cube.
        cube_2_cfg: Scene entity configuration for the cube above cube_1.
        cube_3_cfg: Scene entity configuration for the cube above cube_2. None disables
            the third-cube checks for a two-cube task.
        xy_threshold: Maximum horizontal distance [m] between adjacent cube centers.
        height_threshold: Upper tolerance [m] on the vertical center separation minus height_diff.
        height_diff: Expected vertical center separation [m] between adjacent cubes.
        atol: Absolute tolerance for gripper joint positions [m or rad, depending on
            joint type] when checking the configured open value.
        rtol: Relative tolerance for gripper joint positions when checking the configured
            open value. Gripper joint tolerances are unused for surface grippers.
        max_lin_vel: Speed [m/s] below which a cube counts as at rest, so that a cube passing through the
            stacked configuration mid-fall is not scored as success. ``None`` disables the check.
        symmetric_height_tolerance: Also require the vertical separation to be within
            ``height_threshold`` below ``height_diff``.

    Returns:
        A boolean tensor of shape [N] indicating success in each environment.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]

    separation_c12 = cube_2.data.root_pos_w.torch - cube_1.data.root_pos_w.torch
    stacked = _cube_pair_is_aligned(
        separation_c12, xy_threshold, height_diff, height_threshold, symmetric_height_tolerance
    )

    if cube_3_cfg is not None:
        cube_3: RigidObject = env.scene[cube_3_cfg.name]
        separation_c23 = cube_3.data.root_pos_w.torch - cube_2.data.root_pos_w.torch
        stacked &= _cube_pair_is_aligned(
            separation_c23, xy_threshold, height_diff, height_threshold, symmetric_height_tolerance
        )

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

    # A cube falling past its target also satisfies the position checks, so require every cube to have settled.
    if max_lin_vel is not None:
        for cube in [cube_1, cube_2] + ([env.scene[cube_3_cfg.name]] if cube_3_cfg is not None else []):
            stacked &= torch.linalg.norm(cube.data.root_lin_vel_w.torch, dim=1) < max_lin_vel

    return stacked


def _cube_pair_is_aligned(
    separation: torch.Tensor,
    xy_threshold: float,
    height_diff: float,
    height_threshold: float,
    symmetric_height_tolerance: bool,
) -> torch.Tensor:
    """Check the world-frame upper-minus-lower cube displacement.

    Args:
        separation: Cube center displacements in meters, with shape [N, 3].
        xy_threshold: Maximum horizontal separation in meters.
        height_diff: Expected vertical separation in meters.
        height_threshold: Tolerance on the vertical separation in meters.
        symmetric_height_tolerance: Apply the tolerance below as well as above ``height_diff``.

    Returns:
        A boolean tensor of shape [N] indicating aligned cube pairs.
    """
    height = separation[:, 2]
    height_error = height - height_diff
    if symmetric_height_tolerance:
        height_error = height_error.abs()
    return (
        (torch.linalg.norm(separation[:, :2], dim=1) < xy_threshold)
        & (height > 0.0)
        & (height_error < height_threshold)
    )
