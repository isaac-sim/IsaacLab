# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

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
) -> torch.Tensor:
    """Whether the cubes are stacked, released, and at rest.

    Args:
        max_lin_vel: Speed, in m/s, below which a cube counts as at rest. The position checks below
            describe an instantaneous configuration, and a cube dropped above its target passes
            through that configuration on the way down -- without this, the drop is scored as a
            successful stack. Pass None to skip the check.

            The default leaves room for the contact solver's residual jitter, which keeps a settled
            cube from ever reading exactly zero: over a sample of 91 sound generated demos, cubes
            resting on the stack read a median 0.013 m/s and peaked at 0.034 m/s, while a cube
            caught mid-fall read a median 0.10 m/s.
    """
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

    # The position checks above are satisfied by a cube falling past its target as well as by one
    # resting on it, so require the cubes to have settled before calling the stack complete.
    if max_lin_vel is not None:
        cubes = [cube_1, cube_2]
        if cube_3_cfg is not None:
            cubes.append(env.scene[cube_3_cfg.name])
        for cube in cubes:
            at_rest = torch.linalg.norm(cube.data.root_lin_vel_w.torch, dim=1) < max_lin_vel
            stacked = torch.logical_and(at_rest, stacked)

    return stacked
