# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene preparation shared by the VLA benchmark success tests."""

import torch

from isaaclab.envs import ManagerBasedRLEnv


def place(
    env: ManagerBasedRLEnv,
    name: str,
    positions: list[tuple[float, float, float]],
    rotations: list[tuple[float, float, float, float]] | None = None,
) -> None:
    """Drop an object in each environment at a position in its environment frame, rotated by a quaternion (xyzw)."""
    pose = torch.zeros(env.num_envs, 7, device=env.device)
    pose[:, :3] = torch.tensor(positions, device=env.device) + env.scene.env_origins
    pose[:, 3:] = torch.tensor(rotations or [(0.0, 0.0, 0.0, 1.0)] * env.num_envs, device=env.device)
    env.scene[name].write_root_pose_to_sim_index(root_pose=pose)
    env.scene[name].write_root_velocity_to_sim_index(root_velocity=torch.zeros(env.num_envs, 6, device=env.device))
