# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for the SO-101 keyboard task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_keyboard_root_state_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | slice,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    command_name: str = "typing",
) -> None:
    """Randomize each keyboard once per environment, including all PhysX articulation partitions."""
    command = env.command_manager.get_term(command_name)
    keyboard = command.keyboard
    selected_env_ids = torch.arange(env.num_envs, device=keyboard.device)[env_ids]
    if selected_env_ids.numel() == 0:
        return

    instance_ids = command.keyboard_instance_ids[selected_env_ids].reshape(-1)
    instances_per_env = command.keyboard_instance_ids.shape[1]
    default_pose = keyboard.data.default_root_pose.torch[instance_ids]
    default_velocity = keyboard.data.default_root_vel.torch[instance_ids]
    pose_samples = math_utils.sample_uniform_from_ranges(
        pose_range, ("x", "y", "z", "roll", "pitch", "yaw"), selected_env_ids.numel(), device=keyboard.device
    ).repeat_interleave(instances_per_env, dim=0)
    origins = env.scene.env_origins[selected_env_ids].repeat_interleave(instances_per_env, dim=0)
    positions = default_pose[:, :3] + origins + pose_samples[:, :3]
    rotation = math_utils.quat_from_euler_xyz(pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5])
    orientations = math_utils.quat_mul(default_pose[:, 3:], rotation)
    velocity_samples = math_utils.sample_uniform_from_ranges(
        velocity_range, ("x", "y", "z", "roll", "pitch", "yaw"), selected_env_ids.numel(), device=keyboard.device
    ).repeat_interleave(instances_per_env, dim=0)

    keyboard.write_root_pose_to_sim_index(root_pose=torch.cat((positions, orientations), dim=-1), env_ids=instance_ids)
    keyboard.write_root_velocity_to_sim_index(root_velocity=default_velocity + velocity_samples, env_ids=instance_ids)
