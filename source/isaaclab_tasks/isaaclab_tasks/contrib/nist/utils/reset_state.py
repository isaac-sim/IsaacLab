# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-env reset-state read/write helpers."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def get_reset_state(env, env_ids: Tensor, reset_assets: Sequence[str], is_relative: bool = False) -> Tensor:
    """Read and concatenate reset-state slices for scene assets."""
    states: list[Tensor] = []
    for name, articulation in env.scene._articulations.items():
        if name in reset_assets:
            root_state = articulation.data.root_state_w.torch[env_ids]
            if is_relative:
                root_state = root_state.clone()
                root_state[:, :3] -= env.scene.env_origins[env_ids]
            states.append(root_state)
            states.append(articulation.data.joint_pos.torch[env_ids])
            states.append(articulation.data.joint_vel.torch[env_ids])

    for name, rigid_object in env.scene._rigid_objects.items():
        if name in reset_assets:
            root_state = rigid_object.data.root_state_w.torch[env_ids]
            if is_relative:
                root_state = root_state.clone()
                root_state[:, :3] -= env.scene.env_origins[env_ids]
            states.append(root_state)
    return torch.cat(states, dim=-1)


def set_reset_state(env, states: Tensor, env_ids: Tensor, reset_assets: Sequence[str], is_relative: bool = False):
    """Split ``states`` by scene asset and write reset-state slices."""
    offset = 0
    for name, articulation in env.scene._articulations.items():
        if name in reset_assets:
            n_joints, width = articulation.num_joints, 13 + 2 * articulation.num_joints
            state = states[:, offset : offset + width]
            root_state = state[:, :13]
            if is_relative:
                root_state = root_state.clone()
                root_state[:, :3] += env.scene.env_origins[env_ids]
            articulation.write_root_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=env_ids)
            articulation.write_root_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)
            articulation.write_joint_position_to_sim_index(position=state[:, 13 : 13 + n_joints], env_ids=env_ids)
            articulation.write_joint_velocity_to_sim_index(
                velocity=state[:, 13 + n_joints : 13 + 2 * n_joints], env_ids=env_ids
            )
            offset += width

    for name, rigid_object in env.scene._rigid_objects.items():
        if name in reset_assets:
            root_state = states[:, offset : offset + 13]
            if is_relative:
                root_state = root_state.clone()
                root_state[:, :3] += env.scene.env_origins[env_ids]
            rigid_object.write_root_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=env_ids)
            rigid_object.write_root_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)
            offset += 13
