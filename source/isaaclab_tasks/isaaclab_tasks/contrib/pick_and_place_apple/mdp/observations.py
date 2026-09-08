# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint state observations for the H2 pick-and-place apple task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp as base_mdp

from isaaclab_tasks.contrib.h2_sharpa.metadata import POLICY_58_ORDER

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_policy_idx_cache: dict[tuple[str, int], torch.Tensor] = {}


def _policy_joint_indices(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return (58,) indices into Isaac articulation joint order."""
    robot = env.scene["robot"]
    key = (tuple(robot.joint_names), str(env.device))
    if key not in _policy_idx_cache:
        names = list(robot.joint_names)
        idx = [names.index(jn) for jn in POLICY_58_ORDER]
        _policy_idx_cache[key] = torch.tensor(idx, dtype=torch.long, device=env.device)
    return _policy_idx_cache[key]


def get_robot_joint_states(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return [pos | vel | torque] concatenated along the joint axis (Isaac order)."""
    joint_pos = env.scene["robot"].data.joint_pos.torch
    joint_vel = env.scene["robot"].data.joint_vel.torch
    joint_torque = env.scene["robot"].data.applied_torque.torch

    return torch.cat([joint_pos, joint_vel, joint_torque], dim=-1)


def get_robot_policy_joint_positions(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return 58-D joint positions in POLICY_58_ORDER for GR00T / RLinf mapping."""
    joint_pos = env.scene["robot"].data.joint_pos.torch
    idx = _policy_joint_indices(env)
    return joint_pos.index_select(dim=1, index=idx)


def warm_rgb_image(
    env: ManagerBasedRLEnv,
    sensor_cfg,
    data_type: str = "rgb",
    normalize: bool = False,
) -> torch.Tensor:
    """Apply the real head camera's warmer, lower-exposure color response."""
    image = base_mdp.image(
        env,
        sensor_cfg=sensor_cfg,
        data_type=data_type,
        normalize=normalize,
    )
    if data_type != "rgb":
        return image

    output_dtype = image.dtype
    image = image.to(torch.float32)

    x = (image / 255.0).clamp(0.0, 1.0)
    x = x - 0.18 * x.pow(3)
    warm_balance = image.new_tensor((1.01, 0.99, 0.96))
    image = x * warm_balance * 255.0
    return image.clamp_(0.0, 255.0).to(output_dtype)
