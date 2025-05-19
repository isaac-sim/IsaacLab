# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab.envs import ManagerBasedEnv


def get_action_shape(env: ManagerBasedEnv, num_steps: int | None = None):
    num_envs, acs_dim = env.num_envs, env.action_manager.total_action_dim
    return (num_steps, num_envs, acs_dim) if num_steps else (num_envs, acs_dim)


def zero_actions(env: ManagerBasedEnv, num_steps: int | None = None):
    """Generate set of environment-specific zero-actions."""
    return torch.zeros(get_action_shape(env, num_steps), device=env.device)


def random_actions(env: ManagerBasedEnv, num_steps: int | None = None):
    """Generate random set of environment-specific actions."""
    return 2 * torch.rand(get_action_shape(env, num_steps), device=env.device) - 1
