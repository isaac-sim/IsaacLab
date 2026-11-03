# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity-tracking tasks for the MicroDuck biped."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="IsaacContrib-Velocity-Flat-MicroDuck-Pollen",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.contrib.microduck.preset_configs:velocity_flat_env_cfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.contrib.microduck.preset_configs:velocity_flat_agent_cfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="IsaacContrib-Velocity-Flat-MicroDuck-Rollers",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.contrib.microduck.preset_configs:velocity_rollers_env_cfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.contrib.microduck.preset_configs:velocity_rollers_agent_cfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="IsaacContrib-Velocity-Swizzle-MicroDuck",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.contrib.microduck.preset_configs:velocity_swizzle_env_cfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.contrib.microduck.preset_configs:velocity_swizzle_agent_cfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="IsaacContrib-Velocity-Rough-MicroDuck-Pollen",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.contrib.microduck.preset_configs:velocity_rough_env_cfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.contrib.microduck.preset_configs:velocity_rough_agent_cfg",
        "default_agent": "rsl_rl",
    },
)
