# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dual-arm environments for coordinated box picking tasks."""

import gymnasium as gym

from . import config

# Register Gym environments
gym.register(
    id="Isaac-DualPick-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{config.franka.__name__}.dual_pick_franka_env_cfg:FrankaDualPickEnvCfg",
        "skrl_cfg_entry_point": f"{config.franka.agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-DualPick-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{config.franka.__name__}.dual_pick_franka_env_cfg:FrankaDualPickEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{config.franka.agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
