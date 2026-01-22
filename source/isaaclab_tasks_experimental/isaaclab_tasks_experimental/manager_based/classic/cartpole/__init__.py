# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment (experimental manager-based entry point).
"""

import gymnasium as gym

gym.register(
    id="Isaac-Cartpole-Managed-Warp-v0",
    entry_point="isaaclab_experimental.envs:ManagerBasedRLEnvWarp",
    disable_env_checker=True,
    kwargs={
        # Use experimental Cartpole cfg (allows isolated modifications).
        "env_cfg_entry_point": (
            "isaaclab_tasks_experimental.manager_based.classic.cartpole.cartpole_env_cfg:CartpoleEnvCfg"
        ),
        # Point agent configs to the existing task package.
        "rl_games_cfg_entry_point": "isaaclab_tasks.manager_based.classic.cartpole.agents:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.manager_based.classic.cartpole.agents.rsl_rl_ppo_cfg:CartpolePPORunnerCfg"
        ),
        "skrl_cfg_entry_point": "isaaclab_tasks.manager_based.classic.cartpole.agents:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": "isaaclab_tasks.manager_based.classic.cartpole.agents:sb3_ppo_cfg.yaml",
    },
)
