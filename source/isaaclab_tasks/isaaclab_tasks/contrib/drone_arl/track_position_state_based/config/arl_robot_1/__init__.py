# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

_AGENTS_MODULE = f"{__name__}.agents"

##
# Register Gym environments.
##

gym.register(
    id="IsaacContrib-TrackPositionNoObstacles-ARL-Robot-1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.no_obstacle_env_cfg:NoObstacleEnvCfg",
        "rl_games_cfg_entry_point": f"{_AGENTS_MODULE}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:TrackPositionNoObstaclesEnvPPORunnerCfg",
        "skrl_cfg_entry_point": f"{_AGENTS_MODULE}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="IsaacContrib-TrackPositionNoObstacles-ARL-Robot-1-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.no_obstacle_env_cfg:NoObstacleEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{_AGENTS_MODULE}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:TrackPositionNoObstaclesEnvPPORunnerCfg",
        "skrl_cfg_entry_point": f"{_AGENTS_MODULE}:skrl_ppo_cfg.yaml",
    },
)
