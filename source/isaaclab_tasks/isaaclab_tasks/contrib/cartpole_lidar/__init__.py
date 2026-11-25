# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cartpole balancing from tiled lidar range images.

The policy reads two stacked range images. The critic still reads the joint state. Tiled capture
requires the ``ovrtx`` extra.
"""

import gymnasium as gym

_CAMERA_AGENTS = "isaaclab_tasks.core.cartpole.agents"

gym.register(
    id="IsaacContrib-Cartpole-Lidar-Direct",
    entry_point=f"{__name__}.cartpole_lidar_direct_env:CartpoleLidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_lidar_direct_env_cfg:CartpoleLidarEnvCfg",
        "rl_games_cfg_entry_point": f"{_CAMERA_AGENTS}:rl_games_camera_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{_CAMERA_AGENTS}.rsl_rl_ppo_cfg:CartpoleCameraDirectPPORunnerCfg",
        "default_agent": "rsl_rl",
        "skrl_cfg_entry_point": f"{_CAMERA_AGENTS}:skrl_direct_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="IsaacContrib-Cartpole-Lidar",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_lidar_env_cfg:CartpoleLidarEnvCfg",
        "rl_games_cfg_entry_point": f"{_CAMERA_AGENTS}:rl_games_camera_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{_CAMERA_AGENTS}.rsl_rl_ppo_cfg:CartpoleCameraPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
