# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demo tasks whose camera image Cosmos generates.

Each task is a task from ``isaaclab_tasks`` with a Cosmos camera. The package registers them through the
``isaaclab.tasks`` entry point, so ``isaaclab train`` and ``isaaclab play`` find them by name. Registration only
names the configurations; ``isaaclab_tasks`` and the Cosmos client are imported when a task is selected.
"""

import gymnasium as gym

_SHADOW_HAND = "isaaclab_tasks.core.reorient.config.shadow_hand"

gym.register(
    id="Isaac-Reorient-Cube-Shadow-Camera-Cosmos-Direct",
    entry_point=f"{_SHADOW_HAND}.shadow_hand_direct_camera_env:ShadowHandCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_cosmos_env_cfg:ShadowHandCameraCosmosEnvCfg",
        "rl_games_cfg_entry_point": f"{_SHADOW_HAND}.agents:rl_games_ppo_camera_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{_SHADOW_HAND}.agents.rsl_rl_ppo_cfg:ShadowHandCameraFFPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Isaac-Reorient-Cube-Shadow-Camera-Cosmos",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_cosmos_env_cfg:ShadowHandCameraManagerCosmosEnvCfg",
        "rl_games_cfg_entry_point": f"{_SHADOW_HAND}.agents:rl_games_ppo_camera_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{_SHADOW_HAND}.agents.rsl_rl_ppo_cfg:ShadowHandCameraFFPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="IsaacContrib-PickPlace-Apple-G1-Dex3-Cosmos",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.apple_pick_place_cosmos_env_cfg:G1ApplePickPlaceCosmosEnvCfg",
    },
)
