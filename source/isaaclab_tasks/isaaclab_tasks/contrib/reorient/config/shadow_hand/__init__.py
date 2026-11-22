# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contributed Shadow Hand tasks: the OpenAI variants, and camera tasks whose image Cosmos generates.

Registering the Cosmos tasks only names their configurations; the Cosmos client is imported when one is selected.
"""

import gymnasium as gym

from isaaclab_tasks.contrib.reorient.config.shadow_hand import agents as contrib_agents

gym.register(
    id="IsaacContrib-Reorient-Cube-Shadow-OpenAI-FF-Direct",
    entry_point="isaaclab_tasks.core.reorient.reorient_direct_env:ReorientDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_openai_env_cfg:ShadowHandOpenAIEnvCfg",
        "rl_games_cfg_entry_point": f"{contrib_agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{contrib_agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
        "skrl_cfg_entry_point": f"{contrib_agents.__name__}:skrl_ff_ppo_cfg.yaml",
    },
)
gym.register(
    id="IsaacContrib-Reorient-Cube-Shadow-OpenAI-LSTM-Direct",
    entry_point="isaaclab_tasks.core.reorient.reorient_direct_env:ReorientDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_openai_env_cfg:ShadowHandOpenAIEnvCfg",
        "rl_games_cfg_entry_point": f"{contrib_agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{contrib_agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymLSTMPPORunnerCfg",
    },
)

_SHADOW_HAND = "isaaclab_tasks.core.reorient.config.shadow_hand"

gym.register(
    id="IsaacContrib-Reorient-Cube-Shadow-Camera-Cosmos-Direct",
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
    id="IsaacContrib-Reorient-Cube-Shadow-Camera-Cosmos",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_cosmos_env_cfg:ShadowHandCameraManagerCosmosEnvCfg",
        "rl_games_cfg_entry_point": f"{_SHADOW_HAND}.agents:rl_games_ppo_camera_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{_SHADOW_HAND}.agents.rsl_rl_ppo_cfg:ShadowHandCameraFFPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
