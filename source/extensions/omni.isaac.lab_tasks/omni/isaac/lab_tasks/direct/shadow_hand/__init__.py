# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment.
"""

import gymnasium as gym

from . import agents
from .shadow_hand_env import ShadowHandEnv
from .shadow_hand_env_cfg import ShadowHandEnvCfg, ShadowHandOpenAIEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Shadow-Hand-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.shadow_hand:ShadowHandEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.ShadowHandPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Shadow-Hand-OpenAI-FF-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.shadow_hand:ShadowHandEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandOpenAIEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.ShadowHandAsymFFPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Shadow-Hand-OpenAI-LSTM-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.shadow_hand:ShadowHandEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandOpenAIEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)
