# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Warp-first Direct variants of the cube reorientation tasks.

The environments reuse the stable configuration and agent definitions from
:mod:`isaaclab_tasks.core.reorient`. The Allegro warp variant keeps its existing
registration in :mod:`isaaclab_tasks_experimental.direct.allegro_hand` (legacy
in-hand implementation), so only the Shadow variants are registered here.
"""

import gymnasium as gym

##
# Register Gym environments.
##

reorient_warp_entry = "isaaclab_tasks_experimental.direct.reorient.reorient_warp_env:ReorientWarpEnv"
stable_shadow = "isaaclab_tasks.core.reorient.config.shadow_hand"
stable_shadow_agents = f"{stable_shadow}.agents"

gym.register(
    id="Isaac-Reorient-Cube-Shadow-Direct-Warp-v0",
    entry_point=reorient_warp_entry,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{stable_shadow}.shadow_hand_env_cfg:ShadowHandEnvCfg",
        "rl_games_cfg_entry_point": f"{stable_shadow_agents}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{stable_shadow_agents}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
        "skrl_cfg_entry_point": f"{stable_shadow_agents}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Reorient-Cube-Shadow-OpenAI-FF-Direct-Warp-v0",
    entry_point=reorient_warp_entry,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{stable_shadow}.shadow_hand_env_cfg:ShadowHandOpenAIEnvCfg",
        "rl_games_cfg_entry_point": f"{stable_shadow_agents}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{stable_shadow_agents}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
        "skrl_cfg_entry_point": f"{stable_shadow_agents}:skrl_ff_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Reorient-Cube-Shadow-OpenAI-LSTM-Direct-Warp-v0",
    entry_point=reorient_warp_entry,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{stable_shadow}.shadow_hand_env_cfg:ShadowHandOpenAIEnvCfg",
        "rl_games_cfg_entry_point": f"{stable_shadow_agents}:rl_games_ppo_lstm_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{stable_shadow_agents}.rsl_rl_ppo_cfg:ShadowHandAsymLSTMPPORunnerCfg",
    },
)
