# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Shadow Hand environment.
"""

import gymnasium as gym

from isaaclab_tasks.core.reorient.config.shadow_hand import agents

##
# Register Gym environments.
##

reorient_direct_entry = "isaaclab_tasks.core.reorient.reorient_direct_env:ReorientDirectEnv"

gym.register(
    id="Isaac-Reorient-Cube-Shadow-Direct",
    entry_point=reorient_direct_entry,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_env_cfg:ShadowHandEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Reorient-Cube-Shadow-OpenAI-FF-Direct",
    entry_point=reorient_direct_entry,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_env_cfg:ShadowHandOpenAIEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ff_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Reorient-Cube-Shadow-OpenAI-LSTM-Direct",
    entry_point=reorient_direct_entry,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_env_cfg:ShadowHandOpenAIEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)

# -------
# Vision
# -------

gym.register(
    id="Isaac-Reorient-Cube-Shadow-Camera-Direct",
    entry_point=f"{__name__}.shadow_hand_camera_env:ShadowHandCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_camera_env_cfg:ShadowHandCameraEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandCameraFFPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_camera_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Reorient-Cube-Shadow-Camera-Direct-Play",
    entry_point=f"{__name__}.shadow_hand_camera_env:ShadowHandCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_camera_env_cfg:ShadowHandCameraEnvPlayCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandCameraFFPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_camera_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Reorient-Cube-Shadow-Camera-Benchmark-Direct",
    entry_point=f"{__name__}.shadow_hand_camera_env:ShadowHandCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_camera_env_cfg:ShadowHandCameraBenchmarkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandCameraFFPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_camera_cfg.yaml",
    },
)
