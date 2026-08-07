# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contributed Shadow Hand reorientation variants.

The OpenAI variants reproduce `Learning Dexterous In-Hand Manipulation`_ rather than the
plain reorientation task: a reduced actor observation, 20 Hz control, action and
observation noise, and an episode budget spent per goal. Those choices are specific to
that paper's sim-to-real setup, so they live here instead of in the core task.

The feed-forward and recurrent policies share one environment and differ only in the
agent configuration -- select the recurrent one with ``--agent``.

.. _Learning Dexterous In-Hand Manipulation: https://arxiv.org/pdf/1808.00177.pdf
"""

import gymnasium as gym

from isaaclab_tasks.core.reorient.config.shadow_hand import agents as core_agents

from . import agents

gym.register(
    id="IsaacContrib-Reorient-Cube-Shadow-Camera-Benchmark-Direct",
    entry_point="isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_direct_camera_env:ShadowHandCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_camera_benchmark_env_cfg:ShadowHandCameraBenchmarkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{core_agents.__name__}.rsl_rl_ppo_cfg:ShadowHandCameraFFPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{core_agents.__name__}:rl_games_ppo_camera_cfg.yaml",
    },
)

gym.register(
    id="IsaacContrib-Reorient-Cube-Shadow-OpenAI",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_openai_manager_env_cfg:ShadowHandOpenAIManagerEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rl_games_lstm_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
        "rsl_rl_lstm_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymLSTMPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ff_ppo_cfg.yaml",
    },
)

gym.register(
    id="IsaacContrib-Reorient-Cube-Shadow-OpenAI-Direct",
    entry_point=f"{__name__}.shadow_hand_openai_env:ShadowHandOpenAIEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_openai_env_cfg:ShadowHandOpenAIEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rl_games_lstm_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
        "rsl_rl_lstm_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymLSTMPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ff_ppo_cfg.yaml",
    },
)
