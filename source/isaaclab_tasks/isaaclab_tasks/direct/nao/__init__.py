# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Humanoid locomotion environment.
"""

import gymnasium as gym

from ..humanoid import agents
from .nao_env import NaoEnv, NaoEnvCfg
from .nao_envb import NaoEnvb, NaoEnvbCfg
from .nao_envc import NaoEnvc, NaoEnvcCfg

##
# Register Gym environments.
##
gym.register(
    id="Nao",
    entry_point="isaaclab_tasks.direct.nao:NaoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": NaoEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
gym.register(
    id="Isaac-Nao-Direct-v0",
    entry_point="isaaclab_tasks.direct.nao:NaoEnvb",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": NaoEnvbCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
gym.register(
    id="Isaac-Nao-Direct-v1",
    entry_point="isaaclab_tasks.direct.nao:NaoEnvc",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": NaoEnvcCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)